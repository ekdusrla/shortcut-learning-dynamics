"""Epoch/layer probes on an exactly balanced independent color protocol."""
import csv
import logging
from pathlib import Path

from followup_common import (K, CrossedColors, common_args, inventory, load_model,
                             load_raw, read_manifest, setup, sha256, start_output, write_json)
import numpy as np
import torch
from torch.utils.data import DataLoader
from evaluate import DecodedCorrelation
from shortcut_experiment import FeatureHookCollector, feature_layer_names, seed_everything


@torch.no_grad()
def encode_layers(model, dataset, layers, device, batch_size, workers):
    collector = FeatureHookCollector(model, layers)
    saved = {layer: [] for layer in layers}
    labels, colors, indices, predictions = [], [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    try:
        for b, batch in enumerate(loader):
            collector.features.clear()
            logits = model(batch['pixel_values'].to(device))
            for layer in layers:
                saved[layer].append(collector.features[layer].numpy())
            labels.append(batch['labels'].numpy())
            colors.append(batch['spurious'].numpy())
            indices.append(batch['index'].numpy())
            predictions.append(logits.argmax(1).cpu().numpy())
            if b % 25 == 0:
                logging.info('Feature batch %d/%d', b + 1, len(loader))
    finally:
        collector.close()
    return dict(layers=np.asarray(layers), labels=np.concatenate(labels),
                spurious=np.concatenate(colors), indices=np.concatenate(indices),
                predictions=np.concatenate(predictions),
                **{f'f{i}': np.concatenate(saved[layer]) for i, layer in enumerate(layers)})


def probe_scores(accuracy, classes=K):
    chance = 1 / classes
    return dict(accuracy=float(accuracy), chance_accuracy=chance,
                chance_adjusted_accuracy=(accuracy - chance) / (1 - chance),
                legacy_decoded_correlation=2 * accuracy - 1)


def main():
    parser = common_args(__doc__)
    parser.add_argument('--probe-epochs', type=int, default=100)
    parser.add_argument('--probe-batch-size', type=int, default=256)
    parser.add_argument('--probe-lr', type=float, default=.001)
    parser.add_argument('--probe-weight-decay', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--all-spatial-layers', action='store_true',
                        help='Include every Conv/BN/ReLU/pool module in addition to all snapshot stages')
    args = parser.parse_args()
    if args.probe_epochs < 1 or args.probe_batch_size < 1 or args.probe_lr <= 0 or args.probe_weight_decay < 0:
        parser.error('Invalid probe hyperparameters')
    device = setup(args)
    out = start_output(args.output, args)
    try:
        records = inventory(args, out)
        manifest = read_manifest(args.manifest)
        protocol_sha = sha256(args.manifest)
        write_json(out / 'protocol.json', manifest)
        raw = load_raw()
        datasets = {s: CrossedColors(raw, manifest, s) for s in ['train', 'test']}
        results = []
        for record in records:
            seed_everything(args.seed)
            model, cfg, checkpoint = load_model(record, manifest, device)
            logging.info('Run %s epoch %d device %s', record['run_id'], record['epoch'], device)
            layers = feature_layer_names(model, cfg['model'])
            if args.all_spatial_layers:
                spatial_types = (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU,
                                 torch.nn.MaxPool2d, torch.nn.AdaptiveAvgPool2d)
                layers = list(dict.fromkeys(layers + [n for n, m in model.named_modules() if isinstance(m, spatial_types)]))
            dest = out / record['run_id'] / f'epoch_{record["epoch"]:03d}'
            dest.mkdir(parents=True)
            features = {}
            checkpoint_sha = sha256(checkpoint)
            for split in ['train', 'test']:
                payload = encode_layers(model, datasets[split], layers, device, args.batch_size, args.workers)
                if any(not np.isfinite(payload[f'f{i}']).all() for i in range(len(layers))):
                    raise ValueError('Non-finite hidden features')
                table = np.zeros((K, K), dtype=int)
                np.add.at(table, (payload['labels'], payload['spurious']), 1)
                if not np.all(table == manifest['per_digit']):
                    raise ValueError('Probe dataset lost exact joint balance')
                np.savez_compressed(dest / f'{split}_features.npz', **payload,
                                    protocol_sha256=np.asarray(protocol_sha), checkpoint_sha256=np.asarray(checkpoint_sha))
                features[split] = payload
            for i, layer in enumerate(layers):
                train_x = torch.from_numpy(features['train'][f'f{i}'])
                test_x = torch.from_numpy(features['test'][f'f{i}'])
                for target, key in [('core', 'labels'), ('shortcut', 'spurious')]:
                    # Reset probe RNG for every layer/target/epoch, independent of model construction.
                    seed_everything(args.seed)
                    probe = DecodedCorrelation(train_x.shape[1], K).to(device)
                    probe.fit(train_x, torch.from_numpy(features['train'][key]), epochs=args.probe_epochs,
                              batch_size=args.probe_batch_size, lr=args.probe_lr, weight_decay=args.probe_weight_decay)
                    metrics = probe.evaluate(test_x, torch.from_numpy(features['test'][key]), args.probe_batch_size)
                    probe.save(dest / f'probe_layer_{i:03d}_{target}.pt')
                    row = dict(run_id=record['run_id'], model=cfg['model'], epoch=record['epoch'],
                               layer=layer, target=target, feature_dim=train_x.shape[1],
                               train_n=len(train_x), test_n=len(test_x), classes=K,
                               train_corr=cfg['train_corr'], model_seed=cfg['seed'], probe_seed=args.seed,
                               probe_epochs=args.probe_epochs, probe_lr=args.probe_lr,
                               probe_weight_decay=args.probe_weight_decay, probe_batch_size=args.probe_batch_size,
                               feature_scaling='none', protocol_sha256=protocol_sha,
                               checkpoint_sha256=checkpoint_sha,
                               **probe_scores(metrics['accuracy']))
                    results.append(row)
                    logging.info('%s epoch %d %s %s acc=%.4f adjusted=%.4f', record['run_id'],
                                 record['epoch'], layer, target, row['accuracy'], row['chance_adjusted_accuracy'])
                    with (out / 'probe_results.csv').open('w', newline='', encoding='utf-8') as file:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerows(results)
            write_json(dest / 'metadata.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                                                     config=cfg, layers=layers, protocol_sha256=protocol_sha))
            del model
        write_json(out / 'status.json', dict(status='partial' if len(records) < len(args.runs) * len(args.epochs) else 'complete',
                                             rows=len(results), available_checkpoints=len(records)))
    except Exception as error:
        logging.exception('Probe analysis failed')
        write_json(out / 'status.json', dict(status='failed', error=str(error)))
        raise


if __name__ == '__main__':
    main()
