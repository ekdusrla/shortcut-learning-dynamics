"""Reproduce epoch-100 Grad-CAM and held-out decoded correlation artifacts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'shortcut_results' / 'analysis_epoch100'
OUT.mkdir(parents=True, exist_ok=True)
os.environ['MPLCONFIGDIR'] = str(OUT / '.matplotlib')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import csv
import json
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import build_datasets
from shortcut_experiment import seed_everything
from evaluate import DecodedCorrelation
from gradcam import GradCAM, default_layer, load_run, save_visualization, unnormalize_image
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(4)
    device = torch.device('cpu')
    results = []
    for run_name, filename in [
        ('resnet18_corr095_seed42_e100', 'resnet18_epoch_100.pt'),
        ('vgg11_corr095_seed42_e100', 'vgg11_epoch_100.pt'),
    ]:
        seed_everything(42)
        run = ROOT / 'shortcut_results' / 'runs' / run_name
        dest = OUT / run_name
        dest.mkdir(exist_ok=True)
        model, cfg, epoch, checkpoint = load_run(run, str(run / filename), device)
        assert epoch == 100
        model.requires_grad_(False)
        print(run_name, 'loaded epoch', epoch, flush=True)
        datasets = build_datasets(cfg)
        # Only original inputs are needed; skip nine unused recolors per input.
        for ds in datasets.values():
            ds.counterfactual = False
        encoded = {}
        for split in ['val', 'test']:
            ds = datasets[split]
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
            x, targets = DecodedCorrelation.encode(model, loader, device)
            encoded[split] = (x, targets)
            np.savez_compressed(dest / f'{split}_features.npz', features=x.numpy(),
                                labels=targets['labels'].numpy(),
                                spurious=targets['spurious'].numpy(), indices=ds.indices)
            print(run_name, split, tuple(x.shape), flush=True)
        row = dict(model=cfg['model'], epoch=epoch, probe_train_split='val',
                   probe_train_n=len(encoded['val'][0]), test_n=len(encoded['test'][0]))
        for name, key in [('core', 'labels'), ('spurious', 'spurious')]:
            probe = DecodedCorrelation(encoded['val'][0].shape[1], 10)
            probe.fit(encoded['val'][0], encoded['val'][1][key], epochs=100)
            probe.save(dest / f'{name}_probe.pt')
            metrics = probe.evaluate(encoded['test'][0], encoded['test'][1][key])
            row[f'{name}_accuracy'] = metrics['accuracy']
            row[f'{name}_decoded_correlation'] = metrics['decoded_correlation']
            print(name, metrics, flush=True)
        head = getattr(model, 'fc', None)
        if head is None:
            head = model.classifier
        with torch.no_grad():
            pred = head(encoded['test'][0]).argmax(1)
        row['test_classifier_accuracy'] = float((pred == encoded['test'][1]['labels']).float().mean())
        layer = default_layer(model, cfg['model'])
        ds = datasets['test']
        records = []
        fig, axes = plt.subplots(10, 4, figsize=(10, 23))
        with GradCAM(model, layer) as gradcam:
            for digit in range(10):
                for col, aligned in enumerate([True, False]):
                    indices = np.flatnonzero((ds.labels == digit) & ((ds.spurious == digit) == aligned))
                    i = int(indices[0])
                    sample = ds[i]
                    tensor = sample['pixel_values']
                    logits, cams = gradcam(tensor.unsqueeze(0))
                    probabilities = logits.softmax(1)[0]
                    predicted = int(probabilities.argmax())
                    image, cam = unnormalize_image(tensor), cams[0].numpy()
                    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
                    stem = f'digit_{digit}_{"aligned" if aligned else "conflicting"}_test_{i:05d}'
                    title = f'{cfg["model"]} epoch 100 | {layer} | y={digit} s={sample["spurious"]} pred={predicted}'
                    save_visualization(image, cam, dest / 'gradcam' / f'{stem}.png', title)
                    np.save(dest / 'gradcam' / f'{stem}_heatmap.npy', cam)
                    meta = dict(sample_index=i, original_index=int(sample['index']), label=digit,
                                spurious=int(sample['spurious']), predicted_class=predicted,
                                target_class=predicted, layer=layer, aligned=aligned,
                                probabilities=probabilities.tolist(), heatmap_max=float(cam.max()))
                    (dest / 'gradcam' / f'{stem}.json').write_text(json.dumps(meta, indent=2))
                    records.append(meta)
                    overlay = np.clip(.55 * image + .45 * plt.get_cmap('jet')(cam)[..., :3], 0, 1)
                    for ax, pixels, label in [(axes[digit, col * 2], image, 'Input'),
                                               (axes[digit, col * 2 + 1], overlay, 'CAM')]:
                        ax.imshow(pixels)
                        ax.set_title(f'{label}: y={digit} s={sample["spurious"]} p={predicted}', fontsize=9)
                        ax.axis('off')
        fig.suptitle(f'{cfg["model"]}, epoch 100, {layer}\nAligned (left) / conflicting (right); predicted-class Grad-CAM')
        fig.tight_layout(rect=(0, 0, 1, .97))
        fig.savefig(dest / 'gradcam_overview.png', dpi=150)
        plt.close(fig)
        provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                          config=cfg, probe_epochs=100, probe_lr=.001, probe_weight_decay=.0001,
                          probe_batch_size=256, seed=42, feature_layer='final classifier input',
                          gradcam_layer=layer, gradcam_samples=records, metrics=row,
                          decoded_correlation_definition='2 * accuracy - 1; ten-class chance baseline -0.8')
        (dest / 'metadata.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
        results.append(row)
        with (OUT / 'decoded_correlation.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(results)
        print('SAVED', dest, flush=True)


if __name__ == '__main__':
    main()
