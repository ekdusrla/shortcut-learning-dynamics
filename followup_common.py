"""Shared, immutable protocols for post-training Colored MNIST analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'outputs' / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch.utils.data import Dataset

EPOCHS = [0, 1, 3, 5, 10, 20, 30, 50, 70, 100]
K = 10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')


def source_hashes():
    root = Path(__file__).resolve().parent
    return {p.name: sha256(p) for p in root.glob('*.py') if p.name in {
        'data.py', 'shortcut_experiment.py', 'evaluate.py', 'gradcam.py',
        'followup_common.py', 'probe_dynamics.py', 'gradcam_dynamics.py', 'integrate_dynamics.py',
        'audit_followup.py', 'prepare_correlations.py', 'validate_followup.py', 'test_followup.py'}}


def start_output(path, args):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(path / 'execution.log', encoding='utf-8'),
                                  logging.StreamHandler(sys.stdout)], force=True)
    write_json(path / 'invocation.json', dict(arguments={k: str(v) if isinstance(v, Path) else v
                                                     for k, v in vars(args).items()},
                                               source_sha256=source_hashes(), torch=torch.__version__,
                                               numpy=np.__version__))
    return path


def common_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; existing paths are rejected')
    p.add_argument('--epochs', type=int, nargs='+', default=EPOCHS)
    p.add_argument('--allow-missing', action='store_true', help='Record missing checkpoints and process available ones')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--offline', action='store_true')
    return p


def setup(args):
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError('batch-size/threads must be positive; workers must be nonnegative')
    if len(set(p.name for p in args.runs)) != len(args.runs):
        raise ValueError('Run directory names must be unique')
    if len(set(args.epochs)) != len(args.epochs) or any(e < 0 for e in args.epochs):
        raise ValueError('epochs must be unique nonnegative integers')
    torch.set_num_threads(args.threads)
    if args.offline:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    from shortcut_experiment import get_device
    return get_device(args.device)


def load_raw():
    from datasets import load_dataset
    return load_dataset('ylecun/mnist')


def checkpoint_for(run, epoch):
    candidates = [run / 'checkpoints' / f'epoch_{epoch:03d}.pt', run / f'epoch_{epoch:03d}.pt']
    candidates += sorted(run.glob(f'*_epoch_{epoch:03d}.pt'))
    found = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    if len(found) > 1:
        raise ValueError(f'Ambiguous checkpoints for {run} epoch {epoch}: {found}')
    return found[0] if found else None


def inventory(args, output):
    found, missing = [], []
    for run in args.runs:
        for epoch in args.epochs:
            path = checkpoint_for(run, epoch)
            record = dict(run=str(run.resolve()), run_id=run.name, epoch=epoch,
                          checkpoint=str(path) if path else None)
            (found if path else missing).append(record)
    write_json(output / 'checkpoint_inventory.json', dict(available=found, missing=missing))
    for record in missing:
        logging.warning('Missing checkpoint: %s epoch %d', record['run_id'], record['epoch'])
    if missing and not args.allow_missing:
        raise FileNotFoundError('Requested checkpoints missing. See checkpoint_inventory.json; use --allow-missing for partial analysis.')
    if not found:
        raise FileNotFoundError('No requested checkpoint exists')
    return found


def read_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest['version'] != 1 or manifest['classes'] != K:
        raise ValueError('Unsupported protocol version/classes')
    identities = []
    for split in ['train', 'test']:
        records = manifest['samples'][split]
        counts = np.bincount([r['label'] for r in records], minlength=K)
        if len(counts) != K or min(counts) < 1 or len(set(counts)) != 1 or counts[0] != manifest['per_digit']:
            raise ValueError(f'Unbalanced or empty {split} labels')
        keys = {(r['source'], r['index']) for r in records}
        if len(keys) != len(records):
            raise ValueError('Duplicate source images')
        identities.append(keys)
    if identities[0] & identities[1]:
        raise ValueError('Source image leakage between probe train and test')
    if {r['pixel_sha256'] for r in manifest['samples']['train']} & {r['pixel_sha256'] for r in manifest['samples']['test']}:
        raise ValueError('Identical grayscale pixels occur in both probe splits')
    return manifest


def check_config(cfg, manifest):
    for key in ['dataset', 'image_size', 'color_mode']:
        if cfg.get(key, 'foreground' if key == 'color_mode' else None) != manifest['config'][key]:
            raise ValueError(f'Protocol/checkpoint mismatch: {key}')
    # Probe-train images must be held out from this checkpoint's model training.
    rng = np.random.default_rng(cfg['seed'])
    val_ids = set(rng.permutation(manifest['raw_train_n'])[:cfg['mnist_val_size']].tolist())
    if any(r['index'] not in val_ids for r in manifest['samples']['train']):
        raise ValueError('Protocol probe-train images overlap this model training split; create a compatible manifest')


def load_model(record, manifest, device):
    from gradcam import load_run
    model, cfg, epoch, path = load_run(Path(record['run']), record['checkpoint'], device)
    if epoch != record['epoch']:
        raise ValueError('Checkpoint epoch does not match filename')
    check_config(cfg, manifest)
    model.eval().requires_grad_(False)
    head = model.fc if hasattr(model, 'fc') else model.classifier
    if head.out_features != K:
        raise ValueError('This analysis requires a 10-class checkpoint')
    return model, cfg, path


class CrossedColors(Dataset):
    """Every selected original image in all ten colors; no resampled assignments."""
    def __init__(self, raw, manifest, split):
        from data import ColoredMNIST
        self.raw = raw
        self.records = manifest['samples'][split]
        cfg = manifest['config']
        self.painters = {}
        for source in set(r['source'] for r in self.records):
            ids = [r['index'] for r in self.records if r['source'] == source]
            self.painters[source] = ColoredMNIST(raw[source], ids, .1, manifest['seed'],
                                                cfg['image_size'], color_mode=cfg['color_mode'])
        for r in self.records:
            gray = raw[r['source']][r['index']]
            digest = hashlib.sha256(np.asarray(gray['image'].convert('L')).tobytes()).hexdigest()
            if int(gray['label']) != r['label'] or digest != r['pixel_sha256']:
                raise ValueError('Source dataset identity differs from saved protocol')

    def __len__(self):
        return len(self.records) * K

    def __getitem__(self, i):
        record = self.records[i // K]
        color = i % K
        gray = self.raw[record['source']][record['index']]['image']
        return dict(pixel_values=self.painters[record['source']].paint(gray, color, i),
                    labels=record['label'], spurious=color, index=record['index'])


def prepare():
    p = argparse.ArgumentParser(description='Freeze a balanced, independent probe and paired-CAM protocol')
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New manifest JSON file')
    p.add_argument('--per-digit', type=int, default=32, help='Distinct original images per digit in EACH split; all 10 colors per image')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if args.per_digit < 1 or not 0 <= args.seed < 2**32:
        p.error('per-digit must be positive; seed must be uint32')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.offline:
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
    cfg = json.loads((args.reference_run / 'config.json').read_text())
    if cfg['dataset'] != 'colored_mnist':
        raise ValueError('Only Colored MNIST is supported')
    raw = load_raw()
    permutation = np.random.default_rng(cfg['seed']).permutation(len(raw['train']))
    val_ids = permutation[:cfg['mnist_val_size']]
    samples = {}
    rng = np.random.default_rng(args.seed)
    for split, source, pool in [('train', 'train', val_ids), ('test', 'test', np.arange(len(raw['test'])))]:
        labels = np.asarray(raw[source]['label'])
        selected = []
        for y in range(K):
            candidates = pool[labels[pool] == y]
            if len(candidates) < args.per_digit:
                raise ValueError(f'Not enough held-out digit {y} images for {split}')
            for index in rng.choice(candidates, args.per_digit, replace=False):
                gray = raw[source][int(index)]['image'].convert('L')
                selected.append(dict(source=source, index=int(index), label=y,
                                     pixel_sha256=hashlib.sha256(np.asarray(gray).tobytes()).hexdigest()))
        samples[split] = selected
    manifest = dict(version=1, classes=K, seed=args.seed, raw_train_n=len(raw['train']),
                    config={k: cfg.get(k, 'foreground') for k in ['dataset', 'image_size', 'color_mode']},
                    reference_model_seed=cfg['seed'], reference_val_size=cfg['mnist_val_size'],
                    colors_per_image=list(range(K)), per_digit=args.per_digit,
                    samples=samples, source_sha256=source_hashes(),
                    description='Probe train uses held-out model validation originals; probe test uses official MNIST test. Every original is crossed with all ten colors.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement even if another job starts.
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
    read_manifest(args.output)
    print(f'Saved {args.output}: {args.per_digit * 100} colored inputs per split; exact 10x10 joint balance')


if __name__ == '__main__':
    prepare()
