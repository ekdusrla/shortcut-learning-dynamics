"""Inventory real checkpoints/NPZ schemas; optionally verify saved split membership."""
import argparse
import logging
from pathlib import Path
import json
import os

from followup_common import EPOCHS, checkpoint_for, start_output, write_json
import numpy as np


def inspect_snapshot(path, datasets=None):
    with np.load(path, allow_pickle=False) as data:
        required = {'layers', 'labels', 'spurious', 'indices'}
        if not required.issubset(data.files):
            raise ValueError(f'Not a snapshot: {path}')
        layers = data['layers'].tolist()
        n = len(data['indices'])
        if not all(len(data[k]) == n for k in ['labels', 'spurious']):
            raise ValueError('Snapshot label/index lengths differ')
        for i in range(len(layers)):
            if data[f'f{i}'].ndim != 2 or len(data[f'f{i}']) != n or not np.isfinite(data[f'f{i}']).all():
                raise ValueError('Invalid pooled feature shape/values')
        result = dict(path=str(path), shapes={k: list(data[k].shape) for k in data.files}, layers=layers,
                      observed_groups=len(set(zip(data['labels'].tolist(), data['spurious'].tolist()))),
                      schema_contains_split_metadata='split' in data.files,
                      suitable_for_independent_probe=False,
                      reason='Historical snapshot has no separate independent probe train/test partition or complete color crossing.')
        if datasets is not None:
            matches = []
            for split, ds in datasets.items():
                lookup = {int(index): pos for pos, index in enumerate(ds.indices)}
                if not all(int(i) in lookup for i in data['indices']):
                    continue
                positions = [lookup[int(i)] for i in data['indices']]
                if np.array_equal(ds.labels[positions], data['labels']) and np.array_equal(ds.spurious[positions], data['spurious']):
                    matches.append(split)
            result['matching_reconstructed_splits'] = matches
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--verify-datasets', action='store_true', help='Load MNIST and match indices/labels/colors against each reconstructed split')
    p.add_argument('--offline', action='store_true')
    args = p.parse_args()
    if args.offline:
        os.environ['HF_DATASETS_OFFLINE'] = os.environ['HF_HUB_OFFLINE'] = '1'
    out = start_output(args.output, args)
    runs = []
    for run in args.runs:
        cfg = json.loads((run / 'config.json').read_text())
        datasets = None
        if args.verify_datasets:
            from data import build_datasets
            datasets = build_datasets(cfg)
        snapshots = [inspect_snapshot(path, datasets) for path in sorted((run / 'features').glob('*.npz'))]
        record = dict(run=str(run.resolve()), config=cfg, snapshots=snapshots,
                      checkpoints={str(epoch): str(path) if (path := checkpoint_for(run, epoch)) else None for epoch in EPOCHS},
                      split_from_training_source='val',
                      split_evidence='shortcut_experiment.run_training passes ds[val] to save_feature_snapshot at epoch 0 and each saved epoch')
        runs.append(record)
        logging.info('%s: %d historical snapshots, %d requested checkpoints', run.name, len(snapshots), sum(v is not None for v in record['checkpoints'].values()))
    write_json(out / 'audit.json', dict(runs=runs, note='No historical 640-row NPZ is fabricated when absent. Matching label/color/index metadata supports split identification; NPZ alone does not encode its origin.'))


if __name__ == '__main__':
    main()
