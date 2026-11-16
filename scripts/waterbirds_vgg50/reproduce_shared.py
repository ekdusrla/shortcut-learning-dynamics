"""Verify public lightweight exports and reproduce CSV figures without torch or raw data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scripts.waterbirds_vgg50.analyze_history import CODES, REPO, derive, sha, tables, plots


def verify():
    root = REPO / 'reports/waterbirds_vgg50'
    registry = json.loads((root / 'shared/export_manifest.json').read_text())
    for entry in registry['files']:
        path = REPO / entry['published']
        if sha(path) != entry['published_sha256']:
            raise ValueError(f'Export hash mismatch: {path}')
    frames = []
    for code in CODES:
        run = root / f'shared/source_metrics/A_corr{code}_seed42_e50_contentv3'
        frame = pd.read_csv(run / 'history.csv')
        if frame.epoch.tolist() != list(range(51)):
            raise ValueError('Expected epochs 0..50')
        cfg = json.loads((run / 'config.json').read_text())
        required = dict(model='vgg11_bn', pretrained=False, epochs=50, seed=42, batch_size=128,
                        image_size=224, lr=.01, momentum=.9, weight_decay=.0001,
                        scheduler='none', amp=True, train_corr=int(code)/100, val_corr=.5, test_corr=.5)
        if any(cfg.get(k) != v for k, v in required.items()):
            raise ValueError('Unexpected published config')
        frame['condition'] = 'A_corr' + code
        frame['corr'] = int(code)/100
        frame['run'] = f'runs/waterbirds_vgg50/A_corr{code}_seed42_e50_contentv3'
        frame = derive(frame)
        for epoch in range(51):
            metrics = json.loads((run / f'epoch_metrics/epoch_{epoch:03d}.json').read_text())['splits']['val']
            for metric in ('acc', 'wga', 'cf_flip_rate', 'cf_correct_to_incorrect_rate'):
                if not np.isclose(frame.loc[epoch, 'val_'+metric], metrics[metric], atol=1e-12, rtol=0):
                    raise ValueError('Published history/epoch JSON mismatch')
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    expected = pd.read_csv(root / 'analysis/merged_history.csv')
    pd.testing.assert_frame_equal(data, expected, check_dtype=False, atol=1e-12, rtol=0)
    return data, len(registry['files'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, help='Optional fresh output directory for regenerated tables/figures')
    args = p.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(args.output)
    data, count = verify()
    if args.output:
        args.output.mkdir(parents=True)
        data.to_csv(args.output/'merged_history.csv', index=False)
        tables(data, args.output)
        plots(data, args.output)
    print(f'PASS: {count} published hashes, 153 epoch rows, three configs, epoch JSON/CSV and merged history')


if __name__ == '__main__':
    main()
