"""Check saved probe predictions independently with NumPy and verify paired artifacts."""
import argparse
import csv
import json
from pathlib import Path

from followup_common import read_manifest, sha256, write_json
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe-results', type=Path, required=True)
    p.add_argument('--gradcam-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New JSON file')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol_path = args.probe_results / 'protocol.json'
    manifest = read_manifest(protocol_path)
    protocol_sha = sha256(protocol_path)
    assert sha256(args.gradcam_results / 'protocol.json') == protocol_sha
    rows = list(csv.DictReader((args.probe_results / 'probe_results.csv').open(encoding='utf-8')))
    assert rows
    groups = {}
    for row in rows:
        assert row['protocol_sha256'] == protocol_sha
        dest = args.probe_results / row['run_id'] / f'epoch_{int(row["epoch"]):03d}'
        with np.load(dest / 'test_features.npz') as data:
            layer_index = data['layers'].tolist().index(row['layer'])
            features = data[f'f{layer_index}']
            assert np.isfinite(features).all()
            state = torch.load(dest / f'probe_layer_{layer_index:03d}_{row["target"]}.pt', weights_only=True, map_location='cpu')
            weights = state['state_dict']['linear.weight'].numpy()
            bias = state['state_dict']['linear.bias'].numpy()
            labels = data['labels' if row['target'] == 'core' else 'spurious']
            predictions = (features @ weights.T + bias).argmax(1)
            accuracy = float(np.mean(predictions == labels))
            assert abs(accuracy - float(row['accuracy'])) < 1e-12
            assert abs((accuracy - .1) / .9 - float(row['chance_adjusted_accuracy'])) < 1e-12
            assert abs(2 * accuracy - 1 - float(row['legacy_decoded_correlation'])) < 1e-12
        key = (row['run_id'], row['epoch'])
        groups[key] = dest
    for (run, epoch), dest in groups.items():
        for split in ['train', 'test']:
            with np.load(dest / f'{split}_features.npz') as data:
                cells = np.zeros((10, 10), int)
                np.add.at(cells, (data['labels'], data['spurious']), 1)
                assert np.all(cells == manifest['per_digit'])
                expected_ids = np.repeat([r['index'] for r in manifest['samples'][split]], 10)
                np.testing.assert_array_equal(data['indices'], expected_ids)
    cams = list(csv.DictReader((args.gradcam_results / 'gradcam_metrics.csv').open(encoding='utf-8')))
    pair_sets = {}
    for row in cams:
        y, index = int(row['target_class']), int(row['original_index'])
        assert int(row['aligned_color']) == y and int(row['conflicting_color']) == (y + 1) % 10
        stem = f'y{y}_index_{index:05d}'
        dest = args.gradcam_results / row['run_id'] / f'epoch_{int(row["epoch"]):03d}'
        with np.load(dest / f'{stem}.npz') as data:
            a, b = data['aligned_cam'], data['conflicting_cam']
            assert a.shape == b.shape == (manifest['config']['image_size'],) * 2
            for cam in [a, b]:
                assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
            assert abs(float(np.abs(a.astype(float) - b.astype(float)).mean()) - float(row['cam_mean_absolute_difference'])) < 1e-12
        for kind in ['aligned', 'conflicting']:
            for artifact in ['input', 'heatmap', 'overlay']:
                assert (dest / f'{stem}_{kind}_{artifact}.png').is_file()
        pair_sets.setdefault((row['run_id'], row['epoch']), set()).add((index, y))
    assert all(pairs == next(iter(pair_sets.values())) for pairs in pair_sets.values())
    result = dict(status='PASS', probe_rows=len(rows), checkpoint_groups=len(groups),
                  paired_gradcam_rows=len(cams), exact_joint_balance=True,
                  disjoint_originals=True, same_cam_originals_and_targets=True,
                  independent_numpy_probe_predictions=True, protocol_sha256=protocol_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    print(result)


if __name__ == '__main__':
    main()
