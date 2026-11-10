"""Offline tests for leakage prevention, balancing, freezing and run safety."""
import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from followup_common import CrossedColors, checkpoint_for, inventory, read_manifest, start_output
from gradcam import GradCAM
from gradcam_dynamics import compare_maps
from probe_dynamics import encode_layers, probe_scores
from evaluate import DecodedCorrelation
from prepare_correlations import training_command


class Raw:
    def __init__(self, offset):
        self.rows = []
        for y in range(10):
            for i in range(2):
                array = np.zeros((28, 28), dtype=np.uint8)
                array[3 + i:20, 5:10 + y] = 80 + y + offset
                self.rows.append(dict(label=y, image=Image.fromarray(array)))

    def __getitem__(self, key):
        return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]


def fixture():
    raw = dict(train=Raw(0), test=Raw(20))
    manifest = dict(version=1, classes=10, seed=42, per_digit=2,
                    config=dict(dataset='colored_mnist', image_size=32, color_mode='foreground'), samples={})
    for split in ['train', 'test']:
        manifest['samples'][split] = [dict(source=split, index=i, label=r['label'],
            pixel_sha256=hashlib.sha256(np.asarray(r['image']).tobytes()).hexdigest())
            for i, r in enumerate(raw[split].rows)]
    return raw, manifest


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        return self.fc(self.pool(self.conv(x).relu()).flatten(1))


class FollowupTests(unittest.TestCase):
    def test_exact_joint_balance_and_same_original(self):
        raw, manifest = fixture()
        ds = CrossedColors(raw, manifest, 'train')
        before = np.asarray(raw['train'][0]['image']).copy()
        cells = np.zeros((10, 10), int)
        for i in range(len(ds)):
            row = ds[i]
            cells[row['labels'], row['spurious']] += 1
            self.assertEqual(row['index'], i // 10)
        np.testing.assert_array_equal(cells, np.full((10, 10), 2))
        self.assertFalse(torch.equal(ds[0]['pixel_values'], ds[1]['pixel_values']))
        np.testing.assert_array_equal(np.asarray(raw['train'][0]['image']), before)

    def test_manifest_leakage_and_duplicate_rejected(self):
        _, manifest = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'protocol.json'
            path.write_text(json.dumps(manifest))
            read_manifest(path)
            bad = copy.deepcopy(manifest)
            bad['samples']['test'][0] = bad['samples']['train'][0]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                read_manifest(path)
            bad = copy.deepcopy(manifest)
            bad['samples']['test'][1] = bad['samples']['test'][0]
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                read_manifest(path)

    def test_changed_source_rejected(self):
        raw, manifest = fixture()
        manifest['samples']['train'][0]['pixel_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'identity'):
            CrossedColors(raw, manifest, 'train')

    def test_freezing_encoding_probes_and_fixed_target_cam(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        raw, manifest = fixture()
        model = Tiny().eval().requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ds = CrossedColors(raw, manifest, 'train')
        features = encode_layers(model, ds, ['conv', 'pool'], torch.device('cpu'), 50, 0)
        self.assertEqual(features['f0'].shape, (200, 4))
        self.assertEqual(features['f1'].shape, (200, 4))
        probe = DecodedCorrelation(4, 10)
        probe.fit(torch.from_numpy(features['f1']), torch.from_numpy(features['labels']), epochs=1)
        with GradCAM(model, 'conv') as cam:
            _, maps = cam(torch.stack([ds[0]['pixel_values'], ds[1]['pixel_values']]), 0)
        self.assertEqual(tuple(maps.shape), (2, 32, 32))
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertEqual(len(model.conv._forward_hooks), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_scores_and_zero_heatmap(self):
        self.assertAlmostEqual(probe_scores(.1)['chance_adjusted_accuracy'], 0)
        self.assertEqual(probe_scores(1)['chance_adjusted_accuracy'], 1)
        self.assertAlmostEqual(probe_scores(.1)['legacy_decoded_correlation'], -.8)
        result = compare_maps(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2), bool))
        self.assertIsNone(result['cam_cosine_similarity'])
        self.assertIsNone(result['aligned_foreground_mass'])
        self.assertEqual(result['cam_mean_absolute_difference'], 0)

    def test_missing_checkpoint_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'run'
            run.mkdir()
            args = argparse.Namespace(runs=[run], epochs=[0, 100], allow_missing=False)
            with self.assertRaises(FileNotFoundError):
                inventory(args, Path(tmp))
            output = Path(tmp) / 'new'
            start_output(output, argparse.Namespace(runs=[run]))
            with self.assertRaises(FileExistsError):
                start_output(output, args)
            (run / 'resnet18_epoch_100.pt').touch()
            self.assertIsNotNone(checkpoint_for(run, 100))
            (run / 'epoch_100.pt').touch()
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                checkpoint_for(run, 100)
            logging.shutdown()  # Release FileHandler before Windows temp cleanup.

    def test_training_config_preserved(self):
        cfg = dict(dataset='colored_mnist', model='resnet18', epochs=100, seed=42,
            image_size=64, train_corr=.8, val_corr=.5, test_corr=.5, mnist_val_size=5000,
            color_mode='foreground', color_pairing='next', lr=.01, momentum=.9,
            scheduler='none', weight_decay=.0001, batch_size=128, num_workers=4,
            features_per_group=32, pretrained=False, augment=False)
        command = training_command(cfg, Path('new-run'))
        self.assertEqual(command[command.index('--scheduler') + 1], 'none')
        self.assertEqual(command[command.index('--batch-size') + 1], '128')
        self.assertNotIn('--pretrained', command)


if __name__ == '__main__':
    unittest.main()
