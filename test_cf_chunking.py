"""CPU equivalence against the original full-batch counterfactual forwards."""
import math
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import shortcut_experiment as exp

class Samples(Dataset):
    num_classes = 10
    groups = [(y, s) for y in range(10) for s in range(10)]
    def __init__(self):
        g = torch.Generator().manual_seed(42)
        self.x = torch.randn(7, 3, 4, 4, generator=g)
        self.cf = torch.randn(7, 9, 3, 4, 4, generator=g)
    def __len__(self): return 7
    def __getitem__(self, i):
        return dict(pixel_values=self.x[i], labels=torch.tensor(i),
                    spurious=torch.tensor((i+1)%10), cf_pixel_values=self.cf[i, 0],
                    cf_all_pixel_values=self.cf[i],
                    cf_all_spurious=torch.tensor([j for j in range(10) if j != (i+1)%10]))

class CheckedModel(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 10))
        self.sizes = []
    def forward(self, x):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.sizes.append(len(x))
        return super().forward(x)

class ChunkingTest(unittest.TestCase):
    def test_predictions_and_all_metrics(self):
        torch.manual_seed(42)
        model = CheckedModel().eval()
        ds = Samples()
        images = ds.cf.flatten(0, 1)
        with torch.inference_mode():
            reference_logits = model(images)
        for size in [1, 5, 9, 128]:
            model.sizes.clear()
            actual = exp.counterfactual_predictions(model, images, 'cpu', chunk_size=size)
            self.assertTrue(torch.equal(actual, reference_logits.argmax(1)))
            self.assertLessEqual(max(model.sizes), size)
            self.assertEqual(sum(model.sizes), 63)
            with torch.inference_mode():
                chunk_logits = torch.cat([model(x) for x in images.split(size)])
            torch.testing.assert_close(chunk_logits, reference_logits)
        def original(model, images, device, amp=False, chunk_size=128):
            return model(images.to(device)).argmax(1)
        with patch.object(exp, 'counterfactual_predictions', original):
            reference = exp.evaluate(model, DataLoader(ds, batch_size=4), 'cpu')
        for size in [1, 5, 128]:
            actual = exp.evaluate(model, DataLoader(ds, batch_size=4), 'cpu', cf_chunk_size=size)
            self.assertEqual(reference.keys(), actual.keys())
            for key, expected in reference.items():
                if math.isnan(expected): self.assertTrue(math.isnan(actual[key]), key)
                else: self.assertEqual(expected, actual[key], key)
    def test_invalid_size(self):
        with self.assertRaises(ValueError):
            exp.counterfactual_predictions(CheckedModel(), torch.zeros(1,3,4,4), 'cpu', chunk_size=0)

if __name__ == '__main__': unittest.main()
