"""Offline sample/metric equivalence across original and bounded loader settings."""
import unittest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from data import ColoredMNIST
from shortcut_experiment import create_loader, loader_memory_options

class Base:
    def __init__(self):
        self.labels = list(range(10))
    def __getitem__(self, key):
        if key == 'label': return self.labels
        return dict(label=self.labels[key], image=Image.fromarray(
            (np.arange(784).reshape(28,28) * (key+1) % 256).astype('uint8')))

class LoaderMemoryTest(unittest.TestCase):
    def test_identical_data(self):
        for mode in ['background', 'foreground']:
            ds = ColoredMNIST(Base(), np.arange(10), .1, 42, 224,
                              counterfactual=True, color_mode=mode, alt_colors=None)
            old = DataLoader(ds, batch_size=2, num_workers=4, prefetch_factor=2)
            new = create_loader(ds, 2, False, 4, device=torch.device('cpu'))
            self.assertEqual(new.num_workers, 4)
            self.assertEqual(new.prefetch_factor, 1)
            self.assertFalse(new.pin_memory)
            self.assertFalse(new.persistent_workers)
            count = 0
            for a,b in zip(old,new):
                self.assertEqual(a.keys(), b.keys())
                for key in a: self.assertTrue(torch.equal(a[key],b[key]), key)
                self.assertEqual(b['cf_all_pixel_values'].shape[1], 9)
                count += len(b['labels'])
            self.assertEqual(count, 10)
    def test_zero_workers(self):
        self.assertNotIn('prefetch_factor', loader_memory_options(0))

if __name__ == '__main__': unittest.main()
