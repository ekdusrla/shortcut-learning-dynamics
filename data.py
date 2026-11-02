"""Dataset construction for shortcut-learning experiments.

Datasets:
- colored_mnist: binary classification (0-4 vs 5-9), configurable color-label correlation.
- waterbirds: grodino/waterbirds, image label and background place.

`build_datasets(cfg)` returns {"train", "val", "test"} datasets whose items are
dicts with `pixel_values`, `labels`, `spurious`, `index` (and optionally
`cf_pixel_values`). Each dataset also exposes `labels` and `spurious` arrays.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEEDS = {"train": 100, "val": 200, "test": 300}


def image_transform(size: int, train: bool = False, augment: bool = False):
    operations = [transforms.Resize((size, size))]
    if train and augment:
        operations.append(transforms.RandomHorizontalFlip())
    operations += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


class ColoredMNIST(Dataset):
    """Stable colors assigned once, independently for each disjoint split.

    y = 0 for digits 0..4, 1 for 5..9. s = 0 red / 1 green.
    P(s == y) = correlation. Counterfactual swaps red and green while
    preserving the original digit pixels (no new digit is generated).
    """
    def __init__(self, base, indices, correlation, seed, size, counterfactual=False):
        if not 0 <= correlation <= 1:
            raise ValueError("correlation must be in [0,1]")
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        digit_labels = np.asarray(base["label"], dtype=np.int64)
        self.labels = (digit_labels[self.indices] >= 5).astype(np.int64)
        rng = np.random.default_rng(seed)
        aligned = rng.random(len(self.indices)) < correlation
        self.spurious = np.where(aligned, self.labels, 1 - self.labels).astype(np.int64)
        self.transform = image_transform(size)
        self.counterfactual = counterfactual

    def __len__(self):
        return len(self.indices)

    def paint(self, grayscale, color):
        a = np.asarray(grayscale.convert("L"), dtype=np.uint8)
        rgb = np.zeros((*a.shape, 3), dtype=np.uint8)
        rgb[..., int(color)] = a  # red or green channel
        return self.transform(Image.fromarray(rgb, mode="RGB"))

    def __getitem__(self, i):
        row = self.base[int(self.indices[i])]
        y, s = int(self.labels[i]), int(self.spurious[i])
        out = {"pixel_values": self.paint(row["image"], s),
               "labels": y, "spurious": s, "index": int(self.indices[i])}
        if self.counterfactual:
            out["cf_pixel_values"] = self.paint(row["image"], 1 - s)
        return out


class Waterbirds(Dataset):
    """Hugging Face grodino/waterbirds: label 0/1 and place 0/1."""
    def __init__(self, hf_split, size, train=False, augment=False):
        self.base = hf_split
        self.labels = np.asarray(hf_split["label"], dtype=np.int64)
        self.spurious = np.asarray(hf_split["place"], dtype=np.int64)
        self.transform = image_transform(size, train=train, augment=augment)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        row = self.base[int(i)]
        return {"pixel_values": self.transform(row["image"].convert("RGB")),
                "labels": int(self.labels[i]), "spurious": int(self.spurious[i]),
                "index": int(i)}


def build_datasets(cfg):
    # Import only when loading a real HF dataset, enabling offline code sanity tests.
    from datasets import load_dataset

    if cfg["dataset"] == "colored_mnist":
        raw = load_dataset("ylecun/mnist")
        n = len(raw["train"])
        rng = np.random.default_rng(cfg["seed"])
        indices = rng.permutation(n)
        n_val = int(cfg["mnist_val_size"])
        if n_val <= 0 or n_val >= n:
            raise ValueError(f"mnist_val_size must be between 1 and {n-1}")
        ids = {"train": indices[n_val:], "val": indices[:n_val],
               "test": np.arange(len(raw["test"]))}
        return {name: ColoredMNIST(
            base=raw["test"] if name == "test" else raw["train"],
            indices=ids[name], correlation=cfg[f"{name}_corr"],
            seed=cfg["seed"] + SPLIT_SEEDS[name], size=cfg["image_size"],
            counterfactual=(name != "train")) for name in ("train", "val", "test")}

    if cfg["dataset"] == "waterbirds":
        raw = load_dataset("grodino/waterbirds")
        return {name: Waterbirds(raw["validation" if name == "val" else name],
                                size=cfg["image_size"], train=(name == "train"),
                                augment=cfg["augment"])
                for name in ("train", "val", "test")}
    raise ValueError(f"Unsupported dataset: {cfg['dataset']}")
