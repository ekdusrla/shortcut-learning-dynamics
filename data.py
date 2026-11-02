"""Dataset construction for shortcut-learning experiments.

Datasets:
- colored_mnist: 10-way digit classification, one color per digit, configurable
  color-label correlation.
- waterbirds: grodino/waterbirds, image label and background place.

`build_datasets(cfg)` returns {"train", "val", "test"} datasets whose items are
dicts with `pixel_values`, `labels`, `spurious`, `index` (and optionally
`cf_pixel_values`). Each dataset also exposes `labels` and `spurious` arrays and
`num_classes`; both `labels` and `spurious` take values in [0, num_classes).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEEDS = {"train": 100, "val": 200, "test": 300}
NUM_CLASSES = {"colored_mnist": 10, "waterbirds": 2}
# One distinct RGB color per digit; spurious attribute s indexes this palette.
MNIST_COLORS = np.array([
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
], dtype=np.float32)


def image_transform(size: int, train: bool = False, augment: bool = False):
    operations = [transforms.Resize((size, size))]
    if train and augment:
        operations.append(transforms.RandomHorizontalFlip())
    operations += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


class ColoredMNIST(Dataset):
    """Stable colors assigned once, independently for each disjoint split.

    y = digit 0..9. s = color index 0..9 into MNIST_COLORS.
    P(s == y) = correlation; otherwise s is uniform over the other 9 colors.
    Counterfactual repaints the same digit pixels with a fixed, per-example
    random color different from s (no new digit is generated).
    """
    num_classes = NUM_CLASSES["colored_mnist"]

    def __init__(self, base, indices, correlation, seed, size, counterfactual=False):
        if not 0 <= correlation <= 1:
            raise ValueError("correlation must be in [0,1]")
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        digit_labels = np.asarray(base["label"], dtype=np.int64)
        self.labels = digit_labels[self.indices]
        k = self.num_classes
        rng = np.random.default_rng(seed)
        aligned = rng.random(len(self.indices)) < correlation
        # A nonzero offset mod k picks uniformly among the colors other than y (or s).
        other = (self.labels + rng.integers(1, k, len(self.indices))) % k
        self.spurious = np.where(aligned, self.labels, other).astype(np.int64)
        self.cf_spurious = (self.spurious + rng.integers(1, k, len(self.indices))) % k
        self.transform = image_transform(size)
        self.counterfactual = counterfactual

    def __len__(self):
        return len(self.indices)

    def paint(self, grayscale, color):
        a = np.asarray(grayscale.convert("L"), dtype=np.float32) / 255
        rgb = (a[..., None] * MNIST_COLORS[int(color)]).round().astype(np.uint8)
        return self.transform(Image.fromarray(rgb, mode="RGB"))

    def __getitem__(self, i):
        row = self.base[int(self.indices[i])]
        y, s = int(self.labels[i]), int(self.spurious[i])
        out = {"pixel_values": self.paint(row["image"], s),
               "labels": y, "spurious": s, "index": int(self.indices[i])}
        if self.counterfactual:
            out["cf_pixel_values"] = self.paint(row["image"], self.cf_spurious[i])
        return out


class Waterbirds(Dataset):
    """Hugging Face grodino/waterbirds: label 0/1 and place 0/1."""
    num_classes = NUM_CLASSES["waterbirds"]

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
