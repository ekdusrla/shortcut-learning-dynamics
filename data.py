"""Dataset construction for shortcut-learning experiments.

Datasets:
- colored_mnist: 10-way digit classification. Each digit has a main and an
  alternate color (see COLOR_PAIRINGS); the main:alternate ratio is configurable
  (see CORR_PRESETS), as is where the color is painted (see COLOR_MODES).
- waterbirds: grodino/waterbirds, image label and background place.

`build_datasets(cfg)` returns {"train", "val", "test"} datasets whose items are
dicts with `pixel_values`, `labels`, `spurious`, `index` (and optionally
`cf_pixel_values`). Each dataset also exposes `labels` and `spurious` arrays and
`num_classes`; both `labels` and `spurious` take values in [0, num_classes).
`groups` lists the (y, s) pairs that can occur by construction.

Run `python data.py` to save example images for each Colored MNIST color mode.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEEDS = {"train": 100, "val": 200, "test": 300}
NUM_CLASSES = {"colored_mnist": 10, "waterbirds": 2}
# Fixed RGB palette; spurious attribute s indexes it and digit k's main color is k.
# Rainbow (ROYGBIV) plus three chromatic colors, so every color shows on black.
MNIST_COLOR_NAMES = ("red", "orange", "yellow", "green", "blue", "indigo", "violet",
                     "pink", "cyan", "brown")
MNIST_COLORS = np.array([
    (255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255),
    (75, 0, 130), (148, 0, 211), (255, 105, 180), (0, 255, 255), (160, 82, 45),
], dtype=np.float32)
# P(main color) for the training split; val/test default to 0.5 (no color cue
# beyond the two-color pairing).
CORR_PRESETS = {"strong": .95, "medium": .80, "weak": .65}
# How digit k's alternate color is chosen:
#   next: color (k + 1) % 10.
#   random: a seeded derangement, so no digit's alternate equals its main color.
COLOR_PAIRINGS = ("next", "random")
WHITE = np.array((255, 255, 255), dtype=np.float32)
BLACK = np.zeros(3, dtype=np.float32)
# Backgrounds are dimmed so the digit always stays brighter than its surroundings.
BACKGROUND_SCALE = 0.5
# Where the shortcut color is painted; every mode keeps the digit shape intact.
#   foreground: colored digit on black.
#   background: white digit on a (dimmed) colored background.
#   border: white digit with a colored outline, on black.
#   foreground_background: colored digit on a (dimmed) background of a second
#     color, MNIST_COLORS[(s + 5) % 10], so digit and background never share a hue.
COLOR_MODES = ("foreground", "background", "border", "foreground_background")


def alternate_colors(pairing="next", seed=0, k=10):
    """Return alt[k]: the alternate color index for each digit (never k itself)."""
    if pairing == "next":
        return (np.arange(k) + 1) % k
    if pairing == "random":
        rng = np.random.default_rng(seed)
        while True:
            alt = rng.permutation(k)
            if not np.any(alt == np.arange(k)):
                return alt
    raise ValueError(f"color pairing must be one of {COLOR_PAIRINGS}")


def image_transform(size: int, train: bool = False, augment: bool = False):
    operations = [transforms.Resize((size, size))]
    if train and augment:
        operations.append(transforms.RandomHorizontalFlip())
    operations += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


class ColoredMNIST(Dataset):
    """Stable colors assigned once, independently for each disjoint split.

    y = digit 0..9. s = color index 0..9 into MNIST_COLORS.
    Each digit y uses only two colors: main color y with probability
    `correlation`, otherwise alternate color `alt_colors[y]`.
    `color_mode` (one of COLOR_MODES) decides which region carries the color.
    Counterfactual repaints the same digit pixels with the digit's other color
    (main <-> alternate); no new digit is generated.
    """
    num_classes = NUM_CLASSES["colored_mnist"]

    def __init__(self, base, indices, correlation, seed, size, counterfactual=False,
                 color_mode="foreground", alt_colors=None):
        if not 0 <= correlation <= 1:
            raise ValueError("correlation must be in [0,1]")
        if color_mode not in COLOR_MODES:
            raise ValueError(f"color_mode must be one of {COLOR_MODES}")
        self.color_mode = color_mode
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        digit_labels = np.asarray(base["label"], dtype=np.int64)
        self.labels = digit_labels[self.indices]
        k = self.num_classes
        alt = alternate_colors() if alt_colors is None else np.asarray(alt_colors, dtype=np.int64)
        self.groups = [(y, int(s)) for y in range(k) for s in (y, alt[y])]
        rng = np.random.default_rng(seed)
        aligned = rng.random(len(self.indices)) < correlation
        self.spurious = np.where(aligned, self.labels, alt[self.labels]).astype(np.int64)
        self.cf_spurious = np.where(aligned, alt[self.labels], self.labels).astype(np.int64)
        self.transform = image_transform(size)
        self.counterfactual = counterfactual

    def __len__(self):
        return len(self.indices)

    def paint(self, grayscale, color):
        gray = grayscale.convert("L")
        a = np.asarray(gray, dtype=np.float32)[..., None] / 255  # digit alpha
        c = MNIST_COLORS[int(color)]
        if self.color_mode == "foreground":
            rgb = a * c + (1 - a) * BLACK
        elif self.color_mode == "background":
            rgb = a * WHITE + (1 - a) * BACKGROUND_SCALE * c
        elif self.color_mode == "border":
            # Outline = 3x3 dilation minus the digit, so a + ring <= 1.
            dilated = np.asarray(gray.filter(ImageFilter.MaxFilter(3)),
                                 dtype=np.float32)[..., None] / 255
            ring = dilated - a
            rgb = a * WHITE + ring * c + (1 - dilated) * BLACK
        else:  # foreground_background
            bg = MNIST_COLORS[(int(color) + 5) % self.num_classes]
            rgb = a * c + (1 - a) * BACKGROUND_SCALE * bg
        rgb = rgb.round().clip(0, 255).astype(np.uint8)
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
    groups = [(y, s) for y in range(2) for s in range(2)]

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
        alt = alternate_colors(cfg.get("color_pairing", "next"), cfg["seed"],
                               NUM_CLASSES["colored_mnist"])
        return {name: ColoredMNIST(
            base=raw["test"] if name == "test" else raw["train"],
            indices=ids[name], correlation=cfg[f"{name}_corr"],
            seed=cfg["seed"] + SPLIT_SEEDS[name], size=cfg["image_size"],
            counterfactual=(name != "train"),
            color_mode=cfg.get("color_mode", "foreground"), alt_colors=alt) for name in ("train", "val", "test")}

    if cfg["dataset"] == "waterbirds":
        raw = load_dataset("grodino/waterbirds")
        return {name: Waterbirds(raw["validation" if name == "val" else name],
                                size=cfg["image_size"], train=(name == "train"),
                                augment=cfg["augment"])
                for name in ("train", "val", "test")}
    raise ValueError(f"Unsupported dataset: {cfg['dataset']}")


def parse_corr(value):
    """argparse type: a CORR_PRESETS name or a float in [0, 1]."""
    if value in CORR_PRESETS:
        return CORR_PRESETS[value]
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a float or one of {list(CORR_PRESETS)}, got {value!r}") from None
    if not 0 <= x <= 1:
        raise argparse.ArgumentTypeError("correlation must be in [0, 1]")
    return x


def to_image(pixel_values):
    """Undo ImageNet normalization: [3,H,W] tensor -> [H,W,3] float array in [0,1]."""
    x = pixel_values.permute(1, 2, 0).numpy()
    return (x * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)).clip(0, 1)


def example_cfg(cfg=None):
    """Training-default Colored MNIST config, overridden by `cfg`."""
    return {"dataset": "colored_mnist", "seed": 42, "mnist_val_size": 5000,
            "train_corr": CORR_PRESETS["strong"], "val_corr": .50, "test_corr": .50,
            "image_size": 64, **(cfg or {})}


def save_colored_mnist_examples(out_file, color_modes=COLOR_MODES, split="train",
                                per_digit=1, cfg=None):
    """Save a grid of Colored MNIST samples: one row block per color mode.

    Columns are digits 0..9 and the same samples are shown in every mode, so
    only the coloring differs between row blocks. Titles give target y and
    shortcut color s; red titles mark conflicting samples (s != y).
    """
    import matplotlib.pyplot as plt

    cfg = example_cfg(cfg)
    k = NUM_CLASSES["colored_mnist"]
    rows = len(color_modes) * per_digit
    fig, axes = plt.subplots(rows, k, figsize=(k * 1.1, rows * 1.3), squeeze=False,
                             layout="constrained")
    for m, mode in enumerate(color_modes):
        ds = build_datasets({**cfg, "color_mode": mode})[split]
        for d in range(k):
            chosen = np.flatnonzero(ds.labels == d)[:per_digit]
            for r in range(per_digit):
                ax = axes[m * per_digit + r, d]
                ax.axis("off")
                if r >= len(chosen):
                    continue
                item = ds[int(chosen[r])]
                y, s = item["labels"], item["spurious"]
                ax.imshow(to_image(item["pixel_values"]))
                ax.set_title(f"y={y} s={s}", fontsize=8, color="red" if s != y else "black")
        axes[m * per_digit, 0].text(-0.15, 0.5, mode, transform=axes[m * per_digit, 0].transAxes,
                                    rotation=90, ha="right", va="center", fontsize=9)
    fig.suptitle(f"Colored MNIST / {split} split / corr={cfg[f'{split}_corr']}")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved {out_file}")


def save_corr_comparison(out_file, color_mode="foreground", samples=20, cols=5, cfg=None):
    """Compare CORR_PRESETS on the train split: one row per preset, one column per digit.

    Each cell is a mosaic of the first `samples` training images of that digit,
    so the share of alternate-color digits is visible at a glance. The same
    samples appear in every row; a sample with the alternate color under a
    stronger preset keeps it under weaker ones (the Bernoulli draws are shared).
    """
    import matplotlib.pyplot as plt

    cfg = example_cfg({"image_size": 28, **(cfg or {}), "color_mode": color_mode})
    k = NUM_CLASSES["colored_mnist"]
    fig, axes = plt.subplots(len(CORR_PRESETS), k, figsize=(k * 1.4, len(CORR_PRESETS) * 1.45),
                             squeeze=False, layout="constrained")
    for r, (name, corr) in enumerate(CORR_PRESETS.items()):
        ds = build_datasets({**cfg, "train_corr": corr})["train"]
        for d in range(k):
            chosen = np.flatnonzero(ds.labels == d)[:samples]
            tiles = [to_image(ds[int(i)]["pixel_values"]) for i in chosen]
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
            mosaic = np.concatenate([np.concatenate(tiles[j:j + cols], axis=1)
                                     for j in range(0, len(tiles), cols)], axis=0)
            n_alt = int((ds.spurious[chosen] != d).sum())
            ax = axes[r, d]
            ax.imshow(mosaic)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{d}: alt {n_alt}/{len(chosen)}", fontsize=8)
        actual = float((ds.labels == ds.spurious).mean())
        axes[r, 0].set_ylabel(f"{name}\n{corr:.0%}:{1 - corr:.0%}\nactual {actual:.3f}", fontsize=9)
    fig.suptitle(f"Colored MNIST train split / main:alternate color ratio / {color_mode} / "
                 f"pairing={cfg.get('color_pairing', 'next')}")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Save Colored MNIST example images: one per color mode, plus a "
                    "strong/medium/weak correlation comparison.")
    parser.add_argument("--output", default="images/colored_mnist_examples.png")
    parser.add_argument("--color-modes", nargs="+", choices=COLOR_MODES, default=list(COLOR_MODES))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--per-digit", type=int, default=1, help="samples shown per digit and mode")
    parser.add_argument("--corr", type=parse_corr, default=None,
                        help="P(main color) for the chosen split: a float or one of "
                             f"{list(CORR_PRESETS)} (default: training default)")
    parser.add_argument("--color-pairing", choices=COLOR_PAIRINGS, default="next")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--corr-output", default="images/colored_mnist_corr_comparison.png",
                        help="strong/medium/weak comparison image; pass '' to skip")
    parser.add_argument("--corr-mode", choices=COLOR_MODES, default="foreground",
                        help="color mode used in the comparison image")
    parser.add_argument("--corr-samples", type=int, default=20,
                        help="samples per digit in each comparison cell")
    args = parser.parse_args()
    if args.per_digit <= 0 or args.corr_samples <= 0:
        parser.error("--per-digit and --corr-samples must be positive")
    cfg = {"seed": args.seed, "image_size": args.image_size, "color_pairing": args.color_pairing}
    if args.corr is not None:
        cfg[f"{args.split}_corr"] = args.corr
    save_colored_mnist_examples(args.output, args.color_modes, args.split, args.per_digit, cfg)
    if args.corr_output:
        save_corr_comparison(args.corr_output, args.corr_mode, args.corr_samples,
                             cfg={"seed": args.seed, "color_pairing": args.color_pairing})


if __name__ == "__main__":
    main()
