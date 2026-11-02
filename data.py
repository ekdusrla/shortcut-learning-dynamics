"""Dataset construction for shortcut-learning experiments.

Datasets:
- colored_mnist: 10-way digit classification. Each digit has a main and an
  alternate color (see COLOR_PAIRINGS); the main:alternate ratio is configurable
  (see CORR_PRESETS), as is where the color is painted (see COLOR_MODES).
- waterbirds: CUB-200-2011 birds composited onto Places365 land/water scenes on the
  fly (group_DRO recipe); configurable bird-background correlation (CORR_PRESETS).
  Raw files come from download_waterbirds.sh.

`build_datasets(cfg)` returns {"train", "val", "test"} datasets whose items are
dicts with `pixel_values`, `labels`, `spurious`, `index` (and optionally
`cf_pixel_values`). Each dataset also exposes `labels` and `spurious` arrays and
`num_classes`; both `labels` and `spurious` take values in [0, num_classes).
`groups` lists the (y, s) pairs that can occur by construction. Setting a
dataset's `stress` to a Stress corrupts every image it returns (stress test).

Run `python data.py` to save example images of both datasets into images/.
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
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
# Training-split correlation: P(main color) for colored_mnist, P(place == label)
# for waterbirds. val/test default to 0.5.
CORR_PRESETS = {"strong": .95, "medium": .80, "weak": .65}
# How digit k's alternate color is chosen:
#   next: color (k + 1) % 10.
#   random: a seeded derangement, so no digit's alternate equals its main color.
COLOR_PAIRINGS = ("next", "random")
# Waterbirds raw data (download_waterbirds.sh): <root>/CUB_200_2011 and <root>/places365.
WATERBIRDS_ROOT = "data/raw"
WATERBIRDS_VAL_FRAC = 0.2  # share of CUB train images used for validation
PLACES_URL = "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
PLACES_PREFIX = "data_256"
# Index = place label: 0 land, 1 water.
TARGET_PLACES = [["bamboo_forest", "forest/broadleaf"], ["ocean", "lake/natural"]]
# Seabirds and waterfowl; a species is a waterbird if its name contains one of these.
WATER_BIRDS = ["Albatross", "Auklet", "Cormorant", "Frigatebird", "Fulmar", "Gull",
               "Jaeger", "Kittiwake", "Pelican", "Puffin", "Tern", "Gadwall", "Grebe",
               "Mallard", "Merganser", "Guillemot", "Pacific_Loon"]
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


# Test-time corruptions, severity 1..5 -> parameter (ImageNet-C style).
# gaussian_noise: std on [0, 1] pixels (ImageNet-C values).
# gaussian_blur: sigma as a fraction of the image side (ImageNet-C's 1..6 px at 224).
# crop: kept side fraction of a random square crop, resized back to full size.
STRESS_SEVERITIES = {
    "gaussian_noise": (.08, .12, .18, .26, .38),
    "gaussian_blur": tuple(px / 224 for px in (1, 2, 3, 4, 6)),
    "crop": (.9, .8, .7, .6, .5),
}
STRESS_TYPES = tuple(STRESS_SEVERITIES)


class Stress:
    """Deterministic corruption of a resized PIL image.

    `key` (the item index) seeds the noise and the crop position, so a sample
    and its counterfactual get exactly the same corruption.
    """

    def __init__(self, kind, severity, seed=0):
        if kind not in STRESS_SEVERITIES:
            raise ValueError(f"stress type must be one of {STRESS_TYPES}")
        if not 1 <= severity <= len(STRESS_SEVERITIES[kind]):
            raise ValueError(f"severity must be in 1..{len(STRESS_SEVERITIES[kind])}")
        self.kind, self.severity, self.seed = kind, int(severity), seed
        self.param = STRESS_SEVERITIES[kind][self.severity - 1]

    def __call__(self, image, key):
        rng = np.random.default_rng([self.seed, int(key)])
        if self.kind == "gaussian_noise":
            x = np.asarray(image, dtype=np.float32) / 255
            x = x + rng.normal(0, self.param, x.shape)
            return Image.fromarray((x.clip(0, 1) * 255).round().astype(np.uint8))
        if self.kind == "gaussian_blur":
            return image.filter(ImageFilter.GaussianBlur(self.param * min(image.size)))
        w, h = image.size
        cw, ch = max(1, round(self.param * w)), max(1, round(self.param * h))
        left, top = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
        return image.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

    def describe(self, size):
        if self.kind == "gaussian_noise":
            return f"std {self.param:g}"
        if self.kind == "gaussian_blur":
            return f"sigma {self.param * size:.2g}px"
        return f"keep {self.param:.0%}"

    def __repr__(self):
        return f"{self.kind}@{self.severity}"


def image_transform(size: int, train: bool = False, augment: bool = False):
    operations = [transforms.Resize((size, size))]
    if train and augment:
        operations.append(transforms.RandomHorizontalFlip())
    operations += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(operations)


def to_tensor(dataset, image, i):
    """dataset.transform(image), corrupted by dataset.stress (if set) after resizing."""
    if dataset.stress is not None:
        # Resizing an already resized image is a no-op inside the transform.
        image = dataset.stress(dataset.resize(image), i)
    return dataset.transform(image)


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
        self.resize = transforms.Resize((size, size))
        self.counterfactual = counterfactual
        self.stress = None

    def __len__(self):
        return len(self.indices)

    def paint(self, grayscale, color, i):
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
        return to_tensor(self, Image.fromarray(rgb, mode="RGB"), i)

    def __getitem__(self, i):
        row = self.base[int(self.indices[i])]
        y, s = int(self.labels[i]), int(self.spurious[i])
        out = {"pixel_values": self.paint(row["image"], s, i),
               "labels": y, "spurious": s, "index": int(self.indices[i])}
        if self.counterfactual:
            out["cf_pixel_values"] = self.paint(row["image"], self.cf_spurious[i], i)
        return out


# Waterbirds compositing, adapted from kohpangwei/group_DRO (MIT License),
# dataset_scripts/generate_waterbirds.py and dataset_utils.py.

def crop_and_resize(source_img, target_img):
    """Resize/crop source_img to exactly target_img's size, keeping aspect ratio."""
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size
    # Upscale first if the source does not fully cover the target.
    if source_width < target_width or source_height < target_height:
        width_resize = (target_width, int(target_width / source_width * source_height))
        if width_resize[1] >= target_height:
            source_img = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int(target_height / source_height * source_width), target_height)
            source_img = source_img.resize(height_resize, Image.LANCZOS)
        return crop_and_resize(source_img, target_img)
    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:  # crop left/right
        new_width = int(target_aspect * source_height)
        offset = (source_width - new_width) // 2
        box = (offset, 0, source_width - offset, source_height)
    else:  # crop top/bottom
        new_height = int(source_width / target_aspect)
        offset = (source_height - new_height) // 2
        box = (0, offset, source_width, source_height - offset)
    return source_img.crop(box).resize((target_width, target_height), Image.LANCZOS)


def combine_and_mask(place, mask, img_black):
    """Paste the masked bird (img_black) onto `place` using the soft mask."""
    place_np = np.asarray(crop_and_resize(place, img_black))
    background = np.around(place_np * (1 - mask)).astype(np.uint8)
    return Image.fromarray(np.asarray(img_black) + background)


def place_files(places_dir, categories):
    """Sorted Places365 paths (relative to data_256/) for the given categories."""
    files = []
    for category in categories:
        folder = Path(places_dir) / PLACES_PREFIX / category[0] / category
        files += sorted(f"{category[0]}/{category}/{p.name}" for p in folder.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No Places365 images for {categories}; run download_waterbirds.sh")
    return files


def waterbirds_metadata(root, correlations, seed, val_frac=WATERBIRDS_VAL_FRAC):
    """One row per CUB image: split, label y, place, and a fixed land and water scene.

    - y: waterbird (seabird/waterfowl species) 1, landbird 0.
    - split: CUB test -> test; CUB train -> train, with `val_frac` moved to val.
    - place: within each (split, y), exactly round(p * n) images get the place that
      matches y, with p = correlations[split] (group_DRO's confounder strength).
    - land_file / water_file: every image owns one scene of each kind, so `place`
      picks the background and the counterfactual uses the other one. Splits and
      scenes depend only on `seed`, not on the correlations.
    """
    cub = Path(root) / "CUB_200_2011"
    df = pd.read_csv(cub / "images.txt", sep=" ", header=None, names=["img_id", "img_filename"])
    split_df = pd.read_csv(cub / "train_test_split.txt", sep=" ", header=None,
                           names=["img_id", "is_train"])
    df = df.merge(split_df, on="img_id").sort_values("img_id").reset_index(drop=True)
    species = df["img_filename"].str.split("/").str[0].str.split(".", n=1).str[1].str.lower()
    water = [w.lower() for w in WATER_BIRDS]
    df["y"] = [int(any(w in s for w in water)) for s in species]

    rng = np.random.default_rng(seed)
    train_ids = np.flatnonzero(df["is_train"] == 1)
    val_ids = rng.choice(train_ids, size=int(np.round(val_frac * len(train_ids))), replace=False)
    df["split"] = np.where(df["is_train"] == 1, "train", "test")
    df.loc[val_ids, "split"] = "val"

    df["place"] = 0
    for split, corr in correlations.items():
        split_rng = np.random.default_rng(seed + SPLIT_SEEDS[split])
        for y in (0, 1):
            ids = np.flatnonzero((df["split"] == split) & (df["y"] == y))
            water_fraction = corr if y == 1 else 1 - corr
            chosen = split_rng.choice(ids, size=int(np.round(water_fraction * len(ids))),
                                      replace=False)
            df.loc[chosen, "place"] = 1

    for place, column in enumerate(("land_file", "water_file")):
        files = place_files(Path(root) / "places365", TARGET_PLACES[place])
        order = np.random.default_rng(seed + 1000 + place).permutation(len(files))
        # Unique scenes while the pool lasts (10,000 per kind for 11,788 birds).
        df[column] = [files[order[i % len(files)]] for i in range(len(df))]
    return df.drop(columns="is_train")


class Waterbirds(Dataset):
    """CUB birds pasted onto Places365 scenes when an item is loaded.

    y = landbird 0 / waterbird 1, s = place: land 0 / water 1. `meta` rows come
    from waterbirds_metadata(). Counterfactual pastes the same bird onto its
    scene of the other place. `index` is the CUB img_id.
    """
    num_classes = NUM_CLASSES["waterbirds"]
    groups = [(y, s) for y in range(2) for s in range(2)]

    def __init__(self, root, meta, size, train=False, augment=False, counterfactual=False):
        self.cub_dir = Path(root) / "CUB_200_2011"
        self.places_dir = Path(root) / "places365" / PLACES_PREFIX
        self.files = meta["img_filename"].tolist()
        self.scenes = meta[["land_file", "water_file"]].to_numpy()
        self.ids = meta["img_id"].to_numpy(dtype=np.int64)
        self.labels = meta["y"].to_numpy(dtype=np.int64)
        self.spurious = meta["place"].to_numpy(dtype=np.int64)
        self.transform = image_transform(size, train=train, augment=augment)
        self.resize = transforms.Resize((size, size))
        self.counterfactual = counterfactual
        self.stress = None

    def __len__(self):
        return len(self.files)

    def composite(self, i, place, bird=None):
        if bird is None:
            rel = self.files[i]
            img = np.asarray(Image.open(self.cub_dir / "images" / rel).convert("RGB"))
            seg = np.asarray(Image.open(self.cub_dir / "segmentations" / rel.replace(".jpg", ".png"))
                             .convert("RGB")) / 255
            bird = (seg, Image.fromarray(np.around(img * seg).astype(np.uint8)))
        scene = Image.open(self.places_dir / self.scenes[i, place]).convert("RGB")
        return to_tensor(self, combine_and_mask(scene, *bird), i), bird

    def __getitem__(self, i):
        y, s = int(self.labels[i]), int(self.spurious[i])
        image, bird = self.composite(i, s)
        out = {"pixel_values": image, "labels": y, "spurious": s, "index": int(self.ids[i])}
        if self.counterfactual:
            out["cf_pixel_values"] = self.composite(i, 1 - s, bird)[0]
        return out


def _category_key(member_name):
    """'data_256/f/forest/broadleaf/00000001.jpg' -> ('f', 'forest', 'broadleaf')."""
    return tuple(member_name.split("/")[1:-1])


def _tar_header_at(url, offset, span=1 << 16):
    """First tar header at or after `offset` (512-aligned): (absolute offset, name)."""
    offset -= offset % 512
    req = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{offset + span - 1}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    for i in range(0, len(data) - 511, 512):
        try:
            info = tarfile.TarInfo.frombuf(data[i:i + 512], "utf-8", "surrogateescape")
        except tarfile.HeaderError:
            continue
        if info.name.startswith(PLACES_PREFIX + "/"):
            return offset + i, info.name
    return _tar_header_at(url, offset, span * 4)  # a large member; widen the window


def fetch_places_category(places_dir, category, url=PLACES_URL):
    """Extract one category from the (sorted, uncompressed) Places365 tar.

    Binary-searches the archive for the category's first header, then streams
    from there and stops after the category ends, so only ~1% of the 26 GB
    archive is downloaded per category.
    """
    out = Path(places_dir) / PLACES_PREFIX / category[0] / category
    marker = out / ".complete"
    if marker.exists():
        print(f"[places] {category}: already downloaded")
        return
    target = (category[0], *category.split("/"))
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as resp:
        total = int(resp.headers["Content-Length"])
    lo, hi = 0, total
    while hi - lo > 1 << 20:
        mid = (lo + hi) // 2
        pos, name = _tar_header_at(url, mid)
        if _category_key(name) < target:
            lo = pos
        else:
            hi = mid
    start, _ = _tar_header_at(url, lo)
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"})
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with urllib.request.urlopen(req, timeout=120) as resp, \
            tarfile.open(fileobj=resp, mode="r|") as archive:
        for member in archive:
            key = _category_key(member.name)
            if key < target or not member.isfile():
                continue
            if key > target:
                break
            (out / Path(member.name).name).write_bytes(archive.extractfile(member).read())
            n += 1
    if n == 0:
        raise RuntimeError(f"No images found for Places365 category {category}")
    marker.write_text(str(n))
    print(f"[places] {category}: {n} images -> {out}")


def fetch_places(places_dir):
    """Download the four Waterbirds scene categories (used by download_waterbirds.sh)."""
    for categories in TARGET_PLACES:
        for category in categories:
            fetch_places_category(places_dir, category)


def build_datasets(cfg):
    if cfg["dataset"] == "colored_mnist":
        # Import only when loading a real HF dataset, enabling offline code sanity tests.
        from datasets import load_dataset

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
        root = Path(cfg.get("waterbirds_root", WATERBIRDS_ROOT))
        if not (root / "CUB_200_2011" / "images.txt").exists():
            raise FileNotFoundError(
                f"{root}/CUB_200_2011 not found; run `bash download_waterbirds.sh` first "
                "or pass --waterbirds-root")
        names = ("train", "val", "test")
        meta = waterbirds_metadata(root, {name: cfg[f"{name}_corr"] for name in names}, cfg["seed"])
        return {name: Waterbirds(root, meta[meta["split"] == name], size=cfg["image_size"],
                                 train=(name == "train"), augment=cfg["augment"],
                                 counterfactual=(name != "train"))
                for name in names}
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
    _save(fig, out_file)


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
    _save(fig, out_file)


WATERBIRDS_GROUP_NAMES = {(0, 0): "landbird / land", (0, 1): "landbird / water",
                          (1, 0): "waterbird / land", (1, 1): "waterbird / water"}


def waterbirds_example_cfg(cfg=None):
    """Training-default Waterbirds config, overridden by `cfg`."""
    return {"dataset": "waterbirds", "seed": 42, "image_size": 128, "augment": False,
            "waterbirds_root": WATERBIRDS_ROOT, "train_corr": CORR_PRESETS["strong"],
            "val_corr": .50, "test_corr": .50, **(cfg or {})}


def save_waterbirds_examples(out_file, split="train", samples=6, cfg=None):
    """Save a grid of Waterbirds samples: one row per (bird, place) group."""
    import matplotlib.pyplot as plt

    cfg = waterbirds_example_cfg(cfg)
    ds = build_datasets(cfg)[split]
    fig, axes = plt.subplots(len(ds.groups), samples, figsize=(samples * 1.6, len(ds.groups) * 1.75),
                             squeeze=False, layout="constrained")
    for r, (y, s) in enumerate(ds.groups):
        members = np.flatnonzero((ds.labels == y) & (ds.spurious == s))
        for c in range(samples):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(members):
                ax.imshow(to_image(ds[int(members[c])]["pixel_values"]))
        axes[r, 0].set_title(f"{WATERBIRDS_GROUP_NAMES[y, s]} (n={len(members)})",
                             fontsize=9, loc="left", color="red" if s != y else "black")
    fig.suptitle(f"Waterbirds / {split} split / P(place == label)={cfg[f'{split}_corr']}")
    _save(fig, out_file)


def save_waterbirds_corr_comparison(out_file, samples=16, cols=4, cfg=None):
    """Compare CORR_PRESETS on the train split: one row per preset, one column per bird class.

    Each cell is a mosaic of the first `samples` training birds of that class.
    Splits do not depend on the correlation, so every row shows the same birds
    and only their backgrounds change; minority-place tiles get a red frame.
    """
    import matplotlib.pyplot as plt

    cfg = waterbirds_example_cfg({"image_size": 64, **(cfg or {})})
    classes = ("landbird", "waterbird")
    fig, axes = plt.subplots(len(CORR_PRESETS), len(classes),
                             figsize=(len(classes) * 3.2, len(CORR_PRESETS) * 3.0),
                             squeeze=False, layout="constrained")
    for r, (name, corr) in enumerate(CORR_PRESETS.items()):
        ds = build_datasets({**cfg, "train_corr": corr})["train"]
        for y, label in enumerate(classes):
            chosen = np.flatnonzero(ds.labels == y)[:samples]
            tiles = []
            for i in chosen:
                tile = to_image(ds[int(i)]["pixel_values"])
                if ds.spurious[i] != y:
                    frame = np.ones(tile.shape[:2], dtype=bool)
                    frame[2:-2, 2:-2] = False
                    tile[frame] = (1, 0, 0)
                tiles.append(tile)
            tiles += [np.ones_like(tiles[0])] * (-len(tiles) % cols)
            mosaic = np.concatenate([np.concatenate(tiles[j:j + cols], axis=1)
                                     for j in range(0, len(tiles), cols)], axis=0)
            n_minor = int((ds.spurious[chosen] != y).sum())
            ax = axes[r, y]
            ax.imshow(mosaic)
            ax.set_xticks([])
            ax.set_yticks([])
            other = "water" if y == 0 else "land"
            ax.set_title(f"{label}: on {other} {n_minor}/{len(chosen)}", fontsize=9)
        actual = float((ds.labels == ds.spurious).mean())
        counts = " / ".join(str(int(((ds.labels == y) & (ds.spurious == s)).sum()))
                            for y, s in ds.groups)
        axes[r, 0].set_ylabel(f"{name}\n{corr:.0%}:{1 - corr:.0%}\nactual {actual:.3f}\n"
                              f"groups {counts}", fontsize=9)
    fig.suptitle("Waterbirds train split / P(place == label) per preset\n"
                 "(groups: landbird-land / landbird-water / waterbird-land / waterbird-water)",
                 fontsize=10)
    _save(fig, out_file)


def save_waterbirds_counterfactuals(out_file, split="test", per_group=2, cfg=None):
    """Original vs counterfactual pairs: the same bird on its land and water scenes."""
    import matplotlib.pyplot as plt

    if split == "train":
        raise ValueError("counterfactual images exist only for val/test")
    ds = build_datasets(waterbirds_example_cfg(cfg))[split]
    chosen = [int(i) for y, s in ds.groups
              for i in np.flatnonzero((ds.labels == y) & (ds.spurious == s))[:per_group]]
    fig, axes = plt.subplots(2, len(chosen), figsize=(len(chosen) * 1.6, 3.7), squeeze=False,
                             layout="constrained")
    for c, i in enumerate(chosen):
        item = ds[i]
        y, s = item["labels"], item["spurious"]
        for r, (key, place) in enumerate((("pixel_values", s), ("cf_pixel_values", 1 - s))):
            ax = axes[r, c]
            ax.imshow(to_image(item[key]))
            ax.axis("off")
            ax.set_title(WATERBIRDS_GROUP_NAMES[y, place], fontsize=8,
                         color="red" if place != y else "black")
    axes[0, 0].text(-0.08, 0.5, "original", transform=axes[0, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=9)
    axes[1, 0].text(-0.08, 0.5, "counterfactual", transform=axes[1, 0].transAxes, rotation=90,
                    ha="right", va="center", fontsize=9)
    fig.suptitle(f"Waterbirds / {split} split / same bird, other place's scene")
    _save(fig, out_file)


def save_stress_examples(out_file, dataset="colored_mnist", split="test", index=0, cfg=None):
    """One sample under every stress type (rows) and severity (columns, 0 = clean)."""
    import matplotlib.pyplot as plt

    cfg = example_cfg(cfg) if dataset == "colored_mnist" else waterbirds_example_cfg(cfg)
    ds = build_datasets(cfg)[split]
    levels = len(next(iter(STRESS_SEVERITIES.values())))
    fig, axes = plt.subplots(len(STRESS_TYPES), levels + 1,
                             figsize=((levels + 1) * 1.5, len(STRESS_TYPES) * 1.65),
                             squeeze=False, layout="constrained")
    for r, kind in enumerate(STRESS_TYPES):
        for c in range(levels + 1):
            ds.stress = Stress(kind, c, cfg["seed"]) if c else None
            ax = axes[r, c]
            ax.imshow(to_image(ds[index]["pixel_values"]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ds.stress.describe(cfg["image_size"]) if c else "clean", fontsize=8)
            if c == 0:
                ax.set_ylabel(kind, fontsize=9)
    ds.stress = None
    fig.suptitle(f"{dataset} / {split} sample {index} / stress severities 1-5 "
                 f"at {cfg['image_size']}px", fontsize=10)
    _save(fig, out_file)


def _save(fig, out_file):
    import matplotlib.pyplot as plt

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"Saved {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Save example images: Colored MNIST (one row per color mode, plus a "
                    "strong/medium/weak comparison), Waterbirds (groups, counterfactuals, "
                    "comparison) and stress-test severities for both.")
    parser.add_argument("--dataset", choices=["all", "colored_mnist", "waterbirds"], default="all",
                        help="'all' skips Waterbirds when its raw data is missing")
    parser.add_argument("--output", default="images/colored_mnist_examples.png")
    parser.add_argument("--color-modes", nargs="+", choices=COLOR_MODES, default=list(COLOR_MODES))
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--per-digit", type=int, default=1, help="samples shown per digit and mode")
    parser.add_argument("--corr", type=parse_corr, default=None,
                        help="P(main color) / P(place == label) for the chosen split: a float "
                             f"or one of {list(CORR_PRESETS)} (default: training default)")
    parser.add_argument("--color-pairing", choices=COLOR_PAIRINGS, default="next")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=None,
                        help="default: 64 for colored_mnist, 128 for waterbirds")
    parser.add_argument("--corr-output", default="images/colored_mnist_corr_comparison.png",
                        help="strong/medium/weak comparison image; pass '' to skip")
    parser.add_argument("--corr-mode", choices=COLOR_MODES, default="foreground",
                        help="color mode used in the comparison image")
    parser.add_argument("--corr-samples", type=int, default=20,
                        help="samples per digit in each comparison cell")
    parser.add_argument("--waterbirds-output", default="images/waterbirds_examples.png",
                        help="one row per group; pass '' to skip")
    parser.add_argument("--waterbirds-cf-output", default="images/waterbirds_counterfactual.png",
                        help="original vs counterfactual pairs (test split); pass '' to skip")
    parser.add_argument("--waterbirds-corr-output", default="images/waterbirds_corr_comparison.png",
                        help="strong/medium/weak comparison image; pass '' to skip")
    parser.add_argument("--waterbirds-corr-samples", type=int, default=16,
                        help="birds per class in each comparison cell")
    parser.add_argument("--waterbirds-samples", type=int, default=6, help="samples per group")
    parser.add_argument("--waterbirds-root", default=WATERBIRDS_ROOT)
    parser.add_argument("--stress-output", default="images/{dataset}_stress.png",
                        help="stress-test severity grid per dataset; pass '' to skip")
    args = parser.parse_args()
    if min(args.per_digit, args.corr_samples, args.waterbirds_samples,
           args.waterbirds_corr_samples) <= 0:
        parser.error("--per-digit and the --*-samples options must be positive")
    cfg = {"seed": args.seed}
    if args.image_size is not None:
        cfg["image_size"] = args.image_size
    if args.corr is not None:
        cfg[f"{args.split}_corr"] = args.corr
    if args.dataset in ("all", "colored_mnist"):
        save_colored_mnist_examples(args.output, args.color_modes, args.split, args.per_digit,
                                    {**cfg, "color_pairing": args.color_pairing})
        if args.corr_output:
            save_corr_comparison(args.corr_output, args.corr_mode, args.corr_samples,
                                 cfg={"seed": args.seed, "color_pairing": args.color_pairing})
        if args.stress_output:
            save_stress_examples(args.stress_output.format(dataset="colored_mnist"),
                                 "colored_mnist", cfg={"seed": args.seed,
                                                       "color_pairing": args.color_pairing})
    if args.dataset in ("all", "waterbirds"):
        wb_cfg = {**cfg, "waterbirds_root": args.waterbirds_root}
        if not (Path(args.waterbirds_root) / "CUB_200_2011" / "images.txt").exists():
            if args.dataset == "waterbirds":
                parser.error(f"no Waterbirds raw data in {args.waterbirds_root}; "
                             "run `bash download_waterbirds.sh`")
            print(f"[skip] Waterbirds: no raw data in {args.waterbirds_root}")
            return
        if args.waterbirds_output:
            save_waterbirds_examples(args.waterbirds_output, args.split, args.waterbirds_samples,
                                     wb_cfg)
        if args.waterbirds_cf_output:
            save_waterbirds_counterfactuals(args.waterbirds_cf_output, cfg=wb_cfg)
        if args.waterbirds_corr_output:
            save_waterbirds_corr_comparison(args.waterbirds_corr_output,
                                            args.waterbirds_corr_samples,
                                            cfg={"seed": args.seed,
                                                 "waterbirds_root": args.waterbirds_root})
        if args.stress_output:
            save_stress_examples(args.stress_output.format(dataset="waterbirds"), "waterbirds",
                                 cfg={"seed": args.seed, "waterbirds_root": args.waterbirds_root})


if __name__ == "__main__":
    main()
