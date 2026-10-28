"""Epoch-wise shortcut-learning experiments: PyTorch models + Hugging Face datasets.

Datasets:
- colored_mnist: binary classification (0-4 vs 5-9), configurable color-label correlation.
- waterbirds: grodino/waterbirds, image label and background place.

Checkpoints, metrics, and group-balanced hidden representations can be inspected
at each epoch. See README.md for commands and protocol details.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEEDS = {"train": 100, "val": 200, "test": 300}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For performance, do not force fully deterministic cuDNN kernels.


def get_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def build_model(cfg):
    architecture = cfg["model"]
    pre = cfg["pretrained"]
    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pre else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif architecture == "vgg11_bn":
        weights = models.VGG11_BN_Weights.DEFAULT if pre else None
        model = models.vgg11_bn(weights=weights)
        # A compact head makes small-image experiments and per-epoch storage feasible.
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Linear(512, 2)
    else:
        raise ValueError(f"Unsupported model: {architecture}")
    return model


def feature_layer_names(model, architecture):
    if architecture == "resnet18":
        return ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    # Each VGG max-pool output corresponds to the end of a feature stage.
    return [f"features.{i}" for i, m in enumerate(model.features)
            if isinstance(m, nn.MaxPool2d)] + ["avgpool"]


class FeatureHookCollector:
    """Forward hooks collect GAP-pooled activations, not enormous pixel maps.

    During a batch, `features[name]` has shape (batch, channels). A user
    who needs raw feature maps can hook the same modules without the GAP.
    """
    def __init__(self, model, names):
        all_modules = dict(model.named_modules())
        self.features = {}
        self.handles = []
        for name in names:
            if name not in all_modules:
                raise KeyError(f"Feature layer not found: {name}")

            def hook(_module, _inputs, output, name=name):
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"Expected tensor output at {name}")
                x = output.detach()
                if x.ndim == 4:
                    x = x.mean(dim=(2, 3))
                else:
                    x = x.flatten(start_dim=1)
                self.features[name] = x.float().cpu()

            self.handles.append(all_modules[name].register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def balanced_indices(dataset, per_group=32, seed=0):
    """Select a fixed subset from each existing (target, spurious) group."""
    rng = np.random.default_rng(seed)
    out = []
    for y in range(2):
        for s in range(2):
            candidates = np.flatnonzero((dataset.labels == y) & (dataset.spurious == s))
            if len(candidates) == 0:
                continue
            n = min(per_group, len(candidates))
            out.extend(rng.choice(candidates, size=n, replace=False).tolist())
    return sorted(out)


@torch.inference_mode()
def save_feature_snapshot(model, dataset, arch, device, batch_size, per_group,
                          out_file, num_workers=0, seed=0):
    ids = balanced_indices(dataset, per_group=per_group, seed=seed)
    if not ids:
        raise RuntimeError("Cannot extract features from an empty dataset")
    names = feature_layer_names(model, arch)
    loader = DataLoader(Subset(dataset, ids), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=(device.type == "cuda"))
    model.eval()
    collector = FeatureHookCollector(model, names)
    saved = {name: [] for name in names}
    ys, ss, indices = [], [], []
    try:
        for batch in loader:
            collector.features.clear()
            _ = model(batch["pixel_values"].to(device, non_blocking=True))
            for name in names:
                saved[name].append(collector.features[name].numpy())
            ys.append(batch["labels"].numpy())
            ss.append(batch["spurious"].numpy())
            indices.append(batch["index"].numpy())
    finally:
        collector.close()
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"layers": np.asarray(names), "labels": np.concatenate(ys),
               "spurious": np.concatenate(ss), "indices": np.concatenate(indices)}
    for k, name in enumerate(names):
        payload[f"f{k}"] = np.concatenate(saved[name], axis=0).astype(np.float32)
    np.savez_compressed(out_file, **payload)
    print(f"[features] {out_file} / N={len(payload['labels'])} / layers={names}")


def create_loader(ds, batch_size, shuffle, workers, gen=None, device=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=(device is not None and device.type == "cuda"),
                      persistent_workers=workers > 0, generator=gen)


def is_amp_enabled(device, cfg):
    return bool(device.type == "cuda" and cfg.get("amp", False))


def autocast_context(device, enabled):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled else nullcontext()


def train_one_epoch(model, loader, optimizer, scaler, device, amp):
    model.train()
    n_correct = n_total = 0
    loss_sum = 0.
    criterion = nn.CrossEntropyLoss()
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n = len(y)
        loss_sum += loss.item() * n
        n_correct += (logits.argmax(dim=1) == y).sum().item()
        n_total += n
    return {"loss": loss_sum / n_total, "acc": n_correct / n_total}


@torch.inference_mode()
def evaluate(model, loader, device, amp=False):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    group_total = np.zeros((2, 2), dtype=np.int64)
    group_correct = np.zeros((2, 2), dtype=np.int64)
    aligned_n = aligned_ok = conflicted_n = conflicted_ok = 0
    cf_n = cf_ok = cf_flips = 0
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        s = batch["spurious"].to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(x)
        pred = logits.argmax(1)
        ok = pred == y
        n = len(y)
        loss_sum += nn.functional.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += ok.sum().item()
        total += n
        for yi in range(2):
            for si in range(2):
                mask = (y == yi) & (s == si)
                group_total[yi, si] += mask.sum().item()
                group_correct[yi, si] += (ok & mask).sum().item()
        aligned = (y == s)
        aligned_n += aligned.sum().item()
        aligned_ok += (ok & aligned).sum().item()
        conflicted_n += (~aligned).sum().item()
        conflicted_ok += (ok & ~aligned).sum().item()
        if "cf_pixel_values" in batch:
            with autocast_context(device, amp):
                cf_pred = model(batch["cf_pixel_values"].to(device, non_blocking=True)).argmax(1)
            cf_n += n
            cf_ok += (cf_pred == y).sum().item()
            cf_flips += (cf_pred != pred).sum().item()

    def ratio(a, b):
        return float(a / b) if b else float("nan")

    result = {"loss": ratio(loss_sum, total), "acc": ratio(correct, total),
              "aligned_acc": ratio(aligned_ok, aligned_n),
              "conflicting_acc": ratio(conflicted_ok, conflicted_n)}
    populated = []
    for yi in range(2):
        for si in range(2):
            key = f"g{yi}{si}"
            v = ratio(int(group_correct[yi, si]), int(group_total[yi, si]))
            result[f"{key}_acc"] = v
            result[f"{key}_n"] = int(group_total[yi, si])
            if group_total[yi, si] > 0:
                populated.append(v)
    # Require all four groups; otherwise WGA is undefined rather than inflated.
    result["wga"] = min(populated) if len(populated) == 4 else float("nan")
    result["cf_acc"] = ratio(cf_ok, cf_n)
    result["cf_flip_rate"] = ratio(cf_flips, cf_n)
    result["cf_acc_drop"] = result["acc"] - result["cf_acc"] if cf_n else float("nan")
    return result


def pretty_metrics(m):
    text = f"acc={m['acc']:.4f}, WGA={m['wga']:.4f}, aligned={m['aligned_acc']:.4f}, conflict={m['conflicting_acc']:.4f}"
    if not math.isnan(m["cf_flip_rate"]):
        text += f", color-flip-rate={m['cf_flip_rate']:.4f}, cf-acc={m['cf_acc']:.4f}"
    return text


def save_checkpoint(path, epoch, model, optimizer, scheduler, scaler, cfg, loader_gen):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"epoch": epoch, "config": cfg, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
             "loader_rng": loader_gen.get_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    pointer = path.parent.parent / "latest.json"
    tmp_pointer = pointer.with_suffix(".tmp")
    tmp_pointer.write_text(json.dumps({"epoch": epoch, "checkpoint": str(path.resolve())}, indent=2),
                           encoding="utf-8")
    os.replace(tmp_pointer, pointer)


def load_checkpoint(path, device):
    # weights_only=True protects loading by not allowing arbitrary Python objects.
    return torch.load(path, map_location=device, weights_only=True)


def append_history(path, epoch, train_stats, val_stats, lr):
    flat = {"epoch": epoch, "lr": lr,
            "train_loss": train_stats.get("loss", float("nan")),
            "train_acc": train_stats.get("acc", float("nan"))}
    flat.update({"val_" + k: v for k, v in val_stats.items()})
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def reconcile_history(path, checkpoint_epoch):
    """Discard a CSV row appended just before an interrupted checkpoint write."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    retained = [r for r in rows if int(r["epoch"]) <= checkpoint_epoch]
    if len(retained) == len(rows):
        return
    if not retained or int(retained[-1]["epoch"]) != checkpoint_epoch:
        raise RuntimeError("History/checkpoint mismatch; inspect the run manually")
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(retained)
    os.replace(temp, path)
    print(f"[resume] Removed {len(rows) - len(retained)} uncommitted history row(s)")


def choose_checkpoint(run_dir, selector):
    if selector == "latest":
        pointer = json.loads((Path(run_dir) / "latest.json").read_text(encoding="utf-8"))
        return Path(run_dir) / "checkpoints" / f"epoch_{int(pointer['epoch']):03d}.pt"
    return Path(selector)


def configure_from_args(args):
    return {"dataset": args.dataset, "model": args.model, "epochs": args.epochs,
            "seed": args.seed, "image_size": args.image_size,
            "train_corr": args.train_corr, "val_corr": args.val_corr,
            "test_corr": args.test_corr, "mnist_val_size": args.mnist_val_size,
            "pretrained": args.pretrained, "augment": args.augment,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "num_workers": args.num_workers,
            "features_per_group": args.features_per_group, "amp": not args.no_amp,
            "save_features": not args.no_features}


def run_training(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    if not args.resume and (output / "history.csv").exists():
        raise FileExistsError(
            f"{output} already has a run. Use --resume latest, or a new --output directory.")
    resume_path = choose_checkpoint(output, args.resume) if args.resume else None
    if resume_path:
        saved = load_checkpoint(resume_path, device)
        pointer = output / "latest.json"
        if pointer.exists():
            latest_epoch = int(json.loads(pointer.read_text(encoding="utf-8"))["epoch"])
            if latest_epoch > int(saved["epoch"]):
                raise ValueError(
                    "Do not resume an older checkpoint into a run containing newer "
                    "checkpoints. Copy/fork the run into a fresh directory first.")
        cfg = saved["config"]
        # The resume uses checkpoint config to prevent accidental changes in the
        # dataset definition, training schedule, architecture, and optimizer.
        print(f"[resume] Configuration loaded from checkpoint: {resume_path}")
        if args.epochs != cfg["epochs"]:
            print("[resume] Ignoring --epochs; using checkpoint's original total epochs")
    else:
        cfg = configure_from_args(args)
    seed_everything(cfg["seed"])
    (output / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    ds = build_datasets(cfg)
    gen = torch.Generator().manual_seed(cfg["seed"] + 777)
    train_loader = create_loader(ds["train"], cfg["batch_size"], True,
                                 cfg["num_workers"], gen, device)
    val_loader = create_loader(ds["val"], cfg["batch_size"], False,
                               cfg["num_workers"], device=device)
    test_loader = create_loader(ds["test"], cfg["batch_size"], False,
                                cfg["num_workers"], device=device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=is_amp_enabled(device, cfg))
    amp = is_amp_enabled(device, cfg)
    ckpt_dir = output / "checkpoints"
    feature_dir = output / "features"
    history_path = output / "history.csv"
    start_epoch = 1
    if resume_path:
        reconcile_history(history_path, int(saved["epoch"]))
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        gen.set_state(saved["loader_rng"].cpu())
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        start_epoch = int(saved["epoch"]) + 1
        print(f"[resume] Continuing from epoch {start_epoch}")
    else:
        # Save the untrained baseline at epoch 0 as well as each learned epoch.
        v0 = evaluate(model, val_loader, device, amp)
        append_history(history_path, 0, {}, v0, optimizer.param_groups[0]["lr"])
        save_checkpoint(ckpt_dir / "epoch_000.pt", 0, model, optimizer, scheduler,
                        scaler, cfg, gen)
        if cfg["save_features"]:
            save_feature_snapshot(model, ds["val"], cfg["model"], device,
                                  cfg["batch_size"], cfg["features_per_group"],
                                  feature_dir / "epoch_000.npz", cfg["num_workers"], cfg["seed"])
        print("epoch=000 | val " + pretty_metrics(v0))

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        stats = train_one_epoch(model, train_loader, optimizer, scaler, device, amp)
        val = evaluate(model, val_loader, device, amp)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        print(f"epoch={epoch:03d} | train acc={stats['acc']:.4f} loss={stats['loss']:.4f} | val "
              + pretty_metrics(val), flush=True)
        # Append first, then checkpoint. The run can be resumed from the last
        # committed checkpoint; see README for handling interrupted writes.
        append_history(history_path, epoch, stats, val, current_lr)
        save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pt", epoch, model, optimizer,
                        scheduler, scaler, cfg, gen)
        if cfg["save_features"]:
            save_feature_snapshot(model, ds["val"], cfg["model"], device,
                                  cfg["batch_size"], cfg["features_per_group"],
                                  feature_dir / f"epoch_{epoch:03d}.npz",
                                  cfg["num_workers"], cfg["seed"])

    if start_epoch <= cfg["epochs"]:
        test = evaluate(model, test_loader, device, amp)
        (output / "test_metrics.json").write_text(
            json.dumps({"epoch": cfg["epochs"], "metrics": test}, indent=2),
            encoding="utf-8")
        print("FINAL TEST: " + pretty_metrics(test))
    print(f"Run directory: {output}")


def run_from_checkpoint(args):
    device = get_device(args.device)
    ckpt = choose_checkpoint(args.run, args.checkpoint)
    saved = load_checkpoint(ckpt, device)
    cfg = saved["config"]
    seed_everything(cfg["seed"])
    ds = build_datasets(cfg)
    model = build_model(cfg).to(device)
    model.load_state_dict(saved["model"])
    print(f"checkpoint={ckpt} epoch={saved['epoch']} {cfg['model']} / {cfg['dataset']}")
    loader = create_loader(ds[args.split], args.batch_size or cfg["batch_size"],
                           False, args.num_workers, device=device)
    if args.command == "eval":
        metrics = evaluate(model, loader, device, is_amp_enabled(device, cfg))
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
    elif args.command == "features":
        file = args.output or (Path(args.run) / "features" / f"{args.split}_epoch_{saved['epoch']:03d}.npz")
        save_feature_snapshot(model, ds[args.split], cfg["model"], device,
                              args.batch_size or cfg["batch_size"], args.features_per_group,
                              file, args.num_workers, cfg["seed"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train.add_argument("--dataset", choices=["colored_mnist", "waterbirds"], required=True)
    train.add_argument("--model", choices=["resnet18", "vgg11_bn"], required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--output", required=True)
    train.add_argument("--resume", default=None,
                       help="checkpoint path or 'latest' in the output directory")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--image-size", type=int, default=None,
                       help="64 for colored_mnist; 128 for waterbirds by default")
    train.add_argument("--train-corr", type=float, default=.95)
    train.add_argument("--val-corr", type=float, default=.50)
    train.add_argument("--test-corr", type=float, default=.10)
    train.add_argument("--mnist-val-size", type=int, default=5000)
    train.add_argument("--pretrained", action="store_true",
                       help="use ImageNet pretrained weights (recommended for Waterbirds)")
    train.add_argument("--augment", action="store_true", help="horizontal flips on Waterbirds train only")
    train.add_argument("--lr", type=float, default=.01)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--num-workers", type=int, default=2)
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--no-features", action="store_true", help="skip per-epoch feature snapshots")
    train.add_argument("--features-per-group", type=int, default=32,
                       help="samples per (target,spurious) group for feature snapshots")
    for cmd in ("eval", "features"):
        p = sub.add_parser(cmd, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        p.add_argument("--run", required=True, help="existing run directory")
        p.add_argument("--checkpoint", default="latest", help="checkpoint file or 'latest'")
        p.add_argument("--split", choices=["train", "val", "test"], default="test" if cmd == "eval" else "val")
        p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=2)
        if cmd == "features":
            p.add_argument("--features-per-group", type=int, default=32)
            p.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.command == "train":
        if args.epochs <= 0:
            parser.error("--epochs must be positive")
        if args.batch_size <= 0:
            parser.error("--batch-size must be positive")
        if args.image_size is None:
            args.image_size = 64 if args.dataset == "colored_mnist" else 128
        if args.image_size < 32:
            parser.error("--image-size must be at least 32 for VGG")
        if any(not 0 <= x <= 1 for x in (args.train_corr, args.val_corr, args.test_corr)):
            parser.error("correlations must be in [0, 1]")
        run_training(args)
    else:
        run_from_checkpoint(args)


if __name__ == "__main__":
    main()
