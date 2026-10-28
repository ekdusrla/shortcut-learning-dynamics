"""Capture and visualize a *raw* intermediate activation map from one example.

Unlike features/epoch_*.npz, this stores the full [C,H,W] tensor for one layer
and one chosen sample. Run `--layer layer3` (ResNet) or `features.14` (VGG).
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from shortcut_experiment import (build_datasets, build_model, choose_checkpoint,
                                  get_device, load_checkpoint, seed_everything,
                                  feature_layer_names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-index", type=int, default=0, help="position within split")
    parser.add_argument("--layer", default="layer3", help="full named-module path; use --list-layers to discover")
    parser.add_argument("--list-layers", action="store_true")
    parser.add_argument("--channels", type=int, default=16, help="number of feature maps to visualize")
    parser.add_argument("--output", default=None, help="output file prefix, without extension")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    device = get_device(args.device)
    ckpt_path = choose_checkpoint(args.run, args.checkpoint)
    checkpoint = load_checkpoint(ckpt_path, device)
    cfg = checkpoint["config"]
    seed_everything(cfg["seed"])
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    all_modules = dict(model.named_modules())
    if args.list_layers:
        print("Suggested feature stages:", feature_layer_names(model, cfg["model"]))
        print("All named modules:")
        print("\n".join(k for k in all_modules if k))
        return
    if args.layer not in all_modules:
        raise KeyError(f"Unknown layer: {args.layer}; use --list-layers")
    dataset = build_datasets(cfg)[args.split]
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(args.sample_index)
    row = dataset[args.sample_index]
    captured = []

    def hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise TypeError("Activation must be a tensor")
        captured.append(output.detach().cpu().float().numpy())

    handle = all_modules[args.layer].register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model(row["pixel_values"].unsqueeze(0).to(device))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one activation but found {len(captured)}")
    activation = captured[0][0]
    prefix = args.output or (Path(args.run) / f"raw_epoch_{checkpoint['epoch']:03d}_"
                             f"{args.split}_{args.sample_index}_{args.layer.replace('.', '_')}")
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(prefix) + ".npz", hidden=activation,
                        label=np.asarray(row["labels"]),
                        shortcut=np.asarray(row["spurious"]),
                        pred=np.asarray(logits.argmax(1).item()),
                        input_index=np.asarray(row["index"]))
    print(f"Shape: {activation.shape}; label={row['labels']} shortcut={row['spurious']}"
          f" pred={logits.argmax(1).item()}")
    print(f"Saved {prefix}.npz")
    if activation.ndim != 3:
        print("Activation is not C×H×W; saved raw values, no feature-map grid generated.")
        return
    n = min(args.channels, activation.shape[0])
    if n <= 0:
        raise ValueError("--channels must be positive")
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.3), squeeze=False,
                             layout="constrained")
    for i, ax in enumerate(axes.flat):
        if i >= n:
            ax.axis("off")
            continue
        im = activation[i]
        ax.imshow(im, cmap="viridis")
        ax.set_title(f"channel {i}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{cfg['model']} | epoch {checkpoint['epoch']} | {args.layer}")
    fig.savefig(str(prefix) + ".png", dpi=150)
    plt.close(fig)
    print(f"Saved {prefix}.png")


if __name__ == "__main__":
    main()
