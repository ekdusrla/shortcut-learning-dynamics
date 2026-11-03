"""Grad-CAM visualization for saved Shortcut Learning experiment checkpoints.

Works with `shortcut_experiment.py` (ResNet-18 / VGG-11-BN) without
changing the training code.  Gradients are computed at inference time;
checkpoint parameters are never updated.

Examples
--------
# On an exact Colored MNIST validation sample (loads dataset from HF):
python gradcam.py --run runs/cm_resnet --checkpoint checkpoints/epoch_005.pt \
    --split val --sample-index 3 --layer layer3

# On an external image (no HF data download):
python gradcam.py --run runs/wb_vgg --checkpoint latest \
    --image bird.jpg --target-class 1 --output gradcam_bird.png

# Print available convolutional layers and recommended default:
python gradcam.py --run runs/cm_vgg --list-layers

Outputs a three-panel PNG (input / heatmap / overlay), a float32 heatmap
(.npy), and prediction metadata (.json).
Colored MNIST uses 10 classes (0..9); Waterbirds uses 2 classes (0..1).
Run alongside the matching shortcut_experiment.py and data.py from this project.
Dataset membership and spurious assignments are reconstructed from checkpoint
config and the same source data. Historical random training flips are not stored;
train samples use the saved seed to make visualization-time flips repeatable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Headless / Google Colab compatible
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

# Keep experiment/data imports inside the CLI helpers: importing GradCAM does
# not load shortcut_experiment.py, data.py, torchvision, or a run config.


class GradCAM:
    """Compute class-conditional Grad-CAM at a named *spatial* layer.

    Parameters
    ----------
    model: nn.Module
        Classification model whose output is [B, num_classes] logits.
    layer_name: str
        Name in dict(model.named_modules()), such as 'layer3' or 'features.25'.

    Input to __call__: image Tensor [B, 3, H, W], class_id int or None.
    Output: (logits [B, num_classes], heatmap [B, H, W]); heatmap in [0,1].

    Does not use torch.inference_mode() because gradients are required.
    Does not call optimizer.step() or modify model weights.
    Import independently: from gradcam import GradCAM
    """

    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model.eval()
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(
                f"Layer {layer_name!r} not found. "
                "Use --list-layers to inspect available convolutional layers."
            )
        self.activations: Optional[torch.Tensor] = None
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Grad-CAM target layer must output a [B, C, H, W] Tensor")
        self.activations = output  # Keep the computation graph (do NOT detach).

    def __call__(self, images: torch.Tensor, class_id: Optional[int] = None):
        if images.ndim != 4:
            raise ValueError("images must have shape [B, 3, H, W]")
        with torch.enable_grad():
            self.activations = None
            # An input requiring gradients also supports fully frozen models.
            logits = self.model(images.detach().requires_grad_(True))
            if self.activations is None:
                raise RuntimeError("The target layer was not executed during forward")
            if logits.ndim != 2:
                raise ValueError("Expected classifier output shape [B, num_classes]")
            if class_id is None:
                targets = logits.argmax(dim=1)
            else:
                if class_id < 0 or class_id >= logits.shape[1]:
                    raise ValueError(f"class_id must be 0..{logits.shape[1] - 1}")
                targets = torch.full(
                    (images.shape[0],), class_id, dtype=torch.long, device=logits.device
                )
            scores = logits.gather(1, targets[:, None]).sum()
            grads = torch.autograd.grad(scores, self.activations)[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)   # [B, C, 1, 1]
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)  # Classical Grad-CAM
            cam = F.interpolate(
                cam, size=images.shape[-2:], mode="bilinear", align_corners=False
            )
            # Min-max normalize independently for each sample.
            b = cam.size(0)
            flattened = cam.reshape(b, -1)
            minimum = flattened.min(dim=1).values.reshape(b, 1, 1, 1)
            maximum = flattened.max(dim=1).values.reshape(b, 1, 1, 1)
            cam = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
            result = cam[:, 0].detach().float().cpu()
            detached_logits = logits.detach().float().cpu()
            self.activations = None  # Free graph references
            return detached_logits, result

    def close(self):
        """Remove the forward hook so repeated inspection does not leak hooks."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        self.activations = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def default_layer(model: nn.Module, arch: str) -> str:
    """Choose ResNet layer4 or VGG's last convolution (before pooling)."""
    if arch == "resnet18":
        return "layer4"
    if arch == "vgg11_bn":
        conv_layers = [n for n, module in model.named_modules()
                       if isinstance(module, nn.Conv2d)]
        if not conv_layers:
            raise ValueError("No convolution layers found")
        return conv_layers[-1]
    raise ValueError(f"Unsupported architecture {arch!r}")


def load_run(run_dir: Path, checkpoint_arg: str, device: torch.device):
    """Load saved experiment config, its model weights, and checkpoint epoch.

    The checkpoint's config restores the model and dataset (no config.json needed).
    `--checkpoint` accepts 'latest', a full path, a path relative to run_dir,
    or a bare filename like 'epoch_005.pt'.
    """
    from shortcut_experiment import build_model, load_checkpoint

    if checkpoint_arg == "latest":
        pointer = json.loads((run_dir / "latest.json").read_text(encoding="utf-8"))
        ckpt = run_dir / "checkpoints" / f"epoch_{int(pointer['epoch']):03d}.pt"
    else:
        candidate = Path(checkpoint_arg).expanduser()
        possibilities = [candidate, run_dir / candidate, run_dir / "checkpoints" / candidate.name]
        ckpt = next((p for p in possibilities if p.is_file()), None)
        if ckpt is None:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")
    # The checkpoint config belongs to these exact weights. config.json may
    # have been overwritten by another run/resume and must not override it.
    state = load_checkpoint(ckpt, device)
    cfg = dict(state["config"])
    # All weights are loaded from the checkpoint; do not download pretrained
    # ImageNet weights even when the training run used --pretrained.
    model_cfg = dict(cfg)
    model_cfg["pretrained"] = False
    model = build_model(model_cfg).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, cfg, int(state["epoch"]), ckpt


def load_sample(cfg: dict, image_file: Optional[Path], split: str, index: int):
    """Return (normalized input [3,H,W], metadata) using data.py's contract."""
    from data import build_datasets, image_transform
    from shortcut_experiment import seed_everything

    if image_file is not None:
        with Image.open(image_file) as im:
            image = im.convert("RGB")
            tensor = image_transform(cfg["image_size"])(image)
        return tensor, {"image": str(image_file), "split": None,
                        "index": None, "label": None, "spurious": None}
    # Reuse all saved settings: split seed, correlations, color mode/pairing,
    # image size, Waterbirds root and augmentation. Do not reimplement data.py.
    dataset = build_datasets(cfg)[split]
    if index < 0 or index >= len(dataset):
        raise IndexError(f"sample-index {index} outside 0..{len(dataset) - 1}")
    # Reproduce this visualization's random train augmentation. Historical
    # per-access flips during training were not saved in the checkpoint.
    seed_everything(cfg["seed"])
    row = dataset[index]
    return row["pixel_values"], {
        "image": None, "split": split, "sample_index": index,
        # index is the original MNIST index or CUB img_id, not split position.
        "index": int(row["index"]),
        "label": int(row["labels"]), "spurious": int(row["spurious"]),
    }


def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Reverse the exact experiment's ImageNet normalization for plotting."""
    from data import IMAGENET_MEAN, IMAGENET_STD

    mean = torch.tensor(IMAGENET_MEAN).reshape(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).reshape(3, 1, 1)
    rgb = (tensor.detach().float().cpu() * std + mean).clamp(0, 1)
    return rgb.permute(1, 2, 0).numpy()


def save_visualization(image: np.ndarray, cam: np.ndarray, output: Path,
                       title: str, alpha: float = 0.45):
    """Save input / heatmap / overlay PNG; returns no value."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps["jet"]
    colors = cmap(cam)[..., :3]
    overlay = np.clip((1 - alpha) * image + alpha * colors, 0, 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, pixels, label in zip(
        axes, (image, colors, overlay), ("Input", "Grad-CAM", "Overlay")
    ):
        ax.imshow(pixels)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    from shortcut_experiment import get_device

    parser = argparse.ArgumentParser(description="Grad-CAM for shortcut_learning_experiment")
    parser.add_argument("--run", required=True, type=Path, help="training run directory (config is loaded from checkpoint)")
    parser.add_argument("--checkpoint", default="latest", help="latest, checkpoint path or epoch_005.pt")
    parser.add_argument("--image", type=Path, default=None, help="external RGB image (avoids HF download)")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--sample-index", type=int, default=0, help="dataset index if --image is omitted")
    parser.add_argument("--layer", default=None, help="model.named_modules() layer name")
    parser.add_argument("--target-class", type=int, default=None,
                        help="class index 0..C-1; default: model's predicted class")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output", type=Path, default=None, help="output PNG path")
    parser.add_argument("--alpha", type=float, default=0.45, help="overlay heatmap opacity [0,1]")
    parser.add_argument("--list-layers", action="store_true", help="list spatial layers and exit")

    from shortcut_experiment import parse_corr, CORR_PRESETS, COLOR_MODES, COLOR_PAIRINGS, WATERBIRDS_ROOT
    data_args = parser.add_argument_group("dataset overrides (default: checkpoint config)")
    data_args.add_argument("--dataset", choices=["colored_mnist", "waterbirds"], required=True)
    data_args.add_argument("--seed", type=int, default=42)
    data_args.add_argument("--image-size", type=int, default=None,
                       help="64 for colored_mnist; 128 for waterbirds by default")
    corr_help = ("colored_mnist: P(main color); waterbirds: P(place == label). "
                 "A float or one of " + str(list(CORR_PRESETS)))
    data_args.add_argument("--train-corr", type=parse_corr, default="strong", help=corr_help)
    data_args.add_argument("--val-corr", type=parse_corr, default=.50, help=corr_help)
    data_args.add_argument("--test-corr", type=parse_corr, default=.50, help=corr_help)
    data_args.add_argument("--mnist-val-size", type=int, default=5000)
    data_args.add_argument("--color-mode", choices=COLOR_MODES, default="foreground",
                       help="colored_mnist only: region painted with the shortcut color")
    data_args.add_argument("--color-pairing", choices=COLOR_PAIRINGS, default="next",
                       help="colored_mnist only: how each digit's alternate color is chosen")
    data_args.add_argument("--waterbirds-root", default=WATERBIRDS_ROOT,
                       help="waterbirds only: directory with CUB_200_2011/ and places365/ "
                            "from download_waterbirds.sh")
    data_args.add_argument("--augment", action="store_true", help="horizontal flips on Waterbirds train only")
    args = parser.parse_args()
    dataset_cfg = {k: getattr(args, k) for k in 
                   ["dataset", 
                    "seed", 
                    "image_size", 
                    "train_corr", 
                    "val_corr", 
                    "test_corr",
                    "mnist_val_size",
                    "color_mode",
                    "color_pairing",
                    "waterbirds_root",
                    "augment"
                    ]}

    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be between 0 and 1")
    run_dir = args.run.expanduser().resolve()
    device = get_device(args.device)
    model, cfg, epoch, ckpt = load_run(run_dir, args.checkpoint, device)
    selected_layer = args.layer or default_layer(model, cfg["model"])

    if args.list_layers:
        print(f"Recommended Grad-CAM layer: {default_layer(model, cfg['model'])}")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.MaxPool2d)) or (
                cfg["model"] == "resnet18" and name in ("layer1", "layer2", "layer3", "layer4")
            ):
                print(f"{name}: {module.__class__.__name__}")
        return

    x, meta = load_sample(dataset_cfg, args.image, args.split, args.sample_index)
    with GradCAM(model, selected_layer) as gradcam:
        logits, cams = gradcam(x.unsqueeze(0).to(device), args.target_class)
    probabilities = torch.softmax(logits[0], dim=0)
    predicted = int(probabilities.argmax().item())
    target = predicted if args.target_class is None else args.target_class
    cam = cams[0].numpy()

    if args.output:
        output = args.output.expanduser()
    else:
        kind = (args.image.stem if args.image is not None
                else f"{args.split}_{args.sample_index:05d}")
        layer_slug = selected_layer.replace(".", "_")
        output = run_dir / "gradcam" / f"epoch_{epoch:03d}_{kind}_{layer_slug}.png"
    title = (f"Epoch {epoch} | {selected_layer} | predicted={predicted} "
             f"(p={probabilities[predicted].item():.3f}) | Grad-CAM target={target}")
    save_visualization(unnormalize_image(x), cam, output, title, args.alpha)

    heatmap_path = output.with_name(output.stem + "_heatmap.npy")
    metadata_path = output.with_suffix(".json")
    np.save(heatmap_path, cam.astype(np.float32))
    metadata = {
        "checkpoint": str(ckpt.resolve()), "epoch": epoch,
        "architecture": cfg["model"], "dataset": cfg["dataset"],
        "layer": selected_layer, "predicted_class": predicted,
        "gradcam_target_class": target,
        "class_probabilities": probabilities.tolist(), **meta,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Grad-CAM] PNG: {output.resolve()}")
    print(f"[Grad-CAM] Heatmap: {heatmap_path.resolve()} shape={cam.shape}")
    print(f"[Grad-CAM] Metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
