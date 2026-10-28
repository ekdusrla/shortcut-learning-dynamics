"""Plot accuracy trajectories and inspect hidden-state PCA from saved epochs."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def curves(run: Path):
    df = pd.read_csv(run / "history.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3), layout="constrained")
    e = df["epoch"]
    for col in ("train_acc", "val_acc", "val_wga"):
        ax[0].plot(e, df[col], label=col)
    ax[0].set_title("Accuracy and worst-group accuracy")
    for col in ("val_aligned_acc", "val_conflicting_acc", "val_g00_acc", "val_g01_acc", "val_g10_acc", "val_g11_acc"):
        ax[1].plot(e, df[col], label=col)
    ax[1].set_title("Group accuracy: target y, shortcut s")
    for col in ("val_cf_flip_rate", "val_cf_acc", "val_cf_acc_drop"):
        if col in df and df[col].notna().any():
            ax[2].plot(e, df[col], label=col)
    ax[2].set_title("Paired color intervention (Colored MNIST)")
    for a in ax:
        a.set_xlabel("Epoch")
        a.set_ylim(-.05, 1.05)
        a.grid(alpha=.25)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=7)
    out = run / "learning_curves.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"Saved {out}")


def pca_snapshot(file: Path, out: Path, layer: str):
    with np.load(file) as z:
        names = [str(n) for n in z["layers"]]
        if layer == "last":
            i = len(names) - 1
        elif layer.isdigit():
            i = int(layer)
        else:
            i = names.index(layer)
        features = np.asarray(z[f"f{i}"], dtype=np.float32)
        y = z["labels"]
        s = z["spurious"]
    if len(features) < 3:
        raise ValueError("Need at least three samples for a PCA plot")
    # PCA is fitted independently for each epoch: compare separation, not raw axes.
    features = StandardScaler().fit_transform(features)
    coords = PCA(n_components=2, random_state=0).fit_transform(features)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for ax, targets, title in zip(axs, (y, s), ("Target label", "Shortcut attribute")):
        for value in (0, 1):
            chosen = targets == value
            ax.scatter(coords[chosen, 0], coords[chosen, 1], s=11, alpha=.60,
                       label=f"{value} (n={chosen.sum()})")
        ax.set_title(f"{names[i]} / {title}")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.legend(fontsize=8)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out} / raw features shape: {features.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="directory containing history.csv")
    parser.add_argument("--epoch", type=int, default=None,
                        help="if set, plot hidden state PCA for this epoch")
    parser.add_argument("--layer", default="last",
                        help="layer name, index, or 'last'; see features NPZ 'layers'")
    args = parser.parse_args()
    run = Path(args.run)
    curves(run)
    if args.epoch is not None:
        src = run / "features" / f"epoch_{args.epoch:03d}.npz"
        if not src.exists():
            raise FileNotFoundError(f"Feature snapshot absent: {src}; run with --no-features disabled")
        pca_snapshot(src, run / f"hidden_pca_epoch_{args.epoch:03d}.png", args.layer)


if __name__ == "__main__":
    main()
