"""Linear probes on the per-epoch Waterbirds feature snapshots (validation, 32 images per bird x background group).

Each group is split 24 / 8 (3 fixed seeds) into probe-train / held-out, so both sides are balanced over the four
groups (held-out n = 32, so one image = 3.1pp). Train-only z-score, ridge on one-hot targets (alpha by 4-fold CV inside the probe-train split), argmax.
Writes probe_curves.csv and figure 10 (bird / background decodability vs the model's own balanced conflicting accuracy)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import CORRS, CORR_COLOR, INK, MUTED, OUT, RUNS, load_history, out_csv, run_name, save, style

LAYERS = {"f0": "features.3", "f1": "features.7", "f2": "features.14", "f3": "features.21", "f4": "features.28 (last conv)"}


def splits(labels, spurious, seeds=(1701, 1702, 1703)):
    out = []
    for s in seeds:
        rng = np.random.default_rng(s); tr = []
        for y in (0, 1):
            for c in (0, 1):
                tr += list(rng.permutation(np.flatnonzero((labels == y) & (spurious == c)))[:24])
        tr = np.array(sorted(tr)); out.append((tr, np.setdiff1d(np.arange(len(labels)), tr)))
    return out


ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def _fit_predict(A, T, B, alpha):
    # Dual ridge (n_train << dim): W = A^T (A A^T + alpha I)^-1 T.
    Tm = T.mean(0); K = A @ A.T
    C = np.linalg.solve(K + alpha * np.eye(len(A)), T - Tm)
    return B @ (A.T @ C) + Tm


def ridge_acc(X, Y, tr, te, folds=4):
    """Held-out accuracy; alpha picked by 4-fold CV inside the probe-train split (96 images, 512+ dims)."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    accs = []
    for y in Y:
        yt = y[tr]; T = np.eye(2)[yt]; fold = np.arange(len(tr)) % folds
        cv = [np.mean([(_fit_predict(A[fold != f], T[fold != f], A[fold == f], a).argmax(1) == yt[fold == f]).mean()
                       for f in range(folds)]) for a in ALPHAS]
        best = ALPHAS[int(np.argmax(cv))]
        accs.append(float((_fit_predict(A, T, B, best).argmax(1) == y[te]).mean()))
    return accs


rows = []
for ctag, corr in CORRS:
    files = sorted((RUNS / run_name(ctag) / "features").glob("epoch_*.npz"))
    sp = None
    for f in files:
        z = np.load(f); ep = int(f.stem.split("_")[1]); lab, spu = z["labels"], z["spurious"]
        if sp is None: sp = splits(lab, spu)
        for key in LAYERS:
            X = z[key].astype(np.float64)
            for k, (tr, te) in enumerate(sp):
                bird, place = ridge_acc(X, [lab, spu], tr, te)
                rows.append(dict(corr=ctag, epoch=ep, layer=key, split=k, bird=bird, place=place))
    print("probed", ctag, len(files), "epochs", flush=True)
df = pd.DataFrame(rows); df.to_csv(out_csv("probe_curves.csv"), index=False)
if df.empty: raise SystemExit
m = df[df.layer == "f4"].groupby(["corr", "epoch"])[["bird", "place"]].agg(["mean", "min", "max"]).reset_index()
H = load_history("rerun")
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, (ctag, corr) in zip(axes, CORRS):
    g = m[m["corr"] == ctag]
    if g.empty: ax.axis("off"); continue
    for key, col, lab in (("bird", "#2da44e", "bird class decodable (probe)"), ("place", "#cf222e", "background decodable (probe)")):
        ax.fill_between(g.epoch, g[(key, "min")] * 100, g[(key, "max")] * 100, color=col, alpha=0.2, lw=0)
        ax.plot(g.epoch, g[(key, "mean")] * 100, color=col, lw=1.5, label=lab)
    if ctag in H:
        ax.plot(H[ctag].index, H[ctag].val_balanced_conflicting_acc * 100, color=INK, lw=1.4, ls="--",
                label="model's balanced conflicting acc")
    ax.axhline(50, color=MUTED, lw=0.8, ls=":")
    style(ax, f"corr {corr:.2f}", "epoch", "%" if ax is axes[0] else None); ax.set_xlim(0, 50); ax.set_ylim(0, 102)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=8, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.9))
fig.suptitle("Information vs use: linear probes on the last conv features (band = 3 splits, held-out n = 32) vs the model's behaviour",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.82)); save(fig, "10_probe_vs_behaviour")
