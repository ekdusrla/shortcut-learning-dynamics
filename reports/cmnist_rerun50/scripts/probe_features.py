"""Linear probes on the per-epoch feature snapshots (validation, 32 digits per digit x color cell).

Each cell is split 24 / 8 (fixed, 3 seeds) into probe-train / held-out, so both sides are exactly
balanced over digit x color. Train-only z-score, ridge (alpha 1) on one-hot targets, argmax.
Writes probe_curves.csv and figure 06 (decodability vs behaviour)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import CORRS, CORR_COLOR, INK, MODES, MUTED, OUT, run_dir, save, style

LAYERS = {"f0": "features.3 (early)", "f1": "features.7", "f2": "features.14", "f3": "features.21", "f4": "features.28 (last conv)"}


def splits(labels, spurious, seeds=(1701, 1702, 1703)):
    out = []
    for s in seeds:
        rng = np.random.default_rng(s); tr = []
        for y in range(10):
            for c in range(10):
                idx = np.flatnonzero((labels == y) & (spurious == c))
                tr += list(rng.permutation(idx)[:24])
        tr = np.array(sorted(tr)); te = np.setdiff1d(np.arange(len(labels)), tr)
        out.append((tr, te))
    return out


def ridge_acc(X, Y, tr, te, alpha=1.0):
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
    accs = []
    for y in Y:
        T = np.eye(10)[y[tr]]; Tm = T.mean(0)
        W = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (T - Tm))
        accs.append(float(((B @ W + Tm).argmax(1) == y[te]).mean()))
    return accs


rows = []
for tag, _, _ in MODES:
    for ctag, corr in CORRS:
        fdir = run_dir(tag, ctag) / "features"
        files = sorted(fdir.glob("epoch_*.npz"))
        if not files: continue
        sp = None
        for f in files:
            z = np.load(f); ep = int(f.stem.split("_")[1])
            lab, spu = z["labels"], z["spurious"]
            if sp is None: sp = splits(lab, spu)
            for key in LAYERS:
                X = z[key].astype(np.float64)
                for k, (tr, te) in enumerate(sp):
                    core, color = ridge_acc(X, [lab, spu], tr, te)
                    rows.append(dict(mode=tag, corr=ctag, epoch=ep, layer=key, split=k, core=core, color=color))
        print("probed", tag, ctag, len(files), "epochs", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "probe_curves.csv", index=False)
m = df.groupby(["mode", "corr", "epoch", "layer"])[["core", "color"]].agg(["mean", "min", "max"]).reset_index()

# Figure 06: last-layer core / color decodability vs behavioural shape use, per run.
from figs_history import load
H = load()
fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), sharex=True, sharey=True)
for r, (tag, _, mname) in enumerate(MODES):
    for c, (ctag, corr) in enumerate(CORRS):
        ax = axes[r, c]; g = m[(m["mode"] == tag) & (m["corr"] == ctag) & (m.layer == "f4")]
        if g.empty: ax.axis("off"); continue
        ep = g.epoch.to_numpy()
        for key, col, lab in (("core", "#2da44e", "digit decodable (probe)"), ("color", "#cf222e", "color decodable (probe)")):
            ax.fill_between(ep, g[(key, "min")] * 100, g[(key, "max")] * 100, color=col, alpha=0.2, lw=0)
            ax.plot(ep, g[(key, "mean")] * 100, color=col, lw=1.6, label=lab)
        if (tag, ctag) in H:
            h = H[(tag, ctag)]
            ax.plot(h.index, h.val_cf_all_acc * 100, color=INK, lw=1.6, ls="--", label="model actually uses shape\n(acc on recolored digits)")
        ax.set_xscale("symlog", linthresh=3); ax.set_xlim(0, 50); ax.set_ylim(0, 102)
        ax.set_xticks([0, 1, 2, 3, 5, 10, 20, 50]); ax.set_xticklabels([0, 1, 2, 3, 5, 10, 20, 50])
        style(ax, f"{mname}, corr {corr:.2f}", "epoch (symlog)" if r == 1 else None, "%" if c == 0 else None)
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("Information vs use: linear probes on the last conv features (band = 3 splits) vs the model's own behaviour",
             fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "06_probe_vs_behaviour")

# Figure 07: digit decodability per layer at selected epochs (.95 runs).
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for c, (tag, _, mname) in enumerate(MODES):
    ax = axes[c]
    # Epoch 0 is the same seed-42 initialisation for every corr and the probe set is the same, so draw it once.
    g0 = m[(m["mode"] == tag) & (m.epoch == 0)]
    if not g0.empty:
        g0 = g0[g0["corr"] == g0["corr"].iloc[0]].set_index("layer").loc[list(LAYERS)]
        ax.plot(range(len(LAYERS)), g0[("core", "mean")] * 100, color=MUTED, ls=":", marker="o", ms=3,
                label="epoch 0 (untrained, same for all corr)")
    for ctag, corr in CORRS:
        for e, ls in ((1, "--"), (50, "-")):
            g = m[(m["mode"] == tag) & (m["corr"] == ctag) & (m.epoch == e)]
            if g.empty: continue
            g = g.set_index("layer").loc[list(LAYERS)]
            ax.plot(range(len(LAYERS)), g[("core", "mean")] * 100, color=CORR_COLOR[ctag], ls=ls, marker="o", ms=3,
                    label=f"corr {corr:.2f}, epoch {e}")
    ax.set_xticks(range(len(LAYERS))); ax.set_xticklabels([v.split(" ")[0] for v in LAYERS.values()], fontsize=8)
    style(ax, f"{mname}: digit decodability by layer", "layer", "probe accuracy (%)" if c == 0 else None)
axes[0].legend(fontsize=7, frameon=False, ncol=1, loc="upper left")
fig.tight_layout(); save(fig, "07_probe_by_layer")
