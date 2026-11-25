"""Figures and summary table: Waterbirds with ImageNet pretraining + all CUB birds + a new scene every epoch (B),
next to the original team Waterbirds runs and Colored MNIST (CPU, seconds).

Settings (seed 42, VGG11_BN, SGD momentum 0.9, wd 1e-4, batch 128, AMP, 224px, 50 epochs, no augmentation):
  Colored MNIST (BG)   team rerun, from scratch, lr 0.01 constant                 reports/core_flip/core_flip_cmnist.csv
  Waterbirds original  team data, from scratch, lr 0.01 constant                  A_corr*_seed42_e50_contentv3
  Waterbirds final     pretrained + cubfull + B, lr 0.01 constant (scheduler none)  A_corr*_seed42_e50_contentv3<suffix>
  Waterbirds cosine    same, lr 0.01 cosine to 0 at epoch 50 (scheduler cosine), figures 10-11
Reads history.csv of each Waterbirds run, reports/core_flip/core_flip_*.csv (core_flip.py) and isolate.csv (isolate.py).
Writes summary.csv and figures/01-04, 10-11.
Usage (from the repo root): python3 reports/waterbirds_cubfull_pretrained/scripts/report.py
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports" / "waterbirds_cubfull_pretrained"
RUNS = REPO / "runs" / "waterbirds_rerun50"; CF = REPO / "reports" / "core_flip"
FINAL = "_cubfull_pretrained_lr0.01_multiscene"
CORRS = [("095", 0.95), ("080", 0.80), ("065", 0.65)]
CORR_COLOR = {"095": "#08306b", "080": "#2171b5", "065": "#6baed6"}
CORE, SPUR = "#1a7f37", "#cf222e"
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
COSINE = "_cubfull_pretrained_lr0.01_cosine_multiscene"
LATE = (31, 50)


def streak(ok, e, dur=3):
    """First epoch of `dur` epochs in a row where ok holds."""
    return next((int(e[i]) for i in range(len(ok) - dur + 1) if ok[i:i + dur].all()), None)


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title: ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def save(fig, name, title, handles=None, labels=None, ncol=6):
    if handles:
        fig.legend(handles, labels, fontsize=8, frameon=False, ncol=ncol, loc="upper left",
                   bbox_to_anchor=(0.01, 1 - 0.55 / fig.get_figheight()))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.8 if handles else 0.45) / fig.get_figheight()))
    (OUT / "figures").mkdir(parents=True, exist_ok=True); fig.savefig(OUT / "figures" / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("saved", OUT / "figures" / f"{name}.png")


def flips(kind, suf=""):
    """Core / spurious flip per epoch 1-50 on the full val set, {corr: frame indexed by epoch}."""
    if kind == "cmnist":
        d = pd.read_csv(CF / "core_flip_cmnist.csv", dtype={"corr": str}); d = d[d["mode"] == "bg"]
    else:
        d = pd.read_csv(CF / f"core_flip_waterbirds{suf}.csv", dtype={"corr": str}); d = d[d.seed == 42]
    d = d[d.subset == "all"].copy(); d["corr"] = d["corr"].str.zfill(3)
    return {c: d[d["corr"] == c].set_index("epoch").sort_index().loc[1:50] for c, _ in CORRS}


def history(c, suf):
    h = pd.read_csv(RUNS / f"A_corr{c}_seed42_e50_contentv3{suf}" / "history.csv").set_index("epoch").loc[1:50]
    n = {k: h[f"val_g{k}_n"] for k in ("00", "01", "10", "11")}; a = {k: h[f"val_g{k}_acc"] for k in n}
    h["balanced_acc"] = ((a["00"] * n["00"] + a["01"] * n["01"]) / (n["00"] + n["01"])
                         + (a["10"] * n["10"] + a["11"] * n["11"]) / (n["10"] + n["11"])) / 2
    h["balanced_conflicting_acc"] = (a["01"] + a["10"]) / 2
    return h


def isolate(suf):
    d = pd.read_csv(OUT / "isolate.csv", dtype={"corr": str, "suffix": str}, keep_default_na=False)
    d = d[(d.suffix == suf) & (d.seed == 42)]
    return {c: d[d["corr"] == c].set_index("epoch").sort_index() for c, _ in CORRS}


def summary(settings):
    rows = []
    for name, suf in settings:
        F, I = flips("waterbirds", suf), isolate(suf)
        for c, corr in CORRS:
            h, f, i = history(c, suf), F[c], I[c]; L = slice(*LATE)
            rows.append(dict(
                setting=name, corr=corr, lr=0.01, scheduler="cosine" if suf == COSINE else "none",
                core_use_onset=streak((f.core_flip > f.spurious_flip).to_numpy(), f.index.to_numpy()),
                robust_criterion_onset=streak(((h.balanced_conflicting_acc >= 0.5) & (h.val_wga >= 0.2)).to_numpy(), h.index.to_numpy(), 5),
                first_train_acc_99=next((int(e) for e in h.index if h.train_acc[e] >= 0.99), None),
                late_core_flip=f.core_flip.loc[L].mean(), late_spurious_flip=f.spurious_flip.loc[L].mean(),
                late_val_acc=h.val_acc.loc[L].mean(), late_balanced_acc=h.balanced_acc.loc[L].mean(), late_val_wga=h.val_wga.loc[L].mean(),
                late_val_wga_min=h.val_wga.loc[L].min(), late_val_wga_max=h.val_wga.loc[L].max(),
                # swing = mean absolute change from one epoch to the next (so a slow trend does not count as swing)
                late_val_wga_swing=h.val_wga.loc[L].diff().abs().mean(), late_core_flip_swing=f.core_flip.loc[L].diff().abs().mean(),
                late_spurious_flip_swing=f.spurious_flip.loc[L].diff().abs().mean(), val_wga_50=h.val_wga[50],
                core_flip_50=f.core_flip[50], spurious_flip_50=f.spurious_flip[50],
                late_bird_only_bal=i.bird_only_bal.loc[L].mean(), late_scene_place_acc=i.scene_place_acc.loc[L].mean()))
    s = pd.DataFrame(rows).astype({"core_use_onset": "Int64", "robust_criterion_onset": "Int64"}); s.to_csv(OUT / "summary.csv", index=False, float_format="%.4f"); print(s.to_string()); return s


def fig_plane(panels, name="01_flip_plane", lr="lr 0.01 constant"):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.7 * len(panels), 5.0))
    for ax, (title, F) in zip(axes, panels):
        for c, corr in CORRS:
            g = F[c]
            x = g.spurious_flip.rolling(3, min_periods=1).mean() * 100; y = g.core_flip.rolling(3, min_periods=1).mean() * 100
            on = streak((g.core_flip > g.spurious_flip).to_numpy(), g.index.to_numpy())
            ax.plot(x, y, color=CORR_COLOR[c], lw=1.4, label=f"corr {corr:.2f}: onset " + (f"E{on}" if on else "none"))
            ax.scatter(x.iloc[-1], y.iloc[-1], color=CORR_COLOR[c], s=22, zorder=3, edgecolor="white", linewidth=1)
        ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=":")
        ax.text(3, 94, "uses the core", fontsize=7, color=CORE); ax.text(97, 3, "uses the shortcut", fontsize=7, color=SPUR, ha="right")
        style(ax, title, "spurious flip (%)", "core flip (%)" if ax is axes[0] else None)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.legend(fontsize=7, frameon=False, loc="center right")
    save(fig, name, "(spurious flip, core flip) trajectory by correlation, epochs 1-50, 3-epoch rolling mean, dot = epoch 50\n"
         "onset = first of 3 epochs in a row with core flip > spurious flip; seed 42, " + lr)


def fig_flip_epochs(panels):
    fig, axes = plt.subplots(2, len(panels), figsize=(4.7 * len(panels), 6.4), sharex=True, sharey=True)
    for j, (title, F) in enumerate(panels):
        for i, (col, name) in enumerate((("core_flip", "core flip (%)"), ("spurious_flip", "spurious flip (%)"))):
            ax = axes[i, j]
            for c, _ in CORRS: ax.plot(F[c].index, F[c][col] * 100, color=CORR_COLOR[c], lw=1.3)
            style(ax, f"{title}\n{name}" if i == 0 else name, "epoch" if i == 1 else None, None)
            ax.set_xlim(1, 50); ax.set_ylim(0, 100)
    handles = [Line2D([], [], color=CORR_COLOR[c], lw=2) for c, _ in CORRS]
    save(fig, "02_flip_epochs", "core flip and spurious flip per epoch by correlation (raw, seed 42)", handles, [f"corr {corr:.2f}" for _, corr in CORRS], ncol=3)


def fig_curves(name, title, panels, settings, source):
    """One panel per metric, one line per (setting, corr); final = solid, original = dashed."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.7), sharex=True)
    ls = {settings[0][1]: "-", settings[1][1]: "--"}
    for ax, (col, label, ref) in zip(axes, panels):
        for setting, suf in settings:
            D = source(suf)
            for c, _ in CORRS:
                ax.plot(D[c].index, D[c][col] * 100, color=CORR_COLOR[c], lw=1.3 if ls[suf] == "-" else 1.0, ls=ls[suf],
                        alpha=1 if ls[suf] == "-" else 0.75)
        if ref: ax.axhline(ref, color=MUTED, lw=0.7, ls=":")
        style(ax, label, "epoch", None); ax.set_xlim(1, 50); ax.set_ylim(0, 101)
    handles = [Line2D([], [], color=CORR_COLOR[c], lw=2) for c, _ in CORRS] + \
              [Line2D([], [], color=MUTED, lw=1.3, ls=ls[suf]) for _, suf in settings]
    labels = [f"corr {corr:.2f}" for _, corr in CORRS] + [s for s, _ in settings]
    save(fig, name, title, handles, labels, ncol=5)


def fig_schedule():
    """Constant lr vs cosine: val WGA, core flip and spurious flip per epoch, one column per correlation."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 8.6), sharex=True, sharey="row")
    runs = [("lr 0.01 constant", FINAL, "#8c959f", 1.1), ("lr 0.01 cosine to 0", COSINE, "#eb6834", 1.7)]
    for j, (c, corr) in enumerate(CORRS):
        for label, suf, col, lw in runs:
            f, h = flips("waterbirds", suf)[c], history(c, suf)
            for i, y in enumerate((h.val_wga, f.core_flip, f.spurious_flip)): axes[i, j].plot(y.index, y * 100, color=col, lw=lw)
        for i, name in enumerate(("val worst-group accuracy (%)", "core flip (%)", "spurious flip (%)")):
            style(axes[i, j], f"corr {corr:.2f}" if i == 0 else None, "epoch" if i == 2 else None, name if j == 0 else None)
            axes[i, j].set_xlim(1, 50); axes[i, j].set_ylim(0, 65 if i == 2 else 100)
    save(fig, "10_constant_vs_cosine", "Constant lr vs cosine, pretrained + all CUB + B (val, seed 42, raw per-epoch values)",
         [Line2D([], [], color=col, lw=lw) for _, _, col, lw in runs], [r[0] for r in runs], ncol=2)


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    settings = [("pretrained + all CUB + B", FINAL), ("original (from scratch)", "")]
    summary(settings + [("pretrained + all CUB + B, cosine", COSINE)])
    panels = [("Colored MNIST (BG), from scratch", flips("cmnist")),
              ("Waterbirds, original (from scratch)", flips("waterbirds", "")),
              ("Waterbirds, pretrained + all CUB + B", flips("waterbirds", FINAL))]
    fig_plane(panels); fig_flip_epochs(panels); fig_schedule()
    cos = panels[:1] + [("Waterbirds, final, lr constant", panels[2][1]), ("Waterbirds, final, lr cosine", flips("waterbirds", COSINE))]
    fig_plane(cos, "11_flip_plane_cosine", "lr 0.01, constant or cosine")
    fig_curves("03_accuracy_epochs", "Waterbirds accuracy per epoch (seed 42, lr 0.01 constant); solid = pretrained + all CUB + B, dashed = original",
               [("train_acc", "train accuracy (%)", 99), ("val_acc", "val accuracy (%)", None), ("val_wga", "val worst-group accuracy (%)", None)],
               settings, lambda suf: {c: history(c, suf) for c, _ in CORRS})
    fig_curves("04_bird_scene_epochs", "What the model uses: bird alone on gray vs scene alone (val, seed 42)",
               [("bird_only_bal", "bird only: balanced accuracy (%)", 50), ("scene_place_acc", "scene only: predicted as its place's class (%)", 50)],
               settings, isolate)


if __name__ == "__main__":
    main()
