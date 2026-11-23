"""Tables and figures from core_flip_{waterbirds,cmnist}.csv (CPU, seconds).

Usage: python3 reports/core_flip/scripts/analyze.py [--probe-csv PATH]
  --probe-csv: team-protocol probe results for the Waterbirds seeds
               (default reports/waterbirds_rerun50/seeds/probe_team_protocol.csv, skipped if missing)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports" / "core_flip"; FIG = OUT / "figures"
CORRS = [("095", 0.95), ("080", 0.80), ("065", 0.65)]
CORR_COLOR = {"095": "#08306b", "080": "#2171b5", "065": "#6baed6"}
CORE, SPUR = "#1a7f37", "#cf222e"  # same hues as the probe figures: bird/digit = green, background/color = red
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
PHASES = [("early", 1, 10), ("middle", 11, 30), ("late", 31, 50)]


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title: ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("saved", FIG / f"{name}.png")


def load():
    parts = []
    for name in ("waterbirds", "cmnist"):
        p = OUT / f"core_flip_{name}.csv"
        if p.exists(): parts.append(pd.read_csv(p, dtype={"corr": str, "mode": str}).fillna({"mode": ""}))
    d = pd.concat(parts, ignore_index=True)
    d["corr"] = d["corr"].str.zfill(3)
    d["run"] = np.where(d.dataset == "waterbirds", "WB seed " + d.seed.astype(str), "CM " + d["mode"].str.upper())
    return d


def first_streak(ok, epochs, dur=3):
    return next((int(epochs[i]) for i in range(len(ok) - dur + 1) if ok[i:i + dur].all()), None)


def tables(d, probe):
    a = d[d.subset == "all"]
    rows = []
    for (ds, run, c), g in a.groupby(["dataset", "run", "corr"]):
        r = dict(dataset=ds, run=run, corr=c)
        for ph, lo, hi in PHASES:
            s = g[(g.epoch >= lo) & (g.epoch <= hi)]
            for m in ("spurious_flip", "core_flip", "core_only", "spurious_only", "both", "neither"):
                r[f"{m}_{ph}"] = s[m].mean()
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "phase_means.csv", index=False)
    rows = []
    for (ds, run, c), g in a.groupby(["dataset", "run", "corr"]):
        g = g[g.epoch >= 1].sort_values("epoch"); e = g.epoch.to_numpy()
        r = dict(dataset=ds, run=run, corr=c,
                 core_gt_spurious_3=first_streak((g.core_flip > g.spurious_flip).to_numpy(), e),
                 core_flip_ge20_3=first_streak((g.core_flip >= 0.2).to_numpy(), e),
                 core_flip_ge30_3=first_streak((g.core_flip >= 0.3).to_numpy(), e))
        if probe is not None and ds == "waterbirds":
            p = probe[(probe.seed == int(run.split()[-1])) & (probe["corr"] == c)].sort_values("epoch")
            r["probe_core_ge60_3"] = first_streak((p.acc >= 0.6).to_numpy(), p.epoch.to_numpy())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "onset.csv", index=False)


def fig_curves(d, ds, rows, name, title):
    a = d[(d.subset == "all") & (d.dataset == ds)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.3 * len(rows) + 0.6), sharex=True, sharey=True, squeeze=False)
    for r, (sel, rlabel) in enumerate(rows):
        for c, (ctag, corr) in enumerate(CORRS):
            ax = axes[r, c]
            for run, g in a[(a["corr"] == ctag) & sel(a)].groupby("run"):
                g = g.sort_values("epoch")
                ax.plot(g.epoch, g.spurious_flip * 100, color=SPUR, lw=1.1, alpha=0.85)
                ax.plot(g.epoch, g.core_flip * 100, color=CORE, lw=1.1, alpha=0.85)
            style(ax, f"{rlabel}corr {corr:.2f}", "epoch" if r == len(rows) - 1 else None,
                  "flip rate (%)" if c == 0 else None)
            ax.set_xlim(0, 50); ax.set_ylim(0, 100)
    h = [plt.Line2D([], [], color=SPUR, lw=1.5), plt.Line2D([], [], color=CORE, lw=1.5)]
    fig.legend(h, ["spurious flip: swap the shortcut, keep the core", "core flip: swap the core, keep the shortcut"],
               fontsize=8, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.55 / fig.get_figheight()))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight())); save(fig, name)


def fig_split(d):
    """Share of validation images whose prediction changes under the core swap only / spurious swap only / both / neither."""
    sets = [("waterbirds", "WB seed 42", "Waterbirds (seed 42)"), ("cmnist", "CM BG", "Colored MNIST BG"), ("cmnist", "CM FG", "Colored MNIST FG")]
    sets = [s for s in sets if ((d.dataset == s[0]) & (d.run == s[1])).any()]
    cols = [("core_only", CORE, "core swap only"), ("both", "#9a6700", "both"), ("spurious_only", SPUR, "spurious swap only"),
            ("neither", "#d0d7de", "neither")]
    fig, axes = plt.subplots(len(sets), 3, figsize=(13, 2.9 * len(sets) + 0.6), sharex=True, sharey=True, squeeze=False)
    for r, (ds, run, label) in enumerate(sets):
        for c, (ctag, corr) in enumerate(CORRS):
            ax = axes[r, c]; g = d[(d.dataset == ds) & (d.run == run) & (d["corr"] == ctag) & (d.subset == "all")].sort_values("epoch")
            ax.stackplot(g.epoch, *[g[k] * 100 for k, _, _ in cols], colors=[col for _, col, _ in cols], lw=0)
            style(ax, f"{label}, corr {corr:.2f}", "epoch" if r == len(sets) - 1 else None, "% of images" if c == 0 else None)
            ax.set_xlim(0, 50); ax.set_ylim(0, 100)
    h = [plt.Rectangle((0, 0), 1, 1, color=col) for _, col, _ in cols]
    fig.legend(h, [lab for _, _, lab in cols], fontsize=8, frameon=False, ncol=4, loc="upper left",
               bbox_to_anchor=(0.01, 1 - 0.55 / fig.get_figheight()))
    fig.suptitle("Which swap changes the prediction? (validation images, every epoch)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight())); save(fig, "03_change_split")


def fig_plane(d):
    """Trajectory in the (spurious flip, core flip) plane, epochs 1-50, 5-epoch rolling mean."""
    sets = [("waterbirds", "WB seed ", "Waterbirds (3 seeds)"), ("cmnist", "CM BG", "Colored MNIST BG"), ("cmnist", "CM FG", "Colored MNIST FG")]
    sets = [s for s in sets if ((d.dataset == s[0]) & d.run.str.startswith(s[1])).any()]
    fig, axes = plt.subplots(1, len(sets), figsize=(4.4 * len(sets), 4.4), squeeze=False)
    for ax, (ds, prefix, label) in zip(axes[0], sets):
        for ctag, corr in CORRS:
            for k, (run, g) in enumerate(d[(d.dataset == ds) & d.run.str.startswith(prefix) & (d["corr"] == ctag) & (d.subset == "all")
                                           & (d.epoch >= 1)].groupby("run")):
                g = g.sort_values("epoch"); x = g.spurious_flip.rolling(5, min_periods=1).mean() * 100
                y = g.core_flip.rolling(5, min_periods=1).mean() * 100
                ax.plot(x, y, color=CORR_COLOR[ctag], lw=1.2, label=f"corr {corr:.2f}" if k == 0 else None)
                ax.scatter(x.iloc[-1], y.iloc[-1], color=CORR_COLOR[ctag], s=18, zorder=3)
        ax.plot([0, 100], [0, 100], color=MUTED, lw=0.7, ls=":")
        ax.text(3, 94, "uses the core", fontsize=7, color=CORE); ax.text(97, 3, "uses the shortcut", fontsize=7, color=SPUR, ha="right")
        ax.text(3, 3, "uses neither", fontsize=7, color=MUTED)
        style(ax, label, "spurious flip (%)", "core flip (%)"); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
        ax.legend(fontsize=7, frameon=False, loc="center right")
    fig.suptitle("Training trajectory in the (spurious flip, core flip) plane; 5-epoch rolling mean, dot = epoch 50",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save(fig, "04_flip_plane")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-csv", type=Path, default=REPO / "reports/waterbirds_rerun50/seeds/probe_team_protocol.csv")
    a = ap.parse_args()
    probe = None
    if a.probe_csv.exists():
        probe = pd.read_csv(a.probe_csv, dtype={"corr": str}); probe = probe[probe.target == "core"]; probe["corr"] = probe["corr"].str.zfill(3)
    d = load(); tables(d, probe)
    if (d.dataset == "waterbirds").any():
        fig_curves(d, "waterbirds", [(lambda x: x.run.notna(), "")], "01_waterbirds_flip",
                   "Waterbirds: spurious flip vs core flip per epoch (validation, 1,199 images; three lines each = seeds 42/43/44)")
    if (d.dataset == "cmnist").any():
        fig_curves(d, "cmnist", [(lambda x: x["mode"] == "bg", "BG, "), (lambda x: x["mode"] == "fg", "FG, ")], "02_cmnist_flip",
                   "Colored MNIST: spurious flip (paired color swap) vs core flip (other digit, same color) per epoch (validation, 5,000 images)")
    fig_split(d); fig_plane(d)
