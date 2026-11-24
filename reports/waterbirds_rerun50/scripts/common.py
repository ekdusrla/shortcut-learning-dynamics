"""Shared paths, run list, derived metrics and plot style for the Waterbirds rerun figures.

WB_SEED (env, default 42) picks which rerun seed probe_features.py / figs_checkpoints.py read. Seed 42 writes the
main CSVs and figures; other seeds write only CSVs into seeds/ (seeds.py makes the cross-seed figures)."""
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "waterbirds_rerun50"
TEAM = REPO / "reports" / "waterbirds_vgg50" / "shared" / "source_metrics"
OUT = REPO / "reports" / "waterbirds_rerun50"
FIG = OUT / "figures"
CORRS = [("095", 0.95), ("080", 0.80), ("065", 0.65)]
# Correlation is ordinal: one hue, darker = stronger correlation.
CORR_COLOR = {"095": "#08306b", "080": "#2171b5", "065": "#6baed6"}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
SEED = int(os.environ.get("WB_SEED", "42"))
SEEDS = (42, 43, 44)
SEED_DIR = OUT / "seeds"
GROUPS = {"g00": "Landbird / Land", "g01": "Landbird / Water", "g10": "Waterbird / Land", "g11": "Waterbird / Water"}


def run_name(ctag, seed=None):
    return f"A_corr{ctag}_seed{SEED if seed is None else seed}_e50_contentv3"


def out_csv(name):
    """Seed 42: <OUT>/<name>. Other seeds: <OUT>/seeds/<name>_seed<N>.csv."""
    if SEED == 42: return OUT / name
    SEED_DIR.mkdir(exist_ok=True)
    return SEED_DIR / name.replace(".csv", f"_seed{SEED}.csv")


def add_derived(h, prefix="val_"):
    """Balanced accuracy, balanced conflicting accuracy and class recalls from the four group accuracies."""
    g = {k: (h[f"{prefix}{k}_acc"], h[f"{prefix}{k}_n"]) for k in GROUPS}
    land = (g["g00"][0] * g["g00"][1] + g["g01"][0] * g["g01"][1]) / (g["g00"][1] + g["g01"][1])
    water = (g["g10"][0] * g["g10"][1] + g["g11"][0] * g["g11"][1]) / (g["g10"][1] + g["g11"][1])
    h[f"{prefix}landbird_recall"], h[f"{prefix}waterbird_recall"] = land, water
    h[f"{prefix}balanced_acc"] = (land + water) / 2
    h[f"{prefix}balanced_conflicting_acc"] = (g["g01"][0] + g["g10"][0]) / 2
    return h


def load_history(source="rerun", seed=None):
    """{ctag: history DataFrame indexed by epoch} for a rerun seed or the team's shared copies (seed 42)."""
    base = RUNS if source == "rerun" else TEAM
    seed = 42 if source == "team" else seed
    out = {}
    for ctag, _ in CORRS:
        p = base / run_name(ctag, seed) / "history.csv"
        if p.exists():
            out[ctag] = add_derived(pd.read_csv(p).set_index("epoch").copy())
    return out


def epoch_json(ctag, epoch, source="rerun", seed=None):
    base = RUNS if source == "rerun" else TEAM
    seed = 42 if source == "team" else seed
    p = base / run_name(ctag, seed) / "epoch_metrics" / f"epoch_{epoch:03d}.json"
    return json.loads(p.read_text()) if p.exists() else None


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title: ax.set_title(title, fontsize=10, color=INK, loc="left")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8, color=MUTED)


def save(fig, name, force=False):
    if SEED != 42 and not force:  # per-seed figures are not kept; seeds.py draws the cross-seed ones
        plt.close(fig); return
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")
