"""Shared paths, run list and plot style for the rerun50 report figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
RUNS = REPO / "runs" / "rerun50"
OUT = REPO / "reports" / "cmnist_rerun50"
FIG = OUT / "figures"
TEAM_HISTORY = REPO / "reports" / "metric_final_analysis_406" / "history_combined.csv"
MODES = [("bg", "background", "Background (BG)"), ("fg", "foreground", "Foreground (FG)")]
CORRS = [("095", 0.95), ("080", 0.80), ("065", 0.65)]
# Correlation is ordinal: one hue, darker = stronger correlation.
CORR_COLOR = {"095": "#08306b", "080": "#2171b5", "065": "#6baed6"}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"


def run_dir(tag, ctag):
    return RUNS / f"vgg11_{tag}_corr{ctag}"


def team_id(tag, ctag):
    return f"vgg11_{tag}_corr{ctag}_seed42_e50_img224_all"


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


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / f"{name}.png")
