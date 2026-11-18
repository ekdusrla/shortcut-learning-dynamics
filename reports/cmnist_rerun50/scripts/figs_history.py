"""Figures from per-epoch validation history (no model needed).

01 learning curves, 02 where recolored predictions go, 03 robust-behaviour epoch (H2),
04 rerun vs team reproduction, 05 epoch-50 summary; also writes summary CSVs.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import (CORRS, CORR_COLOR, FIG, INK, MODES, MUTED, OUT, TEAM_HISTORY,
                    run_dir, save, style, team_id)


def load():
    out = {}
    for tag, _, _ in MODES:
        for ctag, corr in CORRS:
            p = run_dir(tag, ctag) / "history.csv"
            if not p.exists():
                continue
            h = pd.read_csv(p).set_index("epoch")
            h["shape_bias"] = h.val_cf_all_acc / (h.val_cf_all_acc + h.val_cf_all_color_rate)
            h["other_rate"] = 1 - h.val_cf_all_acc - h.val_cf_all_color_rate
            out[(tag, ctag)] = h
    return out


def robust_epoch(h, t):
    """First epoch of 3 consecutive epochs with conflicting acc>=t, 9-color acc>=t, 9-color flip<=1-t."""
    ok = ((h.val_conflicting_acc >= t) & (h.val_cf_all_acc >= t) & (h.val_cf_all_flip_rate <= 1 - t)).loc[1:]
    e = ok.index.to_numpy(); v = ok.to_numpy()
    for i in range(len(v) - 2):
        if v[i] and v[i + 1] and v[i + 2] and e[i + 2] - e[i] == 2:
            return int(e[i])
    return None


def fig_curves(H):
    metrics = [("val_acc", "Val accuracy (%)"),
               ("val_cf_all_flip_rate", "9-color flip rate (%)"),
               ("shape_bias", "Shape bias (%)"),
               ("val_wga", "Worst-group acc (%)")]
    fig, axes = plt.subplots(len(metrics), 2, figsize=(11, 11), sharex=True)
    for c, (tag, _, mname) in enumerate(MODES):
        for r, (m, label) in enumerate(metrics):
            ax = axes[r, c]
            for ctag, corr in CORRS:
                if (tag, ctag) in H:
                    h = H[(tag, ctag)]
                    ax.plot(h.index, h[m] * 100, color=CORR_COLOR[ctag], lw=1.8, label=f"train corr {corr:.2f}")
            style(ax, mname if r == 0 else None, "epoch" if r == len(metrics) - 1 else None, label if c == 0 else None)
            ax.set_xlim(0, 50)
            if m == "val_cf_all_flip_rate": ax.set_yscale("symlog", linthresh=5); ax.set_ylim(0, 100)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Colored MNIST x VGG11_BN (224px, pairing all, lr 0.01 constant, seed 42): validation metrics per epoch",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); save(fig, "01_learning_curves")


def fig_outcomes(H):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    cols = {"correct digit": "#2da44e", "color's class": "#cf222e", "another digit": "#afb8c1"}
    for r, (tag, _, mname) in enumerate(MODES):
        for c, (ctag, corr) in enumerate(CORRS):
            ax = axes[r, c]
            if (tag, ctag) not in H: ax.axis("off"); continue
            h = H[(tag, ctag)].loc[0:]
            ys = [h.val_cf_all_acc * 100, h.val_cf_all_color_rate * 100, h.other_rate * 100]
            ax.stackplot(h.index, *ys, colors=list(cols.values()), labels=list(cols), edgecolor="white", linewidth=0.4)
            ax.set_xscale("symlog", linthresh=3); ax.set_xlim(0, 50); ax.set_ylim(0, 100)
            ax.set_xticks([0, 1, 2, 3, 5, 10, 20, 50]); ax.set_xticklabels([0, 1, 2, 3, 5, 10, 20, 50])
            style(ax, f"{mname}, corr {corr:.2f}", "epoch (symlog)" if r == 1 else None, "share of recolored digits (%)" if c == 0 else None)
    axes[0, 0].legend(loc="lower right", fontsize=8, frameon=True)
    fig.suptitle("What the model predicts when a validation digit is repainted with each color other than its own (color != digit)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "02_recolor_outcomes")


def fig_robust(H):
    rows = []
    for (tag, ctag), h in H.items():
        for t in (0.90, 0.95, 0.98):
            rows.append(dict(mode=tag, corr=ctag, threshold=t, epoch=robust_epoch(h, t), last_epoch=int(h.index.max())))
    df = pd.DataFrame(rows); df.to_csv(OUT / "robust_epochs.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, t in enumerate((0.90, 0.95, 0.98)):
        ax = axes[k]; x = np.arange(2); w = 0.26
        for j, (ctag, corr) in enumerate(CORRS):
            vals = []
            for tag, _, _ in MODES:
                r = df[(df["mode"] == tag) & (df["corr"] == ctag) & (df.threshold == t)]
                vals.append(np.nan if r.empty or pd.isna(r.epoch.iloc[0]) else float(r.epoch.iloc[0]))
            last = max((int(H[k2].index.max()) for k2 in H), default=50)
            for i, v in enumerate(vals):
                xx = x[i] + (j - 1) * w
                if np.isnan(v):
                    ax.bar(xx, last, w, color="white", edgecolor=CORR_COLOR[ctag], hatch="//", lw=1)
                    ax.text(xx, last + 1, "not\nreached", ha="center", va="bottom", fontsize=7, color=MUTED)
                else:
                    ax.bar(xx, v, w, color=CORR_COLOR[ctag], label=f"corr {corr:.2f}" if i == 0 else None)
                    ax.text(xx, v + 1, f"{int(v)}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(["BG", "FG"])
        style(ax, f"threshold {int(t * 100)}%: first epoch of 3 robust epochs", None, "epoch" if k == 0 else None)
        ax.set_ylim(0, 62)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("H2: epoch at which behaviour becomes color-robust (conflicting acc >= t, 9-color acc >= t, 9-color flip <= 1 - t)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "03_robust_epoch")


def fig_repro(H):
    if not TEAM_HISTORY.exists(): return
    team = pd.read_csv(TEAM_HISTORY)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True)
    rows = []
    for r, (tag, _, mname) in enumerate(MODES):
        for c, (ctag, corr) in enumerate(CORRS):
            ax = axes[r, c]; t = team[team.run_id == team_id(tag, ctag)].set_index("epoch").loc[0:50]
            ax.plot(t.index, t.val_acc * 100, color=MUTED, lw=2.5, alpha=0.6, label="team (server GPU)")
            ax.plot(t.index, t.val_cf_all_flip_rate * 100, color="#d4a72c", lw=2.5, alpha=0.6)
            if (tag, ctag) in H:
                h = H[(tag, ctag)]
                ax.plot(h.index, h.val_acc * 100, color=CORR_COLOR["095"], lw=1.2, ls="--", label="rerun (RTX 5070 Ti)")
                ax.plot(h.index, h.val_cf_all_flip_rate * 100, color="#9a6700", lw=1.2, ls="--")
                common = t.index.intersection(h.index).drop(0, errors="ignore")
                for m in ("val_acc", "val_conflicting_acc", "val_wga", "val_cf_flip_rate", "val_cf_all_flip_rate", "val_cf_all_acc"):
                    d = (h.loc[common, m] - t.loc[common, m]) * 100
                    rows.append(dict(mode=tag, corr=ctag, metric=m, epochs=len(common), mean_abs_diff_pp=d.abs().mean(), max_abs_diff_pp=d.abs().max()))
            ax.set_xscale("symlog", linthresh=3); ax.set_xlim(0, 50)
            style(ax, f"{mname}, corr {corr:.2f}", "epoch (symlog)" if r == 1 else None, "%" if c == 0 else None)
            ax.text(0.98, 0.80, "validation accuracy", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
            ax.text(0.98, 0.12, "9-color flip", transform=ax.transAxes, ha="right", fontsize=7, color="#9a6700")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.45))
    fig.suptitle("Reproduction: same command and code, team run (solid) vs this rerun (dashed)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "04_reproduction")
    if rows: pd.DataFrame(rows).to_csv(OUT / "reproduction_diff.csv", index=False)


def fig_final(H):
    rows = []
    for tag, mode, _ in MODES:
        for ctag, corr in CORRS:
            p = run_dir(tag, ctag) / "test_metrics.json"
            if not p.exists(): continue
            m = json.loads(p.read_text())["metrics"]
            rows.append(dict(mode=tag, corr=ctag, test_acc=m["acc"], test_conflicting=m["conflicting_acc"], test_wga=m["wga"],
                             test_flip=m["cf_flip_rate"], test_flip9=m["cf_all_flip_rate"], test_acc9=m["cf_all_acc"],
                             test_color9=m["cf_all_color_rate"],
                             test_shape_bias=m["cf_all_acc"] / (m["cf_all_acc"] + m["cf_all_color_rate"])))
    if not rows: return
    df = pd.DataFrame(rows); df.to_csv(OUT / "test_epoch50.csv", index=False)
    panels = [("test_acc", "Test accuracy (%)"), ("test_wga", "Test worst-group accuracy (%)"),
              ("test_flip9", "Test 9-color flip (%)"), ("test_color9", "Test: recolors predicted as the color's class (%)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for k, (m, title) in enumerate(panels):
        ax = axes[k]; x = np.arange(2); w = 0.26
        for j, (ctag, corr) in enumerate(CORRS):
            for i, (tag, _, _) in enumerate(MODES):
                r = df[(df["mode"] == tag) & (df.corr == ctag)]
                if r.empty: continue
                v = r[m].iloc[0] * 100
                ax.bar(x[i] + (j - 1) * w, v, w, color=CORR_COLOR[ctag], label=f"corr {corr:.2f}" if i == 0 else None)
                ax.text(x[i] + (j - 1) * w, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(["BG", "FG"]); style(ax, title)
        if m in ("test_acc", "test_wga"): ax.set_ylim(80, 101)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Epoch 50, official test split (10,000 digits, color independent of digit)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save(fig, "05_epoch50_test")


if __name__ == "__main__":
    H = load()
    print("runs with history:", sorted(H), {k: int(v.index.max()) for k, v in H.items()})
    fig_curves(H); fig_outcomes(H); fig_robust(H); fig_repro(H); fig_final(H)
