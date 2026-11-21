"""Figures and tables from history.csv / epoch_metrics only (CPU, seconds).

01 learning curves, 02 reproduction vs team, 03 group accuracy, 04 epoch-50 val/test vs team;
CSV: reproduction_diff.csv, phase_means.csv, robust_proxy.csv, epoch50_compare.csv, wga_peaks.csv."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import CORRS, CORR_COLOR, GROUPS, INK, MUTED, OUT, add_derived, epoch_json, load_history, save, style

PHASES = [("early", 1, 10), ("middle", 11, 30), ("late", 31, 50)]
METRICS = [("val_acc", "Val accuracy (%)"), ("val_balanced_acc", "Balanced accuracy (%)"), ("val_wga", "Worst-group acc (%)"),
           ("val_cf_flip_rate", "Background-swap flip (%)"), ("val_cf_correct_to_incorrect_rate", "Correct→incorrect (%)"),
           ("val_balanced_conflicting_acc", "Balanced conflicting acc (%)")]


def fig_curves(R):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True)
    for ax, (m, lab) in zip(axes.flat, METRICS):
        for ctag, corr in CORRS:
            if ctag in R: ax.plot(R[ctag].index, R[ctag][m] * 100, color=CORR_COLOR[ctag], lw=1.5, label=f"train corr {corr:.2f}")
        style(ax, lab, "epoch" if ax in axes[1] else None); ax.set_xlim(0, 50)
    axes[0, 0].axhline(77.15, color=MUTED, ls=":", lw=1); axes[0, 0].text(50, 77.6, "all-landbird 77.2%", fontsize=7, color=MUTED, ha="right")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Waterbirds × VGG11_BN rerun (lr 0.01 constant, seed 42): validation metrics per epoch", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "01_learning_curves")


def fig_repro(R, T):
    rows = []
    for ctag, _ in CORRS:
        if ctag not in R or ctag not in T: continue
        n = min(R[ctag].index.max(), T[ctag].index.max())
        for m, _ in METRICS:
            d = (R[ctag][m] - T[ctag][m]).loc[1:n].abs() * 100
            rows.append(dict(corr=ctag, metric=m, epochs=int(n), mean_abs_diff_pp=d.mean(), max_abs_diff_pp=d.max(),
                             epoch0_equal=bool(abs(R[ctag][m].loc[0] - T[ctag][m].loc[0]) < 1e-9)))
    pd.DataFrame(rows).to_csv(OUT / "reproduction_diff.csv", index=False)
    show = [("val_acc", "Val accuracy (%)"), ("val_wga", "Worst-group acc (%)"), ("val_cf_flip_rate", "Background-swap flip (%)")]
    fig, axes = plt.subplots(len(show), 3, figsize=(13, 7.5), sharex=True, sharey="row")
    for c, (ctag, corr) in enumerate(CORRS):
        for r, (m, lab) in enumerate(show):
            ax = axes[r, c]
            if ctag in T: ax.plot(T[ctag].index, T[ctag][m] * 100, color="#afb8c1", lw=2.4, label="team (server GPU)")
            if ctag in R: ax.plot(R[ctag].index, R[ctag][m] * 100, color=CORR_COLOR[ctag], lw=1.2, ls="--", label="rerun (RTX 5070 Ti)")
            style(ax, f"corr {corr:.2f}" if r == 0 else None, "epoch" if r == len(show) - 1 else None, lab if c == 0 else None)
            ax.set_xlim(0, 50)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Reproduction: same code and configuration, team run (gray) vs this rerun (dashed)", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "02_reproduction")


def fig_groups(R):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    cols = {"g00": "#1a7f37", "g01": "#9a6700", "g10": "#cf222e", "g11": "#0969da"}
    for ax, (ctag, corr) in zip(axes, CORRS):
        if ctag not in R: continue
        for g, name in GROUPS.items():
            ax.plot(R[ctag].index, R[ctag][f"val_{g}_acc"] * 100, color=cols[g], lw=1.3, label=name)
        style(ax, f"corr {corr:.2f}", "epoch", "group accuracy (%)" if ax is axes[0] else None); ax.set_xlim(0, 50)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.9))
    fig.suptitle("Validation accuracy per bird × background group (conflicting groups: Landbird/Water, Waterbird/Land)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.82)); save(fig, "03_group_accuracy")


def split_row(js, split):
    s = add_derived(pd.DataFrame([js["splits"][split]]), prefix="").iloc[0]
    return dict(acc=s["acc"], balanced_acc=s["balanced_acc"], wga=s["wga"], flip=s["cf_flip_rate"],
                c2i=s["cf_correct_to_incorrect_rate"], waterbird_recall=s["waterbird_recall"])


def fig_epoch50():
    rows = []
    for ctag, _ in CORRS:
        for src in ("team", "rerun"):
            js = epoch_json(ctag, 50, src)
            if js is None: continue
            for split in ("val", "test"):
                rows.append(dict(corr=ctag, source=src, split=split, **split_row(js, split)))
    df = pd.DataFrame(rows); df.to_csv(OUT / "epoch50_compare.csv", index=False)
    if df.empty: return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=False)
    for ax, (m, lab) in zip(axes, [("acc", "Test accuracy"), ("balanced_acc", "Test balanced acc"), ("wga", "Test WGA"), ("flip", "Test background-swap flip")]):
        x = np.arange(3); w = 0.38
        for j, src in enumerate(("team", "rerun")):
            vals = [df[(df["corr"] == c) & (df.source == src) & (df.split == "test")][m].mean() * 100 for c, _ in CORRS]
            ax.bar(x + (j - 0.5) * w, vals, w, color="#afb8c1" if src == "team" else "#2171b5", label=src)
            for xi, v in zip(x, vals):
                if not np.isnan(v): ax.text(xi + (j - 0.5) * w, v + 0.8, f"{v:.1f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x); ax.set_xticklabels([f"corr {c:.2f}" for _, c in CORRS], fontsize=8); style(ax, f"{lab} (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Epoch 50 on the test set: team run vs rerun", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88)); save(fig, "04_epoch50_test")


def tables(R, T):
    rows = []
    for src, H in (("team", T), ("rerun", R)):
        for ctag, _ in CORRS:
            if ctag not in H: continue
            h = H[ctag]
            for ph, a, b in PHASES:
                seg = h.loc[a:b]
                rows.append(dict(source=src, corr=ctag, phase=ph, epochs=f"{a}-{b}", **{m: seg[m].mean() for m, _ in METRICS}))
    pd.DataFrame(rows).to_csv(OUT / "phase_means.csv", index=False)
    rows = []
    for src, H in (("team", T), ("rerun", R)):
        for ctag, _ in CORRS:
            if ctag not in H: continue
            h = H[ctag].loc[1:]
            for wga_t in (0.2, 0.3):
                ok = ((h.val_balanced_conflicting_acc >= 0.5) & (h.val_wga >= wga_t)).to_numpy(); e = h.index.to_numpy()
                for dur in (3, 5, 10):
                    first = next((int(e[i]) for i in range(len(ok) - dur + 1) if ok[i:i + dur].all()), None)
                    rows.append(dict(source=src, corr=ctag, wga_threshold=wga_t, duration=dur, first_epoch=first))
    pd.DataFrame(rows).to_csv(OUT / "robust_proxy.csv", index=False)
    rows = []
    for src, H in (("team", T), ("rerun", R)):
        for ctag, _ in CORRS:
            if ctag not in H: continue
            w = H[ctag].val_wga; pk = int(w.idxmax())
            rows.append(dict(source=src, corr=ctag, peak_epoch=pk, peak_wga=w[pk], next_wga=w.get(pk + 1, np.nan),
                             increases_flip=int((H[ctag].val_cf_flip_rate.loc[1:].diff() > 0).sum())))
    pd.DataFrame(rows).to_csv(OUT / "wga_peaks.csv", index=False)


if __name__ == "__main__":
    R, T = load_history("rerun"), load_history("team")
    print("rerun epochs:", {k: int(v.index.max()) for k, v in R.items()})
    fig_curves(R); fig_repro(R, T); fig_groups(R); fig_epoch50(); tables(R, T)
