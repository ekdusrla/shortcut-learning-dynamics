"""Cross-seed summary: the team run and rerun seeds 42 / 43 / 44 (same configuration, only the training seed differs).

Reads history.csv and epoch_metrics of every run, plus the per-seed checkpoint CSVs
(seed 42: <OUT>/*.csv, seeds 43/44: <OUT>/seeds/*_seed<N>.csv from WB_SEED=<N> figs_checkpoints.py / probe_features.py).
Writes seeds/seeds_*.csv and figures 11 (curves), 12 (spread of summary values), 13 (probes)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import CORRS, INK, MUTED, OUT, SEED_DIR, SEEDS, add_derived, epoch_json, load_history, save, style

PHASES = [("early", 1, 10), ("middle", 11, 30), ("late", 31, 50)]
METRICS = ["val_acc", "val_balanced_acc", "val_wga", "val_cf_flip_rate", "val_cf_correct_to_incorrect_rate",
           "val_balanced_conflicting_acc", "val_waterbird_recall"]
RUNS = [("team", "team", 42)] + [(f"seed {s}", "rerun", s) for s in SEEDS]
# Seeds are categorical within a correlation panel; the team run is the gray reference.
RUN_COLOR = {"team": "#afb8c1", "seed 42": "#2171b5", "seed 43": "#d95f02", "seed 44": "#7570b3"}
SEED_DIR.mkdir(exist_ok=True)


def histories():
    return {run: load_history(src, seed) for run, src, seed in RUNS}


def first_run(ok, epochs, dur):
    return next((int(epochs[i]) for i in range(len(ok) - dur + 1) if ok[i:i + dur].all()), None)


def tables(H):
    phase, robust, collapse, peaks = [], [], [], []
    for run, _, _ in RUNS:
        for ctag, _ in CORRS:
            h = H[run][ctag]
            for ph, a, b in PHASES:
                phase.append(dict(run=run, corr=ctag, phase=ph, **{m: h.loc[a:b, m].mean() for m in METRICS}))
            g = h.loc[1:]; e = g.index.to_numpy()
            for wt in (0.2, 0.3):
                ok = ((g.val_balanced_conflicting_acc >= 0.5) & (g.val_wga >= wt)).to_numpy()
                for dur in (3, 5, 10):
                    robust.append(dict(run=run, corr=ctag, wga_threshold=wt, duration=dur, first_epoch=first_run(ok, e, dur)))
            wb = g.val_waterbird_recall
            collapse.append(dict(run=run, corr=ctag, all_landbird_epochs=" ".join(map(str, wb.index[wb == 0])),
                                 wb_below5_epochs=" ".join(map(str, wb.index[wb < 0.05])), n_wb_below5=int((wb < 0.05).sum()),
                                 min_wb_recall=wb.min(), min_wb_epoch=int(wb.idxmin())))
            w = g.val_wga; pk = int(w.idxmax()); run_ok = (w >= 0.2).to_numpy()
            longest = max((len(s) for s in "".join("1" if x else "0" for x in run_ok).split("0")), default=0)
            peaks.append(dict(run=run, corr=ctag, peak_epoch=pk, peak_wga=w[pk], next_wga=w.get(pk + 1, np.nan),
                              longest_wga20_streak=longest))
    out = {}
    for name, rows in (("phase_means", phase), ("robust_proxy", robust), ("collapse", collapse), ("wga_peaks", peaks)):
        out[name] = pd.DataFrame(rows); out[name].to_csv(SEED_DIR / f"seeds_{name}.csv", index=False)
    rows = []
    for run, src, seed in RUNS:
        for ctag, _ in CORRS:
            js = epoch_json(ctag, 50, src, seed)
            s = add_derived(pd.DataFrame([js["splits"]["test"]]), prefix="").iloc[0]
            rows.append(dict(run=run, corr=ctag, acc=s["acc"], balanced_acc=s["balanced_acc"], wga=s["wga"],
                             flip=s["cf_flip_rate"], c2i=s["cf_correct_to_incorrect_rate"], waterbird_recall=s["waterbird_recall"]))
    out["epoch50_test"] = pd.DataFrame(rows); out["epoch50_test"].to_csv(SEED_DIR / "seeds_epoch50_test.csv", index=False)
    return out


def merged(name):
    """Concatenate the seed-42 CSV and the seeds/ copies with a seed column."""
    parts = []
    for s in SEEDS:
        p = OUT / name if s == 42 else SEED_DIR / name.replace(".csv", f"_seed{s}.csv")
        if p.exists():
            d = pd.read_csv(p, dtype={"corr": str}); d["corr"] = d["corr"].str.zfill(3); d.insert(0, "seed", s); parts.append(d)
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.to_csv(SEED_DIR / f"seeds_{name}", index=False)
    return df


def fig_curves(H):
    show = [("val_wga", "worst-group acc (%)"), ("val_balanced_conflicting_acc", "balanced conflicting acc (%)"),
            ("val_cf_flip_rate", "background-swap flip (%)")]
    fig, axes = plt.subplots(len(show), 3, figsize=(13, 7.8), sharex=True, sharey="row")
    for c, (ctag, corr) in enumerate(CORRS):
        for r, (m, lab) in enumerate(show):
            ax = axes[r, c]
            for run, _, _ in RUNS:
                h = H[run][ctag]
                ax.plot(h.index, h[m] * 100, color=RUN_COLOR[run], lw=2.4 if run == "team" else 1.2, label=run, zorder=1 if run == "team" else 2)
            if m == "val_balanced_conflicting_acc": ax.axhline(50, color=MUTED, lw=0.8, ls=":")
            if m == "val_wga": ax.axhline(20, color=MUTED, lw=0.8, ls=":")
            style(ax, f"corr {corr:.2f}" if r == 0 else None, "epoch" if r == len(show) - 1 else None, lab if c == 0 else None)
            ax.set_xlim(0, 50)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Same configuration, different training seed: validation curves (dotted: robust-proxy thresholds WGA 20%, balanced conflicting 50%)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.925)); save(fig, "11_seed_curves")


def fig_spread(T):
    pm = T["phase_means"]; late = pm[pm.phase == "late"]; e50 = T["epoch50_test"]
    show = [(late, "val_wga", "Val WGA, epochs 31–50 mean"), (late, "val_cf_flip_rate", "Val flip, epochs 31–50 mean"),
            (e50, "wga", "Test WGA, epoch 50"), (e50, "flip", "Test flip, epoch 50")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for ax, (df, m, title) in zip(axes, show):
        for k, (run, _, _) in enumerate(RUNS):
            vals = [df[(df["corr"] == c) & (df.run == run)][m].iloc[0] * 100 for c, _ in CORRS]
            ax.scatter(np.arange(3) + (k - 1.5) * 0.12, vals, s=22, color=RUN_COLOR[run], label=run, zorder=2)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"corr {c:.2f}" for _, c in CORRS], fontsize=8); ax.set_xlim(-0.5, 2.5)
        style(ax, f"{title} (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0.01, 0.92))
    fig.suptitle("Spread over runs: late-phase means and epoch-50 test values", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.87)); save(fig, "12_seed_spread")


def fig_probe(P, H):
    if P.empty: return
    f4 = P[P.layer == "f4"].groupby(["seed", "corr", "epoch"])[["bird", "place"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, (ctag, corr) in zip(axes, CORRS):
        for s in SEEDS:
            g = f4[(f4.seed == s) & (f4["corr"] == ctag)]; col = RUN_COLOR[f"seed {s}"]
            ax.plot(g.epoch, g.place * 100, color=col, lw=1.2, label=f"seed {s}: background" if ax is axes[0] else None)
            ax.plot(g.epoch, g.bird * 100, color=col, lw=1.2, ls="--", label=f"seed {s}: bird class" if ax is axes[0] else None)
        ax.axhline(50, color=MUTED, lw=0.8, ls=":")
        style(ax, f"corr {corr:.2f}", "epoch", "probe accuracy, held-out (%)" if ax is axes[0] else None); ax.set_xlim(0, 50); ax.set_ylim(20, 102)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.01, 0.92))
    fig.suptitle("Linear probes on the last conv features per seed (solid: background, dashed: bird class; mean of 3 splits, held-out n = 32)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.84)); save(fig, "13_seed_probe")


if __name__ == "__main__":
    H = histories(); T = tables(H)
    stress, cam, probe = merged("stress_epoch50.csv"), merged("gradcam_bird_share.csv"), merged("probe_curves.csv")
    if not probe.empty:
        f4 = probe[probe.layer == "f4"]
        rows = [dict(seed=s, corr=c, phase=ph, bird=g.bird.mean(), place=g.place.mean())
                for (s, c), d in f4.groupby(["seed", "corr"]) for ph, a, b in [("epoch0", 0, 0)] + PHASES
                for g in [d[(d.epoch >= a) & (d.epoch <= b)]]]
        pd.DataFrame(rows).to_csv(SEED_DIR / "seeds_probe_phase.csv", index=False)
    fig_curves(H); fig_spread(T); fig_probe(probe, H)
    print("stress seeds:", sorted(stress.seed.unique()) if not stress.empty else [],
          "cam seeds:", sorted(cam.seed.unique()) if not cam.empty else [])
