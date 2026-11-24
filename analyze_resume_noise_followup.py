#!/usr/bin/env python3
"""
analyze_resume_noise_followup.py

Decide whether the Epoch50->100 "shortcut dependence goes back up" effect
(observed for corr=.95 BG/FG in reports/metric_final_analysis_406) is a real
training-dynamics signal or resume-specific computational noise.

Inputs are the per-run history CSVs produced by shortcut_experiment.py
(history.csv inside each run directory), or a combined CSV that already has
a `run_id` column (e.g. history_combined.csv / key_epochs.csv style).

You tell this script, for each run_id, which *role* it plays:
  - original        : the run you already have (e.g. vgg11_bg_corr095_seed42_e50_img224_all
                       merged with its resume-to-100 extension)
  - resume_repeat    : an independent re-resume of the SAME epoch-50 checkpoint to epoch 100
  - continuous       : a from-scratch, never-resumed run covering the same epoch range
  - seed_variant     : a different-seed continuous run (for the "is it a general pattern" check)

Usage example
-------------
python analyze_resume_noise_followup.py \
  --csv runs/vgg11_bg_corr095_seed42_e50_img224_all/history.csv:original \
  --csv runs/vgg11_bg_corr095_seed42_resume50to100_rep1/history.csv:resume_repeat \
  --csv runs/vgg11_bg_corr095_seed42_resume50to100_rep2/history.csv:resume_repeat \
  --csv runs/vgg11_bg_corr095_seed42_continuous100/history.csv:continuous \
  --metric val_cf_all_flip_rate --metric val_cf_all_acc \
  --early-start 46 --early-end 50 --late-start 96 --late-end 100 \
  --out-dir analysis_out/resume_noise_bg095

Each --csv argument is "path:role". Path may be a per-run history.csv (run_id
is inferred from its parent directory name) or a combined CSV with a run_id
column (all run_ids in it get the same role, unless you pass --run-id to
filter to just one).

What it prints / saves
-----------------------
- analysis_out/.../phase_means.csv      : per-run early/late phase means + delta, by metric
- analysis_out/.../verdict.txt          : plain-language verdict per metric
- analysis_out/.../trajectory_<metric>.png : epoch-wise overlay of all runs, colored by role
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

ROLE_ORDER = ["original", "resume_repeat", "continuous", "seed_variant"]
ROLE_COLORS = {
    "original": "#1f77b4",
    "resume_repeat": "#ff7f0e",
    "continuous": "#2ca02c",
    "seed_variant": "#9467bd",
}


def load_one(path_role: str, run_id_filter: str | None) -> pd.DataFrame:
    path_str, role = path_role.rsplit(":", 1)
    if role not in ROLE_ORDER:
        raise SystemExit(f"Unknown role '{role}' in --csv {path_role}. Must be one of {ROLE_ORDER}.")
    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "run_id" not in df.columns:
        # Per-run history.csv with no run_id column: infer from the run directory name.
        inferred = path.parent.name
        df["run_id"] = inferred
    if run_id_filter:
        df = df[df["run_id"] == run_id_filter]
        if df.empty:
            raise SystemExit(f"run_id '{run_id_filter}' not found in {path}")
    df = df.copy()
    df["role"] = role
    df["source_path"] = str(path)
    return df


def phase_mean(df: pd.DataFrame, metric: str, start: int, end: int) -> float:
    window = df[(df["epoch"] >= start) & (df["epoch"] <= end)]
    if window.empty or metric not in window.columns:
        return float("nan")
    return float(window[metric].mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", action="append", required=True, dest="csvs",
                     help="path/to/history.csv:role (role in {original,resume_repeat,continuous,seed_variant}); repeatable")
    ap.add_argument("--run-id", default=None,
                     help="if a --csv points at a combined CSV with multiple run_ids, filter to this one")
    ap.add_argument("--metric", action="append", required=True, dest="metrics",
                     help="column name to analyze, e.g. val_cf_all_flip_rate; repeatable")
    ap.add_argument("--early-start", type=int, default=46)
    ap.add_argument("--early-end", type=int, default=50)
    ap.add_argument("--late-start", type=int, default=96)
    ap.add_argument("--late-end", type=int, default=100)
    ap.add_argument("--out-dir", default="analysis_out/resume_noise_followup")
    args = ap.parse_args()

    frames = [load_one(c, args.run_id) for c in args.csvs]
    all_df = pd.concat(frames, ignore_index=True)
    if "epoch" not in all_df.columns:
        raise SystemExit("No 'epoch' column found in the input CSV(s).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metric in args.metrics:
        if metric not in all_df.columns:
            print(f"[warn] metric '{metric}' not present in the data; skipping.", file=sys.stderr)
            continue
        for run_id, g in all_df.groupby("run_id"):
            role = g["role"].iloc[0]
            early = phase_mean(g, metric, args.early_start, args.early_end)
            late = phase_mean(g, metric, args.late_start, args.late_end)
            rows.append({
                "metric": metric,
                "run_id": run_id,
                "role": role,
                f"early_{args.early_start}-{args.early_end}": early,
                f"late_{args.late_start}-{args.late_end}": late,
                "delta": late - early,
            })

    phase_df = pd.DataFrame(rows)
    phase_csv = out_dir / "phase_means.csv"
    phase_df.to_csv(phase_csv, index=False)
    print(f"Wrote {phase_csv}")

    verdict_lines = []
    for metric in args.metrics:
        sub = phase_df[phase_df["metric"] == metric]
        if sub.empty:
            continue
        verdict_lines.append(f"=== {metric} ===")

        orig = sub[sub["role"] == "original"]
        reps = sub[sub["role"] == "resume_repeat"]
        cont = sub[sub["role"] == "continuous"]
        seedv = sub[sub["role"] == "seed_variant"]

        if not orig.empty:
            orig_delta = orig["delta"].iloc[0]
            verdict_lines.append(f"original run delta (late - early): {orig_delta:+.4f}")
        else:
            orig_delta = None
            verdict_lines.append("no 'original' run supplied for this metric.")

        if not reps.empty:
            rep_deltas = reps["delta"].to_numpy()
            noise_std = float(np.std(rep_deltas, ddof=1)) if len(rep_deltas) > 1 else float("nan")
            noise_range = (float(np.min(rep_deltas)), float(np.max(rep_deltas)))
            verdict_lines.append(
                f"resume_repeat deltas (n={len(rep_deltas)}): {np.round(rep_deltas, 4).tolist()} "
                f"-> std={noise_std:.4f}, range={noise_range}"
            )
            if orig_delta is not None and not np.isnan(noise_std) and noise_std > 0:
                ratio = abs(orig_delta) / noise_std
                verdict_lines.append(f"|original delta| / repeat-std = {ratio:.2f}x")
                if ratio >= 3:
                    verdict_lines.append(
                        "-> original delta is >=3x the resume-repeat noise std: "
                        "likely a REAL signal, not resume noise alone."
                    )
                elif ratio <= 1.5:
                    verdict_lines.append(
                        "-> original delta is within ~1.5x of resume-repeat noise std: "
                        "consistent with being mostly NOISE, not a reliable effect."
                    )
                else:
                    verdict_lines.append(
                        "-> ambiguous: original delta is somewhat larger than repeat noise "
                        "but not by a comfortable margin. More repeats would help."
                    )
            elif orig_delta is not None and len(rep_deltas) >= 1:
                in_range = noise_range[0] <= orig_delta <= noise_range[1]
                verdict_lines.append(
                    f"only one repeat available, so no std: original delta "
                    f"{'falls inside' if in_range else 'falls OUTSIDE'} the single repeat's value."
                )
        else:
            verdict_lines.append("no 'resume_repeat' runs supplied: cannot estimate resume noise floor.")

        if not cont.empty:
            cont_deltas = cont["delta"].to_numpy()
            verdict_lines.append(
                f"continuous (never-resumed) run delta(s): {np.round(cont_deltas, 4).tolist()}"
            )
            if orig_delta is not None and len(cont_deltas) > 0:
                same_sign = all(np.sign(d) == np.sign(orig_delta) for d in cont_deltas if d != 0)
                verdict_lines.append(
                    f"-> continuous run(s) {'agree' if same_sign else 'do NOT agree'} in direction "
                    "with the original (resumed) run's delta."
                )
        else:
            verdict_lines.append(
                "no 'continuous' run supplied: cannot tell whether the effect needs a resume "
                "boundary to appear at all."
            )

        if not seedv.empty:
            seed_deltas = seedv["delta"].to_numpy()
            verdict_lines.append(f"other-seed continuous run delta(s): {np.round(seed_deltas, 4).tolist()}")

        verdict_lines.append("")

    verdict_txt = out_dir / "verdict.txt"
    verdict_txt.write_text("\n".join(verdict_lines), encoding="utf-8")
    print(f"Wrote {verdict_txt}")
    print("\n".join(verdict_lines))

    if plt is not None:
        for metric in args.metrics:
            if metric not in all_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for run_id, g in all_df.groupby("run_id"):
                g = g.sort_values("epoch")
                role = g["role"].iloc[0]
                ax.plot(g["epoch"], g[metric], label=f"{run_id} ({role})",
                        color=ROLE_COLORS.get(role, "gray"), alpha=0.85,
                        linewidth=2 if role == "original" else 1.3,
                        linestyle="-" if role in ("original", "continuous") else "--")
            ax.axvspan(args.early_start, args.early_end, color="gray", alpha=0.08)
            ax.axvspan(args.late_start, args.late_end, color="gray", alpha=0.08)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}: original vs resume repeats vs continuous")
            ax.legend(fontsize=7, loc="best")
            fig.tight_layout()
            fig_path = out_dir / f"trajectory_{metric}.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Wrote {fig_path}")
    else:
        print("[warn] matplotlib not available; skipped trajectory plots.", file=sys.stderr)


if __name__ == "__main__":
    main()
