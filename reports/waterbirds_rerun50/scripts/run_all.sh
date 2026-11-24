#!/bin/bash
# Usage: bash reports/waterbirds_rerun50/scripts/run_all.sh [seed ...]   (default seed: 42)
#   CORRS="0.8" bash ... 43 44   limits the correlations (default "0.95 0.8 0.65"); one driver per corr can run in parallel.
#   resumes unfinished runs, skips finished ones.
# Train Waterbirds Condition A corr .95/.80/.65 one at a time with the team configuration (lr 0.01 constant).
cd "$(dirname "$0")/../../.."  # repo root
mkdir -p runs/waterbirds_rerun50
seeds=("$@"); [[ ${#seeds[@]} -eq 0 ]] && seeds=(42)
for seed in "${seeds[@]}"; do
  for corr in ${CORRS:-0.95 0.8 0.65}; do
    code=$(printf '%03d' "$(python3 -c "print(round($corr*100))")")
    out=runs/waterbirds_rerun50/A_corr${code}_seed${seed}_e50_contentv3
    [[ -f $out/TRAIN_SUCCESS ]] && { echo "skip $out"; continue; }
    resume=""; [[ -f $out/latest.json ]] && resume="--resume"
    [[ -d $out && -z $resume ]] && rm -rf "$out"
    python3 reports/waterbirds_rerun50/scripts/local_pipeline.py train --corr $corr --seed $seed $resume >> $out.log 2>&1
    echo "done $out exit=$? $(date +%H:%M)"
  done
done
