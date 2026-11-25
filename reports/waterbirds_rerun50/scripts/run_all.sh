#!/bin/bash
# Usage: bash reports/waterbirds_rerun50/scripts/run_all.sh [seed ...]   (default seed: 42)
#   CORRS="0.8" bash ... 43 44   limits the correlations (default "0.95 0.8 0.65"); one driver per corr can run in parallel.
#   DATA=cubfull bash ... 42     trains on manifest_cubfull (about 2x train birds) into *_cubfull folders.
#   PRETRAINED=1 bash ... 42     starts from ImageNet VGG11_BN weights (--pretrained) into *_pretrained folders.
#   LR=0.01 bash ... 42          initial lr (--lr) into *_lr0.01 folders.
#   SCHED=cosine bash ... 42     cosine lr decay from 0.01 (--scheduler cosine) into *_cosine folders.
#   MULTISCENE=1 bash ... 42     option B: a new scene per bird every epoch (--scene-resample) into *_multiscene folders.
#   CACHE=1 PREFETCH=4 WORKERS=12: speed only, same samples and metrics.
#   Final Waterbirds setting: DATA=cubfull PRETRAINED=1 LR=0.01 MULTISCENE=1 CACHE=1 bash ... 42
#   resumes unfinished runs, skips finished ones.
# Train Waterbirds Condition A corr .95/.80/.65 one at a time with the team configuration (lr 0.01 constant unless SCHED=cosine).
cd "$(dirname "$0")/../../.."  # repo root
mkdir -p runs/waterbirds_rerun50
seeds=("$@"); [[ ${#seeds[@]} -eq 0 ]] && seeds=(42)
suffix=""; flags=""; [[ -n $DATA && $DATA != team ]] && { suffix="_$DATA"; flags="--data $DATA"; }
[[ -n $PRETRAINED ]] && { suffix="${suffix}_pretrained"; flags="$flags --pretrained"; }
[[ -n $LR ]] && { suffix="${suffix}_lr$(python3 -c "print(f'{$LR:g}')")"; flags="$flags --lr $LR"; }
[[ $SCHED == cosine ]] && { suffix="${suffix}_cosine"; flags="$flags --scheduler cosine"; }
[[ -n $MULTISCENE ]] && { suffix="${suffix}_multiscene"; flags="$flags --scene-resample"; }
[[ -n $CACHE ]] && flags="$flags --cache-eval"  # CACHE=1: evaluation images composited once, same images
[[ -n $PREFETCH ]] && flags="$flags --prefetch $PREFETCH"  # PREFETCH=4: more batches ahead, same samples
[[ -n $WORKERS ]] && flags="$flags --workers $WORKERS"  # WORKERS=12: more loader processes, same samples
for seed in "${seeds[@]}"; do
  for corr in ${CORRS:-0.95 0.8 0.65}; do
    code=$(printf '%03d' "$(python3 -c "print(round($corr*100))")")
    out=runs/waterbirds_rerun50/A_corr${code}_seed${seed}_e50_contentv3${suffix}
    [[ -f $out/TRAIN_SUCCESS ]] && { echo "skip $out"; continue; }
    resume=""; [[ -f $out/latest.json ]] && resume="--resume"
    [[ -d $out && -z $resume ]] && rm -rf "$out"
    python3 reports/waterbirds_rerun50/scripts/local_pipeline.py train --corr $corr --seed $seed $flags $resume >> $out.log 2>&1
    echo "done $out exit=$? $(date +%H:%M)"
  done
done
