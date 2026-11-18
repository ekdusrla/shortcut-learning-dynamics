#!/bin/bash
# Usage: bash reports/cmnist_rerun50/scripts/run_all.sh  (resumes unfinished runs, skips finished ones)
# Re-run the team's six Colored MNIST x VGG11_BN epoch-50 experiments (scripts/cmnist_vgg50/task.sh), 1 at a time.
cd "$(dirname "$0")/../../.."  # repo root
run_one() {
  mode=$1; tag=$2; corr=$3; ctag=$4
  mkdir -p runs/rerun50; out=runs/rerun50/vgg11_${tag}_corr${ctag}
  [[ -f $out/test_metrics.json ]] && { echo "skip $out"; return; }
  resume=""
  if [[ -f $out/latest.json ]]; then resume="--resume latest"; else rm -rf "$out"; fi
  python3 shortcut_experiment.py train --dataset colored_mnist --seed 42 --image-size 224 --train-corr $corr \
    --val-corr 0.10 --test-corr 0.10 --mnist-val-size 5000 --color-mode $mode --color-pairing all --model vgg11_bn \
    --epochs 50 --lr 0.01 --momentum 0.9 --weight-decay 0.0001 --scheduler none --batch-size 128 --num-workers 4 \
    --features-per-group 32 --cf-chunk-size 128 --device cuda --output $out $resume >> $out.log 2>&1
  echo "done $out exit=$?"
}
export -f run_one
printf '%s\n' "background bg 0.95 095" "foreground fg 0.95 095" "background bg 0.80 080" \
  "foreground fg 0.80 080" "background bg 0.65 065" "foreground fg 0.65 065" | xargs -P 1 -L 1 bash -c 'run_one $0 $1 $2 $3'
