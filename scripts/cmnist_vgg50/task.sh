#!/bin/bash
#SBATCH --job-name=cmnist-vgg50
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --array=0-5%6
#SBATCH --gres=gpu:normal:1
#SBATCH --nodelist=moana-u2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/cmnist_%A_%a.out
#SBATCH --error=logs/cmnist_%A_%a.err
set -euo pipefail
cd /data/dada0228/shortcut-learning-dynamics
PYTHON_BIN=${PYTHON_BIN:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
: "${SLURM_ARRAY_TASK_ID:?Submit as job array}"
stage=${1:?train, eval or gradcam}
i=$SLURM_ARRAY_TASK_ID
[[ $i =~ ^[0-5]$ ]] || exit 2
modes=(background background background foreground foreground foreground)
tags=(bg bg bg fg fg fg)
corrs=(0.95 0.80 0.65 0.95 0.80 0.65)
codes=(095 080 065 095 080 065)
run="$PWD/runs/vgg11_${tags[$i]}_corr${codes[$i]}_seed42_e50_img224_all"
common=(--dataset colored_mnist --seed 42 --image-size 224 --train-corr "${corrs[$i]}" --val-corr 0.10 --test-corr 0.10 --mnist-val-size 5000 --color-mode "${modes[$i]}" --color-pairing all)
date; hostname
printenv | sort | sed -n '/^CUDA/p; /^SLURM/p'
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch, torchvision
print('torch',torch.__version__,'torchvision',torchvision.__version__,'CUDA',torch.version.cuda)
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, 'Expected one visible GPU'
p=torch.cuda.get_device_properties(0)
print(p)
assert 'A2000' in p.name, 'Only RTX A2000 allowed'
PY
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cmnist-mpl-${SLURM_JOB_ID}"
export OMP_NUM_THREADS=4
case "$stage" in
train)
 mkdir -p runs
 mkdir "$run" # atomic refusal if any prior result exists
 "$PYTHON_BIN" shortcut_experiment.py train "${common[@]}" --model vgg11_bn --epochs 50 --lr 0.01 --momentum 0.9 --weight-decay 0.0001 --scheduler none --batch-size 128 --num-workers 4 --features-per-group 32 --cf-chunk-size 128 --device cuda --output "$run"
 touch "$run/TRAIN_SUCCESS"
 ;;
eval)
 test -f "$run/TRAIN_SUCCESS"
 test -f "$run/checkpoints/epoch_050.pt"
 mkdir "$run/evaluation_e050"
 "$PYTHON_BIN" "$PWD/evaluate.py" --run "$run" --checkpoint "$run/checkpoints/epoch_050.pt" "${common[@]}" --device cuda --batch-size 128 --workers 4 --output "$run/evaluation_e050/metrics.csv" --probe-dir "$run/evaluation_e050/probes"
 ;;
gradcam)
 test -f "$run/TRAIN_SUCCESS"
 mkdir "$run/gradcam_comparison"
 for epoch in 002 025 050; do
   ckpt="$run/checkpoints/epoch_${epoch}.pt"
   if [[ ! -f $ckpt ]]; then echo "Missing checkpoint: $ckpt; skipping"; continue; fi
   mkdir "$run/gradcam_comparison/epoch_$epoch"
   "$PYTHON_BIN" gradcam.py --run "$run" --checkpoint "$ckpt" "${common[@]}" --split val --sample-index 0 --target-class 0 --layer features.18 --device cuda --output "$run/gradcam_comparison/epoch_$epoch/val_00000_class0.png"
 done
 ;;
*) exit 2;;
esac
