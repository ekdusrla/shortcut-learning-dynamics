#!/bin/bash
#SBATCH --job-name=cmnist-vgg100-resume
#SBATCH --partition=batch_ce_ugrad
#SBATCH --account=ugrad_ce
#SBATCH --qos=qos_dada0228_2026_2
#SBATCH --array=0-1%2
#SBATCH --gres=gpu:normal:1
#SBATCH --nodelist=moana-u2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=6-00:00:00
#SBATCH --output=runs/cmnist_vgg100_resume/logs/cmnist_%A_%a.out
#SBATCH --error=runs/cmnist_vgg100_resume/logs/cmnist_%A_%a.err
set -euo pipefail
cd /data/dada0228/shortcut-learning-dynamics
PYTHON_BIN=${PYTHON_BIN:-/data/dada0228/conda_envs/shortcut-xai/bin/python}
[[ $# == 0 || ( $# == 1 && ( $1 == --dry-run || $1 == --check-data ) ) ]] || exit 2
if [[ ${1:-} == --check-data ]]; then
 i=${SLURM_ARRAY_TASK_ID:-0}
else
 : "${SLURM_ARRAY_TASK_ID:?Submit as array or set ID for --dry-run}"
 i=$SLURM_ARRAY_TASK_ID
fi
[[ $i =~ ^[01]$ ]] || { echo 'Task must be 0 or 1' >&2; exit 2; }
modes=(background foreground)
tags=(bg fg)
source_run="$PWD/runs/vgg11_${tags[$i]}_corr095_seed42_e50_img224_all"
checkpoint="$source_run/checkpoints/epoch_050.pt"
run="$PWD/runs/cmnist_vgg100_resume/vgg11_${tags[$i]}_corr095_seed42_e100_from050_img224_all_retry191154"
test -f "$source_run/TRAIN_SUCCESS"
test -f "$checkpoint"
[[ ! -e $run ]] || { echo "Output already exists: $run" >&2; exit 2; }
cmd=("$PYTHON_BIN" -u shortcut_experiment.py train
 --dataset colored_mnist --model vgg11_bn --seed 42 --image-size 224
 --train-corr 0.95 --val-corr 0.10 --test-corr 0.10 --mnist-val-size 5000
 --color-mode "${modes[$i]}" --color-pairing all
 --epochs 100 --extend-to-epoch 100 --resume "$checkpoint"
 --lr 0.01 --momentum 0.9 --weight-decay 0.0001 --scheduler none
 --batch-size 128 --num-workers 4 --features-per-group 32
 --cf-chunk-size 128 --device cuda --output "$run")
# Both array tasks use one explicit cache root, independent of node-local HOME.
# Prepare a pinned official cache with mnist_cache.py; never download in a GPU task.
export HF_HOME="${CMNIST_HF_HOME:-$PWD/.cache/huggingface}"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
printf 'HF_HOME=%s\nHF_DATASETS_CACHE=%s\nHF_HUB_CACHE=%s\nHF_HUB_OFFLINE=%s\nHF_DATASETS_OFFLINE=%s\n' \
 "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_HUB_OFFLINE" "$HF_DATASETS_OFFLINE"
printf 'Task %s: %s corr=.95, epoch 51-100\n' "$i" "${modes[$i]}"
printf 'Command: '; printf '%q ' "${cmd[@]}"; printf '\n'
if [[ ${1:-} == --dry-run ]]; then exit 0; fi
# CPU-only preflight uses exactly the same load_dataset call as data.py.
# Fail before creating an output directory or touching CUDA if the cache is absent.
[[ -d $HF_DATASETS_CACHE ]] || { echo "Missing shared datasets cache: $HF_DATASETS_CACHE; run mnist_cache.py prepare first" >&2; exit 1; }
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 "$PYTHON_BIN" - <<'PYDATA'
import os
import json
import hashlib
from pathlib import Path
from datasets import load_dataset
from datasets import config
from huggingface_hub import constants
assert config.HF_DATASETS_OFFLINE and constants.HF_HUB_OFFLINE
cache = Path(os.environ['HF_HOME']).resolve()
manifest = json.loads((cache / 'mnist_manifest.json').read_text())
validated = json.loads((cache / 'offline_validation.json').read_text())
assert manifest['repo'] == 'ylecun/mnist' and validated['success']
assert manifest['revision'] == validated['revision']
assert hashlib.sha256(Path('data.py').read_bytes()).hexdigest() == validated['data_py_sha256']
for entry in manifest['cache_files']:
    assert hashlib.sha256((cache / entry['path']).read_bytes()).hexdigest() == entry['sha256']
print('[offline MNIST] pinned revision=' + manifest['revision'])
raw = load_dataset("ylecun/mnist")
assert len(raw["train"]) == 60000 and len(raw["test"]) == 10000
for split in ("train", "test"):
    assert raw[split]._fingerprint == validated['offline_raw_splits'][split]['fingerprint']
    for item in raw[split].cache_files:
        relative = str(Path(item['filename']).resolve().relative_to(cache))
        assert relative in {entry['path'] for entry in manifest['cache_files']}
    assert raw[split].column_names == ["image", "label"]
    image = raw[split][0]["image"]
    assert image.size == (28, 28)
    print(f"[offline MNIST] {split}: N={len(raw[split])}, fingerprint={raw[split]._fingerprint}, files={raw[split].cache_files}")
print("[offline MNIST] PASS; no GPU used")
PYDATA
if [[ ${1:-} == --check-data ]]; then exit 0; fi
# No CUDA checks or run-directory writes occur above this point.
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/cmnist-resume-mpl-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
date
hostname
nvidia-smi
"$PYTHON_BIN" - <<'PY'
import torch, torchvision
print('torch',torch.__version__,'torchvision',torchvision.__version__,'CUDA',torch.version.cuda)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p=torch.cuda.get_device_properties(0)
print(p)
assert 'A2000' in p.name
PY
mkdir "$run" # atomic refusal if another submission created the same output
"${cmd[@]}"
touch "$run/TRAIN_SUCCESS"
