#!/bin/bash
#SBATCH --job-name=wb-a095-pilot
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/waterbirds_epochpilot_%j.out
#SBATCH --error=logs/waterbirds_epochpilot_%j.err
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SLURM_JOB_ID:?Allocated SLURM session required}"
: "${WB_DATA_NODE:?Set the node holding the verified local dataset and submit with --nodelist}"
[[ $(hostname -s) == "$WB_DATA_NODE" ]] || exit 2
PYTHON_BIN=${PYTHON_BIN:-python3}
manifest=${WB_MANIFEST:-$PWD/reports/waterbirds_vgg50/dataset_moana_u2_content_v3/protocol.json}
epochs=${WB_PILOT_EPOCHS:-2}
run="$PWD/runs/waterbirds_vgg50_pilots/A_corr095_contentv3_job${SLURM_JOB_ID}_e${epochs}"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/wb-compile-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/wb-pilot-mpl-${SLURM_JOB_ID}"
test ! -e "$run"
resource="logs/waterbirds_epochpilot_${SLURM_JOB_ID}.time.txt"
test ! -e "$resource"
nvidia-smi
/usr/bin/time -v -o "$resource" "$PYTHON_BIN" scripts/waterbirds_vgg50/pilot_epochs.py \
  --manifest "$manifest" --epochs "$epochs" --output "$run"
