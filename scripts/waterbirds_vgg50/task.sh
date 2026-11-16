#!/bin/bash
#SBATCH --job-name=waterbirds-vgg50
#SBATCH --gres=gpu:normal:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/waterbirds50_%j.out
#SBATCH --error=logs/waterbirds50_%j.err
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${SLURM_JOB_ID:?Allocated SLURM job required}"
: "${WB_DATA_NODE:?Set the node holding the verified local dataset and submit with --nodelist}"
[[ $(hostname -s) == "$WB_DATA_NODE" ]] || exit 2
stage=${1:?train, resume, pilot or analyze}
condition=${2:-${SLURM_ARRAY_TASK_ID:-}}
case "$condition" in
0|095|.95|0.95) corr=.95; code=095;;
1|080|.80|0.80) corr=.80; code=080;;
2|065|.65|0.65) corr=.65; code=065;;
*) echo 'Specify A correlation .95/.80/.65, or array indices 0-2; B undefined' >&2; exit 2;;
esac
PYTHON_BIN=${PYTHON_BIN:-python3}
manifest=${WB_MANIFEST:-$PWD/reports/waterbirds_vgg50/dataset_moana_u2_content_v3/protocol.json}
run="$PWD/runs/waterbirds_vgg50/A_corr${code}_seed42_e50_contentv3"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/wb50-compile-${SLURM_JOB_ID}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/waterbirds-${SLURM_JOB_ID}"
nvidia-smi
case "$stage" in
train|resume)
 : "${WB_PILOT:?Successful full-epoch A095 pilot summary.json required}"
 extra=(); [[ $stage != resume ]] || extra=(--resume)
 "$PYTHON_BIN" scripts/waterbirds_vgg50/run.py train --manifest "$manifest" --corr "$corr" \
   --pilot "$WB_PILOT" --output "$run" "${extra[@]}" ;;
pilot)
 "$PYTHON_BIN" scripts/waterbirds_vgg50/pilot.py --manifest "$manifest" --corr "$corr" \
   --output "reports/waterbirds_vgg50/pilot_${SLURM_JOB_ID}_${code}.json" ;;
analyze)
 for epoch in ${WB_EPOCHS:-0 2 25 50}; do
 "$PYTHON_BIN" scripts/waterbirds_vgg50/run.py analyze --run "$run" --epoch "$epoch" \
   --output "$run/analysis/epoch_$(printf '%03d' "$epoch")"
 done ;;
*) exit 2;;
esac
