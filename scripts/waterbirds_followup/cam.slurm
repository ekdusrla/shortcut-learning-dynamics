#!/bin/bash
#SBATCH --job-name=wb-followup-cam
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wb_followup_cam_%A_%a.out
set -euo pipefail
cd "${WB_PROJECT_ROOT:?Set absolute repository path}"
: "${SLURM_JOB_ID:?Allocated SLURM job required}"
: "${WB_DATA_NODE:?Node holding the verified dataset}"
: "${WB_ANALYSIS:?Absolute followup analysis path}"
[[ $(hostname -s) == "$WB_DATA_NODE" ]]
codes=(095 080 065)
i=${SLURM_ARRAY_TASK_ID:?Submit array 0-2 after checking available resources}
[[ $i =~ ^[012]$ ]]
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_HOME="${HOME}/.cache/torch"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/wb-cam-${SLURM_JOB_ID}"
case "${WB_MODE:-cam}" in
cam) "${PYTHON_BIN:-python3}" scripts/waterbirds_followup/cam.py --analysis "$WB_ANALYSIS" --code "${codes[$i]}" ;;
extract) "${PYTHON_BIN:-python3}" scripts/waterbirds_followup/extract.py --code "${codes[$i]}" --epochs "${WB_EXTRACT_EPOCHS:-0:50}" --output "${WB_EXTRACT_OUTPUT:?Fresh extraction parent directory}/A_corr${codes[$i]}" ;;
*) exit 2 ;;
esac
