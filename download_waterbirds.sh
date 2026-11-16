#!/usr/bin/env bash
# Safe Waterbirds preparation. No download occurs with --check-only or --help.
# PYTHON=/path/to/python bash download_waterbirds.sh --root /local_datasets/USER/waterbirds_v1 --transport curl
# Legacy CUB_SOURCE=auto|caltech uses the official source; Kaggle is not supported.
set -euo pipefail
cd "$(dirname "$0")"
case "${CUB_SOURCE:-caltech}" in
  auto|caltech) ;;
  *) echo 'Use CUB_SOURCE=caltech (official archives); Kaggle preparation is no longer supported.' >&2; exit 2 ;;
esac
exec "${PYTHON:-python3}" scripts/waterbirds_vgg50/download.py "$@"
