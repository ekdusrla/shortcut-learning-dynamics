#!/usr/bin/env bash
# Download CUB-200-2011 and the Places365 scenes needed for Waterbirds.
# data.py composites birds onto scenes on the fly, so nothing else is generated.
#
#   bash download_waterbirds.sh
#
# Everything is written under data/raw/ (git-ignored):
#   data/raw/CUB_200_2011/          images/, segmentations/, images.txt, ...
#   data/raw/places365/data_256/    bamboo_forest, forest/broadleaf, ocean, lake/natural
#
# Environment variables:
#   CUB_SOURCE=auto|kaggle|caltech   auto uses Kaggle (wenewone/cub2002011) when the
#                                    kaggle CLI and credentials are available,
#                                    otherwise the official Caltech release.
#   PYTHON=python3
set -euo pipefail
cd "$(dirname "$0")"

RAW=data/raw
CUB_DIR="$RAW/CUB_200_2011"
PLACES_DIR="$RAW/places365"
CUB_SOURCE="${CUB_SOURCE:-auto}"
PYTHON="${PYTHON:-python3}"
CUB_URL="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
SEG_URL="https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
mkdir -p "$RAW"

have_kaggle() {
  command -v kaggle >/dev/null 2>&1 &&
    { [ -f "$HOME/.kaggle/kaggle.json" ] || [ -n "${KAGGLE_USERNAME:-}" ]; }
}

download_cub_kaggle() {
  echo "[cub] downloading wenewone/cub2002011 from Kaggle"
  kaggle datasets download -d wenewone/cub2002011 -p "$RAW/kaggle_cub" --unzip
  local images_txt
  images_txt=$(find "$RAW/kaggle_cub" -name images.txt -path '*CUB_200_2011*' | head -1)
  [ -n "$images_txt" ] || { echo "images.txt not found in Kaggle download" >&2; exit 1; }
  mv "$(dirname "$images_txt")" "$CUB_DIR"
  local seg
  seg=$(find "$RAW/kaggle_cub" -type d -name segmentations | head -1)
  [ -n "$seg" ] && [ ! -d "$CUB_DIR/segmentations" ] && mv "$seg" "$CUB_DIR/segmentations"
  rm -rf "$RAW/kaggle_cub"
}

download_cub_caltech() {
  echo "[cub] downloading CUB_200_2011.tgz from Caltech (~1.1 GB)"
  curl -fL --progress-bar "$CUB_URL" | tar -xz -C "$RAW"
}

# 1. CUB-200-2011 images and labels
if [ -f "$CUB_DIR/images.txt" ]; then
  echo "[cub] $CUB_DIR already exists"
else
  case "$CUB_SOURCE" in
    kaggle) download_cub_kaggle ;;
    caltech) download_cub_caltech ;;
    auto) if have_kaggle; then download_cub_kaggle; else download_cub_caltech; fi ;;
    *) echo "CUB_SOURCE must be auto, kaggle or caltech" >&2; exit 1 ;;
  esac
fi

# 2. CUB segmentation masks (separate archive unless the source already had them)
if [ -d "$CUB_DIR/segmentations" ]; then
  echo "[cub] segmentations already exist"
else
  echo "[cub] downloading segmentations.tgz from Caltech"
  curl -fL --progress-bar "$SEG_URL" | tar -xz -C "$CUB_DIR"
fi

# 3. Places365 scenes: only the four categories are fetched from the 26 GB archive
"$PYTHON" -c "import data; data.fetch_places('$PLACES_DIR')"
