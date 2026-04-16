#!/bin/bash
# Upload all 4 dataset variants to modal volume `plants-dataset`.
# Dataset files in local FS are symlinks to all_dieseas_class/origins/;
# we dereference here before uploading.
#
# Prereq: `modal token set --token-id ... --token-secret ...`

set -euo pipefail

PROJECT=/home/vanusha/diplom/diploma-plant-disease
STAGING=/tmp/plants_datasets_deref

echo "[1/3] Dereferencing datasets to staging dir ($STAGING)..."
rm -rf "$STAGING"
mkdir -p "$STAGING"
for V in dataset dataset_augmented dataset_balanced dataset_final; do
  if [ -d "$PROJECT/code/data/$V" ]; then
    echo "    $V"
    mkdir -p "$STAGING/$V"
    # rsync with -L to follow symlinks
    rsync -aL "$PROJECT/code/data/$V/" "$STAGING/$V/"
  else
    echo "    $V — MISSING, skipping"
  fi
done

du -sh "$STAGING/"* | head

echo "[2/3] Ensuring modal volume `plants-dataset` exists..."
PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
$PY -m modal volume create plants-dataset 2>&1 | head -3 || true

echo "[3/3] Uploading to volume..."
$PY -m modal volume put plants-dataset "$STAGING" /
echo "Upload done. Verify with:"
echo "    $PY -m modal volume ls plants-dataset /"
