#!/bin/bash
# Upload datasets to modal volume one at a time via CLI (avoids symlinks).
# Stages each dataset through /tmp/plants_stage_<name>/ and uploads via CLI.

set -euo pipefail

PROJECT=/home/vanusha/diplom/diploma-plant-disease
PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
STAGE_ROOT=/tmp/plants_upload_stage

mkdir -p "$STAGE_ROOT"

# Remove placeholder from earlier
$PY -m modal volume rm plants-dataset /test.txt 2>/dev/null || true

for V in dataset dataset_augmented dataset_balanced dataset_final; do
  if [ ! -d "$PROJECT/code/data/$V" ]; then
    echo "skip $V"
    continue
  fi
  STAGE="$STAGE_ROOT/$V"
  echo "=== $V ==="
  echo "[1/3] dereferencing to $STAGE"
  rm -rf "$STAGE"
  rsync -aL "$PROJECT/code/data/$V/" "$STAGE/"
  echo "    size: $(/usr/bin/du -sh $STAGE | cut -f1)"

  # Ensure data.yaml points to /data/<V> on remote
  cat > "$STAGE/data.yaml" <<EOF
path: /data/$V
train: train/images
val: val/images
test: test/images
nc: 9
names:
- Недостаток P2O5
- Листовая (бурая) ржавчина
- Мучнистая роса
- Пиренофороз
- Фузариоз
- Корневая гниль
- Септориоз
- Недостаток N
- Повреждение заморозками
EOF

  echo "[2/3] uploading to modal volume"
  $PY -m modal volume put --force plants-dataset "$STAGE" /$V 2>&1 | tail -5
  echo "[3/3] cleanup $STAGE"
  rm -rf "$STAGE"
  echo ""
done

echo "=== all uploaded ==="
$PY -m modal volume ls plants-dataset / 2>&1 | head
