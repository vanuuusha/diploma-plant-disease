#!/bin/bash
# Fetch training results from modal volume `plants-results` back to local.
# Usage: ./fetch_results.sh [task_NN | all]

set -euo pipefail

PROJECT=/home/vanusha/diplom/diploma-plant-disease
PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
TARGET="${1:-all}"

mkdir -p "$PROJECT/code/results"

if [ "$TARGET" = "all" ]; then
  for T in task_07 task_08 task_09 task_10; do
    echo "[fetch] $T"
    $PY -m modal volume get plants-results /$T "$PROJECT/code/results/" 2>&1 | tail -3 || true
  done
else
  echo "[fetch] $TARGET"
  $PY -m modal volume get plants-results /$TARGET "$PROJECT/code/results/" 2>&1 | tail -3
fi

echo "Done. Check: ls $PROJECT/code/results/"
