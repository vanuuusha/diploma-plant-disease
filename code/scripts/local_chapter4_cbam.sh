#!/bin/bash
# Локальный запуск CBAM-neck (параллельно CGFM на A100).
# batch=8 для RTX 5070 Ti (15GB).
set -u
cd /home/vanusha/diplom/diploma-plant-disease
PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python

PIDFILE=/tmp/chapter4_cbam.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "already running (pid $(cat $PIDFILE))"; exit 1
fi
echo $$ > "$PIDFILE"
cleanup() {
    pkill -9 -f "chapter4_runner.*cbam" 2>/dev/null || true
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

echo "[$(date)] START CBAM-Neck local (batch=8)"
$PY -u code/notebooks/chapter4_runner.py \
    --config cbam_neck \
    --out code/results/task_13 \
    --batch 8 \
    --epochs 100 --patience 15 \
    --device 0 --seed 42
echo "[$(date)] CBAM-Neck done"
