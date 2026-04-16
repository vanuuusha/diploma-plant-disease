#!/bin/bash
# Chapter 3 chained training: RT-DETR, Faster R-CNN, DETR per-variant with
# git commit + push after each variant.
# YOLOv12 is expected to be already done by the time this script is launched.

set -u
cd /home/vanusha/diplom/diploma-plant-disease

PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
VARIANTS=(baseline aug_geom aug_oversample aug_diffusion)

log() { echo "[chapter3 $(date +%H:%M:%S)] $*"; }

wait_for_yolov12() {
    while pgrep -f "chapter3_ultralytics_runner.py.*--detector yolov12" >/dev/null 2>&1; do
        log "ждём YOLOv12..."
        sleep 60
    done
    log "YOLOv12 finished"
    git add code/results/task_07 2>/dev/null
    git commit -m "chapter3_yolov12: all variants done (task_07)" 2>&1 | tail -3 || true
    git push 2>&1 | tail -3 || true
}

run_detector() {
    local det=$1        # yolov12 | rtdetr | faster_rcnn | detr
    local task=$2       # task_07 | ... | task_10
    local script=$3     # path to runner
    local extra_env=$4  # env vars (space-sep var=val)
    local log_file=/tmp/train_${det}.log
    > "$log_file"
    for variant in "${VARIANTS[@]}"; do
        log "=== $det / $variant ==="
        env $extra_env $PY "$script" --variant "$variant" 2>&1 | tee -a "$log_file" | tail -30
        git add "code/results/$task" 2>/dev/null
        if git commit -m "chapter3_${det}: ${variant} done" 2>/dev/null; then
            git push 2>&1 | tail -3 || true
        else
            log "nothing to commit for $det/$variant"
        fi
    done
}

log "START"

wait_for_yolov12

# RT-DETR
run_detector rtdetr task_08 code/notebooks/chapter3_ultralytics_runner.py "" || true

# Faster R-CNN
run_detector faster_rcnn task_09 code/notebooks/chapter3_torchvision_runner.py "" || true

# DETR (HuggingFace, offline)
run_detector detr task_10 code/notebooks/chapter3_detr_runner.py "HF_HUB_OFFLINE=1" || true

# Task 11 summary (placeholder — actual aggregation runs separately)
log "all detectors finished"
