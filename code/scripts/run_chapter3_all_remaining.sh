#!/bin/bash
# Master script: runs remaining chapter 3 training after terminal crash.
# Resume points:
#   - yolov12: baseline DONE, resume from aug_geom
#   - rtdetr, faster_rcnn, detr: run all 4 variants
# Per-variant git commit + push. Survives SIGHUP via nohup invocation.

set -u
cd /home/vanusha/diplom/diploma-plant-disease

PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
VARIANTS=(baseline aug_geom aug_oversample aug_diffusion)

log() { echo "[chapter3 $(date +%H:%M:%S)] $*"; }

run_variant() {
    local det=$1
    local task=$2
    local script=$3
    local variant=$4
    shift 4
    local extra_env="$*"
    local variant_log=/tmp/train_${det}.log
    log "=== $det / $variant ==="
    env $extra_env $PY "$script" --variant "$variant" 2>&1 | tee -a "$variant_log" | tail -30
    git add "code/results/$task" 2>/dev/null
    if git commit -m "chapter3_${det}: ${variant} done" 2>/dev/null; then
        git push 2>&1 | tail -3 || true
    else
        log "nothing to commit for $det/$variant"
    fi
}

run_detector_from() {
    local det=$1
    local task=$2
    local script=$3
    local start_idx=$4
    shift 4
    local extra_env="$*"
    > /tmp/train_${det}.log 2>/dev/null
    for (( i=$start_idx; i<${#VARIANTS[@]}; i++ )); do
        run_variant "$det" "$task" "$script" "${VARIANTS[$i]}" $extra_env
    done
}

log "RESUME START"

# YOLOv12: baseline already done, continue from index 1 (aug_geom)
for det_args in \
    "yolov12 task_07 code/notebooks/chapter3_ultralytics_runner.py --detector yolov12" \
    ; do
    :
done

# YOLOv12 — нужно передать --detector в wrapper. Упрощённо: запускаем per-variant
for (( i=1; i<${#VARIANTS[@]}; i++ )); do
    v=${VARIANTS[$i]}
    log "=== yolov12 / $v ==="
    $PY code/notebooks/chapter3_ultralytics_runner.py --detector yolov12 --variant "$v" 2>&1 | tee -a /tmp/train_yolov12.log | tail -30
    git add code/results/task_07 2>/dev/null
    git commit -m "chapter3_yolov12: $v done" 2>/dev/null && git push 2>&1 | tail -3 || log "no commit yolov12/$v"
done

# Run grand summary for yolov12 after all 4 variants done
$PY code/notebooks/chapter3_ultralytics_runner.py --detector yolov12 --variant summary 2>&1 | tail -5
git add code/results/task_07 2>/dev/null; git commit -m "chapter3_yolov12: summary" 2>/dev/null && git push 2>&1 | tail -3 || true

# RT-DETR — все 4 варианта
for v in "${VARIANTS[@]}"; do
    log "=== rtdetr / $v ==="
    $PY code/notebooks/chapter3_ultralytics_runner.py --detector rtdetr --variant "$v" 2>&1 | tee -a /tmp/train_rtdetr.log | tail -30
    git add code/results/task_08 2>/dev/null
    git commit -m "chapter3_rtdetr: $v done" 2>/dev/null && git push 2>&1 | tail -3 || log "no commit rtdetr/$v"
done
$PY code/notebooks/chapter3_ultralytics_runner.py --detector rtdetr --variant summary 2>&1 | tail -5
git add code/results/task_08 2>/dev/null; git commit -m "chapter3_rtdetr: summary" 2>/dev/null && git push 2>&1 | tail -3 || true

# Faster R-CNN
for v in "${VARIANTS[@]}"; do
    log "=== faster_rcnn / $v ==="
    $PY code/notebooks/chapter3_torchvision_runner.py --variant "$v" 2>&1 | tee -a /tmp/train_faster_rcnn.log | tail -30
    git add code/results/task_09 2>/dev/null
    git commit -m "chapter3_faster_rcnn: $v done" 2>/dev/null && git push 2>&1 | tail -3 || log "no commit faster_rcnn/$v"
done
$PY code/notebooks/chapter3_torchvision_runner.py --variant summary 2>&1 | tail -5
git add code/results/task_09 2>/dev/null; git commit -m "chapter3_faster_rcnn: summary" 2>/dev/null && git push 2>&1 | tail -3 || true

# DETR (offline)
for v in "${VARIANTS[@]}"; do
    log "=== detr / $v ==="
    HF_HUB_OFFLINE=1 $PY code/notebooks/chapter3_detr_runner.py --variant "$v" 2>&1 | tee -a /tmp/train_detr.log | tail -30
    git add code/results/task_10 2>/dev/null
    git commit -m "chapter3_detr: $v done" 2>/dev/null && git push 2>&1 | tail -3 || log "no commit detr/$v"
done
$PY code/notebooks/chapter3_detr_runner.py --variant summary 2>&1 | tail -5
git add code/results/task_10 2>/dev/null; git commit -m "chapter3_detr: summary" 2>/dev/null && git push 2>&1 | tail -3 || true

# Task 11 summary aggregation
log "running task_11 aggregation"
$PY code/notebooks/chapter3_summary_script.py 2>&1 | tail -30
git add code/results/task_11 2>/dev/null
git commit -m "chapter3: task_11 grand summary aggregation" 2>/dev/null && git push 2>&1 | tail -3 || true

log "ALL DONE"
