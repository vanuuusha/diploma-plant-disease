#!/bin/bash
# Task 20: HP-sweep YOLOv12m aug_diffusion. Запускается на A100.
# Последовательно обучает 4 конфигурации с разными гиперпараметрами.
set -u
PROJ=~/plants_ch4
cd "$PROJ"
export PATH=$HOME/.local/bin:$PATH

mkdir -p out/task_20 logs

RUN() {
    local desc="$1"; shift
    echo "==================================================================="
    echo "[$(date -u +%H:%M:%S)] START: $desc"
    echo "==================================================================="
    "$@"
    echo "[$(date -u +%H:%M:%S)] END: $desc (exit=$?)"
}

PY=python3

# Конфиг 1: HSV + flip аугментации (без mosaic)
RUN "hp_hsv_flip" $PY -c "
from ultralytics import YOLO
y = YOLO('yolo12m.pt')
y.train(
    data='/tmp/data.yaml', epochs=100, patience=15, imgsz=640, batch=16,
    device=0, seed=42, project='out/task_20', name='yolov12_hp_hsv_flip',
    exist_ok=True, pretrained=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, erasing=0.0,
)
"

# Конфиг 2: AdamW + cosine lr schedule
RUN "hp_adamw_cos" $PY -c "
from ultralytics import YOLO
y = YOLO('yolo12m.pt')
y.train(
    data='/tmp/data.yaml', epochs=100, patience=15, imgsz=640, batch=16,
    device=0, seed=42, project='out/task_20', name='yolov12_hp_adamw_cos',
    exist_ok=True, pretrained=True,
    optimizer='AdamW', lr0=0.001, cos_lr=True,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, erasing=0.0,
)
"

# Конфиг 3: dropout + default hyperparams
RUN "hp_dropout" $PY -c "
from ultralytics import YOLO
y = YOLO('yolo12m.pt')
y.train(
    data='/tmp/data.yaml', epochs=100, patience=15, imgsz=640, batch=16,
    device=0, seed=42, project='out/task_20', name='yolov12_hp_dropout',
    exist_ok=True, pretrained=True,
    dropout=0.1,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, erasing=0.0,
)
"

# Конфиг 4: большой batch + scaled lr
RUN "hp_big_batch" $PY -c "
from ultralytics import YOLO
y = YOLO('yolo12m.pt')
y.train(
    data='/tmp/data.yaml', epochs=100, patience=15, imgsz=640, batch=48,
    device=0, seed=42, project='out/task_20', name='yolov12_hp_big_batch',
    exist_ok=True, pretrained=True,
    lr0=0.02,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0, erasing=0.0,
)
"

echo "[$(date -u +%H:%M:%S)] TASK 20 COMPLETE"
