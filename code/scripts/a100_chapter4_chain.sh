#!/bin/bash
# Запускается на A100 (shadeform@216.81.248.198).
# Последовательно обучает все конфигурации главы 4:
#   SE → CBAM → CGFM (main) → ablations → RT-DETR+CGFM
# После каждого прогона — архивирует результаты в tar.gz, коммит опционально.
#
# Usage (на A100):
#   bash ~/a100_chapter4_chain.sh > ~/chapter4_chain.log 2>&1 &
#
# Настройки:
set -u
PROJ=~/plants_ch4
cd "$PROJ"
export PATH=$HOME/.local/bin:$PATH

RUN() {
    local desc="$1"; shift
    echo "==================================================================="
    echo "[$(date -u +%H:%M:%S)] START: $desc"
    echo "==================================================================="
    "$@"
    local rc=$?
    echo "[$(date -u +%H:%M:%S)] END: $desc (exit=$rc)"
    return $rc
}

# task_13 SE-Neck (стартует отдельно — запущен первым, до этого chain'а)
# CBAM-Neck запускается ЛОКАЛЬНО (параллельно CGFM на A100), поэтому его здесь НЕТ.

# task_15 CGFM main (без warm-up, FiLM инициализирован как identity γ=1,β=0)
RUN "CGFM main (MobileNetV3, P3+P4+P5)" bash ~/run_chapter4.sh \
    --config cgfm --encoder mobilenetv3_small_100 \
    --levels P3 P4 P5 \
    --out out/task_15 --name yolov12_cgfm \
    --batch 16 --epochs 100 --patience 15 --device 0 --skip_warmup

# task_16 аблации (по protocol 4): 4 конфигурации, patience=8 (экономия времени)
RUN "CGFM abl P5-only" bash ~/run_chapter4.sh \
    --config cgfm --encoder mobilenetv3_small_100 --levels P5 \
    --out out/task_16 --name yolov12_cgfm_abl_p5only \
    --batch 16 --epochs 100 --patience 8 --device 0 --skip_warmup

RUN "CGFM abl P3-only" bash ~/run_chapter4.sh \
    --config cgfm --encoder mobilenetv3_small_100 --levels P3 \
    --out out/task_16 --name yolov12_cgfm_abl_p3only \
    --batch 16 --epochs 100 --patience 8 --device 0 --skip_warmup

RUN "CGFM abl EfficientNet-B0" bash ~/run_chapter4.sh \
    --config cgfm --encoder efficientnet_b0 --levels P3 P4 P5 \
    --out out/task_16 --name yolov12_cgfm_abl_effb0 \
    --batch 16 --epochs 100 --patience 8 --device 0 --skip_warmup

RUN "CGFM abl ViT-Tiny" bash ~/run_chapter4.sh \
    --config cgfm --encoder vit_tiny_patch16_224 --levels P3 P4 P5 \
    --out out/task_16 --name yolov12_cgfm_abl_vittiny \
    --batch 16 --epochs 100 --patience 8 --device 0 --skip_warmup

# task_17 RT-DETR + CGFM
RUN "RT-DETR + CGFM" python3 -u notebooks/chapter4_rtdetr_cgfm.py \
    --out out/task_17 --variant cgfm --data ~/dataset_final \
    --epochs 50 --patience 8 --batch 8

echo "[$(date -u +%H:%M:%S)] CHAIN COMPLETE"
