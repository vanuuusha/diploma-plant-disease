#!/bin/bash
# Local chain: DETR all 4 variants with per-variant commit+push.
# survives terminal death via setsid invocation.

set -u
cd /home/vanusha/diplom/diploma-plant-disease

PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
VARIANTS=(baseline aug_geom aug_oversample aug_diffusion)

log() { echo "[detr-chain $(date +%H:%M:%S)] $*"; }

log "START local DETR chain"
> /tmp/train_detr.log

for v in "${VARIANTS[@]}"; do
    log "=== detr / $v ==="
    $PY code/notebooks/chapter3_detr_runner.py --variant "$v" --batch 12 2>&1 | tee -a /tmp/train_detr.log | tail -30
    git add code/results/task_10 2>/dev/null
    if git commit -m "chapter3_detr: $v done" 2>/dev/null; then
        git push 2>&1 | tail -3 || true
    else
        log "no commit for detr/$v"
    fi
done

$PY code/notebooks/chapter3_detr_runner.py --variant summary 2>&1 | tail -5
git add code/results/task_10 2>/dev/null
git commit -m "chapter3_detr: summary" 2>/dev/null && git push 2>&1 | tail -3 || true

log "LOCAL DETR CHAIN DONE"
