#!/bin/bash
# Local chain: DETR all 4 variants, batch=8 (safe for 16GB GPU).
# Guards: single orchestrator via pidfile, kills children on exit.

set -u
cd /home/vanusha/diplom/diploma-plant-disease

PY=/home/vanusha/.pyenv/versions/3.12.10/envs/diplom/bin/python
VARIANTS=(baseline aug_geom aug_oversample aug_diffusion)
BATCH=${BATCH:-8}
PIDFILE=/tmp/detr_orchestrator.pid

log() { echo "[detr-chain $(date +%H:%M:%S)] $*"; }

# Single-orchestrator guard
if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    log "already running (pid $(cat $PIDFILE)), exiting"
    exit 1
fi
echo $$ > "$PIDFILE"

# Kill all detr child processes on exit
cleanup() {
    log "cleanup: killing detr workers"
    pkill -9 -P $$ 2>/dev/null || true
    pkill -9 -f "chapter3_detr_runner" 2>/dev/null || true
    rm -f "$PIDFILE"
}
trap cleanup EXIT INT TERM

log "START local DETR chain (batch=$BATCH)"
> /tmp/train_detr.log

for v in "${VARIANTS[@]}"; do
    log "=== detr / $v ==="
    # Pre-flight: ensure no stale detr python processes
    if pgrep -f "chapter3_detr_runner" >/dev/null 2>&1; then
        log "killing stale detr before starting $v"
        pkill -9 -f "chapter3_detr_runner" 2>/dev/null || true
        sleep 3
    fi
    # Run synchronously; wait for exit before next
    $PY code/notebooks/chapter3_detr_runner.py --variant "$v" --batch $BATCH 2>&1 | tee -a /tmp/train_detr.log | tail -30
    # Ensure all children dead before commit
    pkill -f "chapter3_detr_runner.*--variant $v" 2>/dev/null || true
    sleep 2
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
