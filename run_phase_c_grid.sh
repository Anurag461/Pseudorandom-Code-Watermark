#!/bin/bash
# Phase C driver: vanilla baseline + KGW attack + PRC attack at alpha=2.0
# on the 0.6B instruct model, 100 harmful prompts, GPUs 1-7.
set -eu

cd /home/anurakas/nanochat
LOG=phase_c_grid.log
: > "$LOG"

run_cell() {
    local watermark=$1 atk=$2 alpha=$3
    echo "=== cell: watermark=$watermark atk=$atk alpha=$alpha ===" | tee -a "$LOG"
    docker exec \
        -e PRC_MODEL_SIZE=0.6B \
        -e PRC_MODEL_VARIANT=instruct \
        -e PRC_GPU_IDS=1,2,3,4,5,6,7 \
        -e SPOOF_WATERMARK="$watermark" \
        -e SPOOF_ATTACK_ACTIVE="$atk" \
        -e SPOOF_ALPHA="$alpha" \
        -w /home/anurakas/nanochat \
        nile-nemo-jupyter \
        python3 -u run_spoof_attack.py --phase C 2>&1 | tee -a "$LOG"
}

run_cell kgw 0 2.0   # vanilla refusal baseline
run_cell kgw 1 2.0   # KGW attack
run_cell prc 1 2.0   # PRC attack (control)

echo "[phase_c_grid] done at $(date -u)" | tee -a "$LOG"
