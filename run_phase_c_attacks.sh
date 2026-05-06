#!/bin/bash
# Re-run the two attack cells that failed in run_phase_c_grid.sh
# (vanilla baseline already saved as spoof_06b_kgw_atk0_alpha2)
set -eu

cd /home/anurakas/nanochat
LOG=phase_c_attacks.log
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

run_cell kgw 1 2.0
run_cell prc 1 2.0

echo "[phase_c_attacks] done at $(date -u)" | tee -a "$LOG"
