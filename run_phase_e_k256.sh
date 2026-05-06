#!/bin/bash
# Launch PRC and KGW Phase E campaigns concurrently. PRC on GPUs 1-3, KGW on GPUs 4-6.
# (GPU 7 left to co-resident KGW slot if PRC finishes first.)
set -eu
cd /home/anurakas/nanochat

LOG_PRC=phase_e_prc_k256.log
LOG_KGW=phase_e_kgw_k256.log
: > "$LOG_PRC"
: > "$LOG_KGW"

# PRC on GPUs 1,2,3,7  (4 GPUs)
docker exec -d \
    -e PE_WATERMARK=prc \
    -e PE_WORKDIR=phase_e_workdir/prc_AB_k256 \
    -e PE_GPU_IDS=1,2,3,7 \
    -e PRC_MODEL_SIZE=8B \
    -e PRC_MODEL_VARIANT=instruct \
    -w /home/anurakas/nanochat \
    nile-nemo-jupyter \
    bash -c "python3 -u phase_e_drive.py > $LOG_PRC 2>&1"

# KGW on GPUs 4,5,6  (3 GPUs)
docker exec -d \
    -e PE_WATERMARK=kgw \
    -e PE_WORKDIR=phase_e_workdir/kgw_AB_k256 \
    -e PE_GPU_IDS=4,5,6 \
    -e PRC_MODEL_SIZE=8B \
    -e PRC_MODEL_VARIANT=instruct \
    -w /home/anurakas/nanochat \
    nile-nemo-jupyter \
    bash -c "python3 -u phase_e_drive.py > $LOG_KGW 2>&1"

echo "[run_phase_e_k256] launched PRC + KGW drivers (detached)"
