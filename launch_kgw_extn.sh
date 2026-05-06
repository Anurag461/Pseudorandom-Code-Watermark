#!/bin/bash
set -x
export EXT_WATERMARK=kgw
export EXT_MODEL_SIZE=0.6B
export EXT_GPU_IDS=1,2,3,4,5,6,7
export EXT_PROMPTS_JSONL=prompts_10k.jsonl
export EXT_N_PROMPTS=4983
export EXT_BASE_WORKDIR=kgw_workdir_qwen06b_base
export EXT_OUT_WORKDIR=kgw_workdir_extn5k_06b
export EXT_MAX_NEW_TOKENS=800
export PRC_MODEL_SIZE=0.6B
export PRC_MODEL_VARIANT=base
cd /home/anurakas/nanochat
exec python3 -u extend_campaign_batched.py
