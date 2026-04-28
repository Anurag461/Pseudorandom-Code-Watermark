"""
Single-job generation worker. Run as a subprocess from run_calibration.py.

Args: job_index art_path out_dir

Reads CUDA_VISIBLE_DEVICES from the env (set by parent), so this worker
only sees the GPU it was assigned. Imports watermark_expt to get the model,
then overrides `partition` with the shared tensor so all workers use an
identical partition map.
"""
import os
import sys
import time
import torch

job_index = int(sys.argv[1])
art_path = sys.argv[2]
out_dir = sys.argv[3]

artifacts = torch.load(art_path, weights_only=False, map_location="cpu")
encoding_key = artifacts["encoding_key"]
decoding_key = artifacts["decoding_key"]
partition_cpu = artifacts["partition"]
prompt_ids_list = artifacts["prompt_ids_list"]
job = artifacts["jobs"][job_index]

import watermark_expt as we

# Replace the worker-local random partition with the shared one.
we.partition = partition_cpu.to(we.device)

prompt_ids = torch.tensor(
    [prompt_ids_list[job["prompt_idx"]]],
    dtype=torch.long,
    device=we.device,
)

t0 = time.time()
gen = we.generate_text_watermark_prc(
    we.model,
    prompt_ids,
    max_new_tokens=job["max_new_tokens"],
    encoding_key=encoding_key,
    partition_map=we.partition,
    eos_token_id=None,  # disable early stop; we want a full p-trace of length >= n
    watermark=job["watermark"],
)
tokens, p_trace = we.generate_and_collect(gen)
dt = time.time() - t0

out_path = os.path.join(out_dir, f"result_{job_index:02d}.pt")
torch.save(
    {
        "job_index": job_index,
        "job": job,
        "tokens": tokens.cpu(),
        "p_trace": p_trace,
        "duration_sec": dt,
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    },
    out_path,
)
print(
    f"[worker job={job_index} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}] "
    f"prompt={job['prompt_idx']} watermark={job['watermark']} "
    f"tokens={tokens.numel()} time={dt:.1f}s "
    f"({tokens.numel()/max(dt,1e-6):.1f} tok/s) -> {out_path}",
    flush=True,
)
