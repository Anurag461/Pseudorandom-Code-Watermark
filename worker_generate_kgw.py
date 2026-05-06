"""
KGW-SelfHash single-job generation worker.

Args: job_index art_path out_dir

artifacts.pt schema (compatible with run_calibration's PRC pattern, plus a
`kgw` block):
    kgw_key          : int
    gamma            : float
    delta            : float
    prompt_ids_list  : list[list[int]]
    jobs             : list[dict(prompt_idx, watermark, max_new_tokens)]
"""
import os
import sys
import time
import torch

job_index = int(sys.argv[1])
art_path = sys.argv[2]
out_dir = sys.argv[3]

artifacts = torch.load(art_path, weights_only=False, map_location="cpu")
kgw_key = int(artifacts["kgw_key"])
gamma = float(artifacts["gamma"])
delta = float(artifacts["delta"])
prompt_ids_list = artifacts["prompt_ids_list"]
job = artifacts["jobs"][job_index]

import watermark_expt as we
import watermark_kgw as kgw

prompt_ids = torch.tensor(
    [prompt_ids_list[job["prompt_idx"]]],
    dtype=torch.long,
    device=we.device,
)

t0 = time.time()
gen = kgw.generate_text_watermark_kgw(
    we.model,
    prompt_ids,
    max_new_tokens=job["max_new_tokens"],
    key=kgw_key,
    gamma=gamma,
    delta=delta,
    eos_token_id=None,
    watermark=job["watermark"],
)
tokens, _ = we.generate_and_collect(gen)
dt = time.time() - t0

out_path = os.path.join(out_dir, f"result_{job_index:04d}.pt")
torch.save(
    {
        "job_index": job_index,
        "job": job,
        "tokens": tokens.cpu(),
        "duration_sec": dt,
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    },
    out_path,
)
print(
    f"[worker_kgw job={job_index} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}] "
    f"prompt={job['prompt_idx']} watermark={job['watermark']} "
    f"tokens={tokens.numel()} time={dt:.1f}s "
    f"({tokens.numel()/max(dt,1e-6):.1f} tok/s) -> {out_path}",
    flush=True,
)
