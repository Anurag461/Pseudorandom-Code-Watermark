"""
Spoofed-generation single-job worker.

Args: job_index art_path out_dir

The orchestrator builds artifacts.pt with:
    bias_table       : dict[ctx_tuple -> (idx_int32, val_float32)] (or None
                        for the attack-disabled control)
    alpha            : float
    context_size     : int
    prompt_ids_list  : list[list[int]] (chat-templated harmful prompts)
    prompt_meta_list : list[dict(prompt_id, category)]
    jobs             : list[dict(prompt_idx, watermark, attack_active,
                                 max_new_tokens)]

We use the chat-tuned model loaded by watermark_expt with
PRC_MODEL_VARIANT=instruct in the environment.
"""
import os
import sys
import time
import torch

job_index = int(sys.argv[1])
art_path = sys.argv[2]
out_dir = sys.argv[3]

artifacts = torch.load(art_path, weights_only=False, map_location="cpu")
bias_table = artifacts.get("bias_table")
alpha = float(artifacts.get("alpha", 0.0))
context_size = int(artifacts.get("context_size", 2))
prompt_ids_list = artifacts["prompt_ids_list"]
prompt_meta_list = artifacts.get("prompt_meta_list", [{} for _ in prompt_ids_list])
job = artifacts["jobs"][job_index]

import watermark_expt as we
import attack_steal as atk

prompt_ids_py = prompt_ids_list[job["prompt_idx"]]
prompt_ids = torch.tensor([prompt_ids_py], dtype=torch.long, device=we.device)

t0 = time.time()
if job["attack_active"] and bias_table is not None:
    gen = atk.generate_spoofed(
        we.model,
        prompt_ids,
        max_new_tokens=job["max_new_tokens"],
        bias_table=bias_table,
        alpha=alpha,
        context_size=context_size,
        eos_token_id=None,
    )
    tokens, _ = we.generate_and_collect(gen)
else:
    # Vanilla unwatermarked generation (refusal baseline).
    we.model.eval()
    from qwen import KVCache
    tok_chunks = []
    with torch.no_grad():
        cache = KVCache()
        logits = we.model(prompt_ids, cache=cache)[:, -1]
        for _ in range(job["max_new_tokens"]):
            probs = torch.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            tok_chunks.append(nxt)
            logits = we.model(nxt, cache=cache)[:, -1]
    tokens = torch.cat(tok_chunks, dim=1).flatten() if tok_chunks else torch.zeros(0, dtype=torch.long)

dt = time.time() - t0

out_path = os.path.join(out_dir, f"result_{job_index:04d}.pt")
torch.save(
    {
        "job_index": job_index,
        "job": job,
        "prompt_meta": prompt_meta_list[job["prompt_idx"]],
        "tokens": tokens.cpu(),
        "duration_sec": dt,
        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "?"),
    },
    out_path,
)
print(
    f"[worker_spoof job={job_index} cuda={os.environ.get('CUDA_VISIBLE_DEVICES')}] "
    f"prompt={job['prompt_idx']} attack_active={job['attack_active']} "
    f"tokens={tokens.numel()} time={dt:.1f}s -> {out_path}",
    flush=True,
)
