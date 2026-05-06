"""
Phase E artifact builder. Creates a workdir/artifacts.pt for a single
structured-query attack cell on a chosen (watermark, w1, w2) pair.

Reused by `batched_worker.py` (no changes required) — the existing schema
(encoding_key/decoding_key/partition/prompt_ids_list/jobs for PRC, or
kgw_key/gamma/delta/prompt_ids_list/jobs for KGW) is exactly what we need.

Env vars:
  PE_WATERMARK         prc | kgw                        (required)
  PE_WORKDIR           output workdir                   (required)
  PE_W1                bucket-1 word string             (required for prc)
  PE_W2                bucket-0 word string             (required for prc)
  PE_K                 number of structured queries     (default 2048)
  PE_T                 max_new_tokens per query         (default 512)
  PE_USE_SPACE         "1" to use leading-space variant (default "1")
  PE_PRC_BASE          existing PRC workdir to copy keys/partition from
                       (default calib_workdir_n400_t3_eta05)
  PE_KGW_BASE          existing KGW workdir for kgw_key (default kgw_workdir_qwen06b_base)
  PE_MODEL_SIZE        for tokenizer path (default 8B)
"""
import json
import os
import sys

import torch


WATERMARK = os.environ["PE_WATERMARK"].lower()
WORKDIR = os.environ["PE_WORKDIR"]
W1 = os.environ["PE_W1"]
W2 = os.environ["PE_W2"]
K = int(os.environ.get("PE_K", "2048"))
T = int(os.environ.get("PE_T", "512"))
USE_SPACE = os.environ.get("PE_USE_SPACE", "1") == "1"
PRC_BASE = os.environ.get("PE_PRC_BASE", "calib_workdir_n400_t3_eta05")
KGW_BASE = os.environ.get("PE_KGW_BASE", "kgw_workdir_qwen06b_base")
MODEL_SIZE = os.environ.get("PE_MODEL_SIZE", "8B")

assert WATERMARK in ("prc", "kgw")


def build_chat_prompt_ids(w1, w2, T):
    """Build the chat-template tokenized prompt for the structured query.
    Mirrors run_spoof_attack._build_chat_prompt_ids:280-290."""
    from transformers import AutoTokenizer
    local = f"/home/anurakas/nanochat/Qwen3-{MODEL_SIZE}"
    if os.path.isdir(local):
        tok = AutoTokenizer.from_pretrained(local)
    else:
        tok = AutoTokenizer.from_pretrained(f"Qwen/Qwen3-{MODEL_SIZE}")

    system = (
        "You are a uniform random binary stream generator. You output only "
        f"the symbols '{w1}' and '{w2}', separated by single spaces, "
        "in a uniformly random order with no other content."
    )
    user = (
        f"Output exactly {T} symbols. Each symbol must be either '{w1}' or "
        f"'{w2}'. Separate symbols with single spaces. Choose each symbol "
        "uniformly at random from {'" + w1 + "', '" + w2 + "'} independently. "
        "Do not add any other words, punctuation, or explanation."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    formatted = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    # Re-encode through the codebase tokenizer (consistent with prompt_to_ids).
    import watermark_expt as we
    return we.tokenizer.encode(formatted), formatted


def main():
    os.makedirs(WORKDIR, exist_ok=True)
    art_path = os.path.join(WORKDIR, "artifacts.pt")

    prompt_ids, formatted = build_chat_prompt_ids(W1, W2, T)
    print(f"[phase_e_build] formatted prompt: {len(formatted)} chars, "
          f"tokenized to {len(prompt_ids)} ids", flush=True)

    wm_flag = os.environ.get("PE_WM", "1") == "1"
    jobs = [
        {"prompt_idx": 0, "watermark": wm_flag, "max_new_tokens": T}
        for _ in range(K)
    ]

    if WATERMARK == "prc":
        base = torch.load(os.path.join(PRC_BASE, "artifacts.pt"),
                           weights_only=False, map_location="cpu")
        artifacts = {
            "encoding_key": base["encoding_key"],
            "decoding_key": base["decoding_key"],
            "partition": base["partition"],
            "prompt_ids_list": [prompt_ids],
            "jobs": jobs,
            "n": base.get("n"),
            "seed": base.get("seed"),
            # Phase-E metadata (ignored by batched_worker but useful later):
            "phase_e": {
                "watermark": "prc",
                "w1": W1, "w2": W2,
                "use_space": USE_SPACE,
                "K": K, "T": T,
                "model_size": MODEL_SIZE,
                "prompt_text": formatted,
            },
        }
    else:
        base = torch.load(os.path.join(KGW_BASE, "artifacts.pt"),
                           weights_only=False, map_location="cpu")
        artifacts = {
            "kgw_key": base["kgw_key"],
            "gamma": base["gamma"],
            "delta": base["delta"],
            "prompt_ids_list": [prompt_ids],
            "jobs": jobs,
            "phase_e": {
                "watermark": "kgw",
                "w1": W1, "w2": W2,
                "use_space": USE_SPACE,
                "K": K, "T": T,
                "model_size": MODEL_SIZE,
                "prompt_text": formatted,
            },
        }

    torch.save(artifacts, art_path)
    print(f"[phase_e_build] saved {WATERMARK.upper()} artifacts: {K} jobs -> {art_path}",
          flush=True)


if __name__ == "__main__":
    main()
