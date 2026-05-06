"""
Capture real Qwen3-0.6B tokens and per-token empirical entropies for the PRC
visualizer.

What this does
--------------
1. Loads Qwen/Qwen3-0.6B (downloads on first run, ~1.2 GB).
2. Generates a response to a prompt one token at a time, sampling from the
   model's own next-token distribution.
3. For each generated token, records:
     - the token text (with leading-space convention preserved)
     - the token id
     - the empirical entropy  H_e = -log_2 p(token | prefix)
       (i.e. surprisal in bits, exactly the quantity used in the PRC
       watermarking section of Christ-Gunn 2024).
4. Writes everything to qwen3_run.json in a shape the React artifact can ingest.

Run with
--------
    pip install transformers torch accelerate
    python capture_qwen3_run.py

Adjust PROMPT, MAX_NEW_TOKENS, and TEMPERATURE below if you like.
"""

import json
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── config ────────────────────────────────────────────────────────────────
MODEL_NAME      = "Qwen/Qwen3-0.6B"
PROMPT          = "Explain in one paragraph why pseudorandom codes give undetectable watermarks."
MAX_NEW_TOKENS  = 60
TEMPERATURE     = 1.0          # use 1.0 to keep entropies meaningful
TOP_P           = 1.0          # 1.0 = no nucleus truncation; surprisal is then exact
SEED            = 7
OUTPUT_PATH     = "qwen3_run.json"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)

    print(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    ).to(DEVICE)
    model.eval()

    # Qwen3 is a chat model; format the prompt with the chat template so the
    # model behaves naturally. We capture entropies only for the generated
    # tokens, not for the prompt.
    messages = [{"role": "user", "content": PROMPT}]
    input_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tok(input_text, return_tensors="pt").input_ids.to(DEVICE)

    generated_tokens = []  # list of dicts: {text, id, H, logp}

    with torch.no_grad():
        cur = input_ids
        for step in range(MAX_NEW_TOKENS):
            out    = model(cur)
            logits = out.logits[0, -1, :] / TEMPERATURE   # (vocab,)
            probs  = torch.softmax(logits, dim=-1)

            # Sample. (Top-p truncation could be added here; we keep p untruncated
            # so that -log p is the true empirical entropy under the sampling
            # distribution.)
            next_id = torch.multinomial(probs, num_samples=1)
            p       = probs[next_id].item()
            log2p   = math.log2(max(p, 1e-12))

            tok_text = tok.decode([next_id.item()])
            generated_tokens.append({
                "text": tok_text,
                "id":   int(next_id.item()),
                "logp": log2p,        # log_2 p
                "H":    -log2p,       # surprisal in bits
            })
            print(f"  [{step:3d}] H={-log2p:6.3f}  '{tok_text}'")

            # Stop if EOS
            if next_id.item() == tok.eos_token_id:
                break

            cur = torch.cat([cur, next_id.unsqueeze(0)], dim=1)

    # Normalize entropies into [0,1] for the visualization. We use a soft cap
    # at ~10 bits (anything above is "essentially uniform").
    max_H_cap = 10.0
    for t in generated_tokens:
        t["H_norm"] = min(t["H"] / max_H_cap, 1.0)

    payload = {
        "model":  MODEL_NAME,
        "prompt": PROMPT,
        "config": {
            "temperature":    TEMPERATURE,
            "top_p":          TOP_P,
            "seed":           SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "tokens": generated_tokens,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {len(generated_tokens)} tokens to {OUTPUT_PATH}")
    print("Mean H (bits):", sum(t["H"] for t in generated_tokens) / len(generated_tokens))


if __name__ == "__main__":
    main()
