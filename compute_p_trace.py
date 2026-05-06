"""
Recompute the PRC `p_trace` (per-position p1 = sum_v p(v) * partition_map[1, v])
for a token sequence the watermark generator did NOT produce.

This is what enables PRC detection on attacker-spoofed tokens: the detector's
entropy-aware fold needs the LM's per-step probability of partition 1, but the
spoofed text was generated without yielding that quantity. We recover it via
one teacher-forced forward pass over [prompt_ids ; gen_tokens] using the same
model weights the API owner runs.

For Qwen3-8B at length 1024 this is well under a second on a single H100.
"""
import torch
import numpy as np


@torch.no_grad()
def compute_p_trace_for_tokens(
    model,
    prompt_ids: torch.Tensor,   # (1, L_prompt) long
    gen_tokens: torch.Tensor,   # (T,) long, the spoofed continuation
    partition_map: torch.Tensor,  # (2, vocab) on `device`
    device,
) -> np.ndarray:
    """Returns p_trace shape (T,), float64. Note: this also returns p1 for
    position L_prompt-1 (predicting gen_tokens[0]), through L_prompt+T-2
    (predicting gen_tokens[T-1])."""
    model.eval()
    if gen_tokens.dim() != 1:
        gen_tokens = gen_tokens.flatten()
    full = torch.cat([prompt_ids.flatten(), gen_tokens], dim=0).unsqueeze(0).to(device)
    logits = model(full)                                  # (1, L+T, V)
    L = prompt_ids.shape[-1]
    T = gen_tokens.shape[0]
    # Position i in gen_tokens (i in 0..T-1) was predicted by logits[L-1+i].
    pred = logits[0, L - 1:L - 1 + T, :]                  # (T, V)
    probs = torch.softmax(pred.float(), dim=-1)
    p1 = (probs * partition_map[1].to(probs.device).float()).sum(dim=-1)  # (T,)
    return p1.detach().cpu().numpy().astype(np.float64)
