"""
KGW-SelfHash watermark (Kirchenbauer-Geiping-Wen 2024).

Each candidate token at position t gets a green/red bit derived from
    hash( key, prev_token_id, candidate_id ) -> uniform in [0, 1)
The candidate is "green" iff that uniform draw is < gamma. At generation, all
green tokens get a +delta logit boost. At detection, we count tokens whose
(prev, current) hash is < gamma and run a Gaussian-tail z-test:

    z = (n_green - gamma * T) / sqrt(T * gamma * (1-gamma))

Detection threshold defaults to z > 4.0 (one-sided p < 3e-5, ~ matches the
PRC FPR target the rest of the experiments use).

The "self" part: the hash includes the candidate id itself, so the green-list
depends on which candidate is being scored. This is the variant the ETH-SRI
stealing attack explicitly evaluates.

Public surface:
    kgw_hash_unit(prev, cand, key) -> float in [0, 1)
    compute_green_mask(prev, key, gamma, vocab_size, device) -> (vocab,) bool
    mark_logits_kgw(logits, prev, key, gamma, delta) -> (vocab,) logits
    generate_text_watermark_kgw(model, token_ids, max_new_tokens, key,
                                gamma, delta, eos_token_id, watermark=True)
        -> generator yielding (next_token, None)
    detect_kgw(tokens, key, gamma=0.25, z_threshold=4.0) -> (decision, z, n_green, T)
"""
import numpy as np
import torch

from qwen import KVCache


# 64-bit avalanche mix; constants taken from MurmurHash3 finalizer.
_M1 = 0xff51afd7ed558ccd
_M2 = 0xc4ceb9fe1a85ec53
_MASK = 0xFFFFFFFFFFFFFFFF


def _avalanche64(x: int) -> int:
    """64-bit avalanche; pure-Python int arithmetic for scalar use."""
    x = (x ^ (x >> 33)) & _MASK
    x = (x * _M1) & _MASK
    x = (x ^ (x >> 33)) & _MASK
    x = (x * _M2) & _MASK
    x = (x ^ (x >> 33)) & _MASK
    return x


def kgw_hash_unit(prev: int, cand: int, key: int) -> float:
    """
    Scalar hash -> uniform float in [0, 1). Used by the detector (per-token).
    Must produce the SAME values as the vectorized version below.
    """
    seed = ((key & _MASK) * 0x9E3779B97F4A7C15
            + (prev & _MASK) * 0xBF58476D1CE4E5B9
            + (cand & _MASK) * 0x94D049BB133111EB) & _MASK
    h = _avalanche64(seed)
    # 53 bits of mantissa; shift to fit double exactly.
    return (h >> 11) / float(1 << 53)


def compute_green_mask(prev: int, key: int, gamma: float, vocab_size: int,
                       device) -> torch.Tensor:
    """
    Vectorized over the entire vocabulary. Returns a bool tensor of shape
    (vocab_size,) where True == green for THIS step (given prev token + key).

    Implements the same hash as kgw_hash_unit. Uses numpy uint64 for modular
    arithmetic with wrap-around (torch int64 cannot store constants > 2^63).
    """
    u64 = np.uint64
    cands = np.arange(vocab_size, dtype=np.uint64)
    base = u64((key * 0x9E3779B97F4A7C15 + prev * 0xBF58476D1CE4E5B9) & _MASK)
    seed = base + cands * u64(0x94D049BB133111EB)            # wraps in uint64
    x = seed
    x = x ^ (x >> u64(33))
    x = x * u64(_M1)
    x = x ^ (x >> u64(33))
    x = x * u64(_M2)
    x = x ^ (x >> u64(33))
    u = (x >> u64(11)).astype(np.float64) / float(1 << 53)
    green = u < gamma
    return torch.from_numpy(green).to(device)


def mark_logits_kgw(logits: torch.Tensor, prev: int, key: int,
                    gamma: float, delta: float) -> torch.Tensor:
    """logits: (vocab,) or (batch, vocab). Adds delta to green positions."""
    vocab_size = logits.shape[-1]
    green = compute_green_mask(prev, key, gamma, vocab_size, logits.device)
    return logits + delta * green.to(logits.dtype)


def generate_text_watermark_kgw(
    model,
    token_ids,
    max_new_tokens,
    key,
    gamma=0.25,
    delta=2.0,
    eos_token_id=None,
    watermark=True,
):
    """
    Mirrors generate_text_watermark_prc's API. Yields (next_token, None)
    per step so the existing generate_and_collect helper can drain it.

    Batch=1 only (matches existing PRC pipeline).
    """
    model.eval()
    if watermark:
        print(f"Watermark Enabled (KGW-SelfHash gamma={gamma} delta={delta})", flush=True)
    else:
        print("Watermark Disabled", flush=True)

    placeholder_p = torch.zeros(token_ids.shape[0], dtype=torch.float32)
    with torch.no_grad():
        cache = KVCache()
        logits = model(token_ids, cache=cache)[:, -1]    # (batch=1, vocab)
        prev = int(token_ids[0, -1].item())

        for pos in range(max_new_tokens):
            if watermark:
                biased = mark_logits_kgw(logits, prev, key, gamma, delta)
            else:
                biased = logits
            probs = torch.softmax(biased.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # (batch=1, 1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            # Yield a zero placeholder for p1 so generate_and_collect can stack
            # uniformly; KGW detection doesn't use p_trace.
            yield next_token, placeholder_p
            prev = int(next_token[0, 0].item())
            logits = model(next_token, cache=cache)[:, -1]


def detect_kgw(tokens, key: int, gamma: float = 0.25, z_threshold: float = 4.0):
    """
    tokens: 1D long tensor or list of ints. Returns:
        (decision_bool, z_score_float, n_green_int, T_int)
    """
    if torch.is_tensor(tokens):
        toks = tokens.flatten().cpu().tolist()
    else:
        toks = list(tokens)
    n_green = 0
    T = 0
    for i in range(1, len(toks)):
        if kgw_hash_unit(int(toks[i - 1]), int(toks[i]), key) < gamma:
            n_green += 1
        T += 1
    if T == 0:
        return False, 0.0, 0, 0
    expected = gamma * T
    var = T * gamma * (1.0 - gamma)
    z = (n_green - expected) / max(var ** 0.5, 1e-12)
    return bool(z > z_threshold), float(z), int(n_green), int(T)
