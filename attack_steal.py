"""
Context-conditional watermark-stealing attack.

Threat model: attacker observes many watermarked outputs (and ideally a smaller
set of unwatermarked outputs from the same model and prompt distribution) but
has no access to the watermark key, model logits, or partition map. The attack
estimates a per-context (last `context_size` tokens) logit shift that, when
applied at attack-time generation, biases the attacker's output toward tokens
the watermark scheme tends to favor.

For KGW-SelfHash, this works: the green-list bit is a deterministic function
of (prev, candidate, key), so per-(prev, candidate) frequencies in observed
outputs encode the green-list. For PRC, this fails: the partition is keyed by
a PRF and the bucket selection at each step is engineered to be marginal-
preserving, so per-context token frequencies in observed PRC output match
unwatermarked output (up to language statistics) and the bias table contains
no usable signal.

Sparse top-K representation: full per-context bias would be |contexts| x V
floats which is prohibitive (V=151,936). We keep only the K tokens with the
largest log-ratio per context.

Public surface:
    train_bias(wm_results, nw_results, context_size=2, top_k=64,
               min_obs=3, smoothing=1.0) -> dict
    apply_conditional_bias(logits, prev_tokens, bias_table, alpha) -> logits
    save_bias(bias_table, path)
    load_bias(path)
    generate_spoofed(model, token_ids, max_new_tokens, bias_table, alpha,
                     context_size, eos_token_id=None)
        -> generator yielding (next_token, None)
"""
from collections import defaultdict
import torch
import numpy as np

from qwen import KVCache


VOCAB_SIZE = 151936


def _accumulate(results, table, context_size):
    """Add token-occurrence counts to `table[ctx][token] += 1`."""
    for r in results:
        toks = r["tokens"].flatten().cpu().tolist() if torch.is_tensor(r["tokens"]) else list(r["tokens"])
        for i in range(context_size, len(toks)):
            ctx = tuple(toks[i - context_size:i])
            table[ctx][toks[i]] += 1.0


def train_bias(
    wm_results,
    nw_results,
    context_size: int = 2,
    top_k: int = 64,
    min_obs: int = 3,
    smoothing: float = 1.0,
):
    """
    Build a sparse conditional-bias table from (watermarked, unwatermarked)
    samples. Each output entry is

        bias[ctx] = (idx: int32[K], val: float32[K])

    where `val` is a logit-shift to be added at attack time.

    Definition: for each context that appears at least `min_obs` times in
    wm + nw counts combined, we compute
        ratio[v] = log( (count_wm[ctx, v] + a) / (count_nw[ctx, v] + a) )
    with additive smoothing a = `smoothing`. We then keep the top-K
    by absolute value, but bound the negative side at 0 (we only want to
    BOOST, not penalize, since attackers want to push toward green tokens).
    """
    counts_wm = defaultdict(lambda: np.zeros(VOCAB_SIZE, dtype=np.float32))
    counts_nw = defaultdict(lambda: np.zeros(VOCAB_SIZE, dtype=np.float32))

    print(f"[attack] accumulating wm counts from {len(wm_results)} samples ...", flush=True)
    _accumulate(wm_results, counts_wm, context_size)
    print(f"[attack] accumulating nw counts from {len(nw_results)} samples ...", flush=True)
    _accumulate(nw_results, counts_nw, context_size)

    contexts = set(counts_wm) | set(counts_nw)
    print(f"[attack] {len(contexts)} distinct contexts observed", flush=True)

    bias = {}
    kept = 0
    for ctx in contexts:
        cw = counts_wm.get(ctx)
        cn = counts_nw.get(ctx)
        nw_total = float(cn.sum()) if cn is not None else 0.0
        wm_total = float(cw.sum()) if cw is not None else 0.0
        if nw_total + wm_total < min_obs:
            continue
        # Build the log-ratio over a sparse union of nonzero indices.
        nz = set()
        if cw is not None:
            nz |= set(np.flatnonzero(cw).tolist())
        if cn is not None:
            nz |= set(np.flatnonzero(cn).tolist())
        if not nz:
            continue
        idx = np.fromiter(nz, dtype=np.int64)
        cw_v = cw[idx].astype(np.float64) if cw is not None else np.zeros(len(idx), dtype=np.float64)
        cn_v = cn[idx].astype(np.float64) if cn is not None else np.zeros(len(idx), dtype=np.float64)
        cw_v = cw_v + smoothing
        cn_v = cn_v + smoothing
        ratio = np.log(cw_v / cn_v).astype(np.float32)
        # Keep only positive boosts (attacker wants green tokens up; negative
        # ratios just punish unwatermarked-leaning tokens, which a refusing
        # safety prior would already do).
        pos_mask = ratio > 0
        if not np.any(pos_mask):
            continue
        idx_p = idx[pos_mask]
        val_p = ratio[pos_mask]
        if len(idx_p) > top_k:
            order = np.argsort(-val_p)[:top_k]
            idx_p = idx_p[order]
            val_p = val_p[order]
        bias[ctx] = (idx_p.astype(np.int32), val_p.astype(np.float32))
        kept += 1

    print(f"[attack] retained {kept} contexts after min_obs/positive-ratio filter", flush=True)
    return bias


def save_bias(bias_table: dict, path: str):
    """Save bias table to a torch .pt file (the dict keys are tuples; torch.save
    handles arbitrary picklable objects)."""
    torch.save({"bias": bias_table, "vocab_size": VOCAB_SIZE}, path)


def load_bias(path: str) -> dict:
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj["bias"]


def apply_conditional_bias(
    logits: torch.Tensor,
    prev_tokens: tuple,
    bias_table: dict,
    alpha: float,
) -> torch.Tensor:
    """
    logits: (vocab,) or (batch=1, vocab).
    prev_tokens: tuple of int Python token ids (length == context_size).
    Adds alpha * bias[ctx][idx] to logits[idx] for the K listed indices, if
    prev_tokens is in the table. Otherwise no-op.
    """
    entry = bias_table.get(prev_tokens)
    if entry is None:
        return logits
    idx, val = entry
    idx_t = torch.from_numpy(idx).to(device=logits.device, dtype=torch.long)
    val_t = torch.from_numpy(val).to(device=logits.device, dtype=logits.dtype)
    # In-place add along the last axis at the listed indices.
    if logits.dim() == 1:
        logits = logits.clone()
        logits.scatter_add_(0, idx_t, alpha * val_t)
    else:
        logits = logits.clone()
        # broadcast over batch=1
        logits[..., idx_t] = logits[..., idx_t] + alpha * val_t
    return logits


def generate_spoofed(
    model,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    bias_table: dict,
    alpha: float,
    context_size: int = 2,
    eos_token_id=None,
):
    """
    Generation with the stealing attack active. Yields (next_token, None) so
    generate_and_collect can drain it the same way it does PRC/KGW outputs.
    """
    model.eval()
    print(f"[attack] generate_spoofed alpha={alpha} ctx={context_size}", flush=True)

    placeholder_p = torch.zeros(token_ids.shape[0], dtype=torch.float32)
    with torch.no_grad():
        cache = KVCache()
        logits = model(token_ids, cache=cache)[:, -1]
        # Seed the rolling context from the tail of the prompt.
        ctx = list(token_ids[0, -context_size:].cpu().tolist())

        for pos in range(max_new_tokens):
            biased = apply_conditional_bias(logits, tuple(ctx), bias_table, alpha)
            probs = torch.softmax(biased.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            yield next_token, placeholder_p
            tok_id = int(next_token[0, 0].item())
            ctx = ctx[1:] + [tok_id] if context_size > 0 else ctx
            logits = model(next_token, cache=cache)[:, -1]
