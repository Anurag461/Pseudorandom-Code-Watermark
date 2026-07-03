"""
Model-free PRC detector helpers.

These are pure numpy/torch functions (no LM, no model load) shared by
watermark_expt.py and the Modal detect pass. Keeping them here lets detection
run on cached generations without importing watermark_expt (which loads the
Qwen model at import time).
"""
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from prc import Detect


def binary_entropy(p):
    """H_2(p) in nats. Vectorized; safe at p=0 and p=1."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def fold_naive(observed_bits: np.ndarray, n: int) -> np.ndarray:
    """Cyclic fold averaging +/-1 signs at each codeword slot."""
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    seq_len = signs.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, signs)
    np.add.at(counts, idx, 1)
    return sums / np.maximum(counts, 1.0)


def fold_entropy_weighted(observed_bits: np.ndarray, p_array: np.ndarray,
                          n: int) -> np.ndarray:
    """
    Entropy-weighted cyclic fold. Each observation contributes its sign
    scaled by H_2(p)/log(2), so deterministic LM steps contribute ~0.

    p_array gives the LM's P[partition 1] at each generation step.
    """
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    weights = binary_entropy(p_array) / np.log(2)         # in [0, 1]
    seq_len = signs.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    norms = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, weights * signs)
    np.add.at(norms, idx, weights)
    return sums / np.maximum(norms, 1e-9)


def fold_soft_token(observed_bits: np.ndarray, p_array: np.ndarray,
                    n: int) -> np.ndarray:
    """
    Soft-token fold for the Hoeffding detector (prc_fpr_proof.pdf).

    Each observation yields a soft-token S = t * H where t = +/-1 is the
    observed sign and H = H_2(p)/log(2) in [0, 1] is the normalized entropy,
    so |S| <= 1 (exactly S_j = t_j * H_j in the proof's notation). Positions
    are cyclically folded to length n by AVERAGING over the tokens that land
    in each slot (divide by count, not by the entropy mass), which keeps every
    folded soft-token in [-1, 1] as the Hoeffding bound requires.
    """
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    weights = binary_entropy(p_array) / np.log(2)         # H in [0, 1]
    soft = signs * weights                                # t * H in [-1, 1]
    seq_len = soft.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, soft)
    np.add.at(counts, idx, 1)
    return sums / np.maximum(counts, 1.0)


def tokens_to_bits(token_ids, partition_map) -> np.ndarray:
    """Look up each token's partition (0 or 1) -> length-T int array."""
    if token_ids.dim() != 1:
        token_ids = token_ids.flatten()
    bit_for_token = partition_map[1].long().to(token_ids.device)
    bits = bit_for_token[token_ids].detach().cpu().numpy().astype(np.int64)
    return bits


def detect_hoeffding(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-9,
    entropy_weighted=True,
    return_info=False,
):
    """Proven-FPR Hoeffding detector, block-OR over length-n blocks.

    The generated text is split into consecutive length-n blocks (each block is
    an independent PRC codeword under the same key). Each block is scored on its
    own soft-tokens and passed through prc.Detect; the document is declared
    watermarked iff ANY block passes. Trailing tokens with T % n != 0 are
    ignored (when T < n, the whole prefix is treated as one partial block).

    To keep the overall false-positive rate <= F under the OR of B blocks, each
    block is tested at F/B (union bound: B * (F/B) = F).

    entropy_weighted=True  -> soft-tokens S_j = t_j * H_j (fold_soft_token).
    entropy_weighted=False -> naive soft-tokens S_j = t_j  (fold_naive, H_j=1).
    Both keep |S_j| <= 1, so the FPR <= F guarantee holds either way.
    """
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )

    T = bits.shape[0]
    if T >= n:
        slices = [slice(b * n, (b + 1) * n) for b in range(T // n)]
    else:
        slices = [slice(0, T)]              # short output: one partial block
    num_blocks = len(slices)
    block_fpr = fpr / num_blocks            # Bonferroni: keep overall FPR <= F

    fold = fold_soft_token if entropy_weighted else (
        lambda b, p, m: fold_naive(b, m))

    decision = False
    blocks_passed = 0
    best = None                             # block with the largest margin
    for b, sl in enumerate(slices):
        post = fold(bits[sl], p_arr[sl], n)
        dec, info = Detect(decoding_key, post, false_positive_rate=block_fpr,
                           return_info=True)
        margin = info["statistic"] - info["threshold"]
        if dec:
            decision = True
            blocks_passed += 1
        if best is None or margin > best[0]:
            best = (margin, b, info)

    if not return_info:
        return decision

    _, best_block, best_info = best
    return decision, {
        "method": "hoeffding_blockwise",
        "statistic": best_info["statistic"],
        "threshold": best_info["threshold"],
        "V": best_info["V"],
        "num_blocks": num_blocks,
        "blocks_passed": blocks_passed,
        "best_block": best_block,
        "block_fpr": block_fpr,
        "fpr": fpr,
    }
