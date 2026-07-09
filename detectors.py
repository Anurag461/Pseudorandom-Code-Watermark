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


def step_weight(entropy: np.ndarray, temp: float = 0.05) -> np.ndarray:
    """Sharp INCREASING gate in [0, 1] as a function of the NORMALIZED ENTROPY
    H = H_2(p)/log2 in [0, 1]:  f(H) = sigmoid((H - 0.5) / temp).

    ~0 for H<0.5, sharp rise across H=0.5, ~1 for H>0.5. A soft entropy gate:
    it keeps high-entropy tokens (where the watermark signal lives) and zeroes
    low-entropy ones -- a thresholded version of the linear entropy weight H.
    Smaller temp -> sharper gate. At temp=0.05: f(0.2)=0.002, f(0.4)=0.12,
    f(0.5)=0.5, f(0.6)=0.88, f(0.8)=0.998 (-> a near-hard threshold at H=0.5).
    The soft-token is then S_j = t_j * f(H_j); f in [0,1] keeps |S_j|<=1 so the
    Hoeffding FPR bound holds.
    """
    H = np.clip(np.asarray(entropy, dtype=np.float64), 0.0, 1.0)
    return 1.0 / (1.0 + np.exp((0.5 - H) / temp))


def _fold_raw(soft: np.ndarray, n: int) -> np.ndarray:
    """Cyclically fold soft-tokens (already in [-1, 1]) to length n by AVERAGING
    over the tokens in each slot -- keeps every folded value in [-1, 1] as the
    Hoeffding bound requires."""
    seq_len = soft.shape[0]
    sums = np.zeros(n, dtype=np.float64)
    counts = np.zeros(n, dtype=np.float64)
    idx = np.arange(seq_len) % n
    np.add.at(sums, idx, soft)
    np.add.at(counts, idx, 1)
    return sums / np.maximum(counts, 1.0)


def _fold_signed_weights(observed_bits: np.ndarray, weights: np.ndarray,
                         n: int) -> np.ndarray:
    """Fold signed, [0,1]-weighted soft-tokens t*w to length n. weights in [0,1]."""
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float64)
    return _fold_raw(signs * weights, n)                  # t * w in [-1, 1]


def fold_soft_token(observed_bits: np.ndarray, p_array: np.ndarray,
                    n: int) -> np.ndarray:
    """
    Soft-token fold for the Hoeffding detector (prc_fpr_proof.pdf).

    Each observation yields a soft-token S = t * H where t = +/-1 is the
    observed sign and H = H_2(p)/log(2) in [0, 1] is the normalized entropy,
    so |S| <= 1 (exactly S_j = t_j * H_j in the proof's notation).
    """
    weights = binary_entropy(p_array) / np.log(2)         # H in [0, 1]
    return _fold_signed_weights(observed_bits, weights, n)


def fold_step_token(observed_bits: np.ndarray, p_array: np.ndarray,
                    n: int, temp: float = 0.05) -> np.ndarray:
    """Soft-token fold using the step gate f(H) on the normalized entropy H."""
    H = binary_entropy(p_array) / np.log(2)               # H in [0, 1]
    return _fold_signed_weights(observed_bits, step_weight(H, temp), n)


# Per-token soft-token weights, all valued in [0, 1] (so |S_j| <= 1 and the
# Hoeffding FPR bound holds). H = H_2(p)/log2 is the normalized entropy;
# gap = 2*min(p, 1-p) = 1 - |1-2p| is the per-token signal amplitude (the
# matched-filter-optimal weight if per-token noise were constant).
WEIGHT_KINDS = ("entropy", "naive", "step",
                "h2", "h3", "sqrt", "gap", "gap2", "gapsqrt",
                "gap075", "gap125", "map")


def map_soft_token(observed_bits: np.ndarray, p_array: np.ndarray) -> np.ndarray:
    """Bayes-optimal bounded soft-token S_j = E[c | observed bit, p] in [-1, 1],
    where c = 1-2*xi is the +/-1 codeword sign. Unlike t*w(p) it is ASYMMETRIC
    in the observed bit: landing in the low-probability partition is a *certain*
    watermark signal (soft = +/-1), landing in the high-prob one is weak.

      bit=0 (partition 0):  p<=0.5 -> +p/(1-p)      ;  p>0.5 -> +1
      bit=1 (partition 1):  p<=0.5 -> -1            ;  p>0.5 -> -(1-p)/p

    |S_j| <= 1 so the Hoeffding FPR bound is preserved.
    """
    p = np.clip(np.asarray(p_array, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    b = observed_bits.astype(np.int64)
    lo = p / (1.0 - p)                                    # in [0,1] for p<=0.5
    hi = (1.0 - p) / p                                    # in [0,1] for p>0.5
    soft_b0 = np.where(p <= 0.5, lo, 1.0)                 # bit=0 -> positive
    soft_b1 = np.where(p <= 0.5, -1.0, -hi)               # bit=1 -> negative
    return np.where(b == 0, soft_b0, soft_b1)


def weights_from_p(p_array: np.ndarray, kind: str) -> np.ndarray:
    p = np.clip(np.asarray(p_array, dtype=np.float64), 0.0, 1.0)
    H = binary_entropy(p) / np.log(2)                     # H in [0, 1]
    if kind == "entropy":
        return H                                          # baseline (linear H)
    if kind == "naive":
        return np.ones_like(H)
    if kind == "step":
        return step_weight(H)                             # sharp gate at H=0.5
    if kind == "h2":
        return H ** 2                                     # convex: suppress low H
    if kind == "h3":
        return H ** 3
    if kind == "sqrt":
        return np.sqrt(H)                                 # concave: boost low H
    gap = 1.0 - np.abs(1.0 - 2.0 * p)                     # 2*min(p,1-p) in [0,1]
    if kind == "gap":
        return gap                                        # matched-filter proxy
    if kind == "gap2":
        return gap ** 2
    if kind == "gapsqrt":
        return np.sqrt(gap)
    if kind == "gap075":
        return gap ** 0.75
    if kind == "gap125":
        return gap ** 1.25
    raise ValueError(f"unknown weight {kind!r}; choose {WEIGHT_KINDS}")


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
    entropy_weighted=None,
    weight=None,
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

    weight selects the per-token soft-token (all keep |S_j| <= 1, so the FPR <= F
    guarantee holds for every option); see WEIGHT_KINDS / weights_from_p / map_soft_token.
    Default: "map", the Bayes-optimal soft-token S_j = E[c | observed bit, p],
    which is a uniform improvement over the linear-entropy weight "entropy".
    For backwards compatibility, if weight is None the legacy entropy_weighted
    flag is honored (True->"entropy", False->"naive"); if it too is None -> "map".
    """
    if weight is None:
        if entropy_weighted is None:
            weight = "map"                       # new default: Bayes-optimal
        else:
            weight = "entropy" if entropy_weighted else "naive"
    if weight not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight {weight!r}; choose {WEIGHT_KINDS}")
    if weight == "map":
        # Bayes-optimal soft-token: depends on the observed bit, not just p.
        fold = lambda b, p, m: _fold_raw(map_soft_token(b, p), m)
    else:
        fold = lambda b, p, m: _fold_signed_weights(
            b, weights_from_p(p, weight), m)

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
