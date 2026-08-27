"""
Model-free PRC detector helpers.

These are pure numpy/torch functions (no LM, no model load) shared by
watermark_expt.py and the Modal detect pass. Keeping them here lets detection
run on cached generations without importing watermark_expt (which loads the
Qwen model at import time).
"""
import hashlib

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from prc import Detect


GENERATION_TRACE_SCHEMA_VERSION = 1


def _as_cpu_token_tensor(token_ids):
    """Return flattened int64 token IDs without retaining accelerator storage."""
    if torch is None:
        raise RuntimeError("torch is required to build generation records")
    if torch.is_tensor(token_ids):
        tensor = token_ids.detach().cpu().flatten().to(dtype=torch.int64)
    else:
        tensor = torch.as_tensor(token_ids, dtype=torch.int64).flatten()
    # A row selected from a batched tensor retains the batch's full backing
    # storage. torch.save() serializes that storage, not only the visible row,
    # unless we force an independent per-record allocation here.
    return tensor.clone()


def tensor_sha256(value) -> str:
    """Stable SHA-256 for a tensor's dtype, shape, and contiguous bytes."""
    if torch is None:
        raise RuntimeError("torch is required to hash tensors")
    tensor = (
        value.detach().cpu().contiguous()
        if torch.is_tensor(value) else torch.as_tensor(value)
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + raw).hexdigest()


def semantic_sha256(value) -> str:
    """Stable SHA-256 for nested experiment objects without saving secrets."""
    digest = hashlib.sha256()

    def update(item):
        if item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: str(key)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        elif torch is not None and torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif hasattr(item, "tocsr"):
            sparse = item.tocsr()
            digest.update(b"sparse-csr")
            update(np.asarray(sparse.shape))
            update(np.asarray(sparse.indptr))
            update(np.asarray(sparse.indices))
            update(np.asarray(sparse.data))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(np.asarray(item))
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _validated_sha256(name: str, value) -> str:
    fingerprint = str(value).lower()
    if (len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return fingerprint


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


def fold_map_soft_token(observed_bits: np.ndarray, p_array: np.ndarray,
                        n: int) -> np.ndarray:
    """Cyclically fold MAP soft tokens to PRC codeword positions."""
    return _fold_raw(map_soft_token(observed_bits, p_array), n)


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


def build_prc_generation_record(
    prompt_token_ids,
    generated_token_ids,
    p_trace,
    partition_map,
    n: int,
    watermark: bool,
    *,
    encoding_key=None,
    encoding_key_fingerprint: str | None = None,
    prc_codeword_bits,
    base_lm_entropy,
    base_token_logprob,
    partition_fingerprint: str | None = None,
) -> dict:
    """Build the self-contained, versioned record saved for every PRC sample.

    ``p_trace`` is the base LM's probability mass on partition 1. Entropies
    are in bits. ``signed_entropy_trace`` uses the observed token partition
    sign (+ for partition 0, - for partition 1), while
    ``prc_codeword_bits`` stores the exact noisy codeword bits that actually
    controlled watermarked generation. Null samples explicitly store no PRC
    codeword.

    The function intentionally stores the compact raw inputs as well as the
    cheap derived traces. This keeps records independently auditable without
    retaining full-vocabulary logits.
    """
    if int(n) <= 0:
        raise ValueError(f"n must be positive, got {n}")

    prompt_tokens = _as_cpu_token_tensor(prompt_token_ids)
    tokens = _as_cpu_token_tensor(generated_token_ids)
    # Copies prevent numpy row views from retaining batched backing buffers in
    # the per-sample record, mirroring the tensor ownership guarantee above.
    p_arr = np.asarray(p_trace, dtype=np.float64).reshape(-1).copy()
    base_entropy = (
        np.asarray(base_lm_entropy, dtype=np.float32).reshape(-1).copy()
    )
    token_logprob = (
        np.asarray(base_token_logprob, dtype=np.float32).reshape(-1).copy()
    )
    T = int(tokens.numel())

    for name, values in (
        ("p_trace", p_arr),
        ("base_lm_entropy", base_entropy),
        ("base_token_logprob", token_logprob),
    ):
        if values.shape != (T,):
            raise ValueError(
                f"{name} shape {values.shape} does not match tokens length {T}"
            )

    observed_bits = tokens_to_bits(tokens, partition_map).astype(np.uint8)
    entropy = (binary_entropy(p_arr) / np.log(2)).astype(np.float32)
    signs = (1 - 2 * observed_bits.astype(np.int64)).astype(np.float32)
    signed_entropy = (signs * entropy).astype(np.float32)
    map_tokens = map_soft_token(observed_bits, p_arr).astype(np.float32)
    folded_entropy = fold_soft_token(observed_bits, p_arr, int(n)).astype(
        np.float32
    )
    folded_map = fold_map_soft_token(observed_bits, p_arr, int(n)).astype(
        np.float32
    )

    if watermark:
        if prc_codeword_bits is None:
            raise ValueError("watermarked records require prc_codeword_bits")
        codeword = (
            np.asarray(prc_codeword_bits, dtype=np.uint8).reshape(-1).copy()
        )
        if codeword.shape != (T,):
            raise ValueError(
                "prc_codeword_bits shape "
                f"{codeword.shape} does not match tokens length {T}"
            )
        if np.any(codeword > 1):
            raise ValueError("prc_codeword_bits must contain only 0 or 1")
        codeword_signs = (
            1 - 2 * codeword.astype(np.int64)
        ).astype(np.float32)
        codeword_signed_entropy = (
            codeword_signs * entropy
        ).astype(np.float32)
    else:
        if prc_codeword_bits is not None:
            raise ValueError("null records must not contain a PRC codeword")
        codeword = None
        codeword_signed_entropy = None

    boundaries = np.asarray(
        [(start, min(start + int(n), T)) for start in range(0, T, int(n))],
        dtype=np.int32,
    ).reshape(-1, 2)
    if encoding_key_fingerprint:
        key_fingerprint = _validated_sha256(
            "encoding_key_fingerprint", encoding_key_fingerprint
        )
    elif encoding_key is not None:
        key_fingerprint = semantic_sha256(encoding_key)
    else:
        raise ValueError(
            "encoding_key or encoding_key_fingerprint is required"
        )

    if partition_fingerprint:
        partition_hash = _validated_sha256(
            "partition_fingerprint", partition_fingerprint
        )
    else:
        partition_hash = tensor_sha256(partition_map)

    return {
        "generation_trace_schema_version": GENERATION_TRACE_SCHEMA_VERSION,
        "watermark": bool(watermark),
        "prc_n": int(n),
        "prompt_token_ids": prompt_tokens,
        # Keep the historical field name used by every detector/analysis.
        "tokens": tokens,
        "p_trace": p_arr,
        "observed_bucket_bits": observed_bits,
        "entropy_trace": entropy,
        "signed_entropy_trace": signed_entropy,
        "codeword_signed_entropy_trace": codeword_signed_entropy,
        "map_soft_tokens": map_tokens,
        "folded_signed_entropy": folded_entropy,
        "folded_map_soft_tokens": folded_map,
        "prc_codeword_bits": codeword,
        "prc_block_boundaries": boundaries,
        "base_lm_entropy": base_entropy,
        "base_token_logprob": token_logprob,
        "partition_sha256": partition_hash,
        "encoding_key_sha256": key_fingerprint,
        "trace_semantics": {
            "p_trace": "base LM P[token in PRC partition 1]",
            "entropy_trace": "binary bucket entropy H_2(p_trace), bits",
            "signed_entropy_trace": (
                "observed-token sign: "
                "(1 - 2*observed_bucket_bits) * entropy_trace"
            ),
            "codeword_signed_entropy_trace": (
                "latent-codeword sign: "
                "(1 - 2*prc_codeword_bits) * entropy_trace; null for "
                "unwatermarked samples"
            ),
            "map_soft_tokens": "E[PRC codeword sign | observed bucket, p_trace]",
            "prc_codeword_bits": (
                "exact noisy Encode() bits used during generation; null for "
                "unwatermarked samples"
            ),
            "prc_block_boundaries": (
                "half-open [start, end) ranges for independently sampled "
                "length-prc_n codewords"
            ),
            "folded_signed_entropy": (
                "signed_entropy_trace cyclically averaged by position mod prc_n"
            ),
            "folded_map_soft_tokens": (
                "map_soft_tokens cyclically averaged by position mod prc_n"
            ),
            "base_lm_entropy": "full-vocabulary pre-watermark entropy, bits",
            "base_token_logprob": (
                "natural-log probability under the pre-watermark base LM"
            ),
        },
    }


def _soft_tokens(bits, p_arr, weight):
    """Per-position soft-token S_j in [-1, 1] (no folding). map is Bayes-optimal;
    other weights use the symmetric t_j * w(p_j)."""
    if weight == "map":
        return map_soft_token(bits, p_arr)
    return (1 - 2 * bits).astype(np.float64) * weights_from_p(p_arr, weight)


def detect_hoeffding_prefix(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-9,
    weight="map",
    return_info=False,
):
    """Prefix-column Hoeffding detector for short outputs (T = k < n).

    A length-k output only exercises codeword positions 0..k-1 (xi = codeword[pos]
    for pos < n), so only the parity checks whose t columns ALL fall in [0, k) are
    fully observed. We score the Hoeffding statistic over exactly those r_eff
    checks using the k per-position soft-tokens directly -- no cyclic folding of a
    partial block, no Bonferroni split (a single test), so the FPR <= `fpr`
    guarantee holds over the random OTP restricted to the used checks.
    """
    (_, parity_check_matrix, one_time_pad, _, _, _, _, _, t) = decoding_key
    r, n = parity_check_matrix.shape
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )
    k = bits.shape[0]

    S = _soft_tokens(bits, p_arr, weight)                    # (k,) in [-1, 1]
    idx = parity_check_matrix.indices.reshape(r, t)          # (r, t) columns/check
    keep = (idx < k).all(axis=1)                             # checks inside [0, k)
    r_eff = int(keep.sum())
    otp = np.asarray(one_time_pad, dtype=np.int64)

    if r_eff == 0:
        info = {"method": "hoeffding_prefix", "statistic": 0.0,
                "threshold": float("inf"), "V": 0.0, "r_eff": 0, "k": k, "fpr": fpr}
        return (False, info) if return_info else False

    idx_k = idx[keep]
    S_w = np.prod(S[idx_k], axis=1)                          # soft-value per check
    a_w = np.prod(1 - 2 * otp[idx_k], axis=1).astype(np.float64)  # OTP parity +/-1
    S_stat = float(np.sum(a_w * S_w))
    V = float(np.sum(S_w ** 2))
    tau = float(np.sqrt(2 * V * np.log(1 / fpr))) if V > 0 else float("inf")
    decision = bool(S_stat >= tau)

    if not return_info:
        return decision
    return decision, {
        "method": "hoeffding_prefix",
        "statistic": S_stat,
        "threshold": tau,
        "V": V,
        "r_eff": r_eff,
        "k": k,
        "fpr": fpr,
    }


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
    if T < n:
        # Short output: detect on the first k=T codeword positions directly, using
        # only parity checks fully supported on columns [0, k) -- no partial-block
        # folding. (See detect_hoeffding_prefix.)
        return detect_hoeffding_prefix(
            decoding_key, generated_token_ids, partition_probs, partition_map,
            fpr=fpr, weight=weight, return_info=return_info,
        )
    slices = [slice(b * n, (b + 1) * n) for b in range(T // n)]
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


def detect_online_hoeffding(
    online_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-9,
    weight="map",
    fpr_policy="one_shot",
    return_info=False,
    numerical_tolerance=1e-15,
):
    """Hoeffding detector for one causal prefix with ``T = n = length``.

    The online construction has one coordinate per generated token, so this
    detector deliberately performs no cyclic folding and no block OR.  It
    regenerates the prefix's causal support table and OTP directly from the
    compact online key.  The one-shot threshold has the same mathematical form
    as :func:`prc.Detect`; only its realized ``V`` and row count are dynamic.

    ``alpha_spending_v1`` is available for callers that repeatedly test every
    arriving prefix: alpha_L = 6 alpha / (pi^2 L^2).  Final-only experiments
    should use the default ``one_shot`` policy.
    """
    from online_prc import (
        OnlinePRCKey,
        materialize_supports,
        otp_prefix,
    )

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    if not isinstance(online_key, OnlinePRCKey):
        raise TypeError("online_key must be OnlinePRCKey or its serialized dict")
    if weight not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight {weight!r}; choose {WEIGHT_KINDS}")
    if not 0.0 < float(fpr) < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    if fpr_policy not in ("one_shot", "alpha_spending_v1"):
        raise ValueError(
            "fpr_policy must be 'one_shot' or 'alpha_spending_v1'"
        )

    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64).reshape(-1)
    if bits.shape != p_arr.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length {p_arr.shape[0]}"
        )
    length = int(bits.shape[0])
    if weight == "map":
        soft = map_soft_token(bits, p_arr)
    else:
        signs = (1 - 2 * bits.astype(np.int64)).astype(np.float64)
        soft = signs * weights_from_p(p_arr, weight)

    supports = materialize_supports(length, online_key)
    effective_fpr = float(fpr)
    if fpr_policy == "alpha_spending_v1" and length > 0:
        effective_fpr = float(6.0 * fpr / (np.pi ** 2 * length ** 2))

    base_info = {
        "method": "hoeffding_online_causal",
        "scheme": online_key.scheme,
        "schedule_version": online_key.schedule_version,
        "support_sampler_version": online_key.support_sampler_version,
        "weight": weight,
        "length": length,
        "n": length,
        "T": length,
        "r": int(supports.shape[0]),
        "free_coordinates": int(length - supports.shape[0]),
        "fpr": float(fpr),
        "effective_fpr": effective_fpr,
        "fpr_policy": fpr_policy,
    }
    if supports.shape[0] == 0:
        info = {
            **base_info,
            "statistic": 0.0,
            "threshold": float("inf"),
            "V": 0.0,
            "status": "insufficient_evidence_no_checks",
        }
        return (False, info) if return_info else False

    check_values = np.prod(soft[supports], axis=1)
    otp = otp_prefix(length, online_key).astype(np.int64)
    otp_signs = np.prod(1 - 2 * otp[supports], axis=1).astype(np.float64)
    statistic = float(np.sum(otp_signs * check_values))
    V = float(np.sum(check_values ** 2))
    if not np.isfinite(V) or not np.isfinite(statistic):
        raise ValueError("non-finite online detector statistic or variance proxy")
    if V <= float(numerical_tolerance):
        info = {
            **base_info,
            "statistic": statistic,
            "threshold": float("inf"),
            "V": V,
            "status": "insufficient_evidence_zero_variance",
        }
        return (False, info) if return_info else False

    threshold = float(np.sqrt(2.0 * V * np.log(1.0 / effective_fpr)))
    decision = bool(statistic >= threshold)
    info = {
        **base_info,
        "statistic": statistic,
        "threshold": threshold,
        "V": V,
        "status": "ok",
    }
    return (decision, info) if return_info else decision


def prepare_online_map_prefix_context(online_key, maximum_length):
    """Materialize prompt-independent MAP supports and OTP signs once."""
    from online_prc import (
        OnlinePRCKey,
        materialize_supports,
        otp_prefix,
    )

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    if not isinstance(online_key, OnlinePRCKey):
        raise TypeError("online_key must be OnlinePRCKey or its serialized dict")
    maximum = int(maximum_length)
    if maximum <= 0:
        raise ValueError("maximum_length must be positive")
    supports = materialize_supports(maximum, online_key)
    otp = otp_prefix(maximum, online_key).astype(np.int64)
    otp_signs = np.prod(
        1 - 2 * otp[supports], axis=1
    ).astype(np.float64)
    return {
        "online_key": online_key,
        "maximum_length": maximum,
        "supports": supports,
        "otp_signs": otp_signs,
    }


def prepare_online_map_prefix_trace(
    online_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    maximum_length,
    prepared_context=None,
):
    """Prepare one MAP trace once for adaptive prefix detection.

    The signed check contribution and squared contribution for every parity
    row through ``maximum_length`` are independent of the eventual stopping
    prefix.  Keeping those arrays lets an adaptive sweep score one descending
    length at a time without repeatedly converting tokens or rebuilding
    supports, while still avoiding work at lengths after the first failure.
    """
    from online_prc import OnlinePRCKey

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    if not isinstance(online_key, OnlinePRCKey):
        raise TypeError("online_key must be OnlinePRCKey or its serialized dict")
    maximum = int(maximum_length)
    if maximum <= 0:
        raise ValueError("maximum_length must be positive")
    context = (
        prepare_online_map_prefix_context(online_key, maximum)
        if prepared_context is None else prepared_context
    )
    if context.get("online_key") != online_key:
        raise ValueError("prepared MAP context uses a different online key")
    if int(context.get("maximum_length", -1)) != maximum:
        raise ValueError("prepared MAP context uses a different maximum length")

    tokens = _as_cpu_token_tensor(generated_token_ids)
    probabilities = np.asarray(
        partition_probs, dtype=np.float64
    ).reshape(-1)
    if int(tokens.numel()) < maximum or probabilities.size < maximum:
        raise ValueError(
            f"record has {min(int(tokens.numel()), probabilities.size)} values; "
            f"need prefix length {maximum}"
        )

    bits = tokens_to_bits(tokens[:maximum], partition_map)
    soft = map_soft_token(bits, probabilities[:maximum])
    longest_supports = context["supports"]
    check_values = np.prod(soft[longest_supports], axis=1)
    signed_check_values = context["otp_signs"] * check_values
    squared_check_values = check_values ** 2
    if (
        not np.all(np.isfinite(signed_check_values))
        or not np.all(np.isfinite(squared_check_values))
    ):
        raise ValueError("non-finite online detector check contribution")
    return {
        "online_key": online_key,
        "maximum_length": maximum,
        "supports": longest_supports,
        "signed_check_values": signed_check_values,
        "squared_check_values": squared_check_values,
    }


def score_prepared_online_map_prefix(
    prepared,
    length,
    fpr=1e-9,
    fpr_policy="one_shot",
    numerical_tolerance=1e-15,
):
    """Score one exact prefix from :func:`prepare_online_map_prefix_trace`."""
    from online_prc import target_row_count

    online_key = prepared["online_key"]
    length = int(length)
    maximum = int(prepared["maximum_length"])
    if length <= 0 or length > maximum:
        raise ValueError(
            f"length must be in [1, maximum_length={maximum}], got {length}"
        )
    if not 0.0 < float(fpr) < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    if fpr_policy not in ("one_shot", "alpha_spending_v1"):
        raise ValueError(
            "fpr_policy must be 'one_shot' or 'alpha_spending_v1'"
        )

    row_count = target_row_count(length, online_key)
    supports = prepared["supports"][:row_count]
    if supports.size and int(np.max(supports)) >= length:
        raise AssertionError(
            "causal support prefix references a coordinate outside its length"
        )

    effective_fpr = float(fpr)
    if fpr_policy == "alpha_spending_v1":
        effective_fpr = float(6.0 * fpr / (np.pi ** 2 * length ** 2))
    base_info = {
        "method": "hoeffding_online_causal",
        "scheme": online_key.scheme,
        "schedule_version": online_key.schedule_version,
        "support_sampler_version": online_key.support_sampler_version,
        "weight": "map",
        "length": length,
        "n": length,
        "T": length,
        "r": int(row_count),
        "free_coordinates": int(length - row_count),
        "fpr": float(fpr),
        "effective_fpr": effective_fpr,
        "fpr_policy": fpr_policy,
    }
    if row_count == 0:
        return {
            "decision": False,
            **base_info,
            "statistic": 0.0,
            "threshold": float("inf"),
            "V": 0.0,
            "status": "insufficient_evidence_no_checks",
        }

    statistic = float(np.sum(prepared["signed_check_values"][:row_count]))
    V = float(np.sum(prepared["squared_check_values"][:row_count]))
    if not np.isfinite(V) or not np.isfinite(statistic):
        raise ValueError("non-finite online detector statistic or variance proxy")
    if V <= float(numerical_tolerance):
        return {
            "decision": False,
            **base_info,
            "statistic": statistic,
            "threshold": float("inf"),
            "V": V,
            "status": "insufficient_evidence_zero_variance",
        }

    threshold = float(np.sqrt(2.0 * V * np.log(1.0 / effective_fpr)))
    return {
        "decision": bool(statistic >= threshold),
        **base_info,
        "statistic": statistic,
        "threshold": threshold,
        "V": V,
        "status": "ok",
    }


def detect_online_map_prefix_grid(
    online_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    prefix_lengths,
    fpr=1e-9,
    fpr_policy="one_shot",
    numerical_tolerance=1e-15,
):
    """Score MAP detection at several exact prefixes of one online record.

    The result at every requested length is mathematically identical to a
    separate :func:`detect_online_hoeffding` call with ``weight="map"``.  The
    token conversion and per-row check contributions are prepared once.
    """
    lengths = [int(length) for length in prefix_lengths]
    if not lengths:
        raise ValueError("prefix_lengths must be nonempty")
    if len(set(lengths)) != len(lengths):
        raise ValueError("prefix_lengths must not contain duplicates")
    if any(length <= 0 for length in lengths):
        raise ValueError("every prefix length must be positive")

    prepared = prepare_online_map_prefix_trace(
        online_key,
        generated_token_ids,
        partition_probs,
        partition_map,
        max(lengths),
    )

    return [
        score_prepared_online_map_prefix(
            prepared,
            length,
            fpr=fpr,
            fpr_policy=fpr_policy,
            numerical_tolerance=numerical_tolerance,
        )
        for length in lengths
    ]
