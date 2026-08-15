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
