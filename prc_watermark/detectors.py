import hashlib
import numpy as np
import torch
from prc_watermark.prc import Detect

WEIGHT_KINDS = ("map", "entropy", "naive")


def tensor_sha256(value) -> str:
    if torch is None:
        raise RuntimeError("torch is required to hash tensors")
    tensor = (
        value.detach().cpu().contiguous()
        if torch.is_tensor(value)
        else torch.as_tensor(value)
    )
    raw = tensor.view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + raw).hexdigest()


def semantic_sha256(value) -> str:
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


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log1p(-p))


def map_soft_token(observed_bits: np.ndarray, p_array: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p_array, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    b = observed_bits.astype(np.int64)
    lo = p / (1.0 - p)
    hi = (1.0 - p) / p
    soft_b0 = np.where(p <= 0.5, lo, 1.0)
    soft_b1 = np.where(p <= 0.5, -1.0, -hi)
    return np.where(b == 0, soft_b0, soft_b1)


def tokens_to_bits(token_ids, partition_map) -> np.ndarray:
    if token_ids.dim() != 1:
        token_ids = token_ids.flatten()
    bit_for_token = partition_map[1].long().to(token_ids.device)
    bits = bit_for_token[token_ids].detach().cpu().numpy().astype(np.int64)
    return bits


def detect_hoeffding_prefix(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-09,
    weight="map",
    return_info=False,
):
    (_, parity_check_matrix, one_time_pad, _, _, _, _, _, t) = decoding_key
    (r, n) = parity_check_matrix.shape
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    k = bits.shape[0]
    S = _soft_tokens(bits, p_arr, weight)
    idx = parity_check_matrix.indices.reshape(r, t)
    keep = (idx < k).all(axis=1)
    r_eff = int(keep.sum())
    otp = np.asarray(one_time_pad, dtype=np.int64)
    if r_eff == 0:
        info = {
            "method": "hoeffding_prefix",
            "statistic": 0.0,
            "threshold": float("inf"),
            "V": 0.0,
            "r_eff": 0,
            "k": k,
            "fpr": fpr,
        }
        return (False, info) if return_info else False
    idx_k = idx[keep]
    S_w = np.prod(S[idx_k], axis=1)
    a_w = np.prod(1 - 2 * otp[idx_k], axis=1).astype(np.float64)
    S_stat = float(np.sum(a_w * S_w))
    V = float(np.sum(S_w**2))
    tau = float(np.sqrt(2 * V * np.log(1 / fpr))) if V > 0 else float("inf")
    decision = bool(S_stat >= tau)
    if not return_info:
        return decision
    return (
        decision,
        {
            "method": "hoeffding_prefix",
            "statistic": S_stat,
            "threshold": tau,
            "V": V,
            "r_eff": r_eff,
            "k": k,
            "fpr": fpr,
        },
    )


def detect_hoeffding(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-09,
    weight="map",
    return_info=False,
):
    if weight not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight {weight!r}; choose {WEIGHT_KINDS}")
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64)
    T = bits.shape[0]
    if T < n:
        return detect_hoeffding_prefix(
            decoding_key,
            generated_token_ids,
            partition_probs,
            partition_map,
            fpr=fpr,
            weight=weight,
            return_info=return_info,
        )
    soft = _soft_tokens(bits, p_arr, weight)
    slices = [slice(b * n, (b + 1) * n) for b in range(T // n)]
    num_blocks = len(slices)
    block_fpr = fpr / num_blocks
    decision = False
    blocks_passed = 0
    best = None
    for b, sl in enumerate(slices):
        post = soft[sl]
        (dec, info) = Detect(
            decoding_key, post, false_positive_rate=block_fpr, return_info=True
        )
        if info["V"] == 0:
            (dec, info) = (False, {**info, "threshold": float("inf")})
        margin = info["statistic"] - info["threshold"]
        if dec:
            decision = True
            blocks_passed += 1
        if best is None or margin > best[0]:
            best = (margin, b, info)
    if not return_info:
        return decision
    (_, best_block, best_info) = best
    return (
        decision,
        {
            "method": "hoeffding_blockwise",
            "statistic": best_info["statistic"],
            "threshold": best_info["threshold"],
            "V": best_info["V"],
            "num_blocks": num_blocks,
            "blocks_passed": blocks_passed,
            "best_block": best_block,
            "block_fpr": block_fpr,
            "fpr": fpr,
        },
    )


def detect_online_hoeffding(
    online_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    fpr=1e-09,
    weight="map",
    return_info=False,
    numerical_tolerance=1e-15,
):
    fpr_policy = "one_shot"
    from prc_watermark.prc import OnlinePRCKey, materialize_supports, otp_prefix

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    if not isinstance(online_key, OnlinePRCKey):
        raise TypeError("online_key must be OnlinePRCKey or its serialized dict")
    if weight not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight {weight!r}; choose {WEIGHT_KINDS}")
    if not 0.0 < float(fpr) < 1.0:
        raise ValueError("fpr must be in (0, 1)")
    bits = tokens_to_bits(generated_token_ids, partition_map)
    p_arr = np.asarray(partition_probs, dtype=np.float64).reshape(-1)
    length = int(bits.shape[0])
    soft = _soft_tokens(bits, p_arr, weight)
    supports = materialize_supports(length, online_key)
    effective_fpr = float(fpr)
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
    V = float(np.sum(check_values**2))
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


def _soft_tokens(bits, probabilities, weight):
    if weight not in WEIGHT_KINDS:
        raise ValueError(f"unknown weight {weight!r}")
    if weight == "naive":
        return (1 - 2 * bits).astype(np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if (
        not len(bits)
        or p.shape != (len(bits) - 1,)
        or (not np.isfinite(p).all())
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Expected T-1 finite probabilities in [0, 1]")
    soft = np.zeros(len(bits), dtype=np.float64)
    soft[1:] = (
        map_soft_token(bits[1:], p)
        if weight == "map"
        else (1 - 2 * bits[1:]).astype(np.float64)
        * (binary_entropy(np.clip(p, 0.0, 1.0)) / np.log(2))
    )
    return soft
