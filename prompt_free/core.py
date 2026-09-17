"""Prompt-free inference and PRC scoring, independent of generation runners."""
from __future__ import annotations

import numpy as np
import torch

from detectors import map_soft_token, tokens_to_bits, weights_from_p
from online_prc import OnlinePRCKey, materialize_supports, otp_prefix
from prc import Detect
from qwen import make_kv_cache

PROTOCOL = "completion_only_raw_abstain_v1"
WEIGHTS = ("map", "entropy")


def validate_tokens(tokens):
    if not isinstance(tokens, torch.Tensor) or tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("stored completion tokens must be integer tensors")
    if tokens.ndim != 1 or tokens.numel() == 0:
        raise ValueError("expected a nonempty one-dimensional completion")
    return tokens.detach().cpu().to(torch.int64).clone()


def validate_partition(partition):
    if not isinstance(partition, torch.Tensor) or partition.ndim != 2 or partition.shape[0] != 2:
        raise ValueError("expected a 2 x vocabulary partition tensor")
    if partition.shape[1] == 0 or not torch.all((partition == 0) | (partition == 1)):
        raise ValueError("partition entries must be binary")
    if not torch.all(partition.sum(0) == 1):
        raise ValueError("partition buckets must be disjoint and cover the vocabulary")


def recover(model, tokens, partition_one, *, cache="static"):
    """Return B x (T-1) probabilities for original PRC coordinates 2..T.

    Actual inputs concatenate to exactly tokens[:, :-1]. No tokenizer, prefix,
    prompt, padding, or generation cache is accepted. Each call has a fresh KV
    cache, and the first completion token is at model position zero.
    """
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.int64:
        raise ValueError("expected int64 completion tokens")
    if tokens.ndim != 2 or min(tokens.shape) == 0:
        raise ValueError("expected nonempty B x T completion tokens")
    if partition_one.ndim != 1 or not torch.all((partition_one == 0) | (partition_one == 1)):
        raise ValueError("expected a one-dimensional binary partition mask")
    if cache not in ("concat", "static"):
        raise ValueError("cache must be concat or static")
    if torch.any(tokens < 0) or torch.any(tokens >= partition_one.numel()):
        raise ValueError("completion token outside detector vocabulary")
    if tokens.shape[1] == 1:
        return torch.empty((tokens.shape[0], 0), dtype=torch.float32)
    model.eval()
    kv = make_kv_cache(cache, max_length=tokens.shape[1]-1)
    part = partition_one.to(tokens.device)
    recovered = []
    with torch.no_grad():
        for previous in range(tokens.shape[1]-1):
            logits = model(tokens[:, previous:previous+1], cache=kv)[:, -1]
            p = (torch.softmax(logits, dim=-1)*part).sum(dim=-1)
            recovered.append(p.detach().cpu())
    trace = torch.stack(recovered, dim=1).float()
    if not torch.isfinite(trace).all() or torch.any((trace < 0) | (trace > 1)):
        raise ValueError("invalid recovered probabilities")
    return trace


def soft_scores(bits, probabilities_2_to_T, weight="map"):
    bits = np.asarray(bits)
    p = np.asarray(probabilities_2_to_T, dtype=np.float64)
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    if bits.ndim != 1 or not len(bits) or p.shape != (len(bits)-1,):
        raise ValueError("expected T bucket bits and T-1 probabilities")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("bucket bits must be binary")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("invalid probabilities")
    bits = bits.astype(np.int64)
    soft = np.zeros(len(bits), dtype=np.float64)
    soft[1:] = (map_soft_token(bits[1:], p) if weight == "map"
                else (1-2*bits[1:])*weights_from_p(p, weight))
    return soft


def _checks(soft, supports, otp, fpr, tolerance=0.):
    products = np.prod(soft[supports], axis=1)
    signs = np.prod(1-2*otp[supports], axis=1).astype(np.float64)
    statistic = float(np.sum(signs*products))
    variance = float(np.sum(products**2))
    if not np.isfinite(statistic) or not np.isfinite(variance):
        raise ValueError("non-finite parity statistic")
    # Abstention can leave every check at zero, especially in short prefixes.
    # Such a candidate has no evidence; 0 >= 0 must not report detection.
    threshold = float(np.sqrt(2*variance*np.log(1/fpr))) if variance > tolerance else None
    return {"decision": bool(threshold is not None and statistic >= threshold),
            "statistic": statistic, "V": variance, "threshold": threshold,
            "r": len(supports), "effective_fpr": float(fpr),
            "status": "ok" if threshold is not None else "insufficient_evidence"}


def _fixed(artifact, soft, fpr):
    key = artifact["decoding_key"]
    matrix, otp, t = key[1], np.asarray(key[2], dtype=np.int64), int(key[-1])
    r, n = matrix.shape
    if not np.all(np.diff(matrix.indptr) == t) or not np.all(matrix.data == 1):
        raise ValueError("invalid fixed PRC parity matrix")
    supports = matrix.indices.reshape(r, t)
    if len(soft) < n:
        selected = supports[(supports < len(soft)).all(axis=1)]
        return {**_checks(soft, selected, otp, fpr), "method": "fixed_prefix",
                "n": n, "length": len(soft), "ignored_trailing_tokens": 0}
    count = len(soft)//n
    blocks = []
    for b in range(count):
        # Only global completion coordinate 1 was zeroed. Later block starts
        # retain their ordinary response-only scores and original PRC indices.
        block = soft[b*n:(b+1)*n]
        decision, info = Detect(key, block, false_positive_rate=fpr/count, return_info=True)
        valid = info["V"] > 0
        blocks.append({"decision": bool(valid and decision),
                       **{k: float(info[k]) for k in ("statistic", "V")},
                       "threshold": float(info["threshold"]) if valid else None,
                       "status": "ok" if valid else "insufficient_evidence"})
    return {"decision": any(b["decision"] for b in blocks), "method": "fixed_block_or",
            "n": n, "length": len(soft), "num_blocks": count, "r": r,
            "block_fpr": fpr/count, "ignored_trailing_tokens": len(soft) % n,
            "blocks": blocks}


def score(artifact, tokens, probabilities_2_to_T, *, construction, fpr, fpr_policy, weight="map"):
    """Score one candidate with its original fixed/online policy and new V."""
    if not 0 < fpr < 1:
        raise ValueError("FPR must be in (0, 1)")
    tokens = validate_tokens(tokens)
    validate_partition(artifact["partition"])
    if torch.any(tokens < 0) or torch.any(tokens >= artifact["partition"].shape[1]):
        raise ValueError("completion token outside partition vocabulary")
    bits = tokens_to_bits(tokens, artifact["partition"])
    soft = soft_scores(bits, probabilities_2_to_T, weight)
    if construction == "fixed":
        if fpr_policy != "block_or_bonferroni":
            raise ValueError("fixed PRC requires its block-OR Bonferroni policy")
        result = _fixed(artifact, soft, fpr)
    elif construction == "online":
        if fpr_policy not in ("one_shot", "alpha_spending_v1"):
            raise ValueError("unsupported online FPR policy")
        key = OnlinePRCKey.from_dict(artifact["online_key"])
        supports = materialize_supports(len(tokens), key)
        alpha = fpr if fpr_policy == "one_shot" else 6*fpr/(np.pi**2*len(tokens)**2)
        result = {**_checks(soft, supports, otp_prefix(len(tokens), key).astype(np.int64), alpha, 1e-15),
                  "method": "online", "n": len(tokens), "length": len(tokens)}
    else:
        raise ValueError("construction must be fixed or online")
    return {**result, "weight": weight, "fpr": fpr, "fpr_policy": fpr_policy,
            "protocol": PROTOCOL, "coordinate_one_score": 0.}
