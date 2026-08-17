"""Cache-only replay for low-entropy detector interventions.

Phase 0 reconstructs the existing online MAP statistic from saved token and
partition-probability traces.  Phase 1 scores the identical check evidence
with both the legacy Hoeffding threshold and the optimized conditional
weighted-Rademacher Chernoff threshold.  Phase 2 separately selects a
probability-only reliability-adaptive basis of the same fixed dual code.

This module deliberately contains no Modal app and no model imports.  It can
be imported by a short-lived remote cache reader without affecting generation
jobs, or run against locally available ``torch.save`` records.
"""
from __future__ import annotations

import math

import numpy as np

from adaptive_parity_basis import (
    DEFAULT_ERASURE_QUANTILES,
    bucket_reliability,
    select_reliability_adaptive_basis,
)
from detectors import (
    fold_map_soft_token,
    map_soft_token,
    prepare_online_map_prefix_context,
    prepare_online_map_prefix_trace,
    tokens_to_bits,
)
from online_prc import OnlinePRCKey, target_row_count
from weighted_rademacher import (
    weighted_rademacher_log_tail_bound,
    weighted_rademacher_threshold,
)


FPR_POLICIES = ("one_shot", "alpha_spending_v1")


def effective_false_positive_rate(
    false_positive_rate: float,
    length: int,
    policy: str = "one_shot",
) -> float:
    """Return the exact per-test alpha used by the current online detector."""
    fpr = float(false_positive_rate)
    length = int(length)
    if not 0.0 < fpr < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")
    if length <= 0:
        raise ValueError("length must be positive")
    if policy not in FPR_POLICIES:
        raise ValueError(f"policy must be one of {FPR_POLICIES}")
    if policy == "alpha_spending_v1":
        return float(6.0 * fpr / (np.pi ** 2 * length ** 2))
    return fpr


def prepare_online_map_evidence(
    online_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    *,
    length: int | None = None,
    prepared_context=None,
) -> dict:
    """Reconstruct the exact per-check MAP evidence for one cached record."""
    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    if not isinstance(online_key, OnlinePRCKey):
        raise TypeError("online_key must be OnlinePRCKey or its serialized dict")

    probabilities = np.asarray(partition_probs, dtype=np.float64).reshape(-1)
    replay_length = probabilities.size if length is None else int(length)
    if replay_length <= 0 or replay_length > probabilities.size:
        raise ValueError(
            f"length must be in [1, {probabilities.size}], got {replay_length}"
        )
    context = prepared_context
    if context is None:
        context = prepare_online_map_prefix_context(online_key, replay_length)
    prepared = prepare_online_map_prefix_trace(
        online_key,
        generated_token_ids,
        probabilities,
        partition_map,
        replay_length,
        prepared_context=context,
    )
    row_count = target_row_count(replay_length, online_key)
    signed = np.asarray(
        prepared["signed_check_values"][:row_count], dtype=np.float64
    ).copy()
    squared = np.asarray(
        prepared["squared_check_values"][:row_count], dtype=np.float64
    ).copy()
    if signed.shape != squared.shape:
        raise AssertionError("prepared signed and squared check arrays disagree")
    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(squared)):
        raise ValueError("prepared evidence contains non-finite check values")
    if np.any(squared < 0.0):
        raise ValueError("squared check values must be nonnegative")

    return {
        "online_key": online_key,
        "length": replay_length,
        "r": int(row_count),
        "supports": np.asarray(prepared["supports"][:row_count]).copy(),
        "signed_check_values": signed,
        # The Rademacher null depends only on |q_a|.  Recovering it from q_a^2
        # lets existing prepared MAP shards be replayed without a schema change.
        "check_weights": np.sqrt(squared),
        "squared_check_values": squared,
    }


def score_online_map_evidence(
    evidence: dict,
    *,
    false_positive_rate: float,
    fpr_policy: str = "one_shot",
    numerical_tolerance: float = 1e-15,
) -> dict:
    """Score identical MAP evidence with Hoeffding and Rademacher calibration."""
    length = int(evidence["length"])
    signed = np.asarray(
        evidence["signed_check_values"], dtype=np.float64
    ).reshape(-1)
    squared = np.asarray(
        evidence["squared_check_values"], dtype=np.float64
    ).reshape(-1)
    weights = np.asarray(evidence["check_weights"], dtype=np.float64).reshape(-1)
    if not (signed.shape == squared.shape == weights.shape):
        raise ValueError("evidence check arrays must have equal shapes")
    if not (
        np.all(np.isfinite(signed))
        and np.all(np.isfinite(squared))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("evidence check arrays must be finite")
    if np.any(squared < 0.0) or np.any(weights < 0.0):
        raise ValueError("squared values and check weights must be nonnegative")
    if not np.allclose(weights ** 2, squared, rtol=1e-12, atol=1e-15):
        raise ValueError("check_weights squared must match squared_check_values")

    effective_fpr = effective_false_positive_rate(
        false_positive_rate, length, fpr_policy
    )
    statistic = float(np.sum(signed))
    variance_proxy = float(np.sum(squared))
    base = {
        "method": "online_map_phase01_replay",
        "length": length,
        "n": length,
        "T": length,
        "r": int(signed.size),
        "fpr": float(false_positive_rate),
        "effective_fpr": effective_fpr,
        "fpr_policy": fpr_policy,
        "statistic": statistic,
        "V": variance_proxy,
        "support_bound": float(np.sum(weights)),
        "nonzero_checks": int(np.count_nonzero(weights)),
    }
    if signed.size == 0:
        empty = {
            "decision": False,
            "threshold": float("inf"),
            "status": "insufficient_evidence_no_checks",
        }
        return {
            **base,
            "status": "insufficient_evidence_no_checks",
            "calibrations": {
                "hoeffding": {"method": "hoeffding", **empty},
                "weighted_rademacher_chernoff": {
                    "method": "weighted_rademacher_chernoff",
                    **empty,
                    "log_pvalue_upper": 0.0,
                    "pvalue_upper": 1.0,
                },
            },
        }
    if variance_proxy <= float(numerical_tolerance):
        empty = {
            "decision": False,
            "threshold": float("inf"),
            "status": "insufficient_evidence_zero_variance",
        }
        return {
            **base,
            "status": "insufficient_evidence_zero_variance",
            "calibrations": {
                "hoeffding": {"method": "hoeffding", **empty},
                "weighted_rademacher_chernoff": {
                    "method": "weighted_rademacher_chernoff",
                    **empty,
                    "log_pvalue_upper": 0.0,
                    "pvalue_upper": 1.0,
                },
            },
        }

    hoeffding_threshold = float(
        np.sqrt(2.0 * variance_proxy * np.log(1.0 / effective_fpr))
    )
    hoeffding_log_p = (
        min(0.0, -statistic ** 2 / (2.0 * variance_proxy))
        if statistic > 0.0 else 0.0
    )
    rademacher_threshold, threshold_info = weighted_rademacher_threshold(
        weights, effective_fpr, return_info=True
    )
    rademacher_log_p, tail_info = weighted_rademacher_log_tail_bound(
        statistic, weights, return_info=True
    )
    rademacher_decision = bool(rademacher_log_p <= math.log(effective_fpr))
    if np.isfinite(rademacher_threshold):
        threshold_decision = bool(statistic >= rademacher_threshold)
        if rademacher_decision != threshold_decision:
            raise AssertionError(
                "Rademacher p-value and threshold decisions disagree"
            )

    return {
        **base,
        "status": "ok",
        "threshold_ratio_rademacher_to_hoeffding": float(
            rademacher_threshold / hoeffding_threshold
        ),
        "calibrations": {
            "hoeffding": {
                "method": "hoeffding",
                "decision": bool(statistic >= hoeffding_threshold),
                "threshold": hoeffding_threshold,
                "log_pvalue_upper": float(hoeffding_log_p),
                "pvalue_upper": float(math.exp(hoeffding_log_p)),
                "status": "ok",
            },
            "weighted_rademacher_chernoff": {
                "method": "weighted_rademacher_chernoff",
                "decision": rademacher_decision,
                "threshold": float(rademacher_threshold),
                "log_pvalue_upper": float(rademacher_log_p),
                "pvalue_upper": float(tail_info["pvalue_upper"]),
                "threshold_lambda": float(threshold_info["lambda_star"]),
                "tail_lambda": float(tail_info["lambda_star"]),
                "support_bound": float(tail_info["support_bound"]),
                "nonzero_checks": int(tail_info["nonzero_checks"]),
                "threshold_optimizer_converged": bool(
                    threshold_info["optimizer_converged"]
                ),
                "tail_optimizer_converged": bool(
                    tail_info["optimizer_converged"]
                ),
                "threshold_status": threshold_info["status"],
                "tail_status": tail_info["status"],
                "status": (
                    "ok"
                    if threshold_info["optimizer_converged"]
                    and tail_info["optimizer_converged"]
                    else "conservative_numerical_fallback"
                ),
            },
        },
    }


def prepare_fixed_map_block_evidence(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    *,
    block_index: int = 0,
) -> dict:
    """Reconstruct one fixed-PRC block's exact MAP check evidence.

    This mirrors ``detect_hoeffding(..., weight="map")`` without applying a
    threshold.  Keeping preparation and calibration separate ensures that the
    Hoeffding and weighted-Rademacher decisions use identical check values.
    """
    if not isinstance(decoding_key, (tuple, list)) or len(decoding_key) != 9:
        raise TypeError("decoding_key must be the nine-field fixed PRC key")
    generator_matrix, parity_matrix, one_time_pad = decoding_key[:3]
    n = int(generator_matrix.shape[0])
    if n <= 0:
        raise ValueError("fixed PRC block length must be positive")

    bits = tokens_to_bits(generated_token_ids, partition_map)
    probabilities = np.asarray(partition_probs, dtype=np.float64).reshape(-1)
    if bits.shape != probabilities.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length "
            f"{probabilities.shape[0]}"
        )
    total_length = int(bits.size)
    if total_length >= n:
        num_blocks = total_length // n
        if not 0 <= int(block_index) < num_blocks:
            raise ValueError(
                f"block_index must be in [0, {num_blocks}), got {block_index}"
            )
        start = int(block_index) * n
        stop = start + n
        soft = map_soft_token(bits[start:stop], probabilities[start:stop])
    else:
        if int(block_index) != 0:
            raise ValueError("a partial fixed block only has block_index 0")
        num_blocks = 1
        soft = fold_map_soft_token(bits, probabilities, n)

    parity = parity_matrix.tocsr()
    if parity.shape[1] != n:
        raise ValueError(
            f"parity matrix has {parity.shape[1]} columns, expected n={n}"
        )
    row_sizes = np.diff(parity.indptr)
    if np.any(row_sizes <= 0):
        raise ValueError("fixed parity matrix contains an empty check")
    check_values = np.asarray(
        [
            np.prod(soft[parity.indices[parity.indptr[row]:parity.indptr[row + 1]]])
            for row in range(parity.shape[0])
        ],
        dtype=np.float64,
    )
    otp = np.asarray(one_time_pad, dtype=np.int64).reshape(-1)
    if otp.size != n:
        raise ValueError(f"one-time pad has length {otp.size}, expected n={n}")
    otp_signs = np.asarray(
        [
            np.prod(
                1 - 2 * otp[
                    parity.indices[parity.indptr[row]:parity.indptr[row + 1]]
                ]
            )
            for row in range(parity.shape[0])
        ],
        dtype=np.float64,
    )
    signed = otp_signs * check_values
    squared = check_values ** 2
    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(squared)):
        raise ValueError("fixed MAP evidence contains non-finite check values")

    return {
        "length": n,
        "n": n,
        "T": total_length,
        "r": int(parity.shape[0]),
        "num_blocks": int(num_blocks),
        "block_index": int(block_index),
        "signed_check_values": signed,
        "check_weights": np.abs(check_values),
        "squared_check_values": squared,
    }


def replay_cached_fixed_map_record(
    artifact: dict,
    record: dict,
    *,
    false_positive_rate: float,
) -> dict:
    """Replay all complete fixed-PRC blocks in one cached generation record."""
    for field in ("decoding_key", "partition"):
        if field not in artifact:
            raise KeyError(f"artifact is missing {field!r}")
    for field in ("tokens", "p_trace"):
        if field not in record:
            raise KeyError(f"record is missing {field!r}")
    fpr = float(false_positive_rate)
    if not 0.0 < fpr < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")

    n = int(artifact.get("n", artifact["decoding_key"][0].shape[0]))
    requested_T = int(artifact.get("T", len(record["p_trace"])))
    available = min(len(record["tokens"]), len(record["p_trace"]))
    if available < requested_T:
        raise ValueError(
            f"cached record has {available} values, artifact requires {requested_T}"
        )
    tokens = record["tokens"][:requested_T]
    probabilities = np.asarray(record["p_trace"][:requested_T])
    num_blocks = requested_T // n if requested_T >= n else 1
    block_fpr = fpr / num_blocks
    block_results = []
    for block_index in range(num_blocks):
        evidence = prepare_fixed_map_block_evidence(
            artifact["decoding_key"],
            tokens,
            probabilities,
            artifact["partition"],
            block_index=block_index,
        )
        scored = score_online_map_evidence(
            evidence,
            false_positive_rate=block_fpr,
            fpr_policy="one_shot",
        )
        scored["method"] = "fixed_map_block_phase01_replay"
        scored["block_index"] = block_index
        block_results.append(scored)

    calibrations = {}
    for method in ("hoeffding", "weighted_rademacher_chernoff"):
        method_blocks = [result["calibrations"][method] for result in block_results]
        margins = [
            block_results[index]["statistic"] - block["threshold"]
            for index, block in enumerate(method_blocks)
        ]
        best_index = int(np.argmax(margins))
        best = method_blocks[best_index]
        calibrations[method] = {
            **best,
            "decision": bool(any(block["decision"] for block in method_blocks)),
            "blocks_passed": int(sum(block["decision"] for block in method_blocks)),
            "best_block": best_index,
            "block_fpr": block_fpr,
        }

    result = {
        "method": "fixed_map_phase01_replay",
        "n": n,
        "T": requested_T,
        "r": int(block_results[0]["r"]),
        "num_blocks": num_blocks,
        "fpr": fpr,
        "block_fpr": block_fpr,
        "calibrations": calibrations,
        "blocks": block_results,
    }
    for field in ("prompt_idx", "watermark"):
        if field in record:
            result[field] = record[field]
    return result


def prepare_reliability_adaptive_fixed_map_block_evidence(
    decoding_key,
    generated_token_ids,
    partition_probs,
    partition_map,
    *,
    block_index: int = 0,
    erasure_quantiles=DEFAULT_ERASURE_QUANTILES,
) -> dict:
    """Build Phase 2 evidence from a probability-selected dual basis.

    Basis selection sees only the parity matrix, the absolute reliability
    implied by ``partition_probs``, and the configured channel noise rate.
    The one-time pad is consulted only after the basis has been fixed, when
    the independent transformed parity signs are computed for scoring.
    """
    if not isinstance(decoding_key, (tuple, list)) or len(decoding_key) != 9:
        raise TypeError("decoding_key must be the nine-field fixed PRC key")
    generator_matrix, parity_matrix, one_time_pad = decoding_key[:3]
    noise_rate = float(decoding_key[4])
    n = int(generator_matrix.shape[0])
    if n <= 0:
        raise ValueError("fixed PRC block length must be positive")

    bits = tokens_to_bits(generated_token_ids, partition_map)
    probabilities = np.asarray(partition_probs, dtype=np.float64).reshape(-1)
    if bits.shape != probabilities.shape:
        raise ValueError(
            f"tokens length {bits.shape[0]} != p_trace length "
            f"{probabilities.shape[0]}"
        )
    total_length = int(bits.size)
    num_blocks = total_length // n
    if num_blocks <= 0:
        raise ValueError(
            "Phase 2 adaptive-basis replay requires a complete fixed block"
        )
    if not 0 <= int(block_index) < num_blocks:
        raise ValueError(
            f"block_index must be in [0, {num_blocks}), got {block_index}"
        )
    start = int(block_index) * n
    stop = start + n
    block_probabilities = probabilities[start:stop]
    soft = map_soft_token(bits[start:stop], block_probabilities)
    reliabilities = bucket_reliability(block_probabilities)

    parity = parity_matrix.tocsr()
    if parity.shape[1] != n:
        raise ValueError(
            f"parity matrix has {parity.shape[1]} columns, expected n={n}"
        )
    basis = select_reliability_adaptive_basis(
        parity,
        reliabilities,
        noise_rate,
        erasure_quantiles=erasure_quantiles,
    )
    supports = basis["supports"]
    attenuation = 1.0 - 2.0 * noise_rate
    degrees = np.asarray([support.size for support in supports], dtype=np.int64)
    noise_weights = np.power(attenuation, degrees, dtype=np.float64)
    check_values = np.asarray(
        [
            noise_weights[row] * np.prod(soft[support])
            for row, support in enumerate(supports)
        ],
        dtype=np.float64,
    )

    otp = np.asarray(one_time_pad, dtype=np.int64).reshape(-1)
    if otp.size != n:
        raise ValueError(f"one-time pad has length {otp.size}, expected n={n}")
    otp_signs = np.asarray(
        [np.prod(1 - 2 * otp[support]) for support in supports],
        dtype=np.float64,
    )
    signed = otp_signs * check_values
    squared = check_values ** 2
    if not np.all(np.isfinite(signed)) or not np.all(np.isfinite(squared)):
        raise ValueError("adaptive fixed MAP evidence contains non-finite values")

    return {
        "method": "fixed_map_reliability_adaptive_basis_evidence_v1",
        "length": n,
        "n": n,
        "T": total_length,
        "r": int(len(supports)),
        "num_blocks": int(num_blocks),
        "block_index": int(block_index),
        "signed_check_values": signed,
        "check_weights": np.abs(check_values),
        "squared_check_values": squared,
        "basis_selection": basis["selection"],
    }


def replay_cached_fixed_map_record_phase2(
    artifact: dict,
    record: dict,
    *,
    false_positive_rate: float,
    erasure_quantiles=DEFAULT_ERASURE_QUANTILES,
) -> dict:
    """Replay the Phase 2 adaptive-basis detector on one cached record."""
    for field in ("decoding_key", "partition"):
        if field not in artifact:
            raise KeyError(f"artifact is missing {field!r}")
    for field in ("tokens", "p_trace"):
        if field not in record:
            raise KeyError(f"record is missing {field!r}")
    fpr = float(false_positive_rate)
    if not 0.0 < fpr < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")

    n = int(artifact.get("n", artifact["decoding_key"][0].shape[0]))
    requested_T = int(artifact.get("T", len(record["p_trace"])))
    available = min(len(record["tokens"]), len(record["p_trace"]))
    if available < requested_T:
        raise ValueError(
            f"cached record has {available} values, artifact requires {requested_T}"
        )
    if requested_T < n:
        raise ValueError(
            "Phase 2 adaptive-basis replay requires at least one complete block"
        )
    tokens = record["tokens"][:requested_T]
    probabilities = np.asarray(record["p_trace"][:requested_T])
    num_blocks = requested_T // n
    block_fpr = fpr / num_blocks
    block_results = []
    for block_index in range(num_blocks):
        evidence = prepare_reliability_adaptive_fixed_map_block_evidence(
            artifact["decoding_key"],
            tokens,
            probabilities,
            artifact["partition"],
            block_index=block_index,
            erasure_quantiles=erasure_quantiles,
        )
        scored = score_online_map_evidence(
            evidence,
            false_positive_rate=block_fpr,
            fpr_policy="one_shot",
        )
        scored["method"] = "fixed_map_block_phase2_adaptive_basis_replay"
        scored["block_index"] = block_index
        scored["basis_selection"] = evidence["basis_selection"]
        block_results.append(scored)

    calibrations = {}
    for method in ("hoeffding", "weighted_rademacher_chernoff"):
        method_blocks = [result["calibrations"][method] for result in block_results]
        margins = [
            block_results[index]["statistic"] - block["threshold"]
            for index, block in enumerate(method_blocks)
        ]
        best_index = int(np.argmax(margins))
        best = method_blocks[best_index]
        calibrations[method] = {
            **best,
            "decision": bool(any(block["decision"] for block in method_blocks)),
            "blocks_passed": int(sum(block["decision"] for block in method_blocks)),
            "best_block": best_index,
            "block_fpr": block_fpr,
        }

    result = {
        "method": "fixed_map_phase2_adaptive_basis_replay",
        "basis_method": "reliability_erasure_elimination_v1",
        "n": n,
        "T": requested_T,
        "r": int(block_results[0]["r"]),
        "num_blocks": num_blocks,
        "fpr": fpr,
        "block_fpr": block_fpr,
        "calibrations": calibrations,
        "blocks": block_results,
    }
    for field in ("prompt_idx", "watermark"):
        if field in record:
            result[field] = record[field]
    return result


def replay_cached_online_map_record(
    artifact: dict,
    record: dict,
    *,
    length: int | None = None,
    false_positive_rate: float | None = None,
    fpr_policy: str = "one_shot",
    prepared_context=None,
) -> dict:
    """Replay one saved Modal record using its compact experiment artifact."""
    for field in ("online_key", "partition"):
        if field not in artifact:
            raise KeyError(f"artifact is missing {field!r}")
    for field in ("tokens", "p_trace"):
        if field not in record:
            raise KeyError(f"record is missing {field!r}")
    replay_length = int(
        artifact.get("T", len(record["p_trace"])) if length is None else length
    )
    if false_positive_rate is None:
        artifact_fpr = artifact.get("target_fpr", artifact.get("fpr"))
        if artifact_fpr is None:
            raise ValueError(
                "false_positive_rate is required when the artifact has no "
                "target_fpr or fpr field"
            )
        fpr = float(artifact_fpr)
    else:
        fpr = float(false_positive_rate)
    evidence = prepare_online_map_evidence(
        artifact["online_key"],
        record["tokens"],
        record["p_trace"],
        artifact["partition"],
        length=replay_length,
        prepared_context=prepared_context,
    )
    result = score_online_map_evidence(
        evidence,
        false_positive_rate=fpr,
        fpr_policy=fpr_policy,
    )
    for field in ("prompt_idx", "watermark"):
        if field in record:
            result[field] = record[field]
    return result
