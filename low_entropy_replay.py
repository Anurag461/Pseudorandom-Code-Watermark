"""Cache-only replay for low-entropy detector interventions.

Phase 0 reconstructs the existing online MAP statistic from saved token and
partition-probability traces.  Phase 1 scores the identical check evidence
with both the legacy Hoeffding threshold and the optimized conditional
weighted-Rademacher Chernoff threshold.

This module deliberately contains no Modal app and no model imports.  It can
be imported by a short-lived remote cache reader without affecting generation
jobs, or run against locally available ``torch.save`` records.
"""
from __future__ import annotations

import math

import numpy as np

from detectors import (
    prepare_online_map_prefix_context,
    prepare_online_map_prefix_trace,
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
