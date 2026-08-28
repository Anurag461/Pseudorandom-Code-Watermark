"""Common exact-prefix scoring and token-level quality/diversity metrics."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable, Sequence

import numpy as np
from scipy import special, stats


def deduplicated_positions(
    token_ids: Sequence[int],
    context_length: int = 3,
    *,
    start_position: int | None = None,
) -> list[int]:
    """Return positions for unique ``(context, token)`` tuples.

    TextSeal commit c60d0d1 starts at ``ngram + 1`` rather than the first
    mathematically eligible position ``ngram``. The default intentionally
    follows that released evaluation path and is recorded as a discrepancy.
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    start = context_length + 1 if start_position is None else int(start_position)
    if start < context_length:
        raise ValueError("start_position cannot precede a full context")
    seen: set[tuple[int, ...]] = set()
    kept: list[int] = []
    for position in range(start, len(token_ids)):
        key = tuple(int(x) for x in token_ids[position - context_length : position + 1])
        if key not in seen:
            seen.add(key)
            kept.append(position)
    return kept


def gamma_survival(statistic: float, shape: float, scale: float = 1.0) -> float:
    if not (math.isfinite(statistic) and math.isfinite(shape) and math.isfinite(scale)):
        raise ValueError("Gamma inputs must be finite")
    if shape <= 0 or scale <= 0:
        return 1.0
    return float(max(special.gammaincc(shape, statistic / scale), 1e-300))


def gamma_threshold(shape: float, scale: float, nominal_fpr: float) -> float:
    if not 0.0 < nominal_fpr < 1.0:
        raise ValueError("nominal_fpr must be in (0, 1)")
    if shape <= 0 or scale <= 0:
        return float("inf")
    return float(stats.gamma.ppf(1.0 - nominal_fpr, a=shape, scale=scale))


def gumbel_gamma_test(scores: Sequence[float], nominal_fpr: float = 1e-3) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return _empty_test("exact Gamma test")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Gumbel scores must be finite and nonnegative")
    statistic = float(values.sum())
    shape = float(values.size)
    scale = 1.0
    p_value = gamma_survival(statistic, shape, scale)
    threshold = gamma_threshold(shape, scale, nominal_fpr)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "threshold": threshold,
        "decision": bool(p_value < nominal_fpr),
        "calibration_type": "exact Gamma test",
        "intermediate": {"gamma_shape": shape, "gamma_scale": scale},
    }


def textseal_gamma_test(
    fused_scores: Sequence[float],
    entropies: Sequence[float],
    alpha: float = 0.1,
    nominal_fpr: float = 1e-3,
) -> dict:
    scores = np.asarray(fused_scores, dtype=np.float64)
    entropy = np.asarray(entropies, dtype=np.float64)
    if scores.size == 0:
        return _empty_test("moment-matched Gamma approximation")
    if scores.shape != entropy.shape:
        raise ValueError("fused scores and entropies must align")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(entropy)):
        raise ValueError("TextSeal inputs must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    entropy_min = float(entropy.min())
    entropy_max = float(entropy.max())
    if entropy_max - entropy_min < 1e-6:
        # This is the pinned code's fallback range.
        entropy_min, entropy_max = 0.0, 5.0
    ratio = np.clip((entropy - entropy_min) / (entropy_max - entropy_min), 0.0, 1.0)
    weights = 0.1 + 0.9 * ratio
    statistic = float(np.sum(weights * scores))
    routing_variance = float(alpha**2 + (1.0 - alpha) ** 2)
    mean = float(weights.sum())
    variance = float(np.sum(weights**2) * routing_variance)
    shape = mean**2 / variance
    scale = variance / mean
    p_value = gamma_survival(statistic, shape, scale)
    threshold = gamma_threshold(shape, scale, nominal_fpr)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "threshold": threshold,
        "decision": bool(p_value < nominal_fpr),
        "calibration_type": "moment-matched Gamma approximation",
        "intermediate": {
            "gamma_shape": shape,
            "gamma_scale": scale,
            "routing_variance": routing_variance,
            "entropy_min": entropy_min,
            "entropy_max": entropy_max,
            "weight_sum": mean,
            "weight_squared_sum": float(np.sum(weights**2)),
            "unweighted_statistic": float(scores.sum()),
            "unweighted_p_value": gamma_survival(
                float(scores.sum()), scores.size / routing_variance, routing_variance
            ),
        },
    }


def synthid_normal_test(
    g_values: np.ndarray,
    *,
    nominal_fpr: float = 1e-3,
    weights: Sequence[float] | None = None,
) -> dict:
    values = np.asarray(g_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("g_values must have shape (samples, depth)")
    samples, depth = values.shape
    if samples == 0:
        return _empty_test("normal approximation")
    if not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError("SynthID g-values must be binary")
    if weights is None:
        layer_weights = np.linspace(10.0, 1.0, depth)
    else:
        layer_weights = np.asarray(weights, dtype=np.float64)
        if layer_weights.shape != (depth,):
            raise ValueError("weights must have one value per layer")
    layer_weights = layer_weights * depth / layer_weights.sum()
    per_token = values @ layer_weights
    statistic = float(per_token.sum())
    null_mean_per_token = 0.5 * depth
    null_variance_per_token = 0.25 * float(np.sum(layer_weights**2))
    z_score = float(
        (statistic - samples * null_mean_per_token)
        / math.sqrt(samples * null_variance_per_token)
    )
    p_value = float(max(special.ndtr(-z_score), 1e-300))
    z_threshold = float(stats.norm.ppf(1.0 - nominal_fpr))
    score_threshold = float(
        samples * null_mean_per_token
        + z_threshold * math.sqrt(samples * null_variance_per_token)
    )
    return {
        "statistic": statistic,
        "p_value": p_value,
        "threshold": score_threshold,
        "decision": bool(p_value < nominal_fpr),
        "calibration_type": "normal approximation",
        "intermediate": {
            "z_score": z_score,
            "z_threshold": z_threshold,
            "null_mean_per_token": null_mean_per_token,
            "null_variance_per_token": null_variance_per_token,
            "layer_weights": layer_weights.tolist(),
            "g_value_sum_by_depth": values.sum(axis=0).tolist(),
        },
    }


def prc_hoeffding_test(
    statistic: float,
    variance_proxy: float,
    nominal_fpr: float = 1e-3,
) -> dict:
    statistic = float(statistic)
    variance_proxy = float(variance_proxy)
    if variance_proxy <= 0.0:
        return _empty_test("Hoeffding p-value upper bound")
    p_upper = 1.0 if statistic <= 0.0 else float(
        min(1.0, math.exp(-(statistic**2) / (2.0 * variance_proxy)))
    )
    threshold = float(math.sqrt(2.0 * variance_proxy * math.log(1.0 / nominal_fpr)))
    return {
        "statistic": statistic,
        "p_value": p_upper,
        "threshold": threshold,
        "decision": bool(p_upper < nominal_fpr),
        "calibration_type": "Hoeffding p-value upper bound",
        "intermediate": {"variance_proxy": variance_proxy},
    }


def _empty_test(calibration_type: str) -> dict:
    return {
        "statistic": 0.0,
        "p_value": 1.0,
        "threshold": float("inf"),
        "decision": False,
        "calibration_type": calibration_type,
        "intermediate": {"status": "insufficient_evidence"},
    }


def ngram_repetition_rate(token_ids: Sequence[int], n: int = 4) -> float:
    """Fraction of token n-grams beyond their first occurrence."""
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    if not grams:
        return 0.0
    return float(1.0 - len(set(grams)) / len(grams))


def distinct_n(token_ids: Sequence[int], n: int) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    return float(len(set(grams)) / len(grams)) if grams else 0.0


def quality_metrics(token_ids: Sequence[int], base_token_logprobs: Sequence[float]) -> dict:
    ids = [int(x) for x in token_ids]
    logprobs = np.asarray(base_token_logprobs, dtype=np.float64)
    if len(ids) != logprobs.size:
        raise ValueError("one base-model log-probability is required per token")
    if not np.all(np.isfinite(logprobs)):
        raise ValueError("base-model log-probabilities must be finite")
    mean_nll = float(-logprobs.mean()) if logprobs.size else 0.0
    return {
        "base_model_nll": mean_nll,
        "base_model_perplexity": float(math.exp(mean_nll)),
        "output_length": len(ids),
        "repetition_rate": ngram_repetition_rate(ids, 4),
        "repetition_metric": "repeated token 4-gram fraction: 1 - unique_4grams/total_4grams",
        "distinct_2": distinct_n(ids, 2),
        "distinct_3": distinct_n(ids, 3),
    }


def pairwise_token_agreement(sequences: Sequence[Sequence[int]]) -> float | None:
    agreements: list[float] = []
    for left_index in range(len(sequences)):
        for right_index in range(left_index + 1, len(sequences)):
            left, right = sequences[left_index], sequences[right_index]
            length = min(len(left), len(right))
            if length:
                agreements.append(sum(left[i] == right[i] for i in range(length)) / length)
    return float(np.mean(agreements)) if agreements else None


def self_bleu_token_ids(sequences: Sequence[Sequence[int]]) -> float | None:
    """Mean SacreBLEU sentence BLEU, treating token IDs as pre-tokenized words."""
    if len(sequences) < 2:
        return None
    import sacrebleu

    scores = []
    rendered = [" ".join(str(int(token)) for token in sequence) for sequence in sequences]
    for index, hypothesis in enumerate(rendered):
        references = [text for other, text in enumerate(rendered) if other != index]
        scores.append(
            sacrebleu.sentence_bleu(
                hypothesis, references, tokenize="none", smooth_method="exp"
            ).score
            / 100.0
        )
    return float(np.mean(scores))
