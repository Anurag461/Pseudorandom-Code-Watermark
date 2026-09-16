"""Aggregate full-token versus fixed-binary-projection entropy traces.

The diagnostic compares full vocabulary entropy H(token) with the entropy of
the existing fixed vocabulary split H(bucket).  Both are measured in bits, so
their difference is the within-bucket conditional entropy H(token | bucket).
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from itertools import combinations

import numpy as np


FULL_ENTROPY_EDGES = np.linspace(0.0, 18.0, 901)
PROJECTED_ENTROPY_EDGES = np.linspace(0.0, 1.0, 501)
GAP_EDGES = np.linspace(0.0, 18.0, 901)
HEATMAP_FULL_EDGES = np.linspace(0.0, 18.0, 73)
HEATMAP_PROJECTED_EDGES = np.linspace(0.0, 1.0, 41)
METRICS = ("full_entropy_bits", "projected_entropy_bits", "gap_bits")


def binary_entropy_bits(probability) -> np.ndarray:
    """Return binary entropy in bits, including exact zero at p in {0, 1}."""
    p = np.asarray(probability, dtype=np.float64)
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("partition probabilities must be finite and in [0, 1]")
    clipped = np.clip(p, np.finfo(np.float64).tiny, 1.0)
    complement = np.clip(1.0 - p, np.finfo(np.float64).tiny, 1.0)
    entropy = -(p * np.log2(clipped) + (1.0 - p) * np.log2(complement))
    entropy[(p == 0.0) | (p == 1.0)] = 0.0
    return entropy


def _as_trace_matrix(name: str, values, trace_t: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != int(trace_t):
        raise ValueError(
            f"{name} must have shape (batch, {trace_t}); got {array.shape}"
        )
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def summarize_trace_batch(
    *,
    model_size: str,
    prompt_indices: list[int],
    full_entropy_bits,
    partition_probability,
    trace_t: int,
    partition_sha256: str,
    source_cache_t: int,
    source_kind: str,
    reference_partition_probability=None,
) -> dict:
    """Build a compact, mergeable summary for one prompt shard."""
    indices = [int(index) for index in prompt_indices]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("prompt indices must be nonempty and unique")
    full = _as_trace_matrix("full_entropy_bits", full_entropy_bits, trace_t)
    p1 = _as_trace_matrix("partition_probability", partition_probability, trace_t)
    if full.shape[0] != len(indices) or p1.shape != full.shape:
        raise ValueError("trace batch size does not match prompt indices")
    if np.any(full < -1e-6):
        raise ValueError("full-token entropy contains materially negative values")

    projected = binary_entropy_bits(p1)
    gap = full - projected
    arrays = {
        "full_entropy_bits": full,
        "projected_entropy_bits": projected,
        "gap_bits": gap,
    }
    flat_full = full.reshape(-1)
    flat_projected = projected.reshape(-1)
    flat_gap = gap.reshape(-1)
    token_count = int(flat_full.size)

    metric_moments = {}
    position_moments = {}
    histograms = {}
    edges_by_metric = {
        "full_entropy_bits": FULL_ENTROPY_EDGES,
        "projected_entropy_bits": PROJECTED_ENTROPY_EDGES,
        "gap_bits": GAP_EDGES,
    }
    for name, array in arrays.items():
        flat = array.reshape(-1)
        metric_moments[name] = {
            "sum": float(flat.sum(dtype=np.float64)),
            "sumsq": float(np.square(flat).sum(dtype=np.float64)),
            "min": float(flat.min()),
            "max": float(flat.max()),
        }
        position_moments[name] = {
            "sum": array.sum(axis=0, dtype=np.float64).tolist(),
            "sumsq": np.square(array).sum(axis=0, dtype=np.float64).tolist(),
        }
        histograms[name] = np.histogram(flat, bins=edges_by_metric[name])[0].tolist()

    heatmap = np.histogram2d(
        flat_full,
        flat_projected,
        bins=(HEATMAP_FULL_EDGES, HEATMAP_PROJECTED_EDGES),
    )[0]
    prompt_summaries = []
    for row, index in enumerate(indices):
        prompt_summaries.append({
            "prompt_idx": int(index),
            **{
                f"mean_{name}": float(arrays[name][row].mean())
                for name in METRICS
            },
            **{
                f"median_{name}": float(np.median(arrays[name][row]))
                for name in METRICS
            },
        })

    reference_error = None
    if reference_partition_probability is not None:
        reference = _as_trace_matrix(
            "reference_partition_probability",
            reference_partition_probability,
            trace_t,
        )
        if reference.shape != p1.shape:
            raise ValueError("reference partition trace has the wrong shape")
        absolute = np.abs(reference - p1)
        reference_projected = binary_entropy_bits(reference)
        projected_difference = reference_projected - projected
        reference_error = {
            "count": int(absolute.size),
            "sum_abs": float(absolute.sum(dtype=np.float64)),
            "sumsq": float(np.square(absolute).sum(dtype=np.float64)),
            "max_abs": float(absolute.max()),
            "count_gt_1e_4": int(np.count_nonzero(absolute > 1e-4)),
            "count_gt_1e_3": int(np.count_nonzero(absolute > 1e-3)),
            "sum_projected_entropy_difference_bits": float(
                projected_difference.sum(dtype=np.float64)
            ),
            "sum_abs_projected_entropy_difference_bits": float(
                np.abs(projected_difference).sum(dtype=np.float64)
            ),
            "max_abs_projected_entropy_difference_bits": float(
                np.abs(projected_difference).max()
            ),
        }

    return {
        "model_size": str(model_size),
        "trace_t": int(trace_t),
        "source_cache_t": int(source_cache_t),
        "source_kind": str(source_kind),
        "partition_sha256": str(partition_sha256),
        "prompt_indices": indices,
        "token_count": token_count,
        "metric_moments": metric_moments,
        "cross_moment_full_projected": float(
            np.multiply(flat_full, flat_projected).sum(dtype=np.float64)
        ),
        "position_moments": position_moments,
        "histograms": histograms,
        "heatmap_full_by_projected": heatmap.astype(np.int64).tolist(),
        "threshold_counts": {
            "full_lt_1_bit": int(np.count_nonzero(flat_full < 1.0)),
            "projected_lt_0p1_bit": int(np.count_nonzero(flat_projected < 0.1)),
            "projected_lt_0p1_and_full_ge_1": int(np.count_nonzero(
                (flat_projected < 0.1) & (flat_full >= 1.0)
            )),
            "projected_lt_0p1_and_full_ge_2": int(np.count_nonzero(
                (flat_projected < 0.1) & (flat_full >= 2.0)
            )),
            "gap_negative_below_1e_4": int(np.count_nonzero(flat_gap < -1e-4)),
        },
        "prompt_summaries": prompt_summaries,
        "reference_partition_error": reference_error,
    }


def _histogram_quantile(counts, edges, quantile: float) -> float:
    counts = np.asarray(counts, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.float64)
    total = int(counts.sum())
    if total <= 0:
        return float("nan")
    target = min(max(float(quantile), 0.0), 1.0) * max(total - 1, 0)
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target + 1, side="left"))
    index = min(index, len(counts) - 1)
    return float((edges[index] + edges[index + 1]) / 2.0)


def _merge_reference_errors(shards: list[dict]) -> dict | None:
    errors = [item["reference_partition_error"] for item in shards]
    errors = [item for item in errors if item is not None]
    if not errors:
        return None
    count = sum(int(item["count"]) for item in errors)
    return {
        "count": count,
        "mean_abs": sum(float(item["sum_abs"]) for item in errors) / count,
        "rmse": math.sqrt(
            sum(float(item["sumsq"]) for item in errors) / count
        ),
        "max_abs": max(float(item["max_abs"]) for item in errors),
        "count_gt_1e_4": sum(int(item["count_gt_1e_4"]) for item in errors),
        "count_gt_1e_3": sum(int(item["count_gt_1e_3"]) for item in errors),
        "mean_projected_entropy_difference_bits": sum(
            float(item["sum_projected_entropy_difference_bits"])
            for item in errors
        ) / count,
        "mean_abs_projected_entropy_difference_bits": sum(
            float(item["sum_abs_projected_entropy_difference_bits"])
            for item in errors
        ) / count,
        "max_abs_projected_entropy_difference_bits": max(
            float(item["max_abs_projected_entropy_difference_bits"])
            for item in errors
        ),
    }


def merge_model_shards(shards: list[dict], expected_prompts: int) -> dict:
    """Merge remote shard summaries into one exact model-level result."""
    if not shards:
        raise ValueError("at least one shard is required")
    scalar_fields = (
        "model_size",
        "trace_t",
        "source_cache_t",
        "source_kind",
        "partition_sha256",
    )
    for field in scalar_fields:
        if len({str(item[field]) for item in shards}) != 1:
            raise ValueError(f"shards disagree on {field}")
    prompt_indices = [
        int(index) for item in shards for index in item["prompt_indices"]
    ]
    if len(prompt_indices) != len(set(prompt_indices)):
        raise ValueError("shards contain duplicate prompt indices")
    expected = list(range(int(expected_prompts)))
    if sorted(prompt_indices) != expected:
        raise ValueError("shards do not cover the expected canonical prompts")

    token_count = sum(int(item["token_count"]) for item in shards)
    model_size = str(shards[0]["model_size"])
    metric_statistics = {}
    position_statistics = {}
    histograms = {}
    edges_by_metric = {
        "full_entropy_bits": FULL_ENTROPY_EDGES,
        "projected_entropy_bits": PROJECTED_ENTROPY_EDGES,
        "gap_bits": GAP_EDGES,
    }
    for metric in METRICS:
        total = sum(float(item["metric_moments"][metric]["sum"]) for item in shards)
        sumsq = sum(
            float(item["metric_moments"][metric]["sumsq"]) for item in shards
        )
        mean = total / token_count
        variance = max(sumsq / token_count - mean * mean, 0.0)
        histogram = sum(
            np.asarray(item["histograms"][metric], dtype=np.int64)
            for item in shards
        )
        histograms[metric] = histogram.tolist()
        metric_statistics[metric] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(
                float(item["metric_moments"][metric]["min"]) for item in shards
            ),
            "max": max(
                float(item["metric_moments"][metric]["max"]) for item in shards
            ),
            "q10_approx": _histogram_quantile(
                histogram, edges_by_metric[metric], 0.10
            ),
            "median_approx": _histogram_quantile(
                histogram, edges_by_metric[metric], 0.50
            ),
            "q90_approx": _histogram_quantile(
                histogram, edges_by_metric[metric], 0.90
            ),
        }
        position_sum = sum(
            np.asarray(item["position_moments"][metric]["sum"], dtype=np.float64)
            for item in shards
        )
        position_sumsq = sum(
            np.asarray(item["position_moments"][metric]["sumsq"], dtype=np.float64)
            for item in shards
        )
        count = int(expected_prompts)
        position_mean = position_sum / count
        position_variance = np.maximum(
            position_sumsq / count - np.square(position_mean), 0.0
        )
        position_statistics[metric] = {
            "mean": position_mean.tolist(),
            "std": np.sqrt(position_variance).tolist(),
        }

    full = metric_statistics["full_entropy_bits"]
    projected = metric_statistics["projected_entropy_bits"]
    cross = sum(float(item["cross_moment_full_projected"]) for item in shards)
    covariance = cross / token_count - full["mean"] * projected["mean"]
    denominator = full["std"] * projected["std"]
    correlation = covariance / denominator if denominator else float("nan")
    heatmap = sum(
        np.asarray(item["heatmap_full_by_projected"], dtype=np.int64)
        for item in shards
    )
    threshold_counts = {
        name: sum(int(item["threshold_counts"][name]) for item in shards)
        for name in shards[0]["threshold_counts"]
    }
    prompt_summaries = sorted(
        [row for item in shards for row in item["prompt_summaries"]],
        key=lambda row: int(row["prompt_idx"]),
    )
    return {
        "model_size": model_size,
        "trace_t": int(shards[0]["trace_t"]),
        "source_cache_t": int(shards[0]["source_cache_t"]),
        "source_kind": str(shards[0]["source_kind"]),
        "partition_sha256": str(shards[0]["partition_sha256"]),
        "prompt_count": int(expected_prompts),
        "token_count": token_count,
        "metric_statistics": metric_statistics,
        "correlation_full_vs_projected": correlation,
        "projection_retention_ratio_of_means": (
            projected["mean"] / full["mean"] if full["mean"] else float("nan")
        ),
        "threshold_counts": threshold_counts,
        "threshold_rates": {
            name: int(value) / token_count for name, value in threshold_counts.items()
        },
        "position_statistics": position_statistics,
        "histogram_edges": {
            "full_entropy_bits": FULL_ENTROPY_EDGES.tolist(),
            "projected_entropy_bits": PROJECTED_ENTROPY_EDGES.tolist(),
            "gap_bits": GAP_EDGES.tolist(),
        },
        "histograms": histograms,
        "heatmap_full_edges": HEATMAP_FULL_EDGES.tolist(),
        "heatmap_projected_edges": HEATMAP_PROJECTED_EDGES.tolist(),
        "heatmap_full_by_projected": heatmap.tolist(),
        "prompt_summaries": prompt_summaries,
        "reference_partition_error": _merge_reference_errors(shards),
    }


def paired_model_comparisons(models: dict[str, dict]) -> list[dict]:
    """Compare per-prompt means using paired normal-approximation intervals."""
    results = []
    for left_name, right_name in combinations(models, 2):
        left_rows = models[left_name]["prompt_summaries"]
        right_rows = models[right_name]["prompt_summaries"]
        if [row["prompt_idx"] for row in left_rows] != [
            row["prompt_idx"] for row in right_rows
        ]:
            raise ValueError("paired model comparison requires aligned prompts")
        metrics = {}
        for metric in METRICS:
            differences = np.asarray([
                float(left[f"mean_{metric}"]) - float(right[f"mean_{metric}"])
                for left, right in zip(left_rows, right_rows)
            ])
            mean = float(differences.mean())
            standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences)))
            metrics[metric] = {
                "left_minus_right_mean": mean,
                "paired_standard_error": standard_error,
                "normal_95ci": [
                    mean - 1.96 * standard_error,
                    mean + 1.96 * standard_error,
                ],
            }
        results.append({
            "left_model": left_name,
            "right_model": right_name,
            "prompt_count": len(left_rows),
            "metrics": metrics,
        })
    return results


def build_comparison_payload(
    models: dict[str, dict], execution: dict, trace_t: int, prompt_count: int
) -> dict:
    partition_hashes = {item["partition_sha256"] for item in models.values()}
    if len(partition_hashes) != 1:
        raise ValueError("models do not use the same fixed vocabulary partition")
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "full_token_entropy_minus_fixed_binary_projection_entropy",
        "entropy_unit": "bits",
        "trace_t": int(trace_t),
        "prompt_count": int(prompt_count),
        "partition_sha256": next(iter(partition_hashes)),
        "interpretation": (
            "gap_bits = H(token) - H(bucket) = H(token | bucket) for the "
            "deterministic fixed binary vocabulary projection"
        ),
        "models": models,
        "paired_model_comparisons": paired_model_comparisons(models),
        "execution": execution,
    }


def write_comparison_outputs(payload: dict, output_dir: str) -> dict:
    """Write JSON plus compact model, prompt, and position CSV tables."""
    os.makedirs(output_dir, exist_ok=True)
    stem = (
        f"entropy_projection_T{payload['trace_t']}_"
        f"prompts{payload['prompt_count']}_0p6b_8b_14b"
    )
    paths = {
        "json": os.path.join(output_dir, f"{stem}.json"),
        "summary_csv": os.path.join(output_dir, f"{stem}_summary.csv"),
        "prompt_csv": os.path.join(output_dir, f"{stem}_prompt.csv"),
        "position_csv": os.path.join(output_dir, f"{stem}_position.csv"),
    }
    with open(paths["json"], "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    summary_fields = [
        "model_size", "prompt_count", "trace_t", "token_count", "source_cache_t",
        "mean_full_entropy_bits", "mean_projected_entropy_bits", "mean_gap_bits",
        "median_full_entropy_bits_approx", "median_projected_entropy_bits_approx",
        "median_gap_bits_approx", "projection_retention_ratio_of_means",
        "correlation_full_vs_projected", "rate_full_lt_1_bit",
        "rate_projected_lt_0p1_bit", "rate_mapping_collapse_full_ge_1",
    ]
    with open(paths["summary_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for model_size, model in payload["models"].items():
            stats = model["metric_statistics"]
            rates = model["threshold_rates"]
            writer.writerow({
                "model_size": model_size,
                "prompt_count": model["prompt_count"],
                "trace_t": model["trace_t"],
                "token_count": model["token_count"],
                "source_cache_t": model["source_cache_t"],
                "mean_full_entropy_bits": stats["full_entropy_bits"]["mean"],
                "mean_projected_entropy_bits": stats["projected_entropy_bits"]["mean"],
                "mean_gap_bits": stats["gap_bits"]["mean"],
                "median_full_entropy_bits_approx": stats["full_entropy_bits"]["median_approx"],
                "median_projected_entropy_bits_approx": stats["projected_entropy_bits"]["median_approx"],
                "median_gap_bits_approx": stats["gap_bits"]["median_approx"],
                "projection_retention_ratio_of_means": model[
                    "projection_retention_ratio_of_means"
                ],
                "correlation_full_vs_projected": model[
                    "correlation_full_vs_projected"
                ],
                "rate_full_lt_1_bit": rates["full_lt_1_bit"],
                "rate_projected_lt_0p1_bit": rates["projected_lt_0p1_bit"],
                "rate_mapping_collapse_full_ge_1": rates[
                    "projected_lt_0p1_and_full_ge_1"
                ],
            })

    prompt_fields = [
        "model_size", "prompt_idx",
        *[f"mean_{metric}" for metric in METRICS],
        *[f"median_{metric}" for metric in METRICS],
    ]
    with open(paths["prompt_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=prompt_fields)
        writer.writeheader()
        for model_size, model in payload["models"].items():
            for row in model["prompt_summaries"]:
                writer.writerow({"model_size": model_size, **row})

    position_fields = [
        "model_size", "position",
        *[f"mean_{metric}" for metric in METRICS],
        *[f"std_{metric}" for metric in METRICS],
    ]
    with open(paths["position_csv"], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=position_fields)
        writer.writeheader()
        for model_size, model in payload["models"].items():
            position = model["position_statistics"]
            for index in range(model["trace_t"]):
                writer.writerow({
                    "model_size": model_size,
                    "position": index + 1,
                    **{
                        f"mean_{metric}": position[metric]["mean"][index]
                        for metric in METRICS
                    },
                    **{
                        f"std_{metric}": position[metric]["std"][index]
                        for metric in METRICS
                    },
                })
    return paths
