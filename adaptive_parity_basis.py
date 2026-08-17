"""Reliability-adaptive full-rank bases for fixed PRC parity checks.

The selector in this module only consumes the public parity matrix, channel
reliabilities derived from the model probability trace, and the configured
noise rate.  In particular, it never sees the one-time pad, parity signs, or
the observed detection statistic.  The selected rows are obtained solely by
invertible GF(2) row operations, so they remain an independent basis of the
same dual code.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np


DEFAULT_ERASURE_QUANTILES = (
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
)


def bucket_reliability(partition_probabilities) -> np.ndarray:
    """Return ``rho = min(p, 1-p) / max(p, 1-p)`` for every position."""
    probabilities = np.asarray(
        partition_probabilities, dtype=np.float64
    ).reshape(-1)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("partition probabilities must be finite")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("partition probabilities must lie in [0, 1]")
    smaller = np.minimum(probabilities, 1.0 - probabilities)
    larger = np.maximum(probabilities, 1.0 - probabilities)
    return np.divide(
        smaller,
        larger,
        out=np.zeros_like(smaller),
        where=larger > 0.0,
    )


def parity_row_masks(parity_matrix) -> tuple[tuple[int, ...], int]:
    """Pack a binary parity matrix into one Python integer per row."""
    parity = parity_matrix.tocsr(copy=True)
    parity.sum_duplicates()
    rows, columns = parity.shape
    if rows <= 0 or columns <= 0:
        raise ValueError("parity matrix must have positive dimensions")
    masks = []
    for row in range(rows):
        mask = 0
        start, stop = parity.indptr[row], parity.indptr[row + 1]
        for column, value in zip(
            parity.indices[start:stop], parity.data[start:stop]
        ):
            rounded = int(round(float(value)))
            if not math.isclose(float(value), rounded, abs_tol=1e-12):
                raise ValueError("parity matrix entries must be integral")
            if rounded % 2:
                mask ^= 1 << int(column)
        if mask == 0:
            raise ValueError("parity matrix contains an empty GF(2) row")
        masks.append(mask)
    return tuple(masks), int(columns)


def gf2_rank_from_masks(row_masks) -> int:
    """Compute GF(2) row rank for integer-packed rows."""
    pivots: dict[int, int] = {}
    for raw_row in row_masks:
        row = int(raw_row)
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    return len(pivots)


def mask_support(mask: int) -> np.ndarray:
    """Return the increasing set-bit positions in an integer-packed row."""
    positions = []
    remaining = int(mask)
    while remaining:
        lowest = remaining & -remaining
        positions.append(lowest.bit_length() - 1)
        remaining ^= lowest
    return np.asarray(positions, dtype=np.int64)


def _basis_sha256(row_masks, n: int) -> str:
    width = (int(n) + 7) // 8
    header = f"gf2-row-basis:r={len(row_masks)}:n={n}:".encode()
    packed = b"".join(int(row).to_bytes(width, "little") for row in row_masks)
    return hashlib.sha256(header + packed).hexdigest()


def _row_log_utility(
    row_mask: int,
    log_reliabilities: np.ndarray,
    log_noise_attenuation: float,
) -> float:
    degree = int(row_mask).bit_count()
    if degree == 0:
        return float("-inf")
    total = 2.0 * degree * log_noise_attenuation
    remaining = int(row_mask)
    while remaining:
        lowest = remaining & -remaining
        total += float(log_reliabilities[lowest.bit_length() - 1])
        remaining ^= lowest
    return float(total)


def _basis_objective(
    row_masks,
    log_reliabilities: np.ndarray,
    log_noise_attenuation: float,
) -> tuple[float, float]:
    row_scores = np.asarray(
        [
            _row_log_utility(
                row, log_reliabilities, log_noise_attenuation
            )
            for row in row_masks
        ],
        dtype=np.float64,
    )
    finite = row_scores[np.isfinite(row_scores)]
    if finite.size == 0:
        return float("-inf"), 0.0
    maximum = float(np.max(finite))
    log_objective = float(
        maximum + np.log(np.sum(np.exp(finite - maximum)))
    )
    return log_objective, float(math.exp(log_objective))


def _candidate_summary(
    row_masks,
    *,
    quantile: float,
    erased_count: int,
    erased_rank: int,
    erased_mask: int,
    reliability_cutoff: float | None,
    log_reliabilities: np.ndarray,
    log_noise_attenuation: float,
) -> dict:
    degrees = np.asarray(
        [int(row).bit_count() for row in row_masks], dtype=np.int64
    )
    log_objective, objective = _basis_objective(
        row_masks, log_reliabilities, log_noise_attenuation
    )
    return {
        "erasure_quantile": float(quantile),
        "erased_columns": int(erased_count),
        "erased_column_rank": int(erased_rank),
        "erasure_free_rows": int(
            sum((int(row) & erased_mask) == 0 for row in row_masks)
        ),
        "reliability_cutoff": reliability_cutoff,
        "log_predicted_J": log_objective,
        "predicted_J": objective,
        "degree_minimum": int(np.min(degrees)),
        "degree_mean": float(np.mean(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_maximum": int(np.max(degrees)),
    }


def select_reliability_adaptive_basis(
    parity_matrix,
    reliabilities,
    noise_rate: float,
    *,
    erasure_quantiles=DEFAULT_ERASURE_QUANTILES,
) -> dict:
    """Select a full-rank dual basis by eliminating weak columns.

    Candidate erasure sets are nested from least to most reliable.  GF(2)
    elimination concentrates the erased columns into as few basis rows as
    their column rank permits.  Among the requested quantiles, the function
    returns the basis maximizing

    ``sum_a (1 - 2 eta) ** (2 |v_a|) * prod_{i in v_a} rho_i``.

    The zero-erasure identity basis is part of the default grid, so the
    predicted objective cannot be worse than the original basis.
    """
    original_rows, n = parity_row_masks(parity_matrix)
    r = len(original_rows)
    if gf2_rank_from_masks(original_rows) != r:
        raise ValueError(
            "adaptive basis requires a full-row-rank parity matrix"
        )
    reliability = np.asarray(reliabilities, dtype=np.float64).reshape(-1)
    if reliability.size != n:
        raise ValueError(
            f"reliabilities have length {reliability.size}, expected {n}"
        )
    if not np.all(np.isfinite(reliability)):
        raise ValueError("reliabilities must be finite")
    if np.any(reliability < 0.0) or np.any(reliability > 1.0):
        raise ValueError("reliabilities must lie in [0, 1]")
    eta = float(noise_rate)
    if not 0.0 <= eta <= 0.5:
        raise ValueError("noise_rate must lie in [0, 0.5]")

    requested = [float(value) for value in erasure_quantiles]
    if not requested:
        raise ValueError("erasure_quantiles must not be empty")
    if any(not 0.0 <= value <= 1.0 for value in requested):
        raise ValueError("erasure quantiles must lie in [0, 1]")
    count_to_quantile = {}
    for quantile in requested:
        count = min(n, int(math.floor(quantile * n + 1e-12)))
        count_to_quantile.setdefault(count, quantile)
    candidates = sorted(count_to_quantile.items())

    with np.errstate(divide="ignore"):
        log_reliabilities = np.log(reliability)
    attenuation = 1.0 - 2.0 * eta
    log_attenuation = (
        float(math.log(attenuation))
        if attenuation > 0.0
        else float("-inf")
    )
    column_order = np.lexsort((np.arange(n, dtype=np.int64), reliability))
    rows = list(original_rows)
    pivot_count = 0
    processed_count = 0
    erased_mask = 0
    summaries = []
    best_summary = None
    best_rows = None

    for erased_count, quantile in candidates:
        for column_value in column_order[processed_count:erased_count]:
            column = int(column_value)
            column_bit = 1 << column
            erased_mask |= column_bit
            pivot_row = next(
                (
                    row
                    for row in range(pivot_count, r)
                    if rows[row] & column_bit
                ),
                None,
            )
            if pivot_row is None:
                continue
            rows[pivot_count], rows[pivot_row] = (
                rows[pivot_row],
                rows[pivot_count],
            )
            pivot_mask = rows[pivot_count]
            for row in range(r):
                if row != pivot_count and rows[row] & column_bit:
                    rows[row] ^= pivot_mask
            pivot_count += 1
        processed_count = erased_count
        cutoff = (
            float(reliability[int(column_order[erased_count - 1])])
            if erased_count
            else None
        )
        summary = _candidate_summary(
            rows,
            quantile=quantile,
            erased_count=erased_count,
            erased_rank=pivot_count,
            erased_mask=erased_mask,
            reliability_cutoff=cutoff,
            log_reliabilities=log_reliabilities,
            log_noise_attenuation=log_attenuation,
        )
        summaries.append(summary)
        if (
            best_summary is None
            or summary["log_predicted_J"]
            > best_summary["log_predicted_J"] + 1e-12
        ):
            best_summary = summary
            best_rows = tuple(rows)

    if best_rows is None or best_summary is None:
        raise AssertionError("adaptive basis candidate search produced no basis")
    if gf2_rank_from_masks(best_rows) != r:
        raise AssertionError("adaptive row operations did not preserve rank")
    selected = {
        **best_summary,
        "basis_rank": r,
        "basis_rows": r,
        "basis_columns": n,
        "basis_sha256": _basis_sha256(best_rows, n),
        "selection_uses": [
            "parity_matrix",
            "partition_probability_reliability",
            "noise_rate",
        ],
        "selection_excludes": [
            "one_time_pad",
            "parity_signs",
            "token_bucket_observations",
            "observed_detection_statistic",
        ],
        "candidate_summaries": summaries,
    }
    return {
        "row_masks": best_rows,
        "supports": tuple(mask_support(row) for row in best_rows),
        "selection": selected,
    }
