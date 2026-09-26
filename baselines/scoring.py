from __future__ import annotations
import math
from typing import Sequence
import torch
from scipy import special, stats


def deduplicated_positions(
    token_ids: Sequence[int],
    context_length: int = 3,
    *,
    start_position: int | None = None,
) -> list[int]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    start = context_length + 1 if start_position is None else int(start_position)
    if start < context_length:
        raise ValueError("start_position cannot precede a full context")
    seen: set[tuple[int, ...]] = set()
    kept: list[int] = []
    for position in range(start, len(token_ids)):
        key = tuple(
            (int(x) for x in token_ids[position - context_length : position + 1])
        )
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


def _empty_test(calibration_type: str) -> dict:
    return {
        "statistic": 0.0,
        "p_value": 1.0,
        "threshold": float("inf"),
        "decision": False,
        "calibration_type": calibration_type,
        "intermediate": {"status": "insufficient_evidence"},
    }


def empirical_p(reference, stat):
    import numpy as np

    return float(np.searchsorted(reference, stat, side="right") / len(reference))


def _windows_targets(
    token_ids: Sequence[int], positions: Sequence[int], context_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = [int(token) for token in token_ids]
    windows = [tokens[pos - context_length : pos] for pos in positions]
    targets = [tokens[pos] for pos in positions]
    return (
        torch.tensor(windows, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )
