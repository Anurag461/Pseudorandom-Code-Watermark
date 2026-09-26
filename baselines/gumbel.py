from __future__ import annotations
from dataclasses import replace
from typing import Sequence
import numpy as np
import torch
from .config import CONTEXT_LENGTH
from .textseal import textseal_config
from .scoring import _empty_test, gamma_survival, gamma_threshold, _windows_targets

GUMBEL_KEY = 42


def gumbel_config():
    config = textseal_config(watermark_type="gumbelmax")
    return replace(config, secret_key=GUMBEL_KEY)


def gumbel_generator():
    from textseal.watermarking.generator import GumbelmaxGenerator

    config = gumbel_config()
    generator = GumbelmaxGenerator.__new__(GumbelmaxGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    return generator


def official_gumbel_scores(
    token_ids: Sequence[int], positions: Sequence[int]
) -> np.ndarray:
    from textseal.watermarking.core import prf_uniform

    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    uniforms = prf_uniform(windows, targets, GUMBEL_KEY)
    return (-torch.log1p(-uniforms)).double().cpu().numpy()


def gumbel_gamma_test(scores: Sequence[float], nominal_fpr: float = 0.001) -> dict:
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
