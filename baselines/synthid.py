from __future__ import annotations
import math
from typing import Sequence
import numpy as np
import torch
from scipy import special, stats
from .config import CONTEXT_LENGTH, TEMPERATURE, EOS
from .scoring import _empty_test

SYNTHID_KEYS = (654, 400, 836, 123, 340, 443, 597, 160, 57, 29)
SYNTHID_CONTEXT_HISTORY_SIZE = 1024

SYNTHID = dict(
    ngram_len=5,
    keys=[
        654,
        400,
        836,
        123,
        340,
        443,
        597,
        160,
        57,
        29,
        590,
        639,
        13,
        715,
        468,
        990,
        966,
        226,
        324,
        585,
        118,
        504,
        421,
        521,
        129,
        669,
        732,
        225,
        90,
        960,
    ],
    sampling_table_size=65536,
    sampling_table_seed=0,
    context_history_size=1024,
)


def _synthid_keys(keys):
    keys = tuple(keys)
    if (
        not keys
        or len(set(keys)) != len(keys)
        or any((type(key) is not int or not 0 <= key < 2**31 for key in keys))
    ):
        raise ValueError("SynthID keys must be distinct nonnegative 31-bit integers")
    return keys


def synthid_processor(
    device: torch.device | str, *, keys: Sequence[int] = SYNTHID_KEYS
):
    from synthid_text.logits_processing import SynthIDLogitsProcessor

    keys = _synthid_keys(keys)
    target = torch.device(device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    processor = SynthIDLogitsProcessor(
        ngram_len=CONTEXT_LENGTH + 1,
        keys=list(keys),
        context_history_size=SYNTHID_CONTEXT_HISTORY_SIZE,
        temperature=float(TEMPERATURE),
        top_k=2,
        device=torch.device("cpu"),
        skip_first_ngram_calls=False,
        apply_top_k=False,
        num_leaves=2,
    )
    processor.keys = processor.keys.to(target)
    processor.device = target
    processor.state = None
    return processor


def official_synthid_g_values(
    token_ids: Sequence[int],
    positions: Sequence[int],
    *,
    device: str = "cpu",
    keys: Sequence[int] = SYNTHID_KEYS,
) -> np.ndarray:
    keys = _synthid_keys(keys)
    if not positions:
        return np.empty((0, len(keys)), dtype=np.int64)
    processor = synthid_processor(device, keys=keys)
    ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    values = processor.compute_g_values(ids)[0]
    rows = torch.tensor(
        [int(position) - CONTEXT_LENGTH for position in positions],
        dtype=torch.long,
        device=values.device,
    )
    return values.index_select(0, rows).long().cpu().numpy()


def attack_processor(device):
    from transformers.generation.logits_process import (
        SynthIDTextWatermarkLogitsProcessor,
    )

    processor = SynthIDTextWatermarkLogitsProcessor(
        **SYNTHID, device=torch_device("cpu")
    )
    processor.keys = processor.keys.to(device)
    processor.sampling_table = processor.sampling_table.to(device)
    processor.device = device
    return processor


def torch_device(name):
    import torch

    return torch.device(name)


def synthid_mean_score(processor, tokens):
    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()
    mask = processor.compute_context_repetition_mask(ids)
    mask = (
        mask * processor.compute_eos_token_mask(ids, EOS)[:, processor.ngram_len - 1 :]
    )
    count = mask.sum() * g.shape[-1]
    return float((g * mask[..., None]).sum() / count) if count else 0.5


def synthid_pvalue(processor, tokens):
    import math
    from scipy.stats import norm

    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()
    mask = processor.compute_context_repetition_mask(ids)
    mask = (
        mask * processor.compute_eos_token_mask(ids, EOS)[:, SYNTHID["ngram_len"] - 1 :]
    )
    count = float(mask.sum()) * g.shape[-1]
    if not count:
        return (1.0, 0.5)
    mean = float((g * mask[..., None]).sum()) / count
    return (float(norm.sf((mean - 0.5) / math.sqrt(0.25 / count))), mean)


def synthid_normal_test(
    g_values: np.ndarray,
    *,
    nominal_fpr: float = 0.001,
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


class Scorer:
    def __init__(self):
        self.processor = attack_processor(torch.device("cpu"))

    def __call__(self, tokens, seed, null=False, reference=False):
        return -synthid_mean_score(self.processor, tokens)
