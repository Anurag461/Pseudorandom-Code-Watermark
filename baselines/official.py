from __future__ import annotations
from dataclasses import replace
import math
from typing import Sequence
import numpy as np
import torch
from .config import (
    CONTEXT_LENGTH,
    GUMBEL_KEY,
    SYNTHID_CONTEXT_HISTORY_SIZE,
    SYNTHID_KEYS,
    TEMPERATURE,
    TEXTSEAL_ALPHA,
    TEXTSEAL_KEY_A,
    TEXTSEAL_KEY_B,
)


def textseal_config(*, watermark_type: str = "textseal", alpha: float = TEXTSEAL_ALPHA):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("TextSeal alpha must be finite and in [0, 1]")
    from textseal.watermarking.config import WatermarkConfig

    return WatermarkConfig(
        secret_key=TEXTSEAL_KEY_A,
        secret_key_b=TEXTSEAL_KEY_B,
        ngram=CONTEXT_LENGTH,
        watermark_type=watermark_type,
        method="uniform",
        mixing_alpha=float(alpha),
        scoring_method="v2",
        depth=len(SYNTHID_KEYS),
    )


def gumbel_config():
    config = textseal_config(watermark_type="gumbelmax")
    return replace(config, secret_key=GUMBEL_KEY)


def textseal_generator(*, alpha: float = TEXTSEAL_ALPHA):
    import textseal.watermarking.generator as generator_module
    from textseal.watermarking.generator import TextSealGenerator

    def eager_fast_prf_dual(w, token_ids, sk_a, sk_b):
        from textseal.watermarking.core import _prf_dual_compiled, _weighted_sum

        original = getattr(
            _prf_dual_compiled, "_torchdynamo_orig_callable", _prf_dual_compiled
        )
        weighted = _weighted_sum(w)
        key_a = torch.tensor(sk_a, dtype=torch.long, device=w.device)
        key_b = torch.tensor(sk_b, dtype=torch.long, device=w.device)
        return original(weighted, token_ids, key_a, key_b)

    generator_module.fast_prf_dual = eager_fast_prf_dual
    config = textseal_config(alpha=alpha)
    generator = TextSealGenerator.__new__(TextSealGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    generator.key_a = config.key_a
    generator.key_b = config.key_b
    generator.mixing_alpha = config.mixing_alpha
    return generator


def gumbel_generator():
    from textseal.watermarking.generator import GumbelmaxGenerator

    config = gumbel_config()
    generator = GumbelmaxGenerator.__new__(GumbelmaxGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    return generator


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


def official_textseal_fused_scores(
    token_ids: Sequence[int], positions: Sequence[int], *, alpha: float = TEXTSEAL_ALPHA
) -> np.ndarray:
    from textseal.watermarking.core import prf_dual

    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("TextSeal alpha must be finite and in [0, 1]")
    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    r_a, r_b = prf_dual(windows, targets, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B)
    score_a = -torch.log1p(-r_a)
    score_b = -torch.log1p(-r_b)
    fused = alpha * score_a + (1.0 - alpha) * score_b
    return fused.double().cpu().numpy()


def official_gumbel_scores(
    token_ids: Sequence[int], positions: Sequence[int]
) -> np.ndarray:
    from textseal.watermarking.core import prf_uniform

    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    uniforms = prf_uniform(windows, targets, GUMBEL_KEY)
    return (-torch.log1p(-uniforms)).double().cpu().numpy()


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
