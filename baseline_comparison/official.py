"""Thin, documented calls into the pinned official baseline implementations."""

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
from .scoring import (
    deduplicated_positions,
    gumbel_gamma_test,
    self_bleu_token_ids,
    textseal_gamma_test,
)


def textseal_config(*, watermark_type: str = "textseal"):
    """Return the paper settings, explicitly overriding released-code defaults."""
    from textseal.watermarking.config import WatermarkConfig

    return WatermarkConfig(
        secret_key=TEXTSEAL_KEY_A,
        secret_key_b=TEXTSEAL_KEY_B,
        ngram=CONTEXT_LENGTH,
        watermark_type=watermark_type,
        method="uniform",
        mixing_alpha=TEXTSEAL_ALPHA,
        scoring_method="v2",
        depth=len(SYNTHID_KEYS),
    )


def gumbel_config():
    config = textseal_config(watermark_type="gumbelmax")
    return replace(config, secret_key=GUMBEL_KEY)


def textseal_generator():
    """Build only the official sampler state; the integration owns Qwen I/O."""
    import textseal.watermarking.generator as generator_module
    from textseal.watermarking.generator import TextSealGenerator

    # TextSeal's CUDA helper is decorated with torch.compile. In the pinned
    # torch 2.4 image that compiled path segfaults on the smoke worker before
    # returning a token. Call the pinned helper's original function body in
    # eager mode; no operation, constant, dtype, or formula is changed.
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

    config = textseal_config()
    generator = TextSealGenerator.__new__(TextSealGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    generator.key_a = config.key_a
    generator.key_b = config.key_b
    generator.mixing_alpha = config.mixing_alpha
    return generator


def gumbel_generator():
    """Build only the official TextSeal comparison-path Gumbel sampler state."""
    from textseal.watermarking.generator import GumbelmaxGenerator

    config = gumbel_config()
    generator = GumbelmaxGenerator.__new__(GumbelmaxGenerator)
    generator.wm_args = config
    generator.ngram = config.ngram
    generator.secret_key = config.secret_key
    return generator


def synthid_processor(device: torch.device | str):
    """Construct Google's processor and apply its minimal CUDA init workaround.

    The pinned constructor hashes ``self.keys.numpy()``. Constructing directly
    on CUDA therefore fails. We construct on CPU, then move the official key
    tensor and device attribute without changing the hash, g-values, or score
    update formulas.
    """
    from synthid_text.logits_processing import SynthIDLogitsProcessor

    target = torch.device(device)
    # The pinned reference checks the literal device string on every call.
    # ``scores.device`` is reported as ``cuda:0``, while ``torch.device("cuda")``
    # stringifies to ``cuda``; resolve the implicit index without changing any
    # SynthID state or formulas.
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    processor = SynthIDLogitsProcessor(
        ngram_len=CONTEXT_LENGTH + 1,
        keys=list(SYNTHID_KEYS),
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
    return torch.tensor(windows, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def official_textseal_fused_scores(
    token_ids: Sequence[int], positions: Sequence[int]
) -> np.ndarray:
    """Compute aligned dual-key scores with TextSeal's official PRF."""
    from textseal.watermarking.core import prf_dual

    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    r_a, r_b = prf_dual(windows, targets, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B)
    score_a = -torch.log1p(-r_a)
    score_b = -torch.log1p(-r_b)
    # Released code defines alpha as Key-A probability/weight. The paper labels
    # the two keys oppositely; the mixture distribution is symmetric.
    fused = TEXTSEAL_ALPHA * score_a + (1.0 - TEXTSEAL_ALPHA) * score_b
    return fused.double().cpu().numpy()


def official_gumbel_scores(
    token_ids: Sequence[int], positions: Sequence[int]
) -> np.ndarray:
    """Compute the exact-Gamma score using TextSeal's official uniform PRF."""
    from textseal.watermarking.core import prf_uniform

    if not positions:
        return np.empty(0, dtype=np.float64)
    windows, targets = _windows_targets(token_ids, positions, CONTEXT_LENGTH)
    uniforms = prf_uniform(windows, targets, GUMBEL_KEY)
    return (-torch.log1p(-uniforms)).double().cpu().numpy()


def official_synthid_g_values(
    token_ids: Sequence[int], positions: Sequence[int], *, device: str = "cpu"
) -> np.ndarray:
    """Compute Google's g-values and select TextSeal-v2 deduplicated positions."""
    if not positions:
        return np.empty((0, len(SYNTHID_KEYS)), dtype=np.int64)
    processor = synthid_processor(device)
    ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    values = processor.compute_g_values(ids)[0]
    # ngram_len=k+1 maps target position p to g-value row p-k.
    rows = torch.tensor(
        [int(position) - CONTEXT_LENGTH for position in positions],
        dtype=torch.long,
        device=values.device,
    )
    return values.index_select(0, rows).long().cpu().numpy()


def run_official_reference_checks() -> dict:
    """Dependency-heavy parity checks intended for the isolated Modal CPU image."""
    from textseal.watermarking.detector import TextSealDetector
    from textseal.watermarking.core import prf_dual

    tokens = [7, 11, 13, 17, 19, 23, 29, 31, 7, 11, 13, 17, 37, 41, 43]
    entropies = [0.25 + 0.07 * index for index in range(len(tokens))]
    positions = deduplicated_positions(tokens, CONTEXT_LENGTH)
    fused = official_textseal_fused_scores(tokens, positions)
    selected_entropies = [entropies[position - 1] for position in positions]
    common = textseal_gamma_test(fused, selected_entropies, TEXTSEAL_ALPHA)

    detector = TextSealDetector(None, textseal_config(), scoring_method="v2")
    reference = detector._score_text(tokens, entropies, scoring_method="v2")
    textseal_delta = abs(common["p_value"] - reference["p_value_weighted"])
    # The official detector converts PRF values through scalar float32 tensors;
    # the common scorer accumulates the same values in float64. The resulting
    # survival probabilities should agree to substantially better than 1e-7.
    if textseal_delta > 1e-7:
        raise AssertionError(f"TextSeal weighted p mismatch: {textseal_delta}")
    if int(reference["n_tokens"]) != len(positions):
        raise AssertionError("TextSeal reference dedup count differs")

    generator = textseal_generator()
    windows = torch.tensor([[2, 3, 5], [7, 11, 13]], dtype=torch.long)
    candidates = torch.tensor([[17, 19, 23, 29], [31, 37, 41, 43]], dtype=torch.long)
    eager_a, eager_b = __import__(
        "textseal.watermarking.generator", fromlist=["fast_prf_dual"]
    ).fast_prf_dual(windows, candidates, TEXTSEAL_KEY_A, TEXTSEAL_KEY_B)
    expected_a = []
    expected_b = []
    for row in range(windows.shape[0]):
        row_a, row_b = prf_dual(
            windows[row : row + 1], candidates[row], TEXTSEAL_KEY_A, TEXTSEAL_KEY_B
        )
        expected_a.append(row_a)
        expected_b.append(row_b)
    if not torch.equal(eager_a, torch.stack(expected_a)) or not torch.equal(
        eager_b, torch.stack(expected_b)
    ):
        raise AssertionError("TextSeal eager CUDA adapter differs from official PRF")

    gumbel_scores = official_gumbel_scores(tokens, positions)
    gumbel = gumbel_gamma_test(gumbel_scores)
    expected_gumbel_p = float(
        __import__("scipy").special.gammaincc(len(positions), gumbel_scores.sum())
    )
    if not math.isclose(gumbel["p_value"], expected_gumbel_p, rel_tol=0, abs_tol=1e-14):
        raise AssertionError("Gumbel exact-Gamma reference differs")

    synthid = synthid_processor("cpu")
    ids = torch.tensor([tokens], dtype=torch.long)
    direct_g = synthid.compute_g_values(ids)
    adapter_g = official_synthid_g_values(tokens, positions)
    rows = torch.tensor([position - CONTEXT_LENGTH for position in positions])
    if not np.array_equal(adapter_g, direct_g[0].index_select(0, rows).numpy()):
        raise AssertionError("SynthID adapter g-values differ from Google reference")

    torch.manual_seed(8128)
    synthetic_logits = torch.linspace(-2.0, 2.0, 97).repeat(2, 1)
    synthetic_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    left = synthid_processor("cpu")
    right = synthid_processor("cpu")
    left_scores, left_indices, _ = left.watermarked_call(synthetic_ids, synthetic_logits)
    right_scores, right_indices, _ = right.watermarked_call(synthetic_ids, synthetic_logits)
    if not torch.equal(left_indices, right_indices) or not torch.equal(left_scores, right_scores):
        raise AssertionError("SynthID identical-input score update is not deterministic")

    self_bleu = self_bleu_token_ids([[1, 2, 3, 4], [1, 2, 3, 4]])
    if not math.isclose(float(self_bleu), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("token-ID Self-BLEU reference case differs")

    return {
        "passed": True,
        "textseal": {
            "official_weighted_p": float(reference["p_value_weighted"]),
            "common_weighted_p": float(common["p_value"]),
            "absolute_difference": float(textseal_delta),
            "deduplicated_sample_count": len(positions),
            "official_detector_min_p": float(reference["p_value"]),
            "official_unweighted_p": float(reference["p_value_unweighted"]),
            "eager_cuda_adapter_prf_equal": True,
            "cuda_adapter": "pinned _prf_dual_compiled original callable; torch.compile bypassed",
        },
        "gumbel_max": {
            "official_score_sum": float(gumbel_scores.sum()),
            "exact_gamma_p": float(gumbel["p_value"]),
        },
        "synthid_text": {
            "g_values_equal": True,
            "score_update_equal": True,
            "synthetic_g_value_sha256": __import__("hashlib").sha256(
                direct_g.numpy().tobytes()
            ).hexdigest(),
        },
        "diversity_metrics": {"identical_pair_self_bleu": float(self_bleu)},
    }
