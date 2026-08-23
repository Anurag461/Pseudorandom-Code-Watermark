"""Frozen configuration and validation helpers for the cache-only 8B proxy study."""

from __future__ import annotations

import hashlib
import math
import os


PROXY_ANALYSIS_SCHEMA_VERSION = 1
PROXY_MODEL_SIZE = "0.6B"
PROXY_MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
PROXY_MODEL_WEIGHTS_ETAG = (
    "cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"
)
PROXY_TOKENIZER_ETAG = "443909a61d429dff23010e5bddd28ff530edda00"
GENERATION_MODEL_SIZE = "8B"
NULL_TRACE_T = 13_088
COMMON_PREFIXES = (128, 256, 400, 512, 768, 1024)
BASELINE_RUN_ID = "qwen3-8b-batch50-validation-20260823-v1"
APPROVAL_TOKEN = "APPROVE_8B_PROXY_ANALYSES_20_USD"
HARD_CAP_USD = 20.0
NOMINAL_FPR = 1e-3
ESTIMATOR_CHUNK_SIZE = 64
PROXY_IMAGE_DEFINITION_SHA256 = (
    "e62c9627f72c0448435665ec51304050838f65ef68b8d77766d6176a03745e48"
)

# eta=.15 is deliberately a censored ceiling, not an exact >90% boundary.
PRC_AUDITS = (
    {
        "label": "eta0.05-boundary640",
        "eta": 0.05,
        "prefix_T": 640,
        "trace_T": 1024,
        "source_T": 1280,
        "source_tag": (
            "online_causal_prc_v1/qwen3_8b_base/"
            "n1280_T1280_t3_eta0.05_rr99of100_sampler-poscdf-v1_"
            "kvcache-static-v1"
        ),
        "boundary_status": "exact_gt_90pct_map_boundary",
    },
    {
        "label": "eta0.10-boundary1407",
        "eta": 0.10,
        "prefix_T": 1407,
        "trace_T": 1407,
        "source_T": 3072,
        "source_tag": (
            "online_causal_prc_v1/qwen3_8b_base/"
            "n3072_T3072_t3_eta0.10_rr99of100_sampler-poscdf-v1_"
            "kvcache-static-v1"
        ),
        "boundary_status": "exact_gt_90pct_map_boundary",
    },
    {
        "label": "eta0.15-ceiling4096",
        "eta": 0.15,
        "prefix_T": 4096,
        "trace_T": 4096,
        "source_T": 4096,
        "source_tag": (
            "online_causal_prc_v1/qwen3_8b_base/"
            "n4096_T4096_t3_eta0.15_rr99of100_sampler-poscdf-v1"
        ),
        "boundary_status": "censored_ceiling_map_89p6pct_n90_gt_4096",
    },
    {
        "label": "eta0.20-boundary13088",
        "eta": 0.20,
        "prefix_T": 13088,
        "trace_T": 13088,
        "source_T": 14336,
        "source_tag": (
            "online_causal_prc_v1/qwen3_8b_base/"
            "n14336_T14336_t3_eta0.20_rr99of100_sampler-poscdf-v1_"
            "kvcache-static-v1"
        ),
        "boundary_status": "exact_gt_90pct_map_boundary",
    },
)


def prc_audits(*, require_full_entropy: bool = True) -> list[dict]:
    return [
        {
            **dict(audit),
            "require_full_entropy": bool(require_full_entropy),
            "estimator_chunk_size": ESTIMATOR_CHUNK_SIZE,
        }
        for audit in PRC_AUDITS
    ]


def textseal_proxy_trace_path(prompt_index: int) -> str:
    return os.path.join(
        "/data/controlled_baseline_full",
        BASELINE_RUN_ID,
        "proxy_entropy",
        "qwen3_0p6b_base",
        f"chunk{ESTIMATOR_CHUNK_SIZE}",
        f"textseal_{int(prompt_index):04d}.pt",
    )


def shared_null_proxy_trace_path(prompt_index: int) -> str:
    return os.path.join(
        "/data/_online_null_cross_model_entropy",
        "qwen3_8b_base",
        "qwen3_0p6b_base",
        f"chunk{ESTIMATOR_CHUNK_SIZE}",
        f"T{NULL_TRACE_T}",
        f"null_{int(prompt_index):04d}.pt",
    )


def textseal_proxy_trace_identity(
    *, prompt_index: int, prompt_sha256: str, tokens_sha256: str
) -> dict:
    return {
        "proxy_analysis_schema_version": PROXY_ANALYSIS_SCHEMA_VERSION,
        "trace_kind": "full_vocab_entropy_nats",
        "method": "textseal",
        "baseline_run_id": BASELINE_RUN_ID,
        "prompt_index": int(prompt_index),
        "trace_T": 1024,
        "generation_model_size": GENERATION_MODEL_SIZE,
        "proxy_model_size": PROXY_MODEL_SIZE,
        "prompt_sha256": str(prompt_sha256),
        "tokens_sha256": str(tokens_sha256),
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
        "estimator_chunk_size": ESTIMATOR_CHUNK_SIZE,
        "estimator_execution": "causal_multi_token_chunks_v1",
    }


def validate_textseal_proxy_trace(payload: dict, **identity_kwargs):
    import numpy as np
    from detectors import semantic_sha256

    expected = textseal_proxy_trace_identity(**identity_kwargs)
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"TextSeal proxy trace {field}={payload.get(field)!r}; "
                f"expected {value!r}"
            )
    values = np.asarray(payload.get("full_entropy_trace"), dtype=np.float64).reshape(-1)
    if values.size != expected["trace_T"]:
        raise ValueError("TextSeal proxy trace must contain exactly 1024 entropies")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("TextSeal proxy entropies must be finite and nonnegative")
    if payload.get("full_entropy_trace_sha256") != semantic_sha256(values):
        raise ValueError("TextSeal proxy entropy hash is inconsistent")
    return values


def campaign_code_fingerprint(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode())
        digest.update(b"\0")
        with open(path, "rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def cached_quality_metrics(
    token_ids, base_token_logprobs, *, prefix_length: int = 1024
) -> dict:
    """Compute the frozen native-generator quality metrics from cached traces."""
    tokens = [int(value) for value in token_ids][: int(prefix_length)]
    logprobs = [float(value) for value in base_token_logprobs][
        : int(prefix_length)
    ]
    if len(tokens) != int(prefix_length) or len(logprobs) != len(tokens):
        raise ValueError("cached quality input does not cover the exact prefix")
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("cached base-model log-probabilities must be finite")

    def distinct_n(n: int) -> float:
        grams = [
            tuple(tokens[index : index + n])
            for index in range(max(0, len(tokens) - n + 1))
        ]
        return float(len(set(grams)) / len(grams)) if grams else 0.0

    nll = float(-sum(logprobs) / len(logprobs)) if logprobs else 0.0
    return {
        "base_model_nll": nll,
        "base_model_perplexity": float(math.exp(nll)),
        "output_length": len(tokens),
        "repetition_rate": float(1.0 - distinct_n(4)),
        "repetition_metric": (
            "repeated token 4-gram fraction: "
            "1 - unique_4grams/total_4grams"
        ),
        "distinct_2": distinct_n(2),
        "distinct_3": distinct_n(3),
    }
