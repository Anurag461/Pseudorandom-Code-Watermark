import numpy as np
import pytest

from detectors import semantic_sha256
from proxy_8b_analysis import (
    cached_quality_metrics,
    COMMON_PREFIXES,
    ESTIMATOR_CHUNK_SIZE,
    PRC_AUDITS,
    PROXY_MODEL_REVISION,
    prc_audits,
    shared_null_proxy_trace_path,
    textseal_proxy_trace_identity,
    textseal_proxy_trace_path,
    validate_textseal_proxy_trace,
)


def test_proxy_campaign_freezes_all_eta_boundaries_and_ceiling_label():
    assert [row["eta"] for row in PRC_AUDITS] == [0.05, 0.10, 0.15, 0.20]
    assert [row["prefix_T"] for row in PRC_AUDITS] == [640, 1407, 4096, 13088]
    assert COMMON_PREFIXES == (128, 256, 400, 512, 768, 1024)
    assert "censored_ceiling" in PRC_AUDITS[2]["boundary_status"]
    assert "exact" not in PRC_AUDITS[2]["boundary_status"]
    assert all(row["require_full_entropy"] for row in prc_audits())
    assert ESTIMATOR_CHUNK_SIZE == 64
    assert len(PROXY_MODEL_REVISION) == 40
    assert all(
        row["estimator_chunk_size"] == ESTIMATOR_CHUNK_SIZE
        for row in prc_audits()
    )


def test_textseal_proxy_trace_validation_and_namespaces():
    identity = {
        "prompt_index": 7,
        "prompt_sha256": "prompt",
        "tokens_sha256": "tokens",
    }
    values = np.linspace(0.0, 10.0, 1024, dtype=np.float64)
    payload = {
        **textseal_proxy_trace_identity(**identity),
        "full_entropy_trace": values,
        "full_entropy_trace_sha256": semantic_sha256(values),
    }
    assert np.array_equal(validate_textseal_proxy_trace(payload, **identity), values)
    assert textseal_proxy_trace_path(7).endswith("chunk64/textseal_0007.pt")
    assert shared_null_proxy_trace_path(7).endswith(
        "chunk64/T13088/null_0007.pt"
    )
    with pytest.raises(ValueError, match="hash is inconsistent"):
        validate_textseal_proxy_trace(
            {**payload, "full_entropy_trace": values + 1.0}, **identity
        )


def test_cached_quality_metrics_matches_frozen_definitions():
    metrics = cached_quality_metrics(
        [1, 2, 1, 2, 1],
        [-1.0, -2.0, -1.0, -2.0, -1.0],
        prefix_length=5,
    )
    assert metrics["base_model_nll"] == 1.4
    assert metrics["output_length"] == 5
    assert metrics["distinct_2"] == 0.5
    assert metrics["distinct_3"] == 2 / 3
    assert metrics["repetition_rate"] == 0.0
    repeated = cached_quality_metrics(
        [1, 1, 1, 1, 1], [-1.0] * 5, prefix_length=5
    )
    assert repeated["repetition_rate"] == 0.5
