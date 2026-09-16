import numpy as np

from entropy_projection_analysis import (
    binary_entropy_bits,
    merge_model_shards,
    summarize_trace_batch,
)


def test_binary_entropy_bits_handles_boundaries():
    np.testing.assert_allclose(
        binary_entropy_bits([0.0, 0.5, 1.0]),
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )


def test_summary_gap_is_full_minus_projected():
    summary = summarize_trace_batch(
        model_size="toy",
        prompt_indices=[0],
        full_entropy_bits=[[2.0, 1.0]],
        partition_probability=[[0.5, 0.0]],
        trace_t=2,
        partition_sha256="a" * 64,
        source_cache_t=2,
        source_kind="test",
    )
    assert summary["metric_moments"]["full_entropy_bits"]["sum"] == 3.0
    assert summary["metric_moments"]["projected_entropy_bits"]["sum"] == 1.0
    assert summary["metric_moments"]["gap_bits"]["sum"] == 2.0


def test_merge_preserves_prompt_and_position_means():
    shards = []
    for index, full in [(0, [[2.0, 1.0]]), (1, [[4.0, 3.0]])]:
        shards.append(summarize_trace_batch(
            model_size="toy",
            prompt_indices=[index],
            full_entropy_bits=full,
            partition_probability=[[0.5, 0.5]],
            trace_t=2,
            partition_sha256="a" * 64,
            source_cache_t=2,
            source_kind="test",
        ))
    merged = merge_model_shards(shards, expected_prompts=2)
    assert merged["token_count"] == 4
    assert merged["metric_statistics"]["full_entropy_bits"]["mean"] == 2.5
    np.testing.assert_allclose(
        merged["position_statistics"]["full_entropy_bits"]["mean"],
        [3.0, 2.0],
    )


def test_reference_trace_reports_projected_entropy_sensitivity():
    summary = summarize_trace_batch(
        model_size="toy",
        prompt_indices=[0],
        full_entropy_bits=[[2.0, 2.0]],
        partition_probability=[[0.5, 0.5]],
        reference_partition_probability=[[0.0, 1.0]],
        trace_t=2,
        partition_sha256="a" * 64,
        source_cache_t=2,
        source_kind="test",
    )
    error = summary["reference_partition_error"]
    assert error["sum_projected_entropy_difference_bits"] == -2.0
    merged = merge_model_shards([summary], expected_prompts=1)
    assert (
        merged["reference_partition_error"]
        ["mean_projected_entropy_difference_bits"]
        == -1.0
    )
