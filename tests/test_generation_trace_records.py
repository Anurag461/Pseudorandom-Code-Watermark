import numpy as np
import pytest
import torch

from detectors import (
    GENERATION_TRACE_SCHEMA_VERSION,
    build_prc_generation_record,
)
from modal_run import validate_generation_record


def _partition():
    # Token 0 is in bucket 0; tokens 1 and 2 are in bucket 1.
    return torch.tensor(
        [[1, 0, 0], [0, 1, 1]],
        dtype=torch.float32,
    )


def _record(watermark=True):
    return build_prc_generation_record(
        prompt_token_ids=[7, 8],
        generated_token_ids=torch.tensor([0, 1, 2]),
        p_trace=np.array([0.5, 0.1, 0.8]),
        partition_map=_partition(),
        n=2,
        watermark=watermark,
        encoding_key=("synthetic-test-key",),
        prc_codeword_bits=(
            np.array([0, 1, 0], dtype=np.uint8) if watermark else None
        ),
        base_lm_entropy=np.array([2.0, 1.5, 1.0]),
        base_token_logprob=np.array([-1.0, -2.0, -3.0]),
    )


def test_prc_generation_record_contains_raw_and_derived_traces():
    record = _record()

    assert record["generation_trace_schema_version"] == (
        GENERATION_TRACE_SCHEMA_VERSION
    )
    assert record["prompt_token_ids"].tolist() == [7, 8]
    assert record["tokens"].tolist() == [0, 1, 2]
    assert record["prc_n"] == 2
    np.testing.assert_array_equal(record["observed_bucket_bits"], [0, 1, 1])
    np.testing.assert_array_equal(record["prc_codeword_bits"], [0, 1, 0])
    np.testing.assert_array_equal(
        record["prc_block_boundaries"], [[0, 2], [2, 3]]
    )

    expected_entropy = np.array([1.0, 0.4689956, 0.7219281])
    np.testing.assert_allclose(
        record["entropy_trace"], expected_entropy, rtol=1e-6
    )
    np.testing.assert_allclose(
        record["signed_entropy_trace"],
        [1.0, -0.4689956, -0.7219281],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["codeword_signed_entropy_trace"],
        [1.0, -0.4689956, 0.7219281],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["map_soft_tokens"], [1.0, -1.0, -0.25], rtol=1e-6
    )
    np.testing.assert_allclose(
        record["folded_signed_entropy"],
        [(1.0 - 0.7219281) / 2.0, -0.4689956],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        record["folded_map_soft_tokens"], [0.375, -1.0], rtol=1e-6
    )
    np.testing.assert_array_equal(record["base_lm_entropy"], [2.0, 1.5, 1.0])
    np.testing.assert_array_equal(
        record["base_token_logprob"], [-1.0, -2.0, -3.0]
    )
    assert len(record["partition_sha256"]) == 64
    assert len(record["encoding_key_sha256"]) == 64


def test_null_record_explicitly_has_no_prc_codeword():
    record = _record(watermark=False)

    assert record["prc_codeword_bits"] is None
    assert record["codeword_signed_entropy_trace"] is None
    record.update({
        "watermark": False,
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
    })
    validate_generation_record(record, "8B", "null", 0)


def test_record_copies_batched_rows_out_of_shared_backing_storage():
    prompt_batch = torch.arange(20, dtype=torch.long).reshape(4, 5)
    token_batch = torch.tensor(
        [[0, 1, 2], [2, 1, 0], [1, 0, 2], [0, 2, 1]],
        dtype=torch.long,
    )
    p_batch = np.linspace(0.1, 0.9, 12).reshape(4, 3)
    entropy_batch = np.linspace(1.0, 2.1, 12, dtype=np.float32).reshape(4, 3)
    logprob_batch = -entropy_batch
    codeword_batch = np.array(
        [[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 1, 0]],
        dtype=np.uint8,
    )

    record = build_prc_generation_record(
        prompt_token_ids=prompt_batch[2],
        generated_token_ids=token_batch[2],
        p_trace=p_batch[2],
        partition_map=_partition(),
        n=2,
        watermark=True,
        encoding_key=("synthetic-test-key",),
        prc_codeword_bits=codeword_batch[2],
        base_lm_entropy=entropy_batch[2],
        base_token_logprob=logprob_batch[2],
    )

    for field in ("prompt_token_ids", "tokens"):
        tensor = record[field]
        assert tensor.storage_offset() == 0
        assert tensor.untyped_storage().nbytes() == (
            tensor.numel() * tensor.element_size()
        )
    for field in (
        "p_trace",
        "base_lm_entropy",
        "base_token_logprob",
        "prc_codeword_bits",
    ):
        assert record[field].flags.owndata


def test_new_record_validation_rejects_wrong_fold_length():
    record = _record()
    record["folded_map_soft_tokens"] = record["folded_map_soft_tokens"][:1]
    record.update({
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
    })

    with pytest.raises(ValueError, match="folded_map_soft_tokens values"):
        validate_generation_record(record, "8B", "wm", 0)


def test_watermarked_record_requires_exact_codeword():
    with pytest.raises(ValueError, match="require prc_codeword_bits"):
        build_prc_generation_record(
            prompt_token_ids=[7, 8],
            generated_token_ids=torch.tensor([0, 1, 2]),
            p_trace=np.array([0.5, 0.1, 0.8]),
            partition_map=_partition(),
            n=2,
            watermark=True,
            encoding_key=("synthetic-test-key",),
            prc_codeword_bits=None,
            base_lm_entropy=np.array([2.0, 1.5, 1.0]),
            base_token_logprob=np.array([-1.0, -2.0, -3.0]),
        )


def test_generation_record_rejects_misaligned_trace_lengths():
    with pytest.raises(ValueError, match="base_lm_entropy shape"):
        build_prc_generation_record(
            prompt_token_ids=[7, 8],
            generated_token_ids=torch.tensor([0, 1, 2]),
            p_trace=np.array([0.5, 0.1, 0.8]),
            partition_map=_partition(),
            n=2,
            watermark=False,
            encoding_key=("synthetic-test-key",),
            prc_codeword_bits=None,
            base_lm_entropy=np.array([2.0, 1.5]),
            base_token_logprob=np.array([-1.0, -2.0, -3.0]),
        )
