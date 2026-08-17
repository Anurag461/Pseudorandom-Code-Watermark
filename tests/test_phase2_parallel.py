import copy

import pytest

from phase2_parallel import (
    PHASE2_SHARD_SCHEMA_VERSION,
    merge_phase2_shard_payloads,
    phase2_config_fingerprint,
    phase2_prompt_shards,
    stable_json_sha256,
    summarize_phase2_records,
    validate_phase2_shard_payload,
)


def _record(prompt_idx, watermark, decisions):
    source = "wm" if watermark else "null"
    return {
        "prompt_idx": prompt_idx,
        "watermark": watermark,
        "phase0_hoeffding_decision": decisions[0],
        "phase1_rademacher_decision": decisions[1],
        "phase2_hoeffding_decision": decisions[2],
        "phase2_adaptive_decision": decisions[3],
        "basis_rank": 7,
        "selected_erasure_quantile": 0.2,
        "erased_columns": 2,
        "erasure_free_rows": 5,
        "degree_median": 3.0,
        "tokens_sha256": f"tokens-{source}-{prompt_idx}",
        "p_trace_sha256": f"probs-{source}-{prompt_idx}",
        "source_path": f"/{source}_{prompt_idx:04d}.pt",
    }


def _shard(prompt_indices, config="config-a"):
    records = []
    for watermark in (True, False):
        for prompt_idx in prompt_indices:
            if watermark:
                decisions = {
                    0: (True, True, True, True),
                    1: (False, True, True, True),
                    2: (True, True, False, False),
                    3: (False, False, True, True),
                }[prompt_idx]
            else:
                decisions = (False, False, False, False)
            records.append(_record(prompt_idx, watermark, decisions))
    return {
        "shard_schema_version": PHASE2_SHARD_SCHEMA_VERSION,
        "config_fingerprint": config,
        "prompt_indices": list(prompt_indices),
        "records_sha256": stable_json_sha256(records),
        "records": records,
    }


def test_phase2_prompt_shards_are_stable_and_complete():
    assert phase2_prompt_shards(7, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    with pytest.raises(ValueError, match="num_prompts"):
        phase2_prompt_shards(0, 3)
    with pytest.raises(ValueError, match="shard_size"):
        phase2_prompt_shards(7, 0)


def test_phase2_config_fingerprint_is_key_order_independent():
    left = {"n": 32, "eta": 0.1, "grid": [0.0, 0.2]}
    right = {"grid": [0.0, 0.2], "eta": 0.1, "n": 32}
    assert phase2_config_fingerprint(left) == phase2_config_fingerprint(right)


def test_phase2_shards_merge_in_canonical_order_and_summarize_exactly():
    left = _shard([0, 1])
    right = _shard([2, 3])
    records = merge_phase2_shard_payloads(
        [right, left],
        [0, 1, 2, 3],
        config_fingerprint="config-a",
        basis_rank=7,
    )
    assert [record["prompt_idx"] for record in records[:4]] == [0, 1, 2, 3]
    assert all(record["watermark"] for record in records[:4])
    assert not any(record["watermark"] for record in records[4:])

    summary = summarize_phase2_records(
        records,
        4,
        expected_hoeffding_tp=2,
        expected_hoeffding_fp=0,
    )
    assert summary["phase0_original_basis_hoeffding"]["tp"] == 2
    assert summary["phase1_original_basis_weighted_rademacher"]["tp"] == 3
    assert summary["phase2_adaptive_basis_hoeffding"]["tp"] == 3
    assert summary["phase2_adaptive_basis_weighted_rademacher"]["tp"] == 3
    transition = summary["paired_transitions"][
        "adaptive_hoeffding_vs_original_hoeffding_watermarked"
    ]
    assert transition == {"gained": 2, "lost": 1, "net": 1}


def test_phase2_shard_validation_rejects_tampering_and_wrong_rank():
    payload = _shard([0, 1])
    tampered = copy.deepcopy(payload)
    tampered["records"][0]["phase2_adaptive_decision"] = False
    with pytest.raises(ValueError, match="checksum"):
        validate_phase2_shard_payload(tampered)
    with pytest.raises(ValueError, match="rank-deficient"):
        validate_phase2_shard_payload(payload, expected_basis_rank=8)


def test_phase2_merge_rejects_overlap_and_gaps():
    left = _shard([0, 1])
    overlap = _shard([1, 2])
    with pytest.raises(ValueError, match="overlap"):
        merge_phase2_shard_payloads(
            [left, overlap],
            [0, 1, 2],
            config_fingerprint="config-a",
            basis_rank=7,
        )
    with pytest.raises(ValueError, match="coverage"):
        merge_phase2_shard_payloads(
            [left],
            [0, 1, 2],
            config_fingerprint="config-a",
            basis_rank=7,
        )
