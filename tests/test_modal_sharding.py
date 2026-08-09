import copy
import hashlib
import json

import pytest

from modal_run import (
    DETECTION_CHECKPOINT_SCHEMA_VERSION,
    SHARD_RESULT_SCHEMA_VERSION,
    _aggregate_shard_payloads,
    _append_summary_row,
    _detection_checkpoint_identity,
    _load_detection_checkpoint,
    _save_detection_checkpoint,
    _summary_row_exists,
    prompt_indices_for_shard,
    resolve_r,
)


def _checkpoint_identity(**overrides):
    values = {
        "config": {
            "n": 8192,
            "T": 8192,
            "t": 3,
            "eta": 0.2,
            "r_value": 8110,
            "target_fpr": 1e-3,
            "entropy_model": "Qwen3-4B-Base",
        },
        "artifact_fingerprint": "artifact-abc",
        "code_fingerprint": {"sha256": "code-abc"},
        "source": "wm",
        "prompt_idx": 125,
        "tokens_sha256": "tokens-abc",
        "p_trace_sha256": "trace-abc",
    }
    values.update(overrides)
    return _detection_checkpoint_identity(**values)


def _checkpoint_record():
    return {
        "prompt_idx": 125,
        "source": "wm",
        "watermark": True,
        "decision_map": True,
        "stat_map": 42.0,
        "thr_map": 10.0,
        "decision_entropy": True,
        "stat_entropy": 40.0,
        "thr_entropy": 11.0,
        "decision_naive": None,
        "tokens_sha256": "tokens-abc",
        "p_trace_sha256": "trace-abc",
    }


def _payload(indices, workspace, total=8):
    records = []
    for index in indices:
        records.extend([
            {
                "prompt_idx": index,
                "source": "wm",
                "watermark": True,
                "decision_map": index % 2 == 0,
                "decision_entropy": index < 3,
                "decision_naive": True,
                "stat_map": float(index),
                "thr_map": 1.0,
                "stat_entropy": float(index),
                "thr_entropy": 1.0,
                "stat_naive": float(index),
                "thr_naive": 1.0,
            },
            {
                "prompt_idx": index,
                "source": "null",
                "watermark": False,
                "decision_map": index == total - 1,
                "decision_entropy": False,
                "decision_naive": False,
                "stat_map": float(index),
                "thr_map": 1.0,
                "stat_entropy": float(index),
                "thr_entropy": 1.0,
                "stat_naive": float(index),
                "thr_naive": 1.0,
            },
        ])
    records_json = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    return {
        "schema_version": SHARD_RESULT_SCHEMA_VERSION,
        "config": {
            "n": 8192,
            "T": 8192,
            "t": 3,
            "eta": 0.2,
            "r_value": 8110,
            "r_setting": "0.99n",
            "target_fpr": 1e-3,
            "generation_model": "Qwen3-0.6B-Base",
            "entropy_model": "Qwen3-0.6B-Base",
            "entropy_trace_source": "cached_generation_p_trace",
            "seed": 12345,
            "canonical_num_prompts": total,
        },
        "artifact_fingerprint": "artifact-abc",
        "code_fingerprint": {"sha256": "code-abc", "git_revision": "deadbeef"},
        "workspace_label": workspace,
        "prompt_indices": list(indices),
        "record_count": len(records),
        "records_sha256": hashlib.sha256(records_json).hexdigest(),
        "parity_check_rank_info": {
            "rows": 8110,
            "rank": 8110,
            "full_rank": True,
        },
        "created_at": "2026-08-02T00:00:00+00:00",
        "records": records,
    }


def _four_payloads():
    return [
        _payload(range(0, 2), "workspace-a"),
        _payload(range(2, 4), "workspace-b"),
        _payload(range(4, 6), "workspace-c"),
        _payload(range(6, 8), "workspace-d"),
    ]


def test_prompt_indices_are_global_and_bounds_checked():
    assert prompt_indices_for_shard(125, 3) == [125, 126, 127]
    with pytest.raises(ValueError, match="exceeds"):
        prompt_indices_for_shard(499, 2)
    with pytest.raises(ValueError, match="must be >= 0"):
        prompt_indices_for_shard(-1, 1)


def test_eta020_runbook_r_fraction_values():
    assert resolve_r(768, r_frac=0.99) == 760
    assert resolve_r(8192, r_frac=0.99) == 8110
    assert resolve_r(4096, r_frac=0.99) == 4055
    assert resolve_r(2048, r_frac=0.99) == 2028


def test_aggregates_integer_decisions_into_one_summary_row():
    row, audit = _aggregate_shard_payloads(_four_payloads(), 8)

    assert row["Map TPR"] == "4/8 (50.0%)"
    assert row["Entropy Aware TPR"] == "3/8 (37.5%)"
    assert row["Naive TPR"] == "8/8 (100.0%)"
    assert row["Map FPR"] == "1/8 (12.5%)"
    assert row["Entropy FPR"] == "0/8 (0.0%)"
    assert row["r setting"] == "0.99n"
    assert "4-shard prompt aggregation across 4 workspaces" in row["Notes"]
    assert "workspace-a" not in row["Notes"]
    assert "workspace-b" not in row["Notes"]
    assert audit["counts"]["wm_total"] == 8
    assert audit["counts"]["null_total"] == 8
    assert len(audit["shards"]) == 4


def test_aggregates_two_prompt_shards_across_two_workspaces():
    payloads = [
        _payload(range(0, 4), "workspace-a"),
        _payload(range(4, 8), "workspace-b"),
    ]

    row, audit = _aggregate_shard_payloads(payloads, 8)

    assert "2-shard prompt aggregation across 2 workspaces" in row["Notes"]
    assert "workspace-a" not in row["Notes"]
    assert "workspace-b" not in row["Notes"]
    assert len(audit["shards"]) == 2


def test_rejects_overlapping_prompt_shards():
    payloads = _four_payloads()
    payloads[-1] = _payload(range(5, 7), "workspace-d")

    with pytest.raises(ValueError, match="multiple shards"):
        _aggregate_shard_payloads(payloads, 8)


def test_rejects_gaps_in_global_prompt_coverage():
    payloads = _four_payloads()[:-1]

    with pytest.raises(ValueError, match="coverage mismatch"):
        _aggregate_shard_payloads(payloads, 8)


def test_rejects_mismatched_artifact_fingerprint():
    payloads = _four_payloads()
    payloads[-1] = copy.deepcopy(payloads[-1])
    payloads[-1]["artifact_fingerprint"] = "different-key"

    with pytest.raises(ValueError, match="artifact/key mismatch"):
        _aggregate_shard_payloads(payloads, 8)


def test_rejects_modified_records():
    payloads = _four_payloads()
    payloads[-1] = copy.deepcopy(payloads[-1])
    payloads[-1]["records"][0]["decision_map"] = False

    with pytest.raises(ValueError, match="checksum mismatch"):
        _aggregate_shard_payloads(payloads, 8)


def test_detects_an_existing_authoritative_row_across_numeric_formatting(tmp_path):
    row, _ = _aggregate_shard_payloads(_four_payloads(), 8)
    csv_path = tmp_path / "summary.csv"
    _append_summary_row(csv_path, row)
    candidate = dict(row)
    candidate["eta"] = "0.20"
    candidate["Target FPR"] = "0.001"

    assert _summary_row_exists(csv_path, candidate)


def test_summary_identity_distinguishes_generation_models(tmp_path):
    row, _ = _aggregate_shard_payloads(_four_payloads(), 8)
    csv_path = tmp_path / "summary.csv"
    _append_summary_row(csv_path, row)
    candidate = dict(row)
    candidate["Generation Model"] = "Qwen3-8B-Base"

    assert not _summary_row_exists(csv_path, candidate)


def test_detection_checkpoint_round_trip(tmp_path):
    path = tmp_path / "wm_0125.json"
    identity = _checkpoint_identity()
    record = _checkpoint_record()

    _save_detection_checkpoint(path, identity, record)

    assert _load_detection_checkpoint(path, identity) == record
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == DETECTION_CHECKPOINT_SCHEMA_VERSION
    assert payload["identity"]["detector_implementation_sha256"] == "code-abc"


@pytest.mark.parametrize("changed", [
    {"artifact_fingerprint": "different-artifact"},
    {"code_fingerprint": {"sha256": "different-code"}},
    {"tokens_sha256": "different-tokens"},
    {"p_trace_sha256": "different-trace"},
    {"source": "null"},
    {"prompt_idx": 126},
])
def test_detection_checkpoint_rejects_incompatible_identity(tmp_path, changed):
    path = tmp_path / "wm_0125.json"
    identity = _checkpoint_identity()
    _save_detection_checkpoint(path, identity, _checkpoint_record())

    assert _load_detection_checkpoint(
        path, _checkpoint_identity(**changed)
    ) is None


def test_detection_checkpoint_rejects_target_fpr_change(tmp_path):
    path = tmp_path / "wm_0125.json"
    identity = _checkpoint_identity()
    _save_detection_checkpoint(path, identity, _checkpoint_record())
    changed_config = copy.deepcopy(identity["config"])
    changed_config["target_fpr"] = 1e-4

    assert _load_detection_checkpoint(
        path, _checkpoint_identity(config=changed_config)
    ) is None


def test_detection_checkpoint_rejects_modified_record(tmp_path):
    path = tmp_path / "wm_0125.json"
    identity = _checkpoint_identity()
    _save_detection_checkpoint(path, identity, _checkpoint_record())
    payload = json.loads(path.read_text())
    payload["record"]["decision_map"] = False
    path.write_text(json.dumps(payload))

    assert _load_detection_checkpoint(path, identity) is None
