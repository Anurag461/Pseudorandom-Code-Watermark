import numpy as np
import pytest
import torch

from detectors import semantic_sha256

from online_prc import OnlinePRCKey, target_row_count

from modal_run import (
    art_path,
    config_tag,
    entropy_trace_source,
    find_complete_cache_T,
    null_dir,
    null_entropy_dir,
    null_trace_dir,
    uses_cached_generation_trace,
    validate_generation_record,
    wm_dir,
)
from modal_online_run import (
    artifact_generation_model_size,
    artifact_kv_cache_implementation,
    artifact_compatibility_error,
    compare_full_audit_results,
    config_tag as online_config_tag,
    cross_model_entropy_estimation_requests,
    cross_model_entropy_trace_identity,
    cross_model_entropy_trace_path,
    cross_model_null_entropy_dir,
    cross_model_wm_entropy_dir,
    descending_prefix_grid,
    discover_online_cache_tags,
    evaluate_prepared_map_prefixes,
    expected_null_cache_manifest,
    full_audit_shard_path,
    increment_payload_from_grid,
    legacy_config_tag as legacy_online_config_tag,
    model_cache_name as online_model_cache_name,
    model_default_batch as online_model_default_batch,
    model_default_gpu as online_model_default_gpu,
    model_default_memory_mib as online_model_default_memory_mib,
    model_cls_options as online_model_cls_options,
    model_display as online_model_display,
    normalize_model_size as normalize_online_model_size,
    merge_prepared_map_shards,
    merge_full_audit_shards,
    merge_cross_model_entropy_audit_shards,
    prepared_map_shard_path,
    prompt_detection_shards,
    rate_strictly_above,
    require_complete_cache_plan,
    resolve_model_runtime,
    resolve_null_kv_cache_implementation,
    shared_null_dir as online_null_dir,
    summarize_generation_cost,
    summarize_cross_model_entropy_workload,
    summarize_map_sweep,
    null_cache_manifest_compatibility_error,
    validate_prepared_map_shard,
    validate_full_audit_shard,
    validate_generation_model_record as validate_online_model_record,
    validate_generation_segments,
    validate_online_null_record,
    validate_online_watermarked_record,
    validate_cross_model_entropy_trace,
    validate_cross_model_entropy_audit_shard,
)


def _make_cache(root, T, prefix, indices):
    cache_dir = root / f"T{T}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index in indices:
        (cache_dir / f"{prefix}_{index:04d}.pt").touch()


def test_returns_none_without_a_complete_cache(tmp_path):
    _make_cache(tmp_path, 2048, "null", [0, 1])

    assert find_complete_cache_T(tmp_path, 1024, 3, "null") is None


def test_selects_smallest_complete_cache_at_least_requested_length(tmp_path):
    _make_cache(tmp_path, 1024, "null", range(3))
    _make_cache(tmp_path, 2048, "null", [0, 1])
    _make_cache(tmp_path, 8192, "null", range(3))

    assert find_complete_cache_T(tmp_path, 512, 3, "null") == 1024
    assert find_complete_cache_T(tmp_path, 1500, 3, "null") == 8192
    assert find_complete_cache_T(tmp_path, 8192, 3, "null") == 8192
    assert find_complete_cache_T(tmp_path, 8193, 3, "null") is None


def test_ignores_malformed_directories_and_other_prefixes(tmp_path):
    (tmp_path / "not-a-cache").mkdir()
    (tmp_path / "Tbad").mkdir()
    _make_cache(tmp_path, 4096, "wm", range(2))
    _make_cache(tmp_path, 8192, "null", range(2))

    assert find_complete_cache_T(tmp_path, 2048, 2, "null") == 8192


def test_checks_an_exact_global_prompt_shard(tmp_path):
    _make_cache(tmp_path, 8192, "null", [125, 126, 127])

    assert find_complete_cache_T(
        tmp_path, 2048, [125, 126, 127], "null"
    ) == 8192
    assert find_complete_cache_T(
        tmp_path, 2048, [124, 125, 126], "null"
    ) is None


def test_generation_model_cache_namespaces_are_isolated():
    legacy_tag = config_tag(768, 3, 0.1, 760, 768)
    eight_b_tag = config_tag(768, 3, 0.1, 760, 768, "8B")

    assert legacy_tag == "n768_t3_eta0.10_T768_r760"
    assert eight_b_tag == (
        "n768_t3_eta0.10_T768__gen-qwen3_8b_base_r760"
    )
    assert art_path(768, 3, 0.1, 760, 768) != art_path(
        768, 3, 0.1, 760, 768, "8B"
    )
    assert wm_dir(768, 3, 0.1, 760, 768) != wm_dir(
        768, 3, 0.1, 760, 768, "8B"
    )
    assert null_dir(768) == "/data/_nulls/T768"
    assert null_dir(768, "8B") == (
        "/data/_nulls/qwen3_8b_base/T768"
    )
    assert null_entropy_dir(768, "4B", "8B") == (
        "/data/_null_entropy/qwen3_8b_base/qwen3_4b_base/T768"
    )
    assert null_trace_dir(768, "8B", "8B") == (
        "/data/_null_detection_traces/qwen3_8b_base/"
        "qwen3_8b_base/T768"
    )


def test_online_replicate_seeds_have_isolated_cache_namespaces():
    baseline = online_config_tag(256, 3, 0.05, 12345)
    replicate = online_config_tag(256, 3, 0.05, 54321)

    assert baseline.endswith(
        "n256_T256_t3_eta0.05_rr99of100_sampler-poscdf-v1"
    )
    assert replicate.endswith(
        "n256_T256_t3_eta0.05_rr99of100_seed54321_sampler-poscdf-v1"
    )
    assert replicate != baseline
    assert legacy_online_config_tag(256, 3, 0.05, 12345).endswith(
        "n256_T256_t3_eta0.05_rr99of100"
    )
    assert legacy_online_config_tag(256, 3, 0.05, 12345) != baseline


def test_online_static_kv_cache_has_an_isolated_versioned_namespace():
    concat = online_config_tag(256, 3, 0.05, 12345)
    static = online_config_tag(
        256, 3, 0.05, 12345, "0.6B", "static"
    )

    assert concat.endswith("_sampler-poscdf-v1")
    assert static.endswith("_sampler-poscdf-v1_kvcache-static-v1")
    assert concat != static


def test_online_generation_model_runtime_and_namespaces_are_isolated():
    point_six_tag = online_config_tag(768, 3, 0.1, 12345, "0.6B")
    eight_b_tag = online_config_tag(768, 3, 0.1, 12345, "8b")
    fourteen_b_tag = online_config_tag(768, 3, 0.1, 12345, "14b")

    assert point_six_tag.startswith(
        "online_causal_prc_v1/qwen3_0p6b_base/"
    )
    assert eight_b_tag.startswith(
        "online_causal_prc_v1/qwen3_8b_base/"
    )
    assert fourteen_b_tag.startswith(
        "online_causal_prc_v1/qwen3_14b_base/"
    )
    assert len({point_six_tag, eight_b_tag, fourteen_b_tag}) == 3
    assert online_null_dir(768) == "/data/_nulls/T768"
    assert online_null_dir(768, "8") == (
        "/data/_nulls/qwen3_8b_base/T768"
    )
    assert online_null_dir(768, "14") == (
        "/data/_nulls/qwen3_14b_base/T768"
    )
    assert normalize_online_model_size("8") == "8B"
    assert online_model_display("8B") == "Qwen3-8B-Base"
    assert online_model_cache_name("0.6") == "qwen3_0p6b_base"
    assert online_model_default_gpu("8B") == "H100"
    assert online_model_default_batch("8B") == 25
    assert normalize_online_model_size("14") == "14B"
    assert online_model_display("14B") == "Qwen3-14B-Base"
    assert online_model_cache_name("14") == "qwen3_14b_base"
    assert online_model_default_gpu("14B") == "H100"
    assert online_model_default_batch("14B") == 10
    assert online_model_default_memory_mib("14B") == 65_536
    assert online_model_default_memory_mib("8B") == 0
    assert resolve_model_runtime("8b") == ("8B", 25, "H100")
    assert resolve_model_runtime("14b") == ("14B", 10, "H100")
    assert resolve_model_runtime("8B", 10, "H100:80GB") == (
        "8B", 10, "H100:80GB"
    )
    assert online_model_cls_options("14B", "H100", 10) == {
        "gpu": "H100",
        "max_containers": 10,
        "memory": 65_536,
    }
    assert online_model_cls_options("8B", "H100", 5) == {
        "gpu": "H100",
        "max_containers": 5,
    }
    with pytest.raises(ValueError, match="must be one of"):
        normalize_online_model_size("4B")
    with pytest.raises(ValueError, match="batch must be nonnegative"):
        resolve_model_runtime("8B", -1, "H100")


def _make_online_cache(root, T, t=3, eta=0.05, seed=12345,
                       prompt_indices=range(3), model_size="0.6B"):
    tag = online_config_tag(T, t, eta, seed, model_size)
    directory = root / tag
    (directory / "wm").mkdir(parents=True)
    (directory / "artifacts.pt").touch()
    for index in prompt_indices:
        (directory / "wm" / f"wm_{index:04d}.pt").touch()
    return tag


def test_online_cache_discovery_classifies_both_reuse_directions(tmp_path):
    tag_256 = _make_online_cache(tmp_path, 256)
    tag_400 = _make_online_cache(tmp_path, 400)
    _make_online_cache(tmp_path, 512, seed=54321)
    malformed = tmp_path / "online_causal_prc_v1" / "qwen3_0p6b_base" / "junk"
    malformed.mkdir(parents=True)

    found = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345,
    )
    assert [(item["T"], item["relation"]) for item in found] == [
        (256, "shorter"),
        (400, "longer"),
    ]
    assert [item["tag"] for item in found] == [tag_256, tag_400]


def test_online_cache_discovery_never_crosses_generation_models(tmp_path):
    point_six_tag = _make_online_cache(tmp_path, 256, model_size="0.6B")
    eight_b_tag = _make_online_cache(tmp_path, 400, model_size="8B")

    point_six = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345, generation_model_size="0.6B",
    )
    eight_b = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345, generation_model_size="8B",
    )

    assert [item["tag"] for item in point_six] == [point_six_tag]
    assert [item["tag"] for item in eight_b] == [eight_b_tag]


def test_online_cache_discovery_never_crosses_kv_implementations(tmp_path):
    concat_tag = _make_online_cache(tmp_path, 256)
    static_tag = online_config_tag(
        400, 3, 0.05, 12345, "0.6B", "static"
    )
    static_dir = tmp_path / static_tag
    (static_dir / "wm").mkdir(parents=True)
    (static_dir / "artifacts.pt").touch()

    concat = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345, kv_cache_implementation="concat",
    )
    static = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345, kv_cache_implementation="static",
    )

    assert [item["tag"] for item in concat] == [concat_tag]
    assert [item["tag"] for item in static] == [static_tag]


def test_online_cache_discovery_requires_artifact_and_exact_config(tmp_path):
    _make_online_cache(tmp_path, 256, eta=0.05)
    missing_artifact_tag = online_config_tag(400, 3, 0.05, 12345)
    missing_artifact_dir = tmp_path / missing_artifact_tag / "wm"
    missing_artifact_dir.mkdir(parents=True)
    _make_online_cache(tmp_path, 512, t=5, eta=0.05)

    found = discover_online_cache_tags(
        str(tmp_path), requested_T=300, t=3, eta=0.05,
        experiment_seed=12345,
    )
    assert [item["T"] for item in found] == [256]


def test_online_cache_discovery_keeps_legacy_caches_available_as_sources(tmp_path):
    legacy_tag = legacy_online_config_tag(400, 3, 0.05, 12345)
    directory = tmp_path / legacy_tag
    (directory / "wm").mkdir(parents=True)
    (directory / "artifacts.pt").touch()

    found = discover_online_cache_tags(
        str(tmp_path), requested_T=256, t=3, eta=0.05,
        experiment_seed=12345,
    )
    assert [(item["T"], item["tag"], item["relation"])
            for item in found] == [(400, legacy_tag, "longer")]


def test_descending_prefix_grid_includes_both_endpoints():
    assert descending_prefix_grid(512, 400, 16) == [
        512, 496, 480, 464, 448, 432, 416, 400,
    ]
    with pytest.raises(ValueError, match="divisible"):
        descending_prefix_grid(512, 401, 16)
    with pytest.raises(ValueError, match="positive"):
        descending_prefix_grid(512, 400, 0)


def test_strict_map_tpr_crossing_requires_451_of_500():
    assert not rate_strictly_above(450, 500, 0.90)
    assert rate_strictly_above(451, 500, 0.90)
    with pytest.raises(ValueError, match="valid rate"):
        rate_strictly_above(501, 500, 0.90)


def test_map_sweep_selects_contiguous_last_pass_and_flags_nonmonotonicity():
    rows = [
        {"n": 512, "tp": 460, "watermarked_total": 500, "tpr": 0.920},
        {"n": 496, "tp": 455, "watermarked_total": 500, "tpr": 0.910},
        {"n": 480, "tp": 451, "watermarked_total": 500, "tpr": 0.902},
        {"n": 464, "tp": 450, "watermarked_total": 500, "tpr": 0.900},
        {"n": 448, "tp": 452, "watermarked_total": 500, "tpr": 0.904},
    ]
    summary = summarize_map_sweep(rows, 0.90)

    assert summary["last_passing_n_descending"] == 480
    assert summary["last_passing_tp"] == 451
    assert summary["next_shorter_n"] == 464
    assert summary["next_shorter_above_target"] is False
    assert summary["lowest_passing_n_anywhere"] == 448
    assert summary["monotonicity_violations"] == [{
        "lower_n": 448,
        "lower_tpr": 0.904,
        "higher_n": 464,
        "higher_tpr": 0.9,
    }]


def test_adaptive_map_sweep_never_scores_after_first_failure(monkeypatch):
    calls = []

    def fake_score(prepared, length, **_kwargs):
        prompt_idx = int(prepared["prompt_idx"])
        calls.append((prompt_idx, int(length)))
        decision = int(length) == 512 or prompt_idx < 3
        return {
            "decision": decision,
            "length": int(length),
            "n": int(length),
            "T": int(length),
            "r": int(length) - 2,
            "free_coordinates": 2,
        }

    monkeypatch.setattr(
        "detectors.score_prepared_online_map_prefix", fake_score
    )
    prepared = [{
        "prompt_idx": index,
        "prepared": {"prompt_idx": index},
    } for index in range(4)]
    result = evaluate_prepared_map_prefixes(
        prepared,
        [512, 496, 480, 464],
        fpr=1e-3,
        target_rate=0.90,
        stop_after_first_below=True,
    )

    assert result["evaluated_lengths"] == [512, 496]
    assert result["unevaluated_lengths"] == [480, 464]
    assert result["first_below_n"] == 496
    assert result["stopped_after_first_below"] is True
    assert result["rows"][0]["tp"] == 4
    assert result["rows"][1]["tp"] == 3
    assert {length for _, length in calls} == {512, 496}
    assert all("480" not in item["map_scores"] for item in result["results"])


def _prepared_shard_payload(key, prompt_indices, maximum=32):
    row_count = target_row_count(maximum, key)
    records = []
    for index in prompt_indices:
        signed = np.linspace(-0.25, 0.75, row_count) + index * 0.01
        squared = np.linspace(0.1, 0.9, row_count) + index * 0.001
        records.append({
            "prompt_idx": int(index),
            "signed_check_values": signed,
            "squared_check_values": squared,
        })
    return {
        "prepared_map_shard_schema_version": 1,
        "result_kind": "online_map_prepared_prompt_shard",
        "source_tag": "test/source",
        "maximum_length": int(maximum),
        "source_artifact_fingerprint": "artifact-fingerprint",
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": "code-fingerprint",
        "prompt_indices": [int(index) for index in prompt_indices],
        "num_prompts": len(prompt_indices),
        "records": records,
    }


def test_prompt_detection_shards_are_stable_and_paths_are_versioned():
    assert prompt_detection_shards(list(range(7)), 3) == [
        [0, 1, 2], [3, 4, 5], [6],
    ]
    with pytest.raises(ValueError, match="unique"):
        prompt_detection_shards([0, 0], 1)
    with pytest.raises(ValueError, match="positive"):
        prompt_detection_shards([0], 0)

    first = prepared_map_shard_path(
        "tag", 64, [0, 1], "artifact", "code"
    )
    repeated = prepared_map_shard_path(
        "tag", 64, [0, 1], "artifact", "code"
    )
    regrouped = prepared_map_shard_path(
        "tag", 64, [1, 0], "artifact", "code"
    )
    assert first == repeated
    assert "prepared_map_v1" in first
    assert regrouped != first


def test_cache_only_guard_refuses_any_missing_generation_records():
    require_complete_cache_plan({"wm_missing": [], "null_missing": []})
    with pytest.raises(FileNotFoundError, match=r"watermarked=\[2\]"):
        require_complete_cache_plan({
            "wm_missing": [2],
            "null_missing": [],
        })
    with pytest.raises(FileNotFoundError, match=r"null=\[4, 5\]"):
        require_complete_cache_plan({
            "wm_missing": [],
            "null_missing": [4, 5],
        })


def _full_audit_shard_payload(prompt_indices, prefix_T=64):
    results = []
    for watermark in (True, False):
        for index in prompt_indices:
            results.append({
                "prompt_idx": int(index),
                "watermark": watermark,
                "scores": {
                    weight: {
                        "decision": bool(watermark and index % 2 == 0),
                        "length": int(prefix_T),
                    }
                    for weight in ("map", "entropy", "naive")
                },
            })
    return {
        "full_audit_shard_schema_version": 1,
        "result_kind": "online_full_audit_prompt_shard",
        "tag": "test/tag",
        "watermarked_source_tag": "test/source",
        "T": int(prefix_T),
        "null_T": 80,
        "target_fpr": 1e-3,
        "fpr_policy": "one_shot",
        "artifact_fingerprint": "artifact",
        "watermarked_source_artifact_fingerprint": "source-artifact",
        "online_key_sha256": "key",
        "code_fingerprint_sha256": "code",
        "prompt_indices": [int(index) for index in prompt_indices],
        "num_prompts": len(prompt_indices),
        "results": results,
    }


def test_full_audit_prompt_shards_validate_and_merge_in_serial_order():
    left = _full_audit_shard_payload([0, 1])
    right = _full_audit_shard_payload([2, 3])
    validation = {
        "tag": "test/tag",
        "watermarked_source_tag": "test/source",
        "prefix_T": 64,
        "null_T": 80,
        "fpr": 1e-3,
        "artifact_fingerprint": "artifact",
        "watermarked_source_fingerprint": "source-artifact",
        "online_key_sha256": "key",
        "code_fingerprint_sha256": "code",
    }
    assert validate_full_audit_shard(left, **validation) == [0, 1]
    assert validate_full_audit_shard(right, **validation) == [2, 3]

    merged = merge_full_audit_shards([right, left], [0, 1, 2, 3])
    assert [
        (result["watermark"], result["prompt_idx"])
        for result in merged
    ] == [
        (True, 0), (True, 1), (True, 2), (True, 3),
        (False, 0), (False, 1), (False, 2), (False, 3),
    ]

    path = full_audit_shard_path(
        "test/tag", 64, 80, [0, 1], "artifact", "source", "code", 1e-3
    )
    assert "full_audit_shards_v1" in path
    assert "shard-0000-0001-count2" in path


def test_full_audit_prompt_shards_reject_duplicates_and_bad_scores():
    left = _full_audit_shard_payload([0, 1])
    duplicate = _full_audit_shard_payload([1, 2])
    with pytest.raises(ValueError, match="duplicate watermark=True"):
        merge_full_audit_shards([left, duplicate], [0, 1, 2])
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_full_audit_shards([left], [0, 1, 2])

    left["results"][0]["scores"]["map"]["length"] = 63
    with pytest.raises(ValueError, match="wrong length"):
        validate_full_audit_shard(
            left,
            tag="test/tag",
            watermarked_source_tag="test/source",
            prefix_T=64,
            null_T=80,
            fpr=1e-3,
            artifact_fingerprint="artifact",
            watermarked_source_fingerprint="source-artifact",
            online_key_sha256="key",
            code_fingerprint_sha256="code",
        )


def test_full_audit_comparison_allows_only_tiny_float_roundoff():
    left = _full_audit_shard_payload([0])["results"]
    right = _full_audit_shard_payload([0])["results"]
    right[0]["scores"]["entropy"]["statistic"] = 1.0
    left[0]["scores"]["entropy"]["statistic"] = 1.0 + 5e-15
    comparison = compare_full_audit_results(left, right)
    assert comparison["equivalent"] is True
    assert comparison["max_abs_float_difference"] == pytest.approx(5e-15)

    right[0]["scores"]["entropy"]["decision"] = False
    comparison = compare_full_audit_results(left, right)
    assert comparison["equivalent"] is False
    assert any(path.endswith("decision") for path in comparison["mismatches"])


def test_prepared_prompt_shards_merge_exactly_in_requested_order():
    key = OnlinePRCKey(3, 0.05, b"support", b"otp")
    left = _prepared_shard_payload(key, [0, 1])
    right = _prepared_shard_payload(key, [2, 3])
    complete = _prepared_shard_payload(key, [0, 1, 2, 3])
    row_count = target_row_count(32, key)
    for payload in (left, right, complete):
        validate_prepared_map_shard(
            payload,
            source_tag="test/source",
            maximum_length=32,
            artifact_fingerprint="artifact-fingerprint",
            online_key_sha256=key.fingerprint,
            code_fingerprint_sha256="code-fingerprint",
            expected_row_count=row_count,
        )

    sharded = merge_prepared_map_shards(
        [right, left], [0, 1, 2, 3], key, 32
    )
    serial = merge_prepared_map_shards(
        [complete], [0, 1, 2, 3], key, 32
    )
    assert [record["prompt_idx"] for record in sharded] == [0, 1, 2, 3]

    sharded_scores = evaluate_prepared_map_prefixes(
        sharded, [32, 24, 16], 1e-3, 0.90,
        stop_after_first_below=False,
    )
    serial_scores = evaluate_prepared_map_prefixes(
        serial, [32, 24, 16], 1e-3, 0.90,
        stop_after_first_below=False,
    )
    assert sharded_scores == serial_scores


def test_prepared_prompt_shard_validation_rejects_gaps_and_duplicates():
    key = OnlinePRCKey(3, 0.05, b"support", b"otp")
    left = _prepared_shard_payload(key, [0, 1])
    duplicate = _prepared_shard_payload(key, [1, 2])

    with pytest.raises(ValueError, match="duplicate prompt index 1"):
        merge_prepared_map_shards(
            [left, duplicate], [0, 1, 2], key, 32
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        merge_prepared_map_shards([left], [0, 1, 2], key, 32)
    with pytest.raises(ValueError, match="code_fingerprint"):
        validate_prepared_map_shard(
            left,
            source_tag="test/source",
            maximum_length=32,
            artifact_fingerprint="artifact-fingerprint",
            online_key_sha256=key.fingerprint,
            code_fingerprint_sha256="different-code",
            expected_row_count=target_row_count(32, key),
        )


def test_generation_cost_summary_tracks_replay_suffix_and_gpu_seconds():
    summary = summarize_generation_cost({
        "wm": [{
            "generated": 64,
            "cached": 0,
            "batch": 64,
            "resumed": True,
            "resume_prefix_T": 400,
            "suffix_tokens_generated": 64 * 112,
            "seconds": 12.5,
        }],
        "null": [{
            "generated": 2,
            "cached": 0,
            "batch": 2,
            "seconds": 1.5,
        }],
    }, source_n=512, gpu="A10G")

    assert summary["gpu"] == "A10G"
    assert summary["measured_gpu_method_seconds"] == pytest.approx(14.0)
    assert summary["replayed_prefix_tokens"] == 64 * 400
    assert summary["generated_suffix_tokens"] == 64 * 112
    assert summary["generated_null_tokens"] == 2 * 512
    assert summary["model_token_positions_processed"] == (
        64 * 400 + 64 * 112 + 2 * 512
    )


def test_increment_payload_extracts_one_complete_prefix_result():
    grid = {
        "result_schema_version": 2,
        "timestamp_utc": "2026-08-16T00:00:00+00:00",
        "scheme": "online_causal_prc_v1",
        "source_tag": "source",
        "source_T": 512,
        "t": 3,
        "eta": 0.05,
        "num_prompts": 2,
        "prompt_indices": [0, 1],
        "rows": [{
            "n": 448,
            "T": 448,
            "r": 444,
            "free_coordinates": 4,
            "tp": 1,
            "watermarked_total": 2,
            "tpr": 0.5,
            "prefix_online_support_sha256": "a" * 64,
        }],
        "results": [
            {
                "prompt_idx": 0,
                "map_scores": {"448": {"decision": True, "length": 448}},
            },
            {
                "prompt_idx": 1,
                "map_scores": {"448": {"decision": False, "length": 448}},
            },
        ],
    }

    increment = increment_payload_from_grid(grid, 448)
    assert increment["result_kind"] == "saved_online_map_prefix"
    assert increment["n"] == increment["T"] == 448
    assert increment["counts"]["map"] == {
        "tp": 1,
        "watermarked_total": 2,
    }
    assert [
        result["scores"]["map"]["decision"]
        for result in increment["results"]
    ] == [True, False]

    with pytest.raises(ValueError, match="exactly one"):
        increment_payload_from_grid(grid, 432)


def test_increment_payload_rejects_incomplete_prompt_scores():
    grid = {
        "rows": [{
            "n": 448,
            "r": 444,
            "free_coordinates": 4,
            "tp": 1,
            "watermarked_total": 1,
            "tpr": 1.0,
            "prefix_online_support_sha256": "a" * 64,
        }],
        "results": [{"prompt_idx": 0, "map_scores": {}}],
    }
    with pytest.raises(ValueError, match="lacks MAP score"):
        increment_payload_from_grid(grid, 448)


def test_generation_segments_must_cover_record_exactly():
    validate_generation_segments([
        {"start": 0, "end": 256, "sampler_version": "legacy"},
        {"start": 256, "end": 400, "sampler_version": "position-v1"},
    ], 400)

    with pytest.raises(ValueError, match="contiguous"):
        validate_generation_segments([
            {"start": 0, "end": 255, "sampler_version": "legacy"},
            {"start": 256, "end": 400, "sampler_version": "position-v1"},
        ], 400)
    with pytest.raises(ValueError, match="expected 400"):
        validate_generation_segments([
            {"start": 0, "end": 256, "sampler_version": "legacy"},
        ], 400)
    with pytest.raises(ValueError, match="sampler_version"):
        validate_generation_segments([
            {"start": 0, "end": 400},
        ], 400)


def test_online_artifact_compatibility_allows_only_length_fields_to_differ():
    import torch

    key = {"check_weight": 3, "noise_rate": 0.05}
    invariant_config = {
        "scheme": "online_causal_prc_v1",
        "check_weight": 3,
        "noise_rate": 0.05,
        "row_rate_numerator": 99,
        "row_rate_denominator": 100,
        "schedule_version": "schedule-v1",
        "support_sampler_version": "support-v1",
        "stopping_policy": "forced_length_v1",
        "generation_model_size": "0.6B",
        "generation_model": "Qwen3-0.6B-Base",
        "keygen_seed": 12345,
        "partition_seed": 12345,
    }
    target = {
        "T": 400,
        "online_key": key,
        "experiment_seed": 12345,
        "prompt_ids_list": [[1, 2], [3, 4]],
        "partition": torch.tensor([[1, 0], [0, 1]]),
        "config_sig": {**invariant_config, "T": 400, "n": 400},
    }
    source = {
        **target,
        "T": 256,
        "config_sig": {**invariant_config, "T": 256, "n": 256},
    }
    assert artifact_compatibility_error(target, source) is None

    wrong_key = {**source, "online_key": {**key, "noise_rate": 0.1}}
    assert artifact_compatibility_error(target, wrong_key) == "online key differs"
    wrong_partition = {
        **source,
        "partition": torch.tensor([[0, 1], [1, 0]]),
    }
    assert artifact_compatibility_error(target, wrong_partition) == (
        "token partition differs"
    )
    assert artifact_kv_cache_implementation(target) == "concat"
    static_source = {
        **source,
        "kv_cache_implementation": "static",
        "config_sig": {
            **source["config_sig"],
            "kv_cache_implementation": "static",
            "kv_cache_version": "static-v1",
        },
    }
    assert artifact_kv_cache_implementation(static_source) == "static"
    assert artifact_compatibility_error(target, static_source) == (
        "KV cache implementation differs"
    )


def test_online_record_validation_accepts_legacy_provenance_and_rejects_key_mix():
    import numpy as np
    import torch
    from detectors import build_prc_generation_record, tensor_sha256
    from online_prc import (
        OnlinePRCEncoder,
        OnlinePRCKey,
        derive_document_seed,
        support_sha256,
        target_row_count,
    )

    length = 4
    experiment_seed = 77
    prompt_index = 0
    key = OnlinePRCKey.from_seed(
        experiment_seed, check_weight=3, noise_rate=0.05
    )
    partition = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    prompt = torch.tensor([1, 0], dtype=torch.long)
    codeword = OnlinePRCEncoder(
        key, [derive_document_seed(experiment_seed, prompt_index)]
    ).encode_to_length(length)[0]
    tokens = torch.as_tensor(codeword, dtype=torch.long)
    artifact = {
        "T": length,
        "online_key": key.to_dict(),
        "experiment_seed": experiment_seed,
        "prompt_ids_list": [prompt.tolist()],
        "partition": partition,
        "support_sha256": support_sha256(length, key),
        "artifact_fingerprint": "a" * 64,
    }
    record = build_prc_generation_record(
        prompt,
        tokens,
        np.full(length, 0.5),
        partition,
        length,
        True,
        encoding_key_fingerprint=key.fingerprint,
        prc_codeword_bits=codeword,
        base_lm_entropy=np.ones(length),
        base_token_logprob=np.zeros(length),
        partition_fingerprint=tensor_sha256(partition),
    )
    record.update({
        "prompt_idx": prompt_index,
        "scheme": "online_causal_prc_v1",
        "stopping_policy": "forced_length_v1",
        "realized_length": length,
        "realized_r": target_row_count(length, key),
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "online_key_sha256": key.fingerprint,
        "online_support_sha256": support_sha256(length, key),
        "generation_model_size": "0.6B",
        "generation_model": "Qwen3-0.6B-Base",
        "artifact_seed": experiment_seed,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
    })

    segments = validate_online_watermarked_record(
        record, artifact, prompt_index
    )
    assert segments == [{
        "start": 0,
        "end": length,
        "sampler_version": "legacy_torch_global_v1",
        "legacy_inferred": True,
    }]

    wrong_key_record = {**record, "online_key_sha256": "b" * 64}
    with pytest.raises(ValueError, match="online_key_sha256"):
        validate_online_watermarked_record(
            wrong_key_record, artifact, prompt_index
        )

    static_artifact = {
        **artifact,
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
    }
    static_record = {
        **record,
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
    }
    validate_online_watermarked_record(
        static_record, static_artifact, prompt_index
    )
    with pytest.raises(ValueError, match="kv_cache_implementation"):
        validate_online_watermarked_record(
            record, static_artifact, prompt_index
        )


def test_online_8b_null_validation_requires_model_partition_and_prompt():
    import numpy as np
    import torch
    from detectors import tensor_sha256

    partition = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    artifact = {
        "T": 4,
        "prompt_ids_list": [[1, 0]],
        "partition": partition,
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
        "config_sig": {
            "generation_model_size": "8B",
            "generation_model": "Qwen3-8B-Base",
        },
    }
    record = {
        "watermark": False,
        "tokens": torch.tensor([0, 1, 0, 1]),
        "p_trace": np.full(4, 0.5),
        "prompt_token_ids": torch.tensor([1, 0]),
        "partition_sha256": tensor_sha256(partition),
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
    }

    assert artifact_generation_model_size(artifact) == "8B"
    validate_online_null_record(record, artifact, 0, 4)
    legacy_partition_record = {
        key: value for key, value in record.items()
        if key != "partition_sha256"
    }
    validate_online_null_record(legacy_partition_record, artifact, 0, 4)
    validate_online_model_record({}, "0.6B", "legacy null", 0)

    with pytest.raises(ValueError, match="lacks generation-model metadata"):
        validate_online_null_record(
            {
                key: value for key, value in record.items()
                if key not in ("generation_model_size", "generation_model")
            },
            artifact,
            0,
            4,
        )
    with pytest.raises(ValueError, match="different token partition"):
        validate_online_null_record(
            {**record, "partition_sha256": "0" * 64}, artifact, 0, 4
        )
    with pytest.raises(ValueError, match="wrong prompt"):
        validate_online_null_record(
            {**record, "prompt_token_ids": torch.tensor([0, 1])},
            artifact,
            0,
            4,
        )


def test_static_null_manifest_is_eta_independent_and_versioned():
    import torch

    artifact = {
        "T": 8,
        "prompt_ids_list": [[1, 2], [3, 4]],
        "partition": torch.tensor([[1, 0], [0, 1]], dtype=torch.float32),
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
        "experiment_seed": 111,
        "online_key": {"noise_rate": 0.20},
    }
    same_null_inputs = {
        **artifact,
        "experiment_seed": 999,
        "online_key": {"noise_rate": 0.05},
    }
    manifest = expected_null_cache_manifest(artifact, 8, "static")

    assert manifest["kv_cache_implementation"] == "static"
    assert manifest["kv_cache_version"] == "static-v1"
    assert manifest["generation_sampler_version"] == (
        "torch_multinomial_global_v1"
    )
    assert manifest == expected_null_cache_manifest(
        same_null_inputs, 8, "static"
    )
    assert null_cache_manifest_compatibility_error(
        manifest, same_null_inputs, 8
    ) is None
    assert "kv_cache_implementation differs" in (
        null_cache_manifest_compatibility_error(
            manifest, artifact, 8, "concat"
        ) or ""
    )


def test_null_cache_implementation_is_explicit_or_inherits_watermarked():
    assert resolve_null_kv_cache_implementation("", "static") == "static"
    assert resolve_null_kv_cache_implementation("concat", "static") == "concat"
    assert resolve_null_kv_cache_implementation("preallocated", "concat") == (
        "static"
    )


def test_versioned_static_null_record_supports_exact_prefix_validation():
    import torch
    from detectors import tensor_sha256

    partition = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    artifact = {
        "T": 4,
        "prompt_ids_list": [[1, 0]],
        "partition": partition,
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
    }
    record = {
        "watermark": False,
        "prc_codeword_bits": None,
        "tokens": torch.tensor([0, 1, 0, 1]),
        "p_trace": np.full(4, 0.5),
        "base_lm_entropy": np.full(4, 1.0, dtype=np.float32),
        "base_token_logprob": np.full(4, -1.0, dtype=np.float32),
        "prompt_token_ids": torch.tensor([1, 0]),
        "partition_sha256": tensor_sha256(partition),
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
        "prompt_idx": 0,
        "stopping_policy": "forced_length_v1",
        "source_T": 4,
        "realized_length": 4,
        "generation_sampler_version": "torch_multinomial_global_v1",
        "generation_rng_policy": "torch_multinomial_global_v1",
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
    }

    validate_online_null_record(
        record,
        artifact,
        0,
        2,
        source_length=4,
        expected_kv_cache_implementation="static",
        require_provenance=True,
    )
    # The fixed runner shares this null namespace and ignores the additional
    # online/static provenance when the model identity matches.
    validate_generation_record(record, "8B", "null", 0)
    with pytest.raises(ValueError, match="kv_cache_implementation"):
        validate_online_null_record(
            {**record, "kv_cache_implementation": "concat"},
            artifact,
            0,
            2,
            source_length=4,
            expected_kv_cache_implementation="static",
            require_provenance=True,
        )
    with pytest.raises(ValueError, match="PRC codeword bits"):
        validate_online_null_record(
            {**record, "prc_codeword_bits": np.zeros(4, dtype=np.uint8)},
            artifact,
            0,
            2,
            source_length=4,
            expected_kv_cache_implementation="static",
            require_provenance=True,
        )


def test_same_8b_model_reuses_generation_probability_trace():
    assert uses_cached_generation_trace("8B", "8B")
    assert uses_cached_generation_trace("8", "8b")
    assert not uses_cached_generation_trace("4B", "8B")
    assert entropy_trace_source("8B", "8B") == (
        "cached_generation_p_trace"
    )
    assert entropy_trace_source("4B", "8B") == "estimated_4B"


def test_8b_cache_records_require_matching_model_metadata():
    record = {
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
    }
    validate_generation_record(record, "8B", "null", 12)
    validate_generation_record({}, "0.6B", "legacy-null", 12)

    with pytest.raises(ValueError, match="lacks generation-model metadata"):
        validate_generation_record({}, "8B", "null", 12)
    with pytest.raises(ValueError, match="generated by Qwen3-4B-Base"):
        validate_generation_record(
            {"generation_model_size": "4B"}, "8B", "null", 12
        )
    with pytest.raises(ValueError, match="generation model label"):
        validate_generation_record(
            {
                "generation_model_size": "8B",
                "generation_model": "Qwen3-4B-Base",
            },
            "8B",
            "null",
            12,
        )


def test_online_cross_model_entropy_trace_namespaces_are_model_qualified():
    source_tag = online_config_tag(
        1280, 3, 0.05, 12345, "14B", "static"
    )
    assert cross_model_wm_entropy_dir(source_tag, 880, "0.6B") == (
        f"/data/{source_tag}/cross_model_entropy_v1/"
        "qwen3_0p6b_base/T880/wm"
    )
    assert cross_model_null_entropy_dir(1808, "0.6B", "14B") == (
        "/data/_online_null_cross_model_entropy/qwen3_14b_base/"
        "qwen3_0p6b_base/T1808"
    )
    assert cross_model_entropy_trace_path(
        "null", 7, 1808, "0.6B", "14B"
    ).endswith("qwen3_0p6b_base/T1808/null_0007.pt")
    with pytest.raises(ValueError, match="require source_tag"):
        cross_model_entropy_trace_path("wm", 7, 880, "0.6B", "14B")


def test_online_cross_model_entropy_trace_validation_binds_inputs_and_hash():
    identity = {
        "source": "wm",
        "prompt_index": 3,
        "trace_T": 4,
        "generation_model_size": "14B",
        "entropy_model_size": "0.6B",
        "partition_sha256": "partition-hash",
        "prompt_sha256": "prompt-hash",
        "tokens_sha256": "token-prefix-hash",
        "source_artifact_fingerprint": "artifact-hash",
    }
    probabilities = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    payload = {
        **cross_model_entropy_trace_identity(**identity),
        "p_trace": probabilities,
        "p_trace_sha256": semantic_sha256(probabilities),
    }
    assert np.array_equal(
        validate_cross_model_entropy_trace(payload, **identity),
        probabilities,
    )
    with pytest.raises(ValueError, match="tokens_sha256"):
        validate_cross_model_entropy_trace(
            payload, **{**identity, "tokens_sha256": "different"}
        )
    with pytest.raises(ValueError, match="hash is inconsistent"):
        validate_cross_model_entropy_trace(
            {**payload, "p_trace": probabilities + 0.01}, **identity
        )
    with pytest.raises(ValueError, match="missing full-vocabulary entropy"):
        validate_cross_model_entropy_trace(
            payload, require_full_entropy=True, **identity
        )
    entropies = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    enriched = {
        **payload,
        "full_entropy_trace": entropies,
        "full_entropy_trace_sha256": semantic_sha256(entropies),
    }
    assert np.array_equal(
        validate_cross_model_entropy_trace(
            enriched, require_full_entropy=True, **identity
        ),
        probabilities,
    )


def test_online_cross_model_entropy_campaign_uses_one_combined_queue():
    plan = {
        "audits": [
            {
                "label": "eta0.05-n880",
                "source_tag": "eta05-source",
                "prefix_T": 880,
                "trace_T": 1024,
                "require_full_entropy": True,
                "wm_trace_missing": [0, 1, 2],
            },
            {
                "label": "eta0.10-n1808",
                "source_tag": "eta10-source",
                "prefix_T": 1808,
                "wm_trace_missing": [0, 1],
            },
        ],
        "null_T": 1808,
        "require_null_full_entropy": True,
        "null_trace_missing": [0, 1, 2, 3],
    }
    requests = cross_model_entropy_estimation_requests(plan, 2)
    assert len(requests) == 5
    assert [request["source"] for request in requests] == [
        "wm", "wm", "wm", "null", "null"
    ]
    assert all(len(request["prompt_indices"]) <= 2 for request in requests)
    assert requests[0]["trace_T"] == 1024
    assert requests[0]["require_full_entropy"] is True
    assert requests[-1]["require_full_entropy"] is True
    assert summarize_cross_model_entropy_workload(plan) == {
        "watermarked_teacher_forced_token_positions": 6688,
        "null_teacher_forced_token_positions": 7232,
        "teacher_forced_token_positions": 13920,
        "watermarked_trace_records_missing": 5,
        "null_trace_records_missing": 4,
    }


def test_online_cross_model_map_entropy_shards_validate_and_merge_order():
    validation = {
        "source_tag": "source-tag",
        "prefix_T": 880,
        "null_T": 1808,
        "null_trace_T": 1808,
        "fpr": 1e-3,
        "generation_model_size": "14B",
        "entropy_model_size": "0.6B",
        "artifact_fingerprint": "artifact",
        "online_key_sha256": "key",
        "code_fingerprint_sha256": "code",
    }

    def shard(index):
        return {
            "cross_model_entropy_audit_shard_schema_version": 2,
            "result_kind": "online_cross_model_map_entropy_prompt_shard",
            "source_tag": "source-tag",
            "T": 880,
            "null_T": 1808,
            "null_trace_T": 1808,
            "target_fpr": 1e-3,
            "fpr_policy": "one_shot",
            "generation_model_size": "14B",
            "entropy_model_size": "0.6B",
            "artifact_fingerprint": "artifact",
            "online_key_sha256": "key",
            "code_fingerprint_sha256": "code",
            "prompt_indices": [index],
            "num_prompts": 1,
            "results": [
                {
                    "prompt_idx": index,
                    "watermark": watermark,
                    "scores": {
                        "map": {
                            "decision": watermark,
                            "length": 880,
                        },
                        "entropy": {
                            "decision": watermark,
                            "length": 880,
                        },
                    },
                }
                for watermark in (True, False)
            ],
        }

    first, second = shard(0), shard(1)
    assert validate_cross_model_entropy_audit_shard(
        first, **validation
    ) == [0]
    merged = merge_cross_model_entropy_audit_shards(
        [second, first], [0, 1]
    )
    assert [
        (result["watermark"], result["prompt_idx"]) for result in merged
    ] == [(True, 0), (True, 1), (False, 0), (False, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        merge_cross_model_entropy_audit_shards([first, first], [0])
