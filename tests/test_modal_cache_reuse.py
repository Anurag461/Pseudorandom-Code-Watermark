import pytest

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
    artifact_compatibility_error,
    config_tag as online_config_tag,
    descending_prefix_grid,
    discover_online_cache_tags,
    increment_payload_from_grid,
    legacy_config_tag as legacy_online_config_tag,
    rate_strictly_above,
    summarize_generation_cost,
    summarize_map_sweep,
    validate_generation_segments,
    validate_online_watermarked_record,
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


def _make_online_cache(root, T, t=3, eta=0.05, seed=12345,
                       prompt_indices=range(3)):
    tag = online_config_tag(T, t, eta, seed)
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
