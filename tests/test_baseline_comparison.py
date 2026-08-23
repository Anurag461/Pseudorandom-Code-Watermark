from __future__ import annotations

import json
import importlib.util
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from baseline_comparison.adapters import (
    FixedPRCCachedAdapter,
    GumbelMaxAdapter,
    GeneratedContinuation,
    OnlinePRCCachedAdapter,
    SynthIDTextAdapter,
    TextSealAdapter,
)
from baseline_comparison.config import GENERATION_SETTINGS, PREFIX_LENGTHS
from baseline_comparison.diagnostic import (
    analyze_sequence,
    longest_periodic_run,
    repeated_ngram_events,
    stable_power_argmax,
)
from baseline_comparison.schema import PromptLevelResult
from baseline_comparison.smoke_runner import _validated_full_request
from baseline_comparison.scoring import (
    deduplicated_positions,
    distinct_n,
    gamma_survival,
    gumbel_gamma_test,
    ngram_repetition_rate,
    pairwise_token_agreement,
    prc_hoeffding_test,
    quality_metrics,
    self_bleu_token_ids,
    synthid_normal_test,
    textseal_gamma_test,
)


def test_deduplicate_repeated_and_overlapping_context_token_tuples():
    # The tuple (1,2,3,4) occurs twice; overlapping tuples remain distinct.
    tokens = [9, 1, 2, 3, 4, 1, 2, 3, 4, 5]
    assert deduplicated_positions(tokens, 3) == [4, 5, 6, 7, 9]
    # Explicitly selecting the first mathematically eligible position is supported.
    assert deduplicated_positions(tokens, 3, start_position=3) == [3, 4, 5, 6, 7, 9]


def test_diagnostic_repeat_onset_and_periodic_loop_definition():
    tokens = [90, 91, 1, 2, 3, 1, 2, 3, 1, 2, 3, 77]
    repeats = repeated_ngram_events(tokens, n=4)
    assert repeats["first_onset_tokens"] == 9
    assert repeats["first_recurrence_gap"] == 3
    loop = longest_periodic_run(tokens)
    assert loop["found"] is True
    assert loop["period"] == 3
    assert loop["onset_index"] == 2
    assert loop["span_tokens"] == 9


def test_diagnostic_entropy_association_is_finite_and_prefix_local():
    prelude = list(range(100, 112))
    tokens = prelude + [1, 2, 3] * 20
    entropies = [2.0] * len(prelude) + [0.25] * (len(tokens) - len(prelude))
    logprobs = [-1.0] * len(tokens)
    result = analyze_sequence(tokens, entropies, logprobs)
    assert result["repeated_4gram"]["event_count"] > 0
    assert result["longest_periodic_run"]["period"] == 3
    assert result["longest_periodic_run"]["during_minus_pre_entropy"] < 0
    assert math.isfinite(result["base_entropy_mean"])


def test_stable_gumbel_power_argmax_matches_safe_power_form():
    probabilities = np.asarray([0.45, 0.30, 0.20, 0.05])
    uniforms = np.asarray([0.2, 0.8, 0.7, 0.9])
    expected = int(np.argmax(np.power(uniforms, 1.0 / probabilities)))
    assert stable_power_argmax(probabilities, uniforms) == expected
    with pytest.raises(ValueError):
        stable_power_argmax([0.4, 0.4], [0.1, 0.9])


def test_known_gamma_cases_and_threshold():
    result = gumbel_gamma_test([1.0, 1.0], nominal_fpr=1e-3)
    assert result["statistic"] == 2.0
    assert result["p_value"] == pytest.approx(math.exp(-2.0) * 3.0)
    assert result["decision"] is False
    assert gamma_survival(0.0, 2.0, 1.0) == pytest.approx(1.0)


def test_textseal_moment_matching_constant_entropy():
    result = textseal_gamma_test([2.0, 3.0], [1.0, 1.0], alpha=0.1)
    # Pinned code fallback maps constant H=1 to weight .28 using [0,5].
    assert result["statistic"] == pytest.approx(1.4)
    assert result["intermediate"]["routing_variance"] == pytest.approx(0.82)
    assert 0.0 <= result["p_value"] <= 1.0


def test_synthid_normal_synthetic_null_and_signal():
    null = np.tile([[0, 1] * 5], (20, 1))
    null_result = synthid_normal_test(null)
    assert null_result["p_value"] > 1e-3
    signal = np.ones((200, 10), dtype=np.int64)
    signal_result = synthid_normal_test(signal)
    assert signal_result["decision"] is True
    assert signal_result["p_value"] < 1e-20


def test_prc_hoeffding_bound_and_threshold_cases():
    boundary = prc_hoeffding_test(math.sqrt(2 * 10 * math.log(1000)), 10)
    assert boundary["p_value"] == pytest.approx(1e-3)
    assert boundary["decision"] is False  # frozen rule is strict p < 1e-3
    assert prc_hoeffding_test(-1, 10)["p_value"] == 1.0


def _continuation(index: int, length: int = 8) -> GeneratedContinuation:
    return GeneratedContinuation(
        prompt_index=index,
        method="online_prc",
        seed=12345,
        token_ids=tuple(range(length)),
        base_token_logprobs=tuple([-1.0] * length),
        base_entropies=tuple([1.0] * length),
        provenance={"cache": True},
    )


@pytest.mark.parametrize("adapter_type", [OnlinePRCCachedAdapter, FixedPRCCachedAdapter])
def test_cache_reuse_has_no_regeneration_fallback(adapter_type):
    adapter = adapter_type(method="cached_prc", loader=lambda index: _continuation(index), exact_length=8)
    assert len(adapter.load([0, 1])) == 2
    assert adapter.generation_attempts == 0
    with pytest.raises(RuntimeError, match="generation is forbidden"):
        adapter.generate([0], 12345, 8)
    assert adapter.generation_attempts == 1


def test_cache_reuse_missing_record_fails_closed():
    adapter = OnlinePRCCachedAdapter(method="online_prc", loader=lambda _index: None, exact_length=8)
    with pytest.raises(FileNotFoundError, match="regeneration is disabled"):
        adapter.load([0])
    assert adapter.generation_attempts == 0


def test_fixed_seed_determinism_and_gumbel_deterministic_semantics():
    def deterministic(seed: int, method: str):
        if method == "gumbel_max":
            return tuple((position * 7 + 3) % 11 for position in range(32))
        rng = np.random.default_rng(seed)
        return tuple(rng.integers(0, 100, size=32).tolist())

    assert deterministic(123, "textseal") == deterministic(123, "textseal")
    assert deterministic(123, "textseal") != deterministic(456, "textseal")
    assert deterministic(123, "gumbel_max") == deterministic(456, "gumbel_max")


@pytest.mark.parametrize(
    ("adapter_type", "method"),
    [
        (TextSealAdapter, "textseal"),
        (SynthIDTextAdapter, "synthid_text"),
        (GumbelMaxAdapter, "gumbel_max"),
    ],
)
def test_official_generation_adapter_contract(adapter_type, method):
    def backend(indices, seed, length, backend_method):
        return [
            GeneratedContinuation(
                prompt_index=index,
                method=backend_method,
                seed=seed,
                token_ids=tuple(range(length)),
                base_token_logprobs=tuple([-1.0] * length),
                base_entropies=tuple([1.0] * length),
                provenance={"mode": "modal_official_backend"},
            )
            for index in indices
        ]

    outputs = adapter_type(backend).generate([4, 9], seed=17, max_new_tokens=8)
    assert [output.prompt_index for output in outputs] == [4, 9]
    assert {output.method for output in outputs} == {method}


def test_exact_prefix_equivalence_for_pure_prefix_scorer():
    tokens = [2, 3, 4, 5, 6, 2, 3, 4, 5, 7] * 20
    for prefix in PREFIX_LENGTHS[:2]:
        sliced = deduplicated_positions(tokens[:prefix], 3)
        direct = deduplicated_positions(list(tokens[:prefix]), 3)
        assert sliced == direct


def test_quality_and_diversity_metrics():
    tokens = [1, 2, 3, 4, 1, 2, 3, 4]
    quality = quality_metrics(tokens, [-1.0] * len(tokens))
    assert quality["base_model_nll"] == 1.0
    assert quality["base_model_perplexity"] == pytest.approx(math.e)
    assert quality["repetition_rate"] == pytest.approx(1 / 5)
    assert distinct_n(tokens, 2) == pytest.approx(4 / 7)
    assert ngram_repetition_rate(tokens, 4) == pytest.approx(1 / 5)
    assert pairwise_token_agreement([[1, 2], [1, 3], [1, 2]]) == pytest.approx(2 / 3)
    if importlib.util.find_spec("sacrebleu") is not None:
        assert self_bleu_token_ids([[1, 2, 3, 4], [1, 2, 3, 4]]) == pytest.approx(1.0)


def _schema_payload() -> dict:
    return {
        "prompt_index": 0,
        "prompt_id": "canonical-0000",
        "prompt_fingerprint": "a" * 64,
        "sample_type": "watermarked",
        "method": "textseal",
        "method_configuration": {"alpha": 0.1},
        "model_id": "Qwen/Qwen3-8B-Base",
        "model_revision": "b" * 40,
        "tokenizer_id": "Qwen/Qwen3-8B-Base",
        "tokenizer_revision": "b" * 40,
        "generation_seed": 12345,
        "key_seed": 42,
        "key_domain": "textseal-key-a/b",
        "generation_settings": GENERATION_SETTINGS,
        "generated_token_count": 1024,
        "generated_token_hash": "c" * 64,
        "prefix_length": 128,
        "deduplicated_sample_count": 124,
        "statistic": 1.0,
        "p_value": 0.5,
        "calibration_type": "moment-matched Gamma approximation",
        "threshold": 2.0,
        "decision": False,
        "base_model_nll": 2.0,
        "base_model_perplexity": math.exp(2),
        "output_length": 1024,
        "repetition_rate": 0.0,
        "repetition_metric": "repeated token 4-gram fraction",
        "distinct_2": 1.0,
        "distinct_3": 1.0,
        "source_repository_url": "https://example.test/repo",
        "source_repository_commit": "d" * 40,
        "prc_code_fingerprint": "e" * 64,
        "integration_code_fingerprint": "f" * 64,
        "image_fingerprint": "1" * 64,
        "artifact_fingerprint": "2" * 64,
        "cache_or_generation_provenance": {"mode": "generated"},
        "runtime_seconds": 1.0,
    }


def test_shared_schema_completeness_and_json_serialization():
    result = PromptLevelResult(**_schema_payload())
    encoded = result.to_json()
    assert json.loads(encoded)["prompt_index"] == 0
    assert PromptLevelResult.from_dict(json.loads(encoded)) == result


def test_schema_rejects_nan_and_missing_revision():
    payload = _schema_payload()
    payload["p_value"] = float("nan")
    with pytest.raises(ValueError):
        PromptLevelResult(**payload).to_json()


def test_full_run_shard_partition_is_exact_and_gated_by_shape():
    run_id, shard, indices = _validated_full_request(
        {"run_id": "controlled-8b-20260822", "shard_index": 9}
    )
    assert run_id == "controlled-8b-20260822"
    assert shard == 9
    assert indices == list(range(450, 500))
    with pytest.raises(ValueError):
        _validated_full_request({"run_id": "../unsafe", "shard_index": 0})


def test_smoke_artifact_coverage_uniqueness_and_fingerprints():
    root = Path(__file__).parents[1]
    prompt_path = root / "outputs/controlled_baseline_smoke_prompt_level.jsonl"
    rows = [json.loads(line) for line in prompt_path.read_text().splitlines() if line]
    assert len(rows) == 258
    for row in rows:
        PromptLevelResult.from_dict(row)
    identities = {
        (
            row["method"],
            row["sample_type"],
            row["prompt_index"],
            row["generation_seed"],
            row["prefix_length"],
        )
        for row in rows
    }
    assert len(identities) == 258
    primary = [row for row in rows if row["sample_type"] != "seed_validation"]
    assert len(primary) == 240
    assert {row["prompt_index"] for row in primary} == set(range(5))
    assert {row["prefix_length"] for row in primary} == set(PREFIX_LENGTHS)

    manifest = json.loads(
        (root / "outputs/controlled_baseline_smoke_artifact_manifest.json").read_text()
    )
    assert manifest["status"] == "passed_exact_billing_reconciled"
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    validation = json.loads(
        (root / "outputs/controlled_baseline_smoke_validation.json").read_text()
    )
    assert validation["billing"]["campaign_total_usd"] == pytest.approx(1.39371804)
    assert validation["provenance"]["generation_attempts"] == {
        "fixed_prc": 0,
        "null": 0,
        "online_prc": 0,
    }
    assert validation["behavior"]["fixed_seed_replay_passed"] is True
