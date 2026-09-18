"""Exercise actual sampler controls on CPU; no pretrained models or Modal."""
from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from baseline_comparison.config import SYNTHID_KEYS
from baseline_comparison.self_bleu_config import (
    REFERENCE_PATH, SYNTHID_KEY_BANK, StudySetting, batch_manifest, pilot_settings,
    verify_reference,
)
from baseline_comparison.self_bleu_generation import generate_response_batch
from online_prc import derive_document_seed, materialize_supports, otp_prefix

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = [[i % 31 for i in range(50)], [(i + 3) % 31 for i in range(50)]]
EXECUTION = {"validation_fixture": "CPU deterministic causal logits; not production 8B"}


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, ids, cache=None):
        vocab = torch.arange(32, device=ids.device)
        return -((vocab - (ids.unsqueeze(-1) + 1).remainder(32)).float() ** 2) / 80


def original_online_sampler():
    # Load the existing function body without importing watermark_expt's
    # module-level pretrained model download/allocation.
    import qwen
    tree = ast.parse((ROOT / "watermark_expt.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "generate_batch_and_collect_online")
    namespace = {"torch": torch, "np": np, "device": torch.device("cpu"),
                 "make_kv_cache": qwen.make_kv_cache,
                 "normalize_kv_cache_implementation": qwen.normalize_kv_cache_implementation,
                 "kv_cache_version": qwen.kv_cache_version}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "watermark_expt.py", "exec"), namespace)
    return namespace[node.name]


def prc_run(setting, seed, response=0, artifact=None, prompts=PROMPTS, indices=(0, 1)):
    if artifact is None:
        mask = torch.tensor([0, 1] * 16, dtype=torch.float32)
        artifact = {"online_key": setting.online_key().to_dict(), "partition": torch.stack((1-mask, mask))}
    return generate_response_batch(ToyModel(), prompts, indices, setting=setting,
                                   sampling_seed=seed, response_index=response, execution=EXECUTION,
                                   max_new_tokens=32, device="cpu", prc_artifact=artifact,
                                   online_sampler=original_online_sampler())


def test_frozen_commit_and_artifact_records_are_verifiable():
    reference = verify_reference()
    assert reference["reference_commit"] == "46963822e9d1f89558337c013b1ff0d47fcc0fb2"
    assert reference["protocol"] == "completion_only_raw_abstain_v1"


def test_sampling_seed_changes_actual_prc_samples_without_changing_key():
    setting = StudySetting("online_prc")
    key = setting.online_key()
    first = prc_run(setting, 12345)
    repeat = prc_run(setting, 12345)
    second = prc_run(setting, 67890, response=1)
    assert first["responses"] == repeat["responses"]
    assert first["manifest"]["batch_id"] == repeat["manifest"]["batch_id"]
    assert first["manifest"]["setting"] == second["manifest"]["setting"]
    assert key == setting.online_key()
    np.testing.assert_array_equal(materialize_supports(32, key), materialize_supports(32, setting.online_key()))
    np.testing.assert_array_equal(otp_prefix(32, key), otp_prefix(32, setting.online_key()))
    for a, b in zip(first["responses"], second["responses"]):
        assert a["response_id"] != b["response_id"]
        assert a["token_ids"] != b["token_ids"]
        assert a["generation_diagnostics"]["prc_codeword_bits"] != b["generation_diagnostics"]["prc_codeword_bits"]
        assert a["document_seed"] == derive_document_seed(12345, a["prompt_index"])
        assert b["document_seed"] == derive_document_seed(67890, b["prompt_index"])


def test_prc_legacy_first_response_stream_is_preserved_and_wrong_key_rejected():
    setting = StudySetting("online_prc")
    result = prc_run(setting, 12345)
    mask = torch.tensor([0, 1] * 16, dtype=torch.float32)
    partition = torch.stack((1-mask, mask))
    expected, _, _ = original_online_sampler()(
        ToyModel(), torch.tensor(PROMPTS), 32, setting.online_key(), partition,
        return_trace_details=True, document_seeds=[derive_document_seed(12345, i) for i in (0, 1)],
        kv_cache_implementation="static",
    )
    assert [r["token_ids"] for r in result["responses"]] == expected.tolist()
    bad = {"online_key": replace(setting, key_seed=999).online_key().to_dict(), "partition": partition}
    with pytest.raises(ValueError, match="fixed key"):
        prc_run(setting, 67890, artifact=bad)
    other_partition = {"online_key": setting.online_key().to_dict(), "partition": partition.flip(0)}
    assert prc_run(setting, 12345, artifact=other_partition)["manifest"]["namespace"] != result["manifest"]["namespace"]


def test_prompt_ids_bind_document_stream_and_batch_identity():
    setting = StudySetting("online_prc")
    original = prc_run(setting, 12345)
    reordered = prc_run(setting, 12345, prompts=PROMPTS[::-1], indices=(1, 0))
    assert [r["token_ids"] for r in original["responses"]] == [r["token_ids"] for r in reordered["responses"]][::-1]
    assert original["manifest"]["batch_id"] != reordered["manifest"]["batch_id"]
    with pytest.raises(ValueError, match="unique canonical"):
        prc_run(setting, 12345, indices=(0, 0))


def test_nested_synthid_bank_and_parameter_identities():
    assert SYNTHID_KEY_BANK[:10] == SYNTHID_KEYS
    assert len(SYNTHID_KEY_BANK) == len(set(SYNTHID_KEY_BANK)) == 30
    for depth in (2, 5, 10, 15, 20, 30):
        setting = StudySetting("synthid_text", depth=depth)
        assert setting.identity()["keys"] == list(SYNTHID_KEY_BANK[:depth])
    assert len({StudySetting("textseal", alpha=a).fingerprint for a in (0, .05, .1, .2, .3, .5)}) == 6
    assert len(pilot_settings()) == 5
    assert len(pilot_settings("B")) == 4
    assert pilot_settings("depth30")[0].depth == 30
    with pytest.raises(ValueError):
        StudySetting("synthid_text", depth=31)
    with pytest.raises(ValueError):
        StudySetting("textseal", alpha=float("nan"))


def test_manifest_separates_keys_samples_and_execution():
    setting = StudySetting("textseal")
    def manifest(**kw):
        return batch_manifest(setting, (0, 1), PROMPTS, sampling_seed=kw.get("seed", 12345),
                              response_index=kw.get("response", 0), execution=kw.get("execution", EXECUTION))
    a, b = manifest(), manifest(seed=67890, response=1)
    assert a["setting_sha256"] == b["setting_sha256"]
    assert a["namespace"] != b["namespace"]
    assert a["namespace"].startswith("self_bleu_v1/")
    assert manifest(execution={"dtype": "changed"})["batch_id"] != a["batch_id"]
    with pytest.raises(ValueError, match="response_index"):
        manifest(response=2)
    with pytest.raises(ValueError, match="execution"):
        manifest(execution={})


def test_ordinary_sampling_control_reproduces_and_uses_full_vocabulary():
    setting = StudySetting("null")
    def run(seed, response):
        return generate_response_batch(ToyModel(), PROMPTS, (0, 1), setting=setting,
                                       sampling_seed=seed, response_index=response, execution=EXECUTION,
                                       max_new_tokens=16, device="cpu")
    a, replay, b = run(12345, 0), run(12345, 0), run(67890, 1)
    assert a["responses"] == replay["responses"]
    assert a["responses"][0]["token_ids"] != b["responses"][0]["token_ids"]
    # Independent direct multinomial baseline with the same model and RNG stream.
    torch.manual_seed(12345)
    model, ids, expected = ToyModel(), torch.tensor(PROMPTS), []
    for _ in range(16):
        next_token = torch.multinomial(torch.softmax(model(ids)[:, -1].float(), dim=-1), 1)
        expected.append(next_token)
        ids = next_token
    assert [r["token_ids"] for r in a["responses"]] == torch.cat(expected, dim=1).tolist()


@pytest.fixture
def upstream():
    from baseline_comparison.textseal_completion import load_upstream_detector
    root = os.environ.get("TEXTSEAL_SOURCE_ROOT")
    detector, _ = load_upstream_detector(root)
    return detector, root


@pytest.mark.parametrize("alpha", [0, .1, .5])
def test_textseal_alpha_reaches_real_upstream_sampler_and_detector(upstream, alpha):
    from baseline_comparison.official import textseal_config, textseal_generator
    from baseline_comparison.textseal_completion import TextSealCompletionDetector

    class HFModel(ToyModel):
        def forward(self, ids):
            return SimpleNamespace(logits=super().forward(ids))

    detector_type, root = upstream
    adapter = TextSealCompletionDetector(HFModel(), source_root=root, alpha=alpha)
    config = textseal_config(alpha=alpha)
    sampler = textseal_generator(alpha=alpha)
    assert sampler.mixing_alpha == adapter._detector.wm_config.mixing_alpha == alpha
    tokens = list(range(25))
    reference = detector_type(None, config, model=HFModel(), scoring_method="v2")
    h = reference._compute_entropies(tokens)
    assert adapter.detect(tokens)["upstream"] == reference._score_text(tokens, h, "v2")
    # Setting a new alpha cannot mutate previously constructed samplers.
    textseal_generator(alpha=.3)
    assert sampler.mixing_alpha == alpha


@pytest.mark.parametrize("depth", [2, 10, 20, 30])
def test_synthid_depth_reaches_real_upstream_generation_and_evidence(depth):
    from baseline_comparison.official import official_synthid_g_values, synthid_processor
    keys = StudySetting("synthid_text", depth=depth).synthid_keys
    processor = synthid_processor("cpu", keys=keys)
    assert processor.keys.tolist() == list(keys)
    tokens = list(range(20))
    expected = processor.compute_g_values(torch.tensor([tokens]))[0]
    actual = official_synthid_g_values(tokens, [4, 5, 6], keys=keys)
    np.testing.assert_array_equal(actual, expected[[1, 2, 3]].numpy())
    assert actual.shape == (3, depth)


@pytest.mark.parametrize("depth", [2, 10, 20, 30])
def test_legacy_synthid_scorer_uses_requested_keys_and_records_actual_depth(depth):
    from baseline_comparison.comparison_runner import _score_baseline_sequence
    from baseline_comparison.config import PREFIX_LENGTHS
    from baseline_comparison.official import official_synthid_g_values
    from baseline_comparison.scoring import deduplicated_positions, synthid_normal_test

    keys = StudySetting("synthid_text", depth=depth).synthid_keys
    tokens = list(range(1024))
    args = dict(method="synthid_text", sequence={"token_ids": tokens,
                "base_token_logprobs": [-1.]*1024, "base_entropies": [1.]*1024},
                prompt_row={"doc_index": 0, "prompt_tokens": [10, 11]}, prompt_index=0,
                sample_type="watermarked", seed=12345, model_revision="fixture",
                integration_fingerprint="fixture", prc_fingerprint="fixture",
                provenance={"mode": "CPU fixture"}, runtime_seconds=0.)
    with pytest.raises(ValueError, match="explicit keys"):
        _score_baseline_sequence(**args)
    rows, checks = _score_baseline_sequence(**args, synthid_keys=keys)
    assert len(rows) == len(PREFIX_LENGTHS)
    for row, length in zip(rows, PREFIX_LENGTHS):
        evidence = official_synthid_g_values(tokens[:length], deduplicated_positions(tokens[:length]), keys=keys)
        expected = synthid_normal_test(evidence)
        assert evidence.shape[1] == depth
        assert row["statistic"] == expected["statistic"]
        assert row["p_value"] == expected["p_value"]
        assert row["decision"] == expected["decision"]
        assert row["method_configuration"]["keys"] == list(keys)
        assert row["method_configuration"]["depth"] == depth
        assert row["key_seed"] == keys[0]
        assert f"{depth}-key domain" in row["key_domain"]
    assert all(check["max_abs_delta"] == 0 for check in checks["exact_prefix_checks"])
    # Shared nulls must be evaluated under the same requested detector keys.
    null_rows, _ = _score_baseline_sequence(**{**args, "sample_type": "null"}, synthid_keys=keys)
    assert [r["p_value"] for r in null_rows] == [r["p_value"] for r in rows]
    # Distinct keys at equal depth must not silently select the default bank.
    alternate = tuple(k + 1 for k in keys)
    other, _ = _score_baseline_sequence(**args, synthid_keys=alternate)
    assert other[0]["method_configuration"]["keys"] == list(alternate)
    assert other[0]["artifact_fingerprint"] != rows[0]["artifact_fingerprint"]
    expected = synthid_normal_test(official_synthid_g_values(
        tokens[:PREFIX_LENGTHS[0]], deduplicated_positions(tokens[:PREFIX_LENGTHS[0]]), keys=alternate))
    assert other[0]["statistic"] == expected["statistic"]


@pytest.mark.parametrize("method", ["textseal", "synthid_text", "gumbel_max"])
def test_default_generation_matches_frozen_function_on_cpu(upstream, monkeypatch, method):
    # Execute the historical function itself. Only device and requested length
    # are adapted for a local fixture; algorithm and seed consumption stay intact.
    from baseline_comparison import comparison_runner as current
    reference = json.loads(REFERENCE_PATH.read_text())
    source = subprocess.check_output(["git", "show", f"{reference['reference_commit']}:baseline_comparison/comparison_runner.py"], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "generate_method")
    class CPUDevice(ast.NodeTransformer):
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant("cpu"), node) if node.value == "cuda" else node
    node = ast.fix_missing_locations(CPUDevice().visit(node))
    namespace = {**vars(current), "MAX_NEW_TOKENS": 12}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "historical_comparison_runner.py", "exec"), namespace)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # Released TextSeal's CPU PRF supports one row here; its distinct CUDA
    # helper handles production batches. Do not rewrite upstream to test CPU.
    prompts = PROMPTS[:1] if method == "textseal" else PROMPTS
    expected, _ = namespace["generate_method"](ToyModel(), prompts, method=method, seed=12345)
    actual, _ = current.generate_method(ToyModel(), prompts, method=method, seed=12345,
                                        max_new_tokens=12, device="cpu")
    assert actual == expected
