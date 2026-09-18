"""Run against the real pinned TextSeal files, without loading an LLM."""
from __future__ import annotations

import json
import importlib.metadata
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from baseline_comparison.textseal_completion import (
    AUDIT_PATH,
    TextSealCompletionDetector,
    load_upstream_detector,
    verify_upstream_source,
)


@pytest.fixture(scope="module")
def upstream_root():
    root = os.environ.get("TEXTSEAL_SOURCE_ROOT")
    if root is None:
        pytest.importorskip("transformers")
        try:
            root = verify_upstream_source()["root"]
        except importlib.metadata.PackageNotFoundError:
            pytest.skip("install pinned TextSeal or set TEXTSEAL_SOURCE_ROOT")
    return Path(root)


class CausalToyModel(torch.nn.Module):
    """Exact deterministic causal logits; logs actual upstream forward inputs."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.calls = []

    def forward(self, tokens):
        self.calls.append(tokens.clone())
        centers = tokens.cumsum(dim=-1).remainder(31).unsqueeze(-1)
        vocab = torch.arange(32).reshape(1, 1, -1)
        logits = -((vocab - centers).float() ** 2) / 40
        return SimpleNamespace(logits=logits)


class IntegerTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [int(x) for x in text.split()]


@pytest.mark.parametrize("tokens", [[], [1], [1, 2, 3, 4], list(range(25)), [1, 2, 3, 4] * 9])
def test_exact_upstream_public_detect_and_actual_raw_inputs(upstream_root, tokens):
    from baseline_comparison.official import textseal_config

    model = CausalToyModel()
    adapter = TextSealCompletionDetector(model, source_root=upstream_root)
    actual = adapter.detect(tokens)
    detector_type, _ = load_upstream_detector(upstream_root)
    reference = detector_type(IntegerTokenizer(), textseal_config(), model=model, scoring_method="v2")
    expected = reference.detect(" ".join(map(str, tokens)))
    assert actual["upstream"] == expected  # Exact dict equality, not a tolerance.
    assert actual["entropy_count"] == max(0, len(tokens) - 1)
    if len(tokens) > 1:
        assert len(model.calls) == 2
        assert all(call.tolist() == [tokens] for call in model.calls)
    else:
        assert model.calls == []
    assert actual["comparison"]["decision"] == (expected.get("p_value_weighted", 1) < .001)


def test_completion_only_signature_rejects_prompt_and_cached_entropy(upstream_root):
    adapter = TextSealCompletionDetector(CausalToyModel(), source_root=upstream_root)
    tokens = [1, 2, 3, 4, 5, 6]
    for forbidden in ({"prompt": [99]}, {"entropies": [100] * 6}, {"past_key_values": object()}):
        with pytest.raises(TypeError):
            adapter.detect(tokens, **forbidden)
    with pytest.raises(TypeError, match="raw completion"):
        adapter.detect({"token_ids": tokens, "prompt": [99], "base_entropies": [100] * 6})


def test_prefix_alignment_causality_and_no_cross_response_state(upstream_root):
    adapter = TextSealCompletionDetector(CausalToyModel(), source_root=upstream_root)
    tokens = list(range(24))
    full = adapter._detector._compute_entropies(tokens)
    changed = adapter._detector._compute_entropies(tokens[:12] + [31] * 12)
    prefix = adapter._detector._compute_entropies(tokens[:12])
    assert full[:11] == changed[:11] == prefix
    before = adapter.detect(tokens)
    adapter.detect([1, 2, 3, 4] * 8)
    assert adapter.detect(tokens) == before


def test_degenerate_entropy_short_input_and_dedup_match_upstream(upstream_root):
    tokens = [1, 2, 3, 4] * 10
    # Constant entropy is supplied by the model, not as a public cached input.
    class ConstantModel(CausalToyModel):
        def forward(self, ids):
            self.calls.append(ids.clone())
            return SimpleNamespace(logits=torch.zeros(1, ids.shape[1], 32))
    adapter = TextSealCompletionDetector(ConstantModel(), source_root=upstream_root)
    result = adapter.detect(tokens)["upstream"]
    h = adapter._detector._compute_entropies(tokens)
    assert len(set(h)) == 1
    assert result == adapter._detector._score_text(tokens, h, "v2")
    assert result["n_tokens"] == 4
    assert result["p_value"] == min(result["p_value_weighted"], result["p_value_unweighted"])
    assert result["detected"] == (result["p_value"] < .01)


def test_changed_upstream_file_is_rejected_before_import(upstream_root, tmp_path):
    audit = json.loads(AUDIT_PATH.read_text())
    for relative in audit["source_files_sha256"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream_root / relative, target)
    with (tmp_path / "textseal/watermarking/detector.py").open("a") as stream:
        stream.write("\n# changed\n")
    with pytest.raises(ValueError, match="upstream source differs"):
        verify_upstream_source(tmp_path)


def test_historical_prompted_textseal_path_is_retired():
    from baseline_comparison.comparison_runner import (
        _score_baseline_sequence, score_textseal_proxy_entropy_shard,
    )
    with pytest.raises(ValueError, match="prompt-conditioned TextSeal"):
        _score_baseline_sequence(method="textseal", sequence={}, prompt_row={}, prompt_index=0,
                                 sample_type="watermarked", seed=0, model_revision="", integration_fingerprint="",
                                 prc_fingerprint="", provenance={}, runtime_seconds=0)
    with pytest.raises(ValueError, match="prompt-conditioned TextSeal"):
        score_textseal_proxy_entropy_shard(None, {})


def test_real_hf_qwen_interface_matches_upstream_without_weights_download(upstream_root):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from baseline_comparison.official import textseal_config

    torch.manual_seed(7)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64,
        attn_implementation="eager",
    )).eval()
    calls = []
    hook = model.register_forward_pre_hook(lambda module, args: calls.append(args[0].tolist()))
    tokens = list(range(24))
    adapter = TextSealCompletionDetector(model, source_root=upstream_root)
    actual = adapter.detect(tokens)
    detector_type, _ = load_upstream_detector(upstream_root)
    reference = detector_type(IntegerTokenizer(), textseal_config(), model=model, scoring_method="v2")
    assert actual["upstream"] == reference.detect(" ".join(map(str, tokens)))
    assert calls == [[tokens], [tokens]]
    hook.remove()


def test_prefix_reuse_performs_one_forward_and_matches_direct_upstream(upstream_root):
    from baseline_comparison.textseal_redetect import run_record
    model = CausalToyModel()
    detector = TextSealCompletionDetector(model, source_root=upstream_root)
    ids, lengths = list(range(24)), [8, 16, 24]
    row = {"method": "textseal", "prompt_index": 0, "token_ids": ids}
    result = run_record(detector, row, lengths, False, prefix_strategy="reuse")
    assert result["forward_lengths"] == [24]
    assert len(model.calls) == 1
    assert result["validation"]["performed"] is False
    for n in lengths:
        assert result["results"][str(n)] == detector.detect(ids[:n])
    pilot = run_record(detector, row, lengths, True, prefix_strategy="reuse")
    assert pilot["forward_lengths"] == [24, 8, 16]
    assert pilot["validation"]["passed"] is True
    assert pilot["results"] == result["results"]


def test_pilot_detects_shape_dependent_entropy_instead_of_silently_reusing(upstream_root):
    class ShapeDependent(CausalToyModel):
        def forward(self, tokens):
            result = super().forward(tokens)
            result.logits = result.logits * tokens.shape[1]
            return result
    detector = TextSealCompletionDetector(ShapeDependent(), source_root=upstream_root)
    ids, lengths = list(range(24)), [8, 16, 24]
    result = detector.detect_prefixes_reusing_longest(ids, prefix_lengths=lengths, validate_prefixes=True)
    assert result["validation"]["passed"] is False
    assert result["validation"]["prefixes"]["8"]["max_abs_entropy_difference"] > 0
    from baseline_comparison.textseal_redetect import run_record
    row = {"method": "textseal", "prompt_index": 0, "token_ids": ids}
    direct = run_record(detector, row, lengths, True)
    assert direct["prefix_strategy"] == "direct"
    assert direct["validation"]["passed"] is True
    assert direct["forward_lengths"] == [8, 8, 16, 16, 24, 24]
    assert direct["entropies_by_prefix"]["8"] != result["entropies_2_to_T"][:7]
    production = run_record(detector, row, lengths, False)
    assert production["forward_lengths"] == lengths
    assert production["results"] == direct["results"]
    for n in lengths:
        assert direct["results"][str(n)] == detector.detect(ids[:n])


def test_forward_observer_rejects_prepended_prompt(upstream_root):
    from baseline_comparison.textseal_redetect import run_record
    class BadAdapter:
        _detector = SimpleNamespace(model=CausalToyModel())
        def detect_prefixes(self, ids, **kwargs):
            self._detector.model(torch.tensor([[99] + ids]))
    with pytest.raises(ValueError, match="raw completion tokens"):
        run_record(BadAdapter(), {"method": "null", "prompt_index": 0, "token_ids": [1, 2, 3]}, [3], False)


def test_cached_completion_entropy_checksums_and_identity(upstream_root, tmp_path):
    from baseline_comparison.textseal_modal import cached_record
    from baseline_comparison.textseal_redetect import digest, run_record, write_json
    detector = TextSealCompletionDetector(CausalToyModel(), source_root=upstream_root)
    row = {"method": "null", "prompt_index": 0, "token_ids": list(range(24))}
    data = run_record(detector, row, [8, 16, 24], True)
    path, identity = tmp_path / "row.json", {"runtime": "fixture"}
    payload = {"identity": identity, "data": data, "data_sha256": digest(data)}
    write_json(path, payload)
    assert cached_record(path, identity, detector, row, [8, 16, 24], "direct") == data
    with pytest.raises(ValueError, match="prefix strategy"):
        cached_record(path, identity, detector, row, [8, 16, 24], "reuse")
    with pytest.raises(ValueError, match="identity or checksum"):
        cached_record(path, {"runtime": "changed"}, detector, row, [8, 16, 24])
    payload["data"]["entropies_by_prefix"]["8"][4] += .1
    write_json(path, payload)
    with pytest.raises(ValueError, match="identity or checksum"):
        cached_record(path, identity, detector, row, [8, 16, 24])
    payload["data_sha256"] = digest(data)
    write_json(path, payload)
    with pytest.raises(ValueError, match="upstream result differs"):
        cached_record(path, identity, detector, row, [8, 16, 24], "direct")
    del payload["data"]["entropies_by_prefix"]["8"]
    payload["data_sha256"] = digest(data)
    write_json(path, payload)
    with pytest.raises(ValueError, match="prefix coverage"):
        cached_record(path, identity, detector, row, [8, 16, 24], "direct")
