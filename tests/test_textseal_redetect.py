"""Replay approval/input/checkpoint contracts; no network or GPU calls."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from baseline_comparison.textseal_redetect import (
    CODE_FILES, REPO, canonical, digest, file_sha, load_request, record_identity,
    require_pilot, validate_request, write_json,
)


def request_fixture():
    rows = [{"method": method, "prompt_index": 0, "token_ids": [1, 2, 3, 4]}
            for method in ("textseal", "null")]
    identities = [record_identity(row) for row in rows]
    manifest = {"prefix_strategy": "direct", "code_sha256": {name: file_sha(REPO / name) for name in CODE_FILES},
                "inputs": {"records": identities}, "pilot_ids": [r["record_id"] for r in identities]}
    return manifest, rows


@pytest.mark.parametrize("strategy", [None, "reuse"])
def test_production_rejects_unvalidated_longest_trace_reuse(strategy):
    manifest, rows = request_fixture()
    manifest["prefix_strategy"] = strategy
    with pytest.raises(ValueError, match="requires direct detection"):
        validate_request(manifest, rows, "pilot", digest(manifest))


def test_approved_manifest_and_only_raw_records():
    manifest, rows = request_fixture()
    approved = digest(manifest)
    validate_request(manifest, rows, "pilot", approved)
    with pytest.raises(ValueError, match="approved manifest"):
        validate_request(manifest, rows, "pilot", "")
    with pytest.raises(ValueError, match="coverage"):
        validate_request(manifest, rows[:1], "pilot", approved)
    bad = copy.deepcopy(rows)
    bad[0]["token_ids"][0] = 99
    with pytest.raises(ValueError, match="input identity"):
        validate_request(manifest, bad, "pilot", approved)
    bad[0]["prompt"] = [7, 8]
    with pytest.raises(ValueError, match="only method"):
        validate_request(manifest, bad, "pilot", approved)


def test_changed_code_and_incomplete_code_identity_rejected():
    manifest, rows = request_fixture()
    first = CODE_FILES[0]
    manifest["code_sha256"][first] = "0" * 64
    with pytest.raises(ValueError, match="execution code changed"):
        validate_request(manifest, rows, "pilot", digest(manifest))
    del manifest["code_sha256"][first]
    with pytest.raises(ValueError, match="incomplete execution"):
        validate_request(manifest, rows, "pilot", digest(manifest))


def test_local_request_export_hash_before_dispatch(tmp_path):
    manifest, rows = request_fixture()
    inputs = tmp_path / "rows.jsonl"
    inputs.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    manifest["inputs"].update(file=inputs.name, sha256=file_sha(inputs))
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    assert load_request(path, "pilot", digest(manifest)) == (manifest, rows)
    inputs.write_bytes(inputs.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="export changed"):
        load_request(path, "pilot", digest(manifest))


def test_full_requires_exact_successful_matching_pilot():
    runtime = {"gpu": "H100", "dtype": "bfloat16"}
    report = {"manifest_sha256": "abc", "stage": "pilot", "passed": True,
              "runtime": runtime, "record_sha256": {"textseal/0000": "x", "null/0000": "y"}}
    ids = list(report["record_sha256"])
    require_pilot(report, "abc", runtime, ids)
    for field, value in (("passed", False), ("manifest_sha256", "changed"),
                         ("runtime", {"gpu": "A10"}), ("record_sha256", {})):
        with pytest.raises(ValueError, match="passing pilot"):
            require_pilot({**report, field: value}, "abc", runtime, ids)
    with pytest.raises(ValueError, match="passing pilot"):
        require_pilot(None, "abc", runtime, ids)


def test_checkpoint_verified_and_hf_load_remains_offline(tmp_path, monkeypatch):
    from transformers import AutoModelForCausalLM
    from baseline_comparison.textseal_modal import load_model
    root = tmp_path / "model"
    root.mkdir()
    files = {"model-00001-of-00001.safetensors": b"fixture", "tokenizer.json": b"{}",
             "config.json": b"{}", "model.safetensors.index.json": canonical({"weight_map": {"x": "model-00001-of-00001.safetensors"}})}
    for name, content in files.items():
        (root / name).write_bytes(content)
        metadata = root / ".cache/huggingface/download" / f"{name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("revision\n")
    spec = {"cache_directory": "model", "revision": "revision",
            "weight_files": {"model-00001-of-00001.safetensors": file_sha(root / "model-00001-of-00001.safetensors")},
            "config_sha256": file_sha(root / "config.json"), "tokenizer_sha256": file_sha(root / "tokenizer.json"),
            "index_sha256": file_sha(root / "model.safetensors.index.json")}
    calls = []
    class FakeModel:
        config = SimpleNamespace(model_type="qwen3", vocab_size=151936, use_cache=True)
        def eval(self): return self
        def to(self, device):
            assert device == "cuda"
            return self
    def fake_load(path, **kwargs):
        calls.append(kwargs)
        return FakeModel()
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_load)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    model = load_model({"model": spec}, tmp_path)
    assert model.config.use_cache is False
    assert calls == [{"torch_dtype": torch.bfloat16, "attn_implementation": "eager",
                      "local_files_only": True, "trust_remote_code": False}]
    (root / "config.json").write_text("changed")
    with pytest.raises(ValueError, match="checkpoint differs"):
        load_model({"model": spec}, tmp_path)
    assert len(calls) == 1
