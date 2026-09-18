"""Fail-closed validation of identities and replicate alignment before GPU spend."""
from copy import deepcopy

import pytest

from baseline_comparison.self_bleu_config import digest
from baseline_comparison.self_bleu_validation import compare_replicates, save, sha, validate_manifest


def batch(tokens, key=12345):
    return {"manifest": {"setting": {"key": key}},
            "responses": [{"token_ids": ids, "prompt_index": i} for i, ids in enumerate(tokens)]}


def test_replicate_checks_reject_rekeying_and_misaligned_rows():
    first, second = batch([[1, 2], [3, 4]]), batch([[1, 3], [3, 5]])
    assert compare_replicates(first, second, first, deterministic=False)["passed"]
    assert compare_replicates(first, first, first, deterministic=True)["passed"]
    assert not compare_replicates(first, first, first, deterministic=False)["passed"]
    for broken in (batch([[1, 3]]), batch([[1, 3], [3, 5]], key=67890)):
        assert not compare_replicates(first, broken, first, deterministic=False)["passed"]
    second["responses"].reverse()
    assert not compare_replicates(first, second, first, deterministic=False)["passed"]


def test_manifest_rejects_modified_sources_scope_and_budget(tmp_path):
    (tmp_path / "code.py").write_text("frozen code")
    (tmp_path / "prompts.jsonl").write_text("frozen prompts")
    value = {"prompt_indices": list(range(50)), "length": 1024, "seeds": [12345, 67890],
             "budget_usd": 10, "generation_timeout": 3000, "textseal_timeout": 600,
             "code_sha256": {"code.py": sha(tmp_path / "code.py")},
             "prompt_sha256": sha(tmp_path / "prompts.jsonl")}
    value["id"] = digest(value)
    validate_manifest(value, tmp_path)
    for field, bad in (("length", 2048), ("budget_usd", 200), ("generation_timeout", 3600)):
        broken = deepcopy(value)
        broken[field] = bad
        broken["id"] = digest({k: v for k, v in broken.items() if k != "id"})
        with pytest.raises(ValueError):
            validate_manifest(broken, tmp_path)
    (tmp_path / "code.py").write_text("changed code")
    with pytest.raises(ValueError, match="source changed"):
        validate_manifest(value, tmp_path)


def test_saved_validation_artifacts_cannot_be_overwritten(tmp_path):
    path = tmp_path / "result.json"
    save(path, {"passed": True})
    save(path, {"passed": True})
    with pytest.raises(FileExistsError):
        save(path, {"passed": False})
