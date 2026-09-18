"""Fail-closed validation of identities and replicate alignment before GPU spend."""
from copy import deepcopy

import pytest

from self_bleu.config import digest
from self_bleu.validation import compare_replicates, save, sha, validate_manifest


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


def test_archived_sources_are_only_accepted_for_reading_old_artifacts():
    import json
    from self_bleu.validation import ROOT, verify_source_hashes
    old = json.loads((ROOT / "outputs/self_bleu_validation/step3-v4/manifest.json").read_text())
    # The exact pre-consolidation tree remains available, including removed files.
    validate_manifest(old, ROOT, allow_archived=True)
    with pytest.raises(ValueError, match="source changed"):
        validate_manifest(old, ROOT)
    with pytest.raises(ValueError, match="source changed"):
        verify_source_hashes({"baseline_comparison/self_bleu_validation_results.py": "0"*64}, allow_archived=True)


def test_pilot_worker_does_not_accept_historical_source_instead_of_current_code():
    import json
    from self_bleu.pilot import validate_request
    from self_bleu.validation import ROOT
    old = json.loads((ROOT / "outputs/self_bleu_pilot/stage_a_v2/manifest.json").read_text())
    with pytest.raises(ValueError, match="worker source differs"):
        validate_request(old, [], "prc", ROOT)
