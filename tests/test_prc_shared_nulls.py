"""Null-cohort replacement must not change a single watermarked score."""
import copy

import pytest
import torch

from baseline_comparison import prc_shared_nulls as shared
from baseline_comparison.config import PREFIX_LENGTHS
from detectors import semantic_sha256, tensor_sha256
from modal_run import REDETECT_PROTOCOL, _redetect_load, _redetect_write


def batch(root, tokens, cache):
    inputs = {"tokens": tokens, "partition": torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bfloat16)}
    identity = {"protocol": REDETECT_PROTOCOL, "run": root, "start": 0,
                "count": len(tokens), "length": tokens.shape[1], "cache": "static",
                "input_sha256": semantic_sha256(inputs)}
    _redetect_write(cache / root / "inputs.pt", inputs)
    return {"root": root, "identity": identity}


def test_reuse_requires_exact_raw_prefix_and_original_trace_hash(tmp_path):
    before = batch("old", torch.tensor([[0, 1, 2, 1, 0], [2, 1, 0, 2, 1]]), tmp_path)
    after = batch("new", torch.tensor([[0, 1, 2], [2, 1, 0]]), tmp_path)
    values = torch.tensor([[.1, .2, .3, .4], [.7, .6, .5, .4]])
    path = tmp_path / "old/trace.pt"
    _redetect_write(path, {"identity": before["identity"], "probabilities_2_to_T": values,
                          "probabilities_sha256": tensor_sha256(values), "full_validation": True})
    hashed = shared.sha256(path)
    shared.reuse_trace(before, after, tmp_path, tmp_path, hashed)
    payload = _redetect_load(tmp_path / "new/trace.pt")
    assert torch.equal(payload["probabilities_2_to_T"], values[:, :2])
    assert payload["reused_from"]["trace_sha256"] == hashed
    with pytest.raises(ValueError, match="source trace changed"):
        shared.reuse_trace(before, after, tmp_path, tmp_path, "0" * 64)
    changed = batch("changed", torch.tensor([[1, 1, 2], [2, 1, 0]]), tmp_path)
    with pytest.raises(ValueError, match="tokens or partition"):
        shared.reuse_trace(before, changed, tmp_path, tmp_path, hashed)
    # An injected prompt is rejected by the integrated clean-input contract.
    inputs = _redetect_load(tmp_path / "new/inputs.pt")
    inputs["prompt"] = [0, 1]
    _redetect_write(tmp_path / "new/inputs.pt", inputs)
    with pytest.raises(ValueError, match="only frozen completion"):
        shared.reuse_trace(before, after, tmp_path, tmp_path, hashed)


def reference():
    scores = {str(n): {w: {"decision": True, "statistic": i + n / 1000}
                      for w in ("map", "entropy")} for n in PREFIX_LENGTHS for i in [1]}
    return {"records": [{"source": "wm", "prompt_idx": i, "scores": copy.deepcopy(scores)} for i in range(500)],
            "counts": {str(n): {w: {"null": {"detected": 2, "count": 500}} for w in ("map", "entropy")}
                       for n in PREFIX_LENGTHS}}


def csv_rows():
    return [{"PRC Construction": "online_causal_prc_v1", "eta": "0.05", "Entropy Model": "Qwen3-8B-Base",
             "Generation Model": "Qwen3-8B-Base", "n": str(n), "Posterior FPR": "0/500 (0.0%)",
             "Entropy FPR": "0/500 (0.0%)", "Posterior TPR": "sentinel", "Old Posterior TPR": "unchanged",
             "Notes": "new inference=0; original null cohort=T1382"} for n in PREFIX_LENGTHS]


def test_publishing_changes_only_null_fields_and_preserves_other_methods():
    rows = csv_rows() + [{"PRC Construction": "textseal", "value": "untouched"}]
    report = reference()
    updated = shared.aligned_rows(rows, report, report, "fixture")
    assert updated[-1] == rows[-1]
    for old, new in zip(rows[:6], updated[:6]):
        assert new["Posterior FPR"] == new["Entropy FPR"] == "2/500 (0.4%)"
        assert "shared null cohort=T13088" in new["Notes"]
        assert {k: v for k, v in old.items() if k not in {"Posterior FPR", "Entropy FPR", "Notes"}} == {
            k: v for k, v in new.items() if k not in {"Posterior FPR", "Entropy FPR", "Notes"}}
    assert rows[0]["Posterior FPR"] == "0/500 (0.0%)"


def test_changed_wm_score_rejected_even_with_unchanged_decision():
    old = reference()
    new = copy.deepcopy(old)
    new["records"][123]["scores"]["256"]["map"]["statistic"] += .01
    with pytest.raises(ValueError, match="watermarked per-record scores"):
        shared.aligned_rows(csv_rows(), new, old, "fixture")


def test_incomplete_or_duplicate_null_coverage_rejected():
    ref = reference()
    report = copy.deepcopy(ref)
    report["counts"]["400"]["entropy"]["null"]["count"] = 499
    with pytest.raises(ValueError, match="incomplete shared null"):
        shared.aligned_rows(csv_rows(), report, ref, "fixture")
    with pytest.raises(ValueError, match="missing PRC"):
        shared.aligned_rows(csv_rows()[:-1], ref, ref, "fixture")
    with pytest.raises(ValueError, match="duplicate PRC"):
        shared.aligned_rows(csv_rows() + csv_rows()[:1], ref, ref, "fixture")


def test_approval_and_frozen_inputs_required_before_remote_actions(tmp_path):
    prepared = {"batches": []}
    shared.write_json(tmp_path / "prepared.json", prepared)
    plan = {"code_sha256": {}, "local_files_sha256": {"prepared.json": shared.sha256(tmp_path / "prepared.json")}}
    shared.write_json(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="explicit approval"):
        shared.load_plan(tmp_path, "")
    assert shared.load_plan(tmp_path, shared.digest(plan)) == (plan, prepared)
    (tmp_path / "prepared.json").write_text('{}')
    with pytest.raises(ValueError, match="frozen setup changed"):
        shared.load_plan(tmp_path, shared.digest(plan))


def test_original_prefix_sweep_cannot_replace_aligned_nulls(tmp_path):
    from baseline_comparison.prc_prefix_comparison import compare_prefixes
    output = tmp_path / "comparison.csv"
    shared.write_json(output.with_suffix(".provenance.json"), {"shared_null_alignment": {"verified": True}})
    with pytest.raises(ValueError, match="already uses shared nulls"):
        compare_prefixes(tmp_path, output)
