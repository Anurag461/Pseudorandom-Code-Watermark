"""Pilot request boundaries and paired statistical units."""
from copy import deepcopy

import pytest

from baseline_comparison.self_bleu_config import digest
from baseline_comparison.self_bleu_pilot import clean_records, validate_request
from baseline_comparison.self_bleu_validation import RATE


def record():
    return {"response_id": "batch/p0000/r0", "method": "online_prc", "prompt_index": 0,
            "response_index": 0, "token_ids": [1]*1024, "completion_sha256": digest([1]*1024)}


def request(rows):
    value = {"protocol": "completion_only_raw_abstain_v1", "stage": "A", "nominal_fpr": .001,
             "code_sha256": {}, "requests": {"prc": {"count": len(rows), "sha256": digest(rows)}},
             "cost": {"timeout_seconds_per_stage": 600, "resource_usd_per_second": RATE,
                      "previous_planning_charge_usd": 4.05505}}
    value["id"] = digest(value)
    return value


def test_worker_request_rejects_prompt_or_generation_traces_even_when_rehashed(tmp_path):
    clean = record()
    validate_request(request([clean]), [clean], "prc", tmp_path)
    for key, value in (("prompt_ids", [5]), ("generation_diagnostics", {}), ("entropy", [.1])):
        bad = {**clean, key: value}
        with pytest.raises(ValueError, match="raw completions"):
            validate_request(request([bad]), [bad], "prc", tmp_path)


def test_request_rejects_changed_tokens_duplicate_ids_and_budget_overrun(tmp_path):
    row = record()
    changed = deepcopy(row)
    changed["token_ids"][0] = 2
    for rows in ([changed], [row, row]):
        with pytest.raises(ValueError):
            validate_request(request(rows), rows, "prc", tmp_path)
    manifest = request([row])
    manifest["cost"]["previous_planning_charge_usd"] = 9
    manifest["id"] = digest({k: v for k, v in manifest.items() if k != "id"})
    with pytest.raises(ValueError, match="allocation"):
        validate_request(manifest, [row], "prc", tmp_path)


def test_generation_diagnostics_do_not_enter_detector_request():
    row = {**record(), "generation_diagnostics": {"entropy": [42]}, "prompt_ids": [23]}
    result = clean_records({("online_prc", 0): {"responses": [row]}})
    assert result == [record()]


def test_paired_bootstrap_keeps_both_responses_in_the_prompt_unit():
    import numpy as np
    from baseline_comparison.self_bleu_pilot import paired_interval
    # Perfect within-prompt disagreement: prompt-average detection is .5 for
    # every sampled prompt, so a correct prompt bootstrap has no variation.
    pairs = np.array([[0., 1.]]*50)
    draws = np.random.default_rng(20260918).integers(0, 50, size=(2000, 50))
    assert paired_interval(pairs.mean(1), draws) == {"mean": .5, "ci95": [.5, .5]}
    with pytest.raises(ValueError, match="one finite value per prompt"):
        paired_interval(pairs.reshape(-1), draws)
    # A paired contrast between identical method vectors remains exactly zero.
    values = np.linspace(0, 1, 50)
    assert paired_interval(values-values, draws) == {"mean": 0., "ci95": [0., 0.]}
