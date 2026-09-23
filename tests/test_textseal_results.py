"""Publication rejects mismatched inputs and changed statistical decisions."""
import copy

import pytest

from baseline_comparison.textseal_redetect import digest, record_identity
from baseline_comparison.textseal_results import validate_record, validate_report


def fixture():
    row = {"method": "null", "prompt_index": 0, "token_ids": list(range(8))}
    manifest = {"pilot_ids": [], "prefix_lengths": [4, 8], "protocol": "raw", "upstream_commit": "pinned"}
    results = {}
    for n,p in ((4,.001), (8,0.0)):
        results[str(n)] = {"completion_length": n, "completion_sha256": digest(row["token_ids"][:n]),
            "entropy_count": n-1, "protocol": "raw", "upstream_commit": "pinned",
            "upstream": {"p_value_weighted": p},
            "comparison": {"score_field": "p_value_weighted", "nominal_fpr": .001,
                           "p_value": p, "decision": p < .001, "abstained": False}}
    data = {"input": record_identity(row), "completion_sha256": digest(row["token_ids"]),
            "completion_length": 8, "protocol": "raw", "prefix_strategy": "direct",
            "actual_model_inputs_verified": True, "forward_lengths": [4,8],
            "results": results, "entropies_by_prefix": {"4": [1.]*3, "8": [1.]*7},
            "validation": {"performed": False, "passed": None, "prefixes": {}}}
    payload = {"data": data, "data_sha256": digest(data),
               "identity": {"manifest_sha256": digest(manifest), "runtime": {}, "input": record_identity(row)}}
    report_row = {"record_id": "null/0000", "method": "null", "results": copy.deepcopy(results),
                  "validation": copy.deepcopy(data["validation"])}
    return payload, row, report_row, manifest


def test_strict_cutoff_and_zero_p_preserved():
    p,row,rr,m = fixture()
    data = validate_record(p,row,rr,m,{})
    assert data["results"]["4"]["comparison"]["decision"] is False
    assert data["results"]["8"]["comparison"]["decision"] is True


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["forward_lengths"].__setitem__(0, 8), "model input schedule"),
    (lambda d: d["entropies_by_prefix"]["4"].append(1.), "entropy length"),
    (lambda d: d["results"]["4"].__setitem__("completion_sha256", "wrong"), "prefix result identity"),
    (lambda d: d["results"]["4"]["comparison"].__setitem__("decision", True), "comparison decision"),
])
def test_corruption_rejected_even_with_recomputed_checksum(mutation, match):
    p,row,rr,m = fixture()
    mutation(p["data"])
    p["data_sha256"] = digest(p["data"])
    rr["results"] = copy.deepcopy(p["data"]["results"])
    with pytest.raises(ValueError, match=match):
        validate_record(p,row,rr,m,{})


def test_duplicate_report_record_rejected():
    p,row,rr,m = fixture()
    pilot = {"manifest_sha256": digest(m), "runtime": {}, "stage": "pilot", "passed": True,
             "record_sha256": {}}
    report = {"manifest_sha256": digest(m), "runtime": {}, "stage": "full", "passed": True,
              "prefix_strategy": "direct", "rows": [rr,rr], "record_sha256": {"null/0000": "hash"},
              "requested_records": 1, "completed_records": 1, "cached_records": 0, "new_model_forwards": 2}
    with pytest.raises(ValueError, match="coverage differs"):
        validate_report(report,m,[row],pilot)
