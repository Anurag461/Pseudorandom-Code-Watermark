"""Read and verify completed TextSeal replay; never launches remote compute.

Publish source counts as JSON for the shared comparison CSV. The authoritative
scores are those produced by pinned upstream code in the frozen GPU runtime;
this collector does not rescore with a different local SciPy build.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
import struct

from .textseal_redetect import (
    REPO, SETUP, digest, file_sha, load_request, record_identity, require_pilot, write_json,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_report(report, manifest, records, pilot):
    sha = digest(manifest)
    require_pilot(pilot, sha, report["runtime"], manifest["pilot_ids"])
    require(report["stage"] == "full" and report["passed"] is True
            and report["manifest_sha256"] == sha and report["prefix_strategy"] == "direct",
            "full report identity or status differs")
    expected = {record_identity(row)["record_id"] for row in records}
    by_id = {row["record_id"]: row for row in report["rows"]}
    require(len(by_id) == len(report["rows"]) == len(expected)
            and set(by_id) == set(report["record_sha256"]) == expected
            and report["completed_records"] == report["requested_records"] == len(expected),
            "full report coverage differs")
    require(report["new_model_forwards"] == (len(expected)-report["cached_records"])*len(manifest["prefix_lengths"]),
            "full model forward count differs")
    for rid, sha in pilot["record_sha256"].items():
        require(report["record_sha256"][rid] == sha, "validated pilot record changed")
    return by_id


def validate_record(payload, record, report_row, manifest, runtime):
    data = payload["data"]
    ids, lengths = record["token_ids"], manifest["prefix_lengths"]
    identity = record_identity(record)
    pilot = identity["record_id"] in manifest["pilot_ids"]
    require(payload["identity"] == {"manifest_sha256": digest(manifest), "runtime": runtime, "input": identity}
            and payload["data_sha256"] == digest(data), "record checksum or execution identity differs")
    require(data["input"] == identity and data["completion_sha256"] == digest(ids)
            and data["completion_length"] == len(ids) and data["protocol"] == manifest["protocol"]
            and data["prefix_strategy"] == "direct", "completion identity or strategy differs")
    require(data["actual_model_inputs_verified"] is True
            and data["forward_lengths"] == [n for n in sorted(lengths) for _ in range(2 if pilot else 1)],
            "actual raw-completion model input schedule differs")
    require(report_row["record_id"] == identity["record_id"] and report_row["method"] == record["method"]
            and report_row["results"] == data["results"] and report_row["validation"] == data["validation"],
            "report row differs from stored record")
    keys = set(map(str, lengths))
    require(set(data["entropies_by_prefix"]) == set(data["results"]) == keys,
            "per-prefix coverage differs")
    if pilot:
        require(data["validation"]["performed"] is True and data["validation"]["passed"] is True
                and set(data["validation"]["prefixes"]) == keys
                and all(c["entropy_exact"] and c["upstream_result_exact"] and c["decision_equal"]
                        for c in data["validation"]["prefixes"].values()), "pilot exact checks differ")
    for n in lengths:
        h, result = data["entropies_by_prefix"][str(n)], data["results"][str(n)]
        require(len(h) == n-1 and all(math.isfinite(x) and x >= 0 for x in h),
                "direct entropy length or values differ")
        require(result["completion_length"] == n and result["entropy_count"] == n-1
                and result["completion_sha256"] == digest(ids[:n])
                and result["upstream_commit"] == manifest["upstream_commit"]
                and result["protocol"] == manifest["protocol"], "prefix result identity differs")
        p = result["upstream"].get("p_value_weighted")
        require(p is None or (math.isfinite(p) and 0 <= p <= 1), "invalid weighted p-value")
        require(result["comparison"] == {"score_field": "p_value_weighted", "nominal_fpr": .001,
                "p_value": p, "decision": p is not None and p < .001, "abstained": p is None},
                "weighted comparison decision differs")
    return data


def historical_rows(manifest, records):
    root = REPO / "outputs/controlled_baseline_full" / manifest["generation_run"]
    name = "controlled_baseline_full_prompt_level.jsonl"
    artifact = json.loads((root / "controlled_baseline_full_artifact_manifest.json").read_text())
    sha = next(item["sha256"] for item in artifact["artifacts"] if item["path"] == name)
    require(file_sha(root / name) == sha, "historical comparison changed")
    inputs = {record_identity(row)["record_id"]: row for row in records}
    by_key = {}
    for line in (root / name).open():
        row = json.loads(line)
        if row["method"] != "textseal":
            continue
        method = "null" if row["sample_type"] == "null" else "textseal"
        rid = f"{method}/{row['prompt_index']:04d}"
        ids = inputs[rid]["token_ids"]
        token_sha = hashlib.sha256(f"int64:({len(ids)},):".encode() + struct.pack(f"<{len(ids)}q", *ids)).hexdigest()
        require(row["generated_token_hash"] == token_sha, "historical TextSeal/null completion differs")
        key = (rid, str(row["prefix_length"]))
        require(key not in by_key and row["prefix_length"] in manifest["prefix_lengths"], "historical duplicate or unexpected prefix")
        require(row["decision"] == (row["p_value"] < .001), "historical decision rule differs")
        by_key[key] = row
    require(set(by_key) == {(rid, str(n)) for rid in inputs for n in manifest["prefix_lengths"]},
            "historical TextSeal coverage differs")
    return by_key, {"path": str((root/name).relative_to(REPO)), "sha256": sha,
                    "scoring": "Historical published decisions using prompt-conditioned generation entropy."}


async def collect(setup=SETUP):
    import modal
    setup = Path(setup)
    manifest = json.loads((setup / "native8b_manifest.json").read_text())
    sha = digest(manifest)
    manifest, records = load_request(setup / "native8b_manifest.json", "full", sha)
    pilot = json.loads((setup / "pilot_report.json").read_text())
    report = json.loads((setup / "full_report.json").read_text())
    by_id = validate_report(report, manifest, records, pilot)
    historical, history_source = historical_rows(manifest, records)
    cohort_path = REPO / "outputs/comparison_redetect/prc_shared_nulls/cohort_audit.json"
    audit = json.loads(cohort_path.read_text())
    null_hashes = {row["prompt_index"]: row["shared_tokens_sha256"] for row in audit["records"]}
    for row in records:
        if row["method"] == "null":
            raw = struct.pack(f"<{len(row['token_ids'])}q", *row["token_ids"])
            require(hashlib.sha256(raw).hexdigest() == null_hashes[row["prompt_index"]], "PRC shared null identity differs")
    root = f"textseal_completion_redetect/{sha}"
    volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
    remote_report = b"".join([chunk async for chunk in volume.read_file.aio(f"{root}/full.json")])
    require(hashlib.sha256(remote_report).hexdigest() == file_sha(setup / "full_report.json"), "remote full report bytes differ")
    semaphore, completed = asyncio.Semaphore(8), 0

    async def fetch(record):
        nonlocal completed
        async with semaphore:
            rid = record_identity(record)["record_id"]
            path = setup / "cache" / f"{rid}.json"
            if not path.exists():
                raw = b"".join([chunk async for chunk in volume.read_file.aio(f"{root}/records/{rid}.json")])
                require(hashlib.sha256(raw).hexdigest() == report["record_sha256"][rid], "remote record hash differs")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            require(file_sha(path) == report["record_sha256"][rid], "local cached record changed")
            data = validate_record(json.loads(path.read_text()), record, by_id[rid], manifest, report["runtime"])
            completed += 1
            if completed % 100 == 0:
                print(f"Verified {completed}/{len(records)} stored TextSeal records", flush=True)
            return rid, data

    verified = dict(await asyncio.gather(*(fetch(row) for row in records)))
    counts = {}
    for n in map(str, manifest["prefix_lengths"]):
        counts[n] = {}
        for method in ("textseal", "null"):
            cohort = [(rid, row) for rid, row in verified.items() if rid.startswith(method+"/")]
            pairs = [(historical[(rid, n)]["decision"], data["results"][n]["comparison"]["decision"])
                     for rid, data in cohort]
            require(len(pairs) == 500, "denominator must include all 500 completions")
            counts[n][method] = {"count": len(pairs), "detected": sum(b for a,b in pairs),
                "old_detected": sum(a for a,b in pairs), "gained": sum(not a and b for a,b in pairs),
                "lost": sum(a and not b for a,b in pairs),
                "abstained": sum(data["results"][n]["comparison"]["abstained"] for rid,data in cohort)}
    summary = {"method": "textseal", "protocol": manifest["protocol"], "prefix_strategy": "direct",
               "score_field": "p_value_weighted", "nominal_fpr": .001, "alpha": .1, "ngram": 3,
               "counts": counts, "upstream_commit": manifest["upstream_commit"], "runtime": report["runtime"],
               "manifest_sha256": sha, "model": manifest["model"], "modal_root": root,
               "result_volume": "prc-completion-only", "full_report_sha256": file_sha(setup/"full_report.json"),
               "historical_source": history_source, "shared_null_source": manifest["null_source"],
               "prc_cohort_audit_sha256": file_sha(cohort_path), "collector_sha256": file_sha(__file__),
               "cost": {k:report[k] for k in ("load_seconds", "total_seconds", "measured_resource_usd", "new_model_forwards", "cached_records")},
               "checks": {"record_checksums_and_inputs_verified": len(verified),
                          "prefix_results_verified": sum(len(row["results"]) for row in verified.values()),
                          "pilot_records_unchanged": len(pilot["record_sha256"]), "shared_nulls_match_prc": True,
                          "denominators_include_abstentions": True, "local_model_forwards": 0,
                          "numerical_scores_from_pinned_upstream_runtime": True}}
    write_json(setup/"full_summary.json", summary)
    print(json.dumps({"counts": counts, "checks": summary["checks"], "cost": summary["cost"]}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=SETUP)
    args = parser.parse_args()
    asyncio.run(collect(args.setup))


if __name__ == "__main__":
    main()
