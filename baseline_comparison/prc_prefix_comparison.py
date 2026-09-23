"""Rescore the indexed native-8B PRC n=1024 cohort at shorter lengths on CPU.

Uses existing raw-completion probabilities and the integrated detector unchanged.
Never loads a model, calls Modal, or changes the original result CSV.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .config import PREFIX_LENGTHS

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "outputs/redetection"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_json(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"source checksum differs: {path}")
    return json.loads(Path(path).read_text())


def compare_prefixes(generation_cache: Path, output: Path, lengths=PREFIX_LENGTHS):
    existing_provenance = output.with_suffix(".provenance.json")
    if existing_provenance.exists() and json.loads(existing_provenance.read_text()).get("shared_null_alignment"):
        raise ValueError("comparison already uses shared nulls; choose a separate --output for the original T1382 cohort")
    import torch
    from detectors import detect_online_hoeffding, semantic_sha256, tensor_sha256
    from modal_run import (
        _append_redetection_csv, _redetect_inputs, _redetect_load,
        _redetect_trace, _redetect_write, _score_redetection,
    )
    from online_prc import OnlinePRCKey, materialize_supports

    lengths = sorted(set(lengths) | {1024})  # Published endpoint is a regression check.
    if any(type(n) is not int or not 1 <= n <= 1024 for n in lengths):
        raise ValueError("prefix lengths must be integers in [1, 1024]")
    index_path = RESULTS / "cache_index.json"
    index = json.loads(index_path.read_text())
    indexed = {row["case"]: row for row in index["runs"]}
    endpoint = indexed["same_8b_eta005_n1024"]
    parent = indexed["same_8b_eta005_n1280"]
    endpoint_path = RESULTS / endpoint["local_result_file"]
    reference = verified_json(endpoint_path, endpoint["result_sha256"])
    reference_old = verified_json(endpoint_path.parent / "old_tpr.json", endpoint["old_tpr_sha256"])
    parent_path = RESULTS / parent["local_result_file"]
    parent_report = verified_json(parent_path, parent["result_sha256"])
    origin = verified_json(endpoint_path.parent / "provenance.json", endpoint["provenance_sha256"])
    prepared_path = parent_path.parent / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    if prepared["root"] != parent["modal_root"] or semantic_sha256(prepared["run"])[:24] != parent["run_id"]:
        raise ValueError("parent run identity differs")
    case = prepared["run"]["case"]
    if (case["construction"] != "online" or case["fpr_policy"] != "one_shot"
            or case["weights"] != ["map", "entropy"] or case["fpr"] != .001):
        raise ValueError("unexpected source detector configuration")
    expected_ids = {(source, i) for source in ("wm", "null") for i in range(500)}
    if len(case["records"]) != 1000 or {(r["source"], r["prompt_idx"]) for r in case["records"]} != expected_ids:
        raise ValueError("source cohort must have 500 original watermarked and null records")
    for filename in ("detectors.py", "online_prc.py"):
        if sha256(REPO / filename) != prepared["run"]["execution"]["files"][filename]:
            raise ValueError(f"scoring implementation differs from parent inference: {filename}")

    cache = parent_path.parent / "cache"
    artifact_path = cache / prepared["root"] / "artifact.pt"
    if sha256(artifact_path) != prepared["artifact_sha256"]:
        raise ValueError("scoring artifact differs")
    artifact = _redetect_load(artifact_path)
    original_artifact_path = generation_cache / case["artifact"]["path"]
    if sha256(original_artifact_path) != case["artifact"]["sha256"]:
        raise ValueError("original generation artifact differs")
    original_artifact = _redetect_load(original_artifact_path)
    if (original_artifact["online_key"] != artifact["online_key"]
            or not torch.equal(original_artifact["partition"], artifact["partition"])):
        raise ValueError("original and redetection PRC artifacts disagree")

    # Freeze all existing batches before any score is computed.
    expected_batches = {batch["root"]: batch for batch in origin["batches"]}
    source_hashes, original_rows = [], []
    for batch in prepared["batches"]:
        expected = expected_batches[batch["root"]]
        trace_path = cache / batch["root"] / "trace.pt"
        trace_hash = sha256(trace_path)
        if (trace_hash != expected["trace_sha256"]
                or trace_hash != parent_report["trace_shard_sha256"][batch["root"]]
                or trace_hash != reference["trace_shard_sha256"][f"cache/{batch['root']}"]
                or batch["identity"]["input_sha256"] != expected["input_sha256"]):
            raise ValueError("cached batch differs from published n=1024 evidence")
        inputs = _redetect_inputs(batch, cache)
        _redetect_trace(trace_path, batch["identity"])
        if tensor_sha256(inputs["partition"]) != prepared["partition_sha256"]:
            raise ValueError("input partition differs")
        for row, tokens in enumerate(inputs["tokens"]):
            ref = case["records"][batch["identity"]["start"] + row]
            if hashlib.sha256(tokens.numpy().tobytes()).hexdigest() != ref["tokens_sha256"]:
                raise ValueError("parent completion token hash differs")
            if ref["source"] != "wm":
                continue
            path = generation_cache / ref["file"]["path"]
            if sha256(path) != ref["file"]["sha256"] or path.stat().st_size != ref["file"]["bytes"]:
                raise ValueError("historical generation record differs")
            old = _redetect_load(path)
            if (old["prompt_idx"] != ref["prompt_idx"] or not old["watermark"]
                    or not torch.equal(old["tokens"][:len(tokens)], tokens)):
                raise ValueError("historical and completion-only candidates differ")
            original_rows.append((ref, old))
        source_hashes.append({"root": batch["root"], "trace_sha256": trace_hash,
                              "input_sha256": expected["input_sha256"]})
    if len(original_rows) != 500:
        raise ValueError("incomplete historical watermarked coverage")

    # Separate namespace: parent reports, traces, and result CSV remain immutable.
    sweep = copy.deepcopy(prepared)
    sweep["root"] = f"{prepared['root']}/prefixes/comparison-" + "-".join(map(str, lengths))
    sweep["run"]["case"]["lengths"] = lengths
    run_root = cache / sweep["root"]
    run_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(artifact_path, run_root / "artifact.pt")
    print(f"Scoring {len(lengths)} prefixes on 1,000 cached raw-completion traces", flush=True)
    report = _score_redetection(sweep, cache)
    saved = {(r["source"], r["prompt_idx"]): r for r in reference["records"]}
    for row in report["records"]:
        if row["scores"]["1024"] != saved[(row["source"], row["prompt_idx"])]["scores"]["1024"]:
            raise ValueError("n=1024 per-candidate score differs from published evidence")
    if report["counts"]["1024"] != reference["counts"]["1024"]:
        raise ValueError("n=1024 aggregate differs")

    # Prompt-conditioned probabilities are used only for labelled old controls.
    old_counts = {str(n): {w: {"detected": 0, "count": 500} for w in case["weights"]} for n in lengths}
    old_records = []
    saved_old = {row["prompt_idx"]: row for row in reference_old["records"]}
    for ref, record in original_rows:
        decisions = {}
        for n in lengths:
            decisions[str(n)] = {}
            for weight in case["weights"]:
                decision = bool(detect_online_hoeffding(
                    artifact["online_key"], record["tokens"][:n], record["p_trace"][:n],
                    artifact["partition"], fpr=case["fpr"], weight=weight,
                    fpr_policy=case["fpr_policy"], completion_only=False,
                ))
                decisions[str(n)][weight] = decision
                old_counts[str(n)][weight]["detected"] += decision
        if decisions["1024"] != saved_old[ref["prompt_idx"]]["decisions"]:
            raise ValueError("historical n=1024 decision differs")
        old_records.append({"prompt_idx": ref["prompt_idx"], "source_sha256": ref["file"]["sha256"],
                            "decisions": decisions})
        if len(old_records) % 100 == 0:
            print(f"Historical control: {len(old_records)}/500", flush=True)
    if old_counts["1024"] != reference_old["counts"]["1024"]:
        raise ValueError("historical n=1024 aggregate differs")
    old_path = run_root / "old_tpr.json"
    old = {"detector_model": prepared["run"]["model"]["id"],
           "source": "cached_generation_p_trace; historical prompted scoring",
           "counts": old_counts, "records": old_records}
    _redetect_write(old_path, old)
    sweep["run"]["case"]["old_tpr"] = {k: v for k, v in old.items() if k != "records"} | {"evidence_sha256": sha256(old_path)}
    _redetect_write(run_root / "prepared.json", sweep)
    source_csv = run_root / "comparison.csv"
    source_csv.unlink(missing_ok=True)
    _append_redetection_csv(sweep, report, source_csv)
    with source_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    fields += ["Posterior TPR change (pp)", "Entropy TPR change (pp)"]
    for row in rows:
        n = row["n"]
        for weight, field in (("map", fields[-2]), ("entropy", fields[-1])):
            row[field] = f"{(report['counts'][n][weight]['wm']['detected'] - old_counts[n][weight]['detected']) / 5:.1f}"
        row["Notes"] += "; local CPU prefix rescore; new inference=0; original null cohort=T1382; each prefix is a separate one-shot test"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    key = OnlinePRCKey.from_dict(artifact["online_key"])
    parent_supports = materialize_supports(1280, key)
    supports = {}
    for n in lengths:
        support = materialize_supports(n, key)
        if not (support == parent_supports[:len(support)]).all():
            raise ValueError("original PRC prefix supports changed")
        supports[str(n)] = {"checks": len(support), "checks_containing_coordinate1": int((support == 0).any(axis=1).sum())}
    provenance = {
        "protocol": report["protocol"], "parent_modal_root": prepared["root"],
        "lengths": lengths, "watermarked_candidates": 500, "null_candidates": 500,
        "null_source": "_nulls/qwen3_8b_base/T1382", "cohort": "same as published native-8B n1024",
        "null_cohort_note": "Different source from the T13088 shared nulls in the full TextSeal comparison; do not substitute these FPR rows without a token-identity check.",
        "model": prepared["run"]["model"], "inference_commit": prepared["run"]["execution"]["git_commit"],
        "execution_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "scoring_files_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                                 (Path(__file__).resolve(), REPO / "detectors.py", REPO / "online_prc.py", REPO / "modal_run.py")},
        "parent_prepared_sha256": sha256(prepared_path), "endpoint_report_sha256": sha256(endpoint_path),
        "endpoint_old_tpr_sha256": sha256(endpoint_path.parent / "old_tpr.json"),
        "artifact_sha256": prepared["artifact_sha256"], "partition_sha256": prepared["partition_sha256"],
        "batches": source_hashes, "prefix_supports": supports,
        "checks": {"all_2000_endpoint_raw_scores_exact": True, "all_1000_endpoint_old_decisions_exact": True,
                   "original_candidates_and_source_hashes_match": True, "original_prefix_supports_match": True},
        "model_inference_calls": 0, "remote_compute_jobs": 0,
        "input_rule": "For length n: original tokens[:n], cached response-only probabilities[:n-1], score coordinate 1=0",
        "fpr_policy": "one_shot at 0.001 for each separately reported prefix; no OR across lengths",
        "csv_sha256": sha256(output), "counts": report["counts"], "old_counts": old_counts,
        "local_detail_root": str(run_root.relative_to(REPO)),
        "detail_sha256": {name: sha256(run_root / name) for name in ("full.json", "old_tpr.json", "prepared.json")},
    }
    _redetect_write(output.with_suffix(".provenance.json"), provenance)
    print(json.dumps({"csv": str(output), "counts": report["counts"], "model_inference_calls": 0}, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/comparison_redetect/baseline_comparisons.csv")
    parser.add_argument("--lengths", type=int, nargs="+", default=PREFIX_LENGTHS)
    args = parser.parse_args()
    compare_prefixes(args.generation_cache, args.output, args.lengths)


if __name__ == "__main__":
    main()
