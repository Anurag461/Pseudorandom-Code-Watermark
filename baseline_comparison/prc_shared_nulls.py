"""Align PRC with the frozen comparison nulls, reusing watermarked raw traces.

prepare is local only. run requires approval of its frozen plan; collect only
downloads existing traces and scores locally. No generation or TextSeal calls.
All model replay and detection use the existing modal_run implementation.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from .config import PREFIX_LENGTHS
from .prc_prefix_comparison import REPO, RESULTS, sha256, verified_json
from .textseal_preflight import NULL_ROOT, historical_token_hashes

OUTPUT = REPO / "outputs/comparison_redetect"
SETUP = OUTPUT / "prc_shared_nulls"
CODE_FILES = ("baseline_comparison/prc_shared_nulls.py", "baseline_comparison/config.py",
              "baseline_comparison/comparison_runner.py",
              "baseline_comparison/prc_prefix_comparison.py", "baseline_comparison/textseal_preflight.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def raw_token_hash(tokens):
    return hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest()


def reuse_trace(parent_batch, target_batch, parent_cache, cache, parent_trace_sha):
    """Crop a verified trace only when the actual raw token prefix is identical."""
    import torch
    from detectors import tensor_sha256
    from modal_run import _redetect_inputs, _redetect_trace, _redetect_write
    source_path = parent_cache / parent_batch["root"] / "trace.pt"
    if sha256(source_path) != parent_trace_sha:
        raise ValueError("watermarked source trace changed")
    before = _redetect_inputs(parent_batch, parent_cache)
    after = _redetect_inputs(target_batch, cache)
    n = target_batch["identity"]["length"]
    if (not torch.equal(before["tokens"][:, :n], after["tokens"])
            or not torch.equal(before["partition"], after["partition"])):
        raise ValueError("watermarked tokens or partition changed")
    saved = _redetect_trace(source_path, parent_batch["identity"])
    probabilities = saved["probabilities_2_to_T"][:, :n-1].clone()
    payload = {**saved, "identity": target_batch["identity"],
               "probabilities_2_to_T": probabilities,
               "probabilities_sha256": tensor_sha256(probabilities),
               "reused_from": {"root": parent_batch["root"], "trace_sha256": parent_trace_sha}}
    _redetect_write(cache / target_batch["root"] / "trace.pt", payload)
    _redetect_trace(cache / target_batch["root"] / "trace.pt", target_batch["identity"])
    return payload["reused_from"]


def verify_wm_scores(report, reference):
    before = {r["prompt_idx"]: r["scores"] for r in reference["records"] if r["source"] == "wm"}
    after = {r["prompt_idx"]: r["scores"] for r in report["records"] if r["source"] == "wm"}
    if before != after or len(after) != 500:
        raise ValueError("watermarked per-record scores changed")


def prepare(generation_cache: Path, setup: Path = SETUP):
    import torch
    from detectors import semantic_sha256
    from modal_run import (EXECUTION_FILES, _prepare_redetection, _redetect_inputs,
                           _score_redetection)
    from .comparison_runner import _token_sha256

    index = json.loads((RESULTS / "cache_index.json").read_text())
    parent_index = next(r for r in index["runs"] if r["case"] == "same_8b_eta005_n1280")
    parent_path = RESULTS / parent_index["local_result_file"]
    parent_report = verified_json(parent_path, parent_index["result_sha256"])
    parent = json.loads((parent_path.parent / "prepared.json").read_text())
    if (semantic_sha256(parent["run"])[:24] != parent_index["run_id"]
            or parent["root"] != parent_index["modal_root"]):
        raise ValueError("parent identity changed")
    parent_cache = parent_path.parent / "cache"
    provenance_path = OUTPUT / "baseline_comparisons.provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("null_source") != "_nulls/qwen3_8b_base/T1382":
        raise ValueError("expected the original T1382 comparison; do not reset an aligned table")
    csv_path = OUTPUT / "baseline_comparisons.csv"
    if sha256(csv_path) != provenance["csv_sha256"]:
        raise ValueError("comparison CSV differs from its provenance")
    reference = verified_json(REPO / provenance["local_detail_root"] / "full.json",
                              provenance["detail_sha256"]["full.json"])
    preflight_path = OUTPUT / "preflight/preflight.json"
    preflight = json.loads(preflight_path.read_text())
    inputs_path = preflight_path.parent / "completion_inputs.jsonl"
    if sha256(inputs_path) != preflight["completion_inputs"]["sha256"]:
        raise ValueError("clean comparison export changed")
    clean = {(r["method"], r["prompt_index"]): r for r in
             map(json.loads, inputs_path.read_text().splitlines())}
    historical = historical_token_hashes()
    expected_ids = {(s, i) for s in ("wm", "null") for i in range(500)}
    old_case = parent["run"]["case"]
    if (len(old_case["records"]) != 1000
            or {(r["source"], r["prompt_idx"]) for r in old_case["records"]} != expected_ids
            or old_case["construction"] != "online" or old_case["weights"] != ["map", "entropy"]
            or old_case["fpr_policy"] != "one_shot" or old_case["fpr"] != .001):
        raise ValueError("unexpected parent cohort or detector")
    files = {name: sha256(REPO / name) for name in (*EXECUTION_FILES, *CODE_FILES)}
    # Replay and scoring numerics must still match the watermarked parent.
    for name in ("qwen.py", "detectors.py", "online_prc.py", "watermark_expt.py", "constants.py"):
        if files[name] != parent["run"]["execution"]["files"][name]:
            raise ValueError(f"parent numerical implementation changed: {name}")
    source_files = {(r["kind"], r["index"]): r for r in preflight["source_files"]}
    records = []
    for source, i in [(r["source"], r["prompt_idx"]) for r in old_case["records"]]:
        kind = "online_prc" if source == "wm" else "null"
        row = clean[(kind, i)]
        if len(row["token_ids"]) != 1024 or _token_sha256(row["token_ids"]) != historical[(kind, i)]:
            raise ValueError("completion differs from the original comparison")
        ref = source_files[(kind, i)]
        if source == "null" and ref["path"] != f"{NULL_ROOT}/null_{i:04d}.pt":
            raise ValueError("unexpected shared null source")
        records.append({"source": source, "prompt_idx": i,
                        "file": {"volume": "data", **{k: ref[k] for k in ("path", "sha256", "bytes")}},
                        "tokens_sha256": raw_token_hash(torch.tensor(row["token_ids"], dtype=torch.int64))})
    case = {**copy.deepcopy(old_case), "id": "comparison_native8b_shared_nulls_n1024",
            "lengths": list(PREFIX_LENGTHS), "records": records}
    case.pop("old_tpr", None)
    execution = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                 "files": files, "gpu": "H100", "allocator": "expandable_segments:True"}
    cache = setup / "cache"
    prepared = _prepare_redetection(case, parent["run"]["model"], execution,
                                    {"data": generation_cache}, cache)
    if len(prepared["batches"]) != 8 or any(b["identity"]["count"] != 125 for b in prepared["batches"]):
        raise ValueError("expected eight batches of 125")
    reused, replay, audit_rows = [], [], []
    for before, after in zip(parent["batches"], prepared["batches"]):
        a = _redetect_inputs(before, parent_cache)["tokens"]
        b = _redetect_inputs(after, cache)["tokens"]
        start = after["identity"]["start"]
        refs = records[start:start + len(b)]
        if {r["source"] for r in refs} == {"wm"}:
            reused.append(reuse_trace(before, after, parent_cache, cache,
                                      parent_report["trace_shard_sha256"][before["root"]]))
        elif {r["source"] for r in refs} == {"null"}:
            replay.append(after["root"])
            for ref, old, new in zip(refs, a, b):
                differences = torch.nonzero(old[:1024] != new).flatten().tolist()
                audit_rows.append({"prompt_index": ref["prompt_idx"],
                                   "old_tokens_sha256": raw_token_hash(old[:1024]),
                                   "shared_tokens_sha256": raw_token_hash(new),
                                   "first_difference": differences[0] if differences else None})
        else:
            raise ValueError("mixed source batch")
    if len(reused) != 4 or len(replay) != 4 or len(audit_rows) != 500:
        raise ValueError("incomplete replay/reuse split")
    # All 6,000 weighted WM results must agree, before spending on any null replay.
    wm = copy.deepcopy(prepared)
    wm["root"] += "/wm_validation"
    wm["batches"] = prepared["batches"][:4]
    wm["run"]["case"]["records"] = records[:500]
    (cache / wm["root"]).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache / prepared["root"] / "artifact.pt", cache / wm["root"] / "artifact.pt")
    verify_wm_scores(_score_redetection(wm, cache), reference)
    write_json(setup / "prepared.json", prepared)
    write_json(setup / "cohort_audit.json", {"old_null_source": "_nulls/qwen3_8b_base/T1382",
               "shared_null_source": NULL_ROOT, "length": 1024,
               "exact_matches": sum(r["first_difference"] is None for r in audit_rows),
               "records": audit_rows})
    shutil.copyfile(csv_path, setup / "before.csv")
    shutil.copyfile(provenance_path, setup / "before.provenance.json")
    write_json(setup / "before.full.json", reference)
    plan = {"schema_version": 1, "status": "prepared_awaiting_launch_approval",
            "protocol": prepared["run"]["protocol"], "null_source": NULL_ROOT,
            "model": prepared["run"]["model"], "lengths": list(PREFIX_LENGTHS),
            "new_null_completions": 500, "reused_watermarked_completions": 500,
            "new_probabilities": 500 * 1023, "reused_batches": reused,
            "replay_batches": replay, "code_sha256": files,
            "local_files_sha256": {name: sha256(setup / name) for name in
                                   ("prepared.json", "cohort_audit.json", "before.csv", "before.provenance.json", "before.full.json")},
            "source_preflight_sha256": sha256(preflight_path),
            "clean_inputs_sha256": sha256(inputs_path),
            "checks": {"wm_all_6000_scores_exact": True, "original_comparison_tokens_exact": True},
            "runtime": {"gpu": "H100", "cpu": 4, "memory_mib": 65536, "max_containers": 1,
                        "retries": 0, "scaledown_seconds": 2, "batch_timeout_seconds": 300},
            "cost": {"estimated_usd": [1, 2], "resource_usd_per_second": .00129148,
                     "checked_date": "2026-09-17",
                     "pricing_source": "https://modal.com/pricing", "planning_ceiling_usd": 3},
            "remote_compute_jobs": 0, "generation_calls": 0,
            "validation": "First null batch: exact independent token-step replay plus batch-order/prefix check; stop on failure."}
    write_json(setup / "plan.json", plan)
    print(json.dumps({"plan_sha256": digest(plan), "null_matches": sum(r["first_difference"] is None for r in audit_rows),
                      "new_null_completions": 500, "reused_watermarked_completions": 500,
                      "wm_all_6000_scores_exact": True, "remote_compute_jobs": 0}, indent=2))
    return plan


def load_plan(setup, approved=None):
    from modal_run import _redetect_inputs
    plan = json.loads((setup / "plan.json").read_text())
    if approved is not None and approved != digest(plan):
        raise ValueError("explicit approval of the current plan SHA-256 is required")
    for name, expected in plan["code_sha256"].items():
        if sha256(REPO / name) != expected:
            raise ValueError(f"execution code changed: {name}")
    for name, expected in plan["local_files_sha256"].items():
        if sha256(setup / name) != expected:
            raise ValueError(f"frozen setup changed: {name}")
    prepared = json.loads((setup / "prepared.json").read_text())
    for batch in prepared["batches"]:
        _redetect_inputs(batch, setup / "cache")
    return plan, prepared


def run(setup, approved):
    """Cache reproducible inputs; send only the four new null batches to the GPU."""
    import modal
    from modal_run import RedetectionModel, app, redetect_results
    plan, prepared = load_plan(setup, approved)
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("use MODAL_PROFILE=new-prc-watermark")
    batches = [b for b in prepared["batches"] if b["root"] in plan["replay_batches"]]
    with redetect_results.batch_upload(force=True) as upload:
        for name in ("manifest.json", "artifact.pt"):
            path = prepared["root"] + "/" + name
            upload.put_file(setup / "cache" / path, path)
        for name in ("plan.json", "prepared.json"):
            upload.put_file(setup / name, prepared["root"] + "/" + name)
        for batch in prepared["batches"]:
            path = batch["root"] + "/inputs.pt"
            upload.put_file(setup / "cache" / path, path)
            if batch["root"] not in plan["replay_batches"]:
                path = batch["root"] + "/trace.pt"
                upload.put_file(setup / "cache" / path, path)
    runtime = plan["runtime"]
    with modal.enable_output(), app.run():
        worker = RedetectionModel.with_options(
            gpu=runtime["gpu"], cpu=(runtime["cpu"], runtime["cpu"]), memory=runtime["memory_mib"],
            max_containers=runtime["max_containers"], retries=runtime["retries"],
            scaledown_window=runtime["scaledown_seconds"], timeout=runtime["batch_timeout_seconds"],
        )(entropy_model_size="8B", generation_model_size="8B", trace_kv_cache_implementation="static",
          completion_model=json.dumps(plan["model"], sort_keys=True))
        for index, batch in enumerate(batches):
            print(json.dumps(worker.redetect_batch.remote(batch, validate=index == 0)), flush=True)
    collect(setup)


def aligned_rows(rows, report, reference, root):
    """Change only the six PRC null results and their provenance note."""
    verify_wm_scores(report, reference)
    updated, seen = copy.deepcopy(rows), set()
    for row in updated:
        if (row["PRC Construction"] != "online_causal_prc_v1" or row["eta"] != "0.05"
                or row["Entropy Model"] != "Qwen3-8B-Base" or row["Generation Model"] != "Qwen3-8B-Base"):
            continue
        n = row["n"]
        if n not in {str(n) for n in PREFIX_LENGTHS} or n in seen:
            raise ValueError("unexpected or duplicate PRC prefix row")
        seen.add(n)
        for weight, field in (("map", "Posterior FPR"), ("entropy", "Entropy FPR")):
            count = report["counts"][n][weight]["null"]
            if count["count"] != 500:
                raise ValueError("incomplete shared null results")
            row[field] = f"{count['detected']}/500 ({count['detected']/5:.1f}%)"
        row["Notes"] = row["Notes"].replace("new inference=0; original null cohort=T1382", "watermarked inference=0; shared null cohort=T13088")
        row["Notes"] += f"; null-only completion replay at n1024; shared-null report={root}/full.json"
    if seen != {str(n) for n in PREFIX_LENGTHS}:
        raise ValueError("missing PRC comparison prefixes")
    return updated


def collect(setup):
    """Read-only remote retrieval; validate and publish on local CPU."""
    from modal_run import _redetect_trace, _score_redetection, redetect_results
    plan, prepared = load_plan(setup)
    cache = setup / "cache"
    for batch in prepared["batches"]:
        if batch["root"] not in plan["replay_batches"]:
            continue
        path = cache / batch["root"] / "trace.pt"
        if not path.exists():
            data = b"".join(redetect_results.read_file(batch["root"] + "/trace.pt"))
            temporary = path.with_suffix(".download")
            temporary.write_bytes(data)
            _redetect_trace(temporary, batch["identity"])
            temporary.replace(path)
        payload = _redetect_trace(path, batch["identity"])
        if batch["root"] == plan["replay_batches"][0] and not payload["full_validation"]:
            raise ValueError("first null batch lacks independent replay validation")
    report = _score_redetection(prepared, cache)
    reference = json.loads((setup / "before.full.json").read_text())
    target = OUTPUT / "baseline_comparisons.csv"
    if (sha256(target) != plan["local_files_sha256"]["before.csv"]
            or sha256(target.with_suffix(".provenance.json")) != plan["local_files_sha256"]["before.provenance.json"]):
        raise ValueError("comparison CSV changed since setup; review the merge before publishing")
    with target.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames), list(reader)
    updated = aligned_rows(rows, report, reference, prepared["root"])
    # All checks pass before either current comparison file is replaced.
    temporary = target.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(updated)
    provenance = json.loads((setup / "before.provenance.json").read_text())
    provenance.update(null_source=NULL_ROOT, cohort="original full comparison: 500 WM + 500 shared nulls",
                      null_cohort_note="T1382 nulls replaced by completion-only replay of the exact frozen T13088 comparison tokens.",
                      counts=report["counts"], csv_sha256=sha256(temporary))
    # Keep the original CPU-only sweep's accounting under its own label.
    provenance["original_prefix_sweep"] = {
        k: provenance.pop(k) for k in ("model_inference_calls", "remote_compute_jobs", "local_detail_root", "detail_sha256", "batches", "checks")}
    provenance.update(local_detail_root=str((cache / prepared["root"]).relative_to(REPO)),
                      detail_sha256={"full.json": sha256(cache / prepared["root"] / "full.json")},
                      checks={"watermarked_all_6000_scores_exact": True,
                              "original_shared_null_tokens_exact": True, "null_replay_first_batch_validated": True},
                      batches=[{"root": b["root"], "input_sha256": b["identity"]["input_sha256"],
                                "trace_sha256": report["trace_shard_sha256"][b["root"]]}
                               for b in prepared["batches"]])
    provenance["shared_null_alignment"] = {
        "plan_sha256": digest(plan), "prepared_sha256": sha256(setup / "prepared.json"),
        "code_sha256": plan["code_sha256"],
        "cohort_audit_sha256": sha256(setup / "cohort_audit.json"),
        "null_trace_shard_sha256": {k: v for k, v in report["trace_shard_sha256"].items() if k in plan["replay_batches"]},
        "local_detail_root": str((cache / prepared["root"]).relative_to(REPO)),
        "full_sha256": sha256(cache / prepared["root"] / "full.json"),
        "watermarked_scores_unchanged": True, "new_null_completions": 500}
    write_json(target.with_suffix(".provenance.json"), provenance)
    temporary.replace(target)
    print(json.dumps({"csv": str(target), "shared_null_source": NULL_ROOT, "counts": report["counts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "collect"), default="prepare")
    parser.add_argument("--generation-cache", type=Path, default=Path("/tmp/comparison-redetect-cache"))
    parser.add_argument("--setup", type=Path, default=SETUP)
    parser.add_argument("--approved-plan-sha256", default="")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.generation_cache, args.setup)
    elif args.stage == "run":
        run(args.setup, args.approved_plan_sha256)
    else:
        collect(args.setup)


if __name__ == "__main__":
    main()
