"""One explicitly approved stage of cached 8B -> 0.6B online PRC replay.

No generation, reference replay, benchmark, adaptive stopping, or automatic retry.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import modal
import modal_run as rt
from fixed_4b_comparison import check_code, file_ref, load_model, write_json

OUT = Path("outputs/online_8b_to_0p6b_redetect_setup")
STAGES = ("prepare", "replay", "score")
app = modal.App("prc-online-8b-to-0p6b", image=rt.image.add_local_python_source(
    "online_8b_to_0p6b", "fixed_4b_comparison", "online_prc_redetection"))


def case_folder(case_id):
    if case_id not in ("eta005_T1280_N500", "eta010_T3072_N500", "eta015_T6144_N100"):
        raise ValueError("unknown reviewed cohort")
    return OUT / "cases" / case_id


def verify_plan(plan, case_id):
    case_folder(case_id)
    if plan["branch"] != "redetection" or plan["null_count"] != 0:
        raise ValueError("this proposal is watermarked-only on redetection")
    if plan["protocol"] != rt.REDETECT_PROTOCOL or plan["automatic_retries"] != 0:
        raise ValueError("protocol or retry policy changed")
    spec = plan["model"]
    if (spec["id"], spec["revision"], spec["dtype"]) != (
            "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "bfloat16"):
        raise ValueError("unreviewed detector checkpoint")
    selected = next(c for c in plan["cases"] if c["id"] == case_id)
    expected = {"eta005_T1280_N500": (.05, 1280, 500), "eta010_T3072_N500": (.10, 3072, 500),
                "eta015_T6144_N100": (.15, 6144, 100)}[case_id]
    if (selected["eta"], selected["T"], selected["N"]) != expected:
        raise ValueError("cohort scope changed")
    if not selected["lengths"] or max(selected["lengths"]) != selected["T"] or 4096 in selected["lengths"]:
        raise ValueError("invalid prefix list")
    return selected


def prepare(plan, selected, source):
    import torch
    from detectors import tensor_sha256
    check_code(plan)
    if rt._redetect_sha(Path("/results") / selected["source_root"] / "manifest.json") != selected["source_manifest_sha256"]:
        raise ValueError("original cloud manifest changed")
    if source != json.loads((Path("/results") / selected["source_root"] / "manifest.json").read_text()):
        raise ValueError("source manifest differs")
    original = source["case"]
    if source["protocol"] != plan["protocol"] or original["construction"] != "online":
        raise ValueError("source protocol differs")
    artifact_ref = selected["source_artifact"]
    if original.get("artifact", artifact_ref) != artifact_ref:
        raise ValueError("original artifact reference differs")
    artifact = rt._redetect_source(artifact_ref, {"data": "/data"})
    key = artifact["online_key"]
    if (artifact["experiment_seed"], artifact["T"], key["noise_rate"], key["check_weight"],
            key["row_rate_numerator"], key["row_rate_denominator"]) != (
            12345, selected["T"], selected["eta"], 3, 99, 100):
        raise ValueError("original online key or generation length differs")
    if tensor_sha256(artifact["partition"]) != plan["partition_sha256"]:
        raise ValueError("partition differs")
    for name, digest in {**rt._redetect_model_spec(plan["model"]), **plan["model"]["metadata_sha256"]}.items():
        path = Path("/cache") / plan["model"]["cache_directory"] / name
        if not path.exists() or rt._redetect_sha(path) != digest:
            raise ValueError("expected pinned checkpoint missing/changed; no download authorized")
    records = [r for r in original["records"] if r["source"] == "wm"]
    if [r["prompt_idx"] for r in records] != list(range(selected["N"])):
        raise ValueError("require every original watermarked candidate in order")
    case = {"id": selected["id"], "generation_model": "Qwen3-8B-Base", "construction": "online",
            "artifact": artifact_ref, "lengths": selected["lengths"], "weights": ["map", "entropy"],
            "fpr": .001, "fpr_policy": "one_shot", "null_policy": "not_evaluated",
            "batch_size": selected["batch_size"], "cache": "static", "records": records}
    execution = {"git_commit": plan["head"], "files": plan["runtime_source_sha256"],
                 "gpu": "A100-80GB", "source_hashes_authoritative": True}
    prepared = rt._prepare_redetection(case, plan["model"], execution, {"data": "/data"}, "/results")
    root = Path("/results") / prepared["root"]
    rt._redetect_write(root / "prepared.json", prepared)
    refs = [file_ref(root / name, "results") for name in ("manifest.json", "prepared.json", "artifact.pt")]
    refs.extend(file_ref(Path("/results") / b["root"] / "inputs.pt", "results") for b in prepared["batches"])
    refs.append(file_ref(Path("/data") / artifact_ref["path"], "data"))
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": refs}


def score(prepared):
    from detectors import prepare_online_map_prefix_context
    from online_prc_redetection import prefix_scores
    root = Path("/results") / prepared["root"]
    if rt._redetect_sha(root / "artifact.pt") != prepared["artifact_sha256"]:
        raise ValueError("scoring artifact changed")
    artifact = rt._redetect_load(root / "artifact.pt")
    case = prepared["run"]["case"]
    context = prepare_online_map_prefix_context(artifact["online_key"], max(case["lengths"]))
    records, hashes = [], {}
    for batch in prepared["batches"]:
        inputs = rt._redetect_inputs(batch, "/results")
        path = Path("/results") / batch["root"] / "trace.pt"
        trace = rt._redetect_trace(path, batch["identity"])
        hashes[batch["root"]] = rt._redetect_sha(path)
        for row, probabilities in enumerate(trace["probabilities_2_to_T"].numpy()):
            ref = case["records"][batch["identity"]["start"] + row]
            scores = prefix_scores(context, inputs["tokens"][row], probabilities, inputs["partition"],
                                   case["lengths"], completion_only=True)
            compact = {n: {w: {k: (None if isinstance(s[k], float) and not math.isfinite(s[k]) else s[k])
                                  for k in ("decision", "statistic", "threshold", "V", "r", "status")}
                           for w, s in values.items()} for n, values in scores.items()}
            records.append({k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")} | {"scores": compact})
    if records and [(r["source"], r["prompt_idx"], r["tokens_sha256"]) for r in records] != [
            (r["source"], r["prompt_idx"], r["tokens_sha256"]) for r in case["records"]]:
        raise ValueError("candidate coverage differs")
    if len(records) != len(case["records"]) or not records:
        raise ValueError("require all original candidates")
    counts = {str(n): {w: {s: {"detected": sum(r["scores"][str(n)][w]["decision"] for r in records if r["source"] == s),
                              "count": sum(r["source"] == s for r in records)} for s in ("wm", "null")}
                           for w in case["weights"]} for n in case["lengths"]}
    key = context["online_key"]
    report = {"passed": True, "protocol": rt.REDETECT_PROTOCOL, "counts": counts,
              "reported_lengths": case["lengths"], "stop_rule": "none; all frozen comparison lengths",
              "settings": {"eta": key.noise_rate, "t": key.check_weight,
                           "r_setting": "causal round(0.99L), startup-clamped"},
              "trace_shard_sha256": hashes, "records": records}
    rt._redetect_write(root / "full.json", report)
    rt._redetect_write(root / "summary.json", {k: v for k, v in report.items() if k != "records"})
    rt.redetect_results.commit()
    return {"counts": counts, "files": [file_ref(root / name, "results") for name in ("full.json", "summary.json")]}


def execute(stage, payload):
    plan = payload["plan"]
    selected = verify_plan(plan, payload["case_id"])
    check_code(plan)
    # Complete scientific/native imports before the compatibility pickle loader.
    import torch
    import scipy.special
    import galois
    import transformers
    import qwen
    import detectors
    torch.set_num_threads(4)
    if stage == "prepare":
        return prepare(plan, selected, payload["source"])
    prepared = payload["prepared"]
    if stage == "score":
        return score(prepared)
    if stage != "replay":
        raise ValueError("unknown stage")
    we = load_model(plan["model"])
    results = []
    for batch in prepared["batches"]:
        result = rt._recover_redetection_batch(we.model, batch, "/results", validate=False)
        rt.redetect_results.commit()  # Preserve each primary batch immediately.
        results.append(result)
        print(json.dumps(result), flush=True)
    return {"batches": results, "files": [file_ref(Path("/results") / b["root"] / "trace.pt", "results")
                                          for b in prepared["batches"]]}


@app.function(cpu=(4, 4), memory=16384, timeout=3390, startup_timeout=30,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results})
def paid_stage(stage, payload):
    selected = verify_plan(payload["plan"], payload["case_id"])
    limit = selected["stages"][stage]["work_timeout_seconds"]
    folder = Path("/results") / "online_8b_to_0p6b_v1" / selected["id"] / "attempts" / stage
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    started = time.monotonic()
    try:
        with (folder / "worker.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", stage,
                                     str(folder / "request.json"), str(folder / "response.json")], stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise TimeoutError(f"{stage} exceeded {limit}s; no retry")
        if status:
            raise RuntimeError(f"{stage} failed with exit {status}; no retry")
        return json.loads((folder / "response.json").read_text())
    finally:
        write_json(folder / "timing.json", {"stage": stage, "case": selected["id"],
                    "wall_seconds": time.monotonic() - started, "resources": selected["stages"][stage]})
        rt.redetect_results.commit()


def collect(case_id, stage):
    folder = case_folder(case_id)
    result = json.loads((folder / f"result_{stage}.json").read_text())
    for ref in result["files"]:
        volume = rt.data_vol if ref["volume"] == "data" else rt.redetect_results
        data = b"".join(volume.read_file(ref["path"]))
        import hashlib
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("download checksum differs")
        path = folder / "cache" / ref["volume"] / ref["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in ("request.json", "response.json", "worker.log", "timing.json"):
        data = b"".join(rt.redetect_results.read_file(f"online_8b_to_0p6b_v1/{case_id}/attempts/{stage}/{name}"))
        path = folder / "evidence" / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if stage == "score":
        prepared = json.loads((folder / "result_prepare.json").read_text())["prepared"]
        report = json.loads((folder / "cache/results" / prepared["root"] / "full.json").read_text())
        rt._append_redetection_csv(prepared, report, rt.REDETECT_CSV)
    write_json(folder / f"collected_{stage}.json", result["files"])


@app.local_entrypoint()
def run(case: str, stage: str, approval_reference: str):
    if stage not in STAGES or not approval_reference.strip():
        raise ValueError("require a named stage and its explicit approval")
    folder = case_folder(case)
    plan = json.loads((OUT / "setup.json").read_text())
    selected = verify_plan(plan, case)
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("wrong Modal profile")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("wrong branch")
    check_code(plan)
    attempt = folder / f"attempt_{stage}.json"
    if attempt.exists():
        raise ValueError("stage already attempted; no automatic retry")
    plan_sha = rt._redetect_sha(OUT / "setup.json")
    payload = {"plan": plan, "case_id": case, "approval_reference": approval_reference}
    if stage == "prepare":
        source_path = OUT / selected["source_manifest_local"]
        if rt._redetect_sha(source_path) != selected["source_manifest_sha256"]:
            raise ValueError("original source manifest changed")
        payload["source"] = json.loads(source_path.read_text())
    else:
        if json.loads((folder / "attempt_prepare.json").read_text())["plan_sha256"] != plan_sha:
            raise ValueError("setup changed after preparation")
        prerequisite = "prepare" if stage == "replay" else "replay"
        if not (folder / f"collected_{prerequisite}.json").exists():
            raise ValueError("retrieve the preceding stage before proceeding")
        payload["prepared"] = json.loads((folder / "result_prepare.json").read_text())["prepared"]
    spec = selected["stages"][stage]
    write_json(attempt, {"case": case, "stage": stage, "approval_reference": approval_reference,
                         "plan_sha256": plan_sha, "resources": spec})
    options = {"memory": spec["memory_mib"], "timeout": spec["work_timeout_seconds"] + 30}
    if stage == "replay":
        options["gpu"] = "A100-80GB"
    result = paid_stage.with_options(**options).remote(stage, payload)
    write_json(folder / f"result_{stage}.json", result)
    collect(case, stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        write_json(sys.argv[4], execute(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
    elif len(sys.argv) == 4 and sys.argv[1] == "collect":
        collect(sys.argv[2], sys.argv[3])
    else:
        print((OUT / "PLAN.md").read_text())
