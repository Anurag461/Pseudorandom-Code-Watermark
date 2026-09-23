"""Nine approval-gated parallel H200 batches; reuse the completed first 50.

Existing preparation, completion-only replay and cloud CPU scoring routines.
No generation, null cohort, 0.6B inference, benchmark or automatic retry.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import modal
import modal_run as rt
from fixed_4b_comparison import check_code, file_ref, load_model, write_json

OUT = Path("outputs/online_8b_eta020_remaining450_setup")
RUN = "online_8b_eta020_T14336_remaining450_v1"
STAGES = ("prepare", "replay", "score")
app = modal.App("prc-online-8b-eta020-remaining450", image=rt.image.add_local_python_source(
    "online_8b_eta020_remaining450", "fixed_4b_comparison", "online_8b_to_0p6b", "online_prc_redetection"))
VOLUMES = {"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results}


def verify_plan(plan):
    if (plan["run_id"], plan["branch"], plan["eta"], plan["T"], plan["N"], plan["batch_size"]) != (
            RUN, "redetection", .2, 14336, 450, 50):
        raise ValueError("unreviewed experiment")
    if (plan["prompt_indices"] != list(range(50, 500)) or plan["final_N"] != 500
            or plan["reporting_lengths"] != [14336]
            or plan["protocol"] != "completion_only_raw_abstain_v1"
            or any(plan[k] for k in ("new_generations", "null_count", "small_detector_records",
                                    "reference_replays", "automatic_retries", "benchmarks"))):
        raise ValueError("unreviewed cohort or extra work")
    if (plan["model"]["id"], plan["model"]["revision"], plan["model"]["dtype"]) != (
            "Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", "bfloat16"):
        raise ValueError("unreviewed checkpoint")
    if (plan["max_memory_fraction"], plan["max_concurrent_GPUs"], plan["allowance_usd"]) != (.95, 9, 75):
        raise ValueError("unreviewed resources or spending allowance")
    if {k: v["work_timeout_seconds"] for k, v in plan["stages"].items()} != {
            "prepare": 600, "replay": 5400, "score": 300}:
        raise ValueError("unreviewed work deadlines")
    if {k: v["allowance_usd"] for k, v in plan["stages"].items()} != {
            "prepare": .15, "replay": 74.70, "score": .15}:
        raise ValueError("unreviewed stage allowances")
    expected = [f'{plan["source_tag"]}/wm/wm_{i:04d}.pt' for i in range(50, 500)]
    if [r["path"] for r in plan["source_files"]] != expected:
        raise ValueError("source cohort differs")


def frozen_json(ref):
    path = Path("/results") / ref["path"]
    if path.stat().st_size != ref["bytes"] or rt._redetect_sha(path) != ref["sha256"]:
        raise ValueError("completed first-batch output changed")
    return json.loads(path.read_text())


def reject_unplanned_traces(plan):
    # Avoid dispatching a GPU if another task completed overlapping records.
    base = Path("/results") / plan["protocol"] / "integrated"
    for path in base.glob("*/manifest.json"):
        run = json.loads(path.read_text())
        case = run.get("case", {})
        if (run.get("model", {}).get("id") != plan["model"]["id"]
                or case.get("artifact", {}).get("sha256") != plan["artifact"]["sha256"]
                or max(case.get("lengths", [0])) < 14336):
            continue
        for trace in path.parent.glob("batches/*/trace.pt"):
            start = int(trace.parent.name)
            records = case["records"][start:start + case["batch_size"]]
            if any(r["source"] == "wm" and 50 <= r["prompt_idx"] < 500 for r in records):
                raise ValueError("additional saved traces found; reuse them before any GPU dispatch")


def prepared_manifest():
    return json.loads((Path("/results") / RUN / "prepared.json").read_text())


def prepare(plan):
    import torch
    from online_prc import OnlinePRCKey
    rt._verify_redetection_checkpoint(plan["model"])
    reject_unplanned_traces(plan)
    old_report = frozen_json(plan["pilot"]["report"])
    if old_report["counts"] != plan["pilot"]["counts"]:
        raise ValueError("completed first-batch scores changed")
    artifact = rt._redetect_source(plan["artifact"], {"data": "/data"})
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if (artifact["T"], artifact["n"], artifact["generation_model"], artifact["experiment_seed"],
            artifact["artifact_fingerprint"], key.check_weight, key.noise_rate,
            key.row_rate_numerator, key.row_rate_denominator, key.fingerprint) != (
            14336, 14336, "Qwen3-8B-Base", 12345, plan["artifact_fingerprint"], 3, .2,
            99, 100, plan["online_key_sha256"]):
        raise ValueError("source artifact/settings differ")
    records = []
    for idx, listed in zip(plan["prompt_indices"], plan["source_files"]):
        source = Path("/data") / listed["path"]
        if source.stat().st_size != listed["bytes"]:
            raise ValueError("listed generation cache changed")
        ref = file_ref(source, "data")
        record = rt._redetect_source(ref, {"data": "/data"})
        if (record["prompt_idx"] != idx or not record["watermark"]
                or record["artifact_fingerprint"] != plan["artifact_fingerprint"]
                or record["online_key_sha256"] != plan["online_key_sha256"]
                or record["generation_model"] != "Qwen3-8B-Base"
                or record["tokens"].shape != (14336,)):
            raise ValueError("source record differs")
        tokens = record["tokens"].to(torch.int64).contiguous()
        records.append({"source": "wm", "prompt_idx": idx, "file": ref,
                        "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest()})
    if len(records) != 450:
        raise ValueError("require exactly 450 remaining source records")
    case = {"id": RUN, "generation_model": "Qwen3-8B-Base", "construction": "online",
            "artifact": plan["artifact"], "lengths": [14336], "weights": ["map", "entropy"],
            "fpr": .001, "fpr_policy": "one_shot", "null_policy": "not_evaluated",
            "batch_size": 50, "cache": "static", "records": records}
    prepared = rt._prepare_redetection(case, plan["model"], plan["execution"],
                                       {"data": "/data"}, "/results")
    if len(prepared["batches"]) != 9 or any(b["identity"]["count"] != 50 for b in prepared["batches"]):
        raise ValueError("require nine batches of 50")
    if prepared["partition_sha256"] != plan["pilot"]["prepared"]["partition_sha256"]:
        raise ValueError("partition differs from the completed first batch")
    root = Path("/results") / RUN
    write_json(root / "prepared.json", prepared)
    files = [file_ref(root / "prepared.json", "results")]
    for name in ("manifest.json", "artifact.pt"):
        files.append(file_ref(Path("/results") / prepared["root"] / name, "results"))
    files.extend(file_ref(Path("/results") / b["root"] / "inputs.pt", "results") for b in prepared["batches"])
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": files}


def merge_reports(old, new):
    if (not old["passed"] or not new["passed"]
            or old["protocol"] != new["protocol"]
            or old["settings"] != new["settings"]
            or old["reported_lengths"] != new["reported_lengths"]):
        raise ValueError("incompatible component scores")
    if old["reported_lengths"] != [14336] or new["reported_lengths"] != [14336]:
        raise ValueError("unreviewed scoring length")
    records = old["records"] + new["records"]
    if ([r["prompt_idx"] for r in old["records"]] != list(range(50))
            or [r["prompt_idx"] for r in new["records"]] != list(range(50, 500))
            or any(r["source"] != "wm" for r in records)):
        raise ValueError("combined cohort must contain each watermarked prompt exactly once")
    counts = {"14336": {w: {"wm": {"detected": sum(r["scores"]["14336"][w]["decision"] for r in records),
                                      "count": 500}, "null": {"detected": 0, "count": 0}}
                         for w in ("map", "entropy")}}
    return {**new, "records": records, "counts": counts,
            "trace_shard_sha256": {**old["trace_shard_sha256"], **new["trace_shard_sha256"]},
            "reuse": {"completed_first_batch": 50, "new": 450, "first_batch_replayed": False,
                      "first_batch_rescored": False}}


def score(plan, prepared):
    from online_8b_to_0p6b import score as existing_score
    result = existing_score(prepared)
    old = frozen_json(plan["pilot"]["report"])
    if old["counts"] != plan["pilot"]["counts"]:
        raise ValueError("completed first-batch scores changed")
    new = json.loads((Path("/results") / prepared["root"] / "full.json").read_text())
    merged = merge_reports(old, new)
    aggregate = copy.deepcopy(prepared)
    aggregate["root"] = RUN + "/combined/8B"
    aggregate["run"]["case"]["records"] = plan["pilot"]["prepared"]["run"]["case"]["records"] + prepared["run"]["case"]["records"]
    aggregate["run"]["case"]["id"] = RUN + "_N500"
    aggregate["batches"] = []  # Original batch offsets belong to component manifests.
    aggregate["aggregate_only"] = True
    aggregate["component_prepared"] = [plan["pilot"]["prepared"], prepared]
    root = Path("/results") / aggregate["root"]
    for name, value in (("prepared.json", aggregate), ("manifest.json", aggregate["run"]),
                        ("full.json", merged), ("summary.json", {k: v for k, v in merged.items() if k != "records"})):
        write_json(root / name, value)
        result["files"].append(file_ref(root / name, "results"))
    rt.redetect_results.commit()
    return {"combined_root": aggregate["root"], "counts": merged["counts"], "files": result["files"]}


def child(stage, payload):
    # Finish native imports before the existing NumPy pickle-compatibility aliases.
    import torch
    import scipy.special
    import galois
    import transformers
    import safetensors.torch
    import qwen
    import detectors
    plan = payload["plan"]
    verify_plan(plan)
    check_code(plan)
    torch.set_num_threads(1 if stage == "replay" else 4)
    rt.redetect_results.reload()
    if stage == "prepare":
        rt.data_vol.reload()
        return prepare(plan)
    prepared = prepared_manifest()
    if prepared != payload["prepared"]:
        raise ValueError("prepared inputs changed")
    if stage == "replay":
        if "H200" not in torch.cuda.get_device_name():
            raise ValueError("H200 required")
        batch = prepared["batches"][payload["batch_index"]]
        path = Path("/results") / batch["root"] / "trace.pt"
        if path.exists():
            rt._redetect_trace(path, batch["identity"])
            return {"replay": {"reused": True}, "files": [file_ref(path, "results")]}
        model = load_model(plan["model"])
        result = rt._recover_redetection_batch(model.model, batch, "/results",
                                               validate=False, max_memory_fraction=.95)
        rt.redetect_results.commit()
        return {"replay": result, "files": [file_ref(path, "results")]}
    if stage == "score":
        return score(plan, prepared)
    raise ValueError("unknown stage")


def bounded(stage, payload):
    verify_plan(payload["plan"])
    check_code(payload["plan"])
    rt.redetect_results.reload()
    folder = Path("/results") / RUN / "attempts" / stage
    if stage == "replay":
        if type(payload.get("batch_index")) is not int or not 0 <= payload["batch_index"] < 9:
            raise ValueError("unreviewed GPU batch")
        folder = folder / f'{payload["batch_index"]:02d}'
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()  # Durable marker blocks a provider restart from repeating work.
    limit = payload["plan"]["stages"][stage]["work_timeout_seconds"]
    started = time.monotonic()
    log_path = Path("/tmp") / f"{RUN}-{stage}-{os.getpid()}.log"
    try:
        with log_path.open("x") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", stage,
                                     str(folder / "request.json"), str(folder / "response.json")],
                                    stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise TimeoutError("approved work deadline reached; no retry")
        if status:
            raise RuntimeError(f"worker exited {status}; no retry")
        result = json.loads((folder / "response.json").read_text())
    except Exception as exc:
        write_json(folder / "failure.json", {"error": str(exc)})
        raise
    finally:
        if log_path.exists():
            (folder / "worker.log").write_bytes(log_path.read_bytes())
            log_path.unlink()
        write_json(folder / "timing.json", {"stage": stage, "wall_seconds": time.monotonic() - started,
                                            "work_timeout_seconds": limit})
        rt.redetect_results.commit()
    result["files"] += [file_ref(folder / name, "results") for name in
                        ("request.json", "response.json", "worker.log", "timing.json")]
    return result


@app.function(cpu=(4, 4), memory=16384, timeout=630, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES)
def cpu_stage(stage, payload):
    if stage not in ("prepare", "score"):
        raise ValueError("unreviewed CPU stage")
    return bounded(stage, payload)


@app.function(gpu="H200", cpu=(4, 4), memory=65536, timeout=5430, startup_timeout=60,
              retries=0, max_containers=9, scaledown_window=2, volumes=VOLUMES)
def gpu_replay(payload):
    return bounded("replay", payload)


def collect(stage, result, task_id=None):
    for ref in result["files"]:
        path = OUT / "cache" / ref["volume"] / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]:
            continue
        content = b"".join(rt.redetect_results.read_file(ref["path"]))
        if len(content) != ref["bytes"] or hashlib.sha256(content).hexdigest() != ref["sha256"]:
            raise ValueError("download changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    name = f"collected_{stage}.json" if task_id is None else f"tasks/replay/collected_{task_id}.json"
    write_json(OUT / name, {"files": result["files"]})


@app.local_entrypoint()
def run(stage: str, approval_reference: str):
    if stage not in STAGES:
        raise ValueError("unknown stage")
    plan = json.loads((OUT / "setup.json").read_text())
    verify_plan(plan)
    check_code(plan)
    approval = json.loads((OUT / f"approval_{stage}.json").read_text())
    if (approval["plan_sha256"] != rt._redetect_sha(OUT / "setup.json")
            or approval["approval_reference"] != approval_reference or not approval["explicit_user_approval"]
            or approval["authorized_spend_usd"] < plan["stages"][stage]["allowance_usd"]
            or os.environ.get("MODAL_PROFILE") != "new-prc-watermark"
            or subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection"):
        raise ValueError("stage approval, profile or branch mismatch")
    if (OUT / f"attempt_{stage}.json").exists():
        raise ValueError("already attempted; collect saved work, no automatic retry")
    for prior in STAGES[:STAGES.index(stage)]:
        if not (OUT / f"collected_{prior}.json").exists():
            raise ValueError("collect preceding stage first")
    payload = {"plan": plan, "approval_reference": approval_reference}
    if stage != "prepare":
        payload["prepared"] = json.loads((OUT / "result_prepare.json").read_text())["prepared"]
    if stage == "score" and not (OUT / "commit_replay.json").exists():
        raise ValueError("save and commit primary replay evidence before scoring")
    write_json(OUT / f"attempt_{stage}.json", {"plan_sha256": rt._redetect_sha(OUT / "setup.json"),
                                             "approval_reference": approval_reference})
    spec = plan["stages"][stage]
    if stage == "replay":
        results, failures = [], []
        with ThreadPoolExecutor(max_workers=9) as pool:
            futures = {pool.submit(gpu_replay.remote, {**payload, "batch_index": i}): i for i in range(9)}
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                    write_json(OUT / f"tasks/replay/result_{index}.json", result)
                    collect(stage, result, index)
                    results.append(result)
                    print(json.dumps({"saved_batch": index, "prompt_ids": [50 + 50*index, 99 + 50*index]}), flush=True)
                except Exception as exc:
                    failures.append({"batch_index": index, "error": str(exc)})
        if failures:
            write_json(OUT / "failures_replay.json", failures)
            raise RuntimeError("incomplete replay; completed batches retained; no automatic retry")
        result = {"batch_count": len(results), "files": [ref for r in results for ref in r["files"]]}
    else:
        result = cpu_stage.with_options(memory=spec["memory_mib"],
                    timeout=spec["work_timeout_seconds"] + 30).remote(stage, payload)
    write_json(OUT / f"result_{stage}.json", result)
    collect(stage, result)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        write_json(sys.argv[4], child(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
    else:
        print((OUT / "PLAN.md").read_text())
