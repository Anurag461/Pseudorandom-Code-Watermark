"""Approval-gated 500-record eta=.20 8B -> 0.6B completion-only replay.

Default invocation checks local metadata only. Paid stages are explicit and
use the established runtime; no generation, reference replay or auto-retry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

OUT = Path("outputs/online_8b_eta020_0p6b_setup")
RUN = "online_8b_eta020_T14336_to_0p6b_N500_v1"
STAGES = ("prepare", "pilot", "replay", "score")
VOLUMES = {"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results}
app = modal.App("prc-online-8b-eta020-to-0p6b", image=rt.image.add_local_python_source(
    "online_8b_eta020_0p6b", "fixed_4b_comparison", "online_8b_to_0p6b", "online_prc_redetection"))


def verify_plan(plan):
    if (plan["run_id"], plan["branch"], plan["profile"], plan["eta"], plan["T"], plan["n"], plan["N"]) != (
            RUN, "redetection", "new-prc-watermark", .2, 14336, 14336, 500):
        raise ValueError("unreviewed experiment")
    if (plan["prompt_indices"] != list(range(500)) or plan["reporting_lengths"] != [14336]
            or plan["weights"] != ["map", "entropy"] or plan["fpr"] != .001
            or plan["fpr_policy"] != "one_shot" or plan["protocol"] != rt.REDETECT_PROTOCOL
            or plan["seed"] != 12345 or plan["t"] != 3 or plan["row_rate"] != [99, 100]
            or plan["r"] != 14193 or plan["empirical_fpr"] != "not_evaluated"
            or any(plan[k] for k in ("new_generations", "null_count", "native_8B_replays",
                                    "reference_replays", "automatic_retries", "benchmarks"))):
        raise ValueError("unreviewed scope or detector protocol")
    if (plan["detector"]["id"], plan["detector"]["revision"], plan["detector"]["dtype"]) != (
            "Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd", "bfloat16"):
        raise ValueError("unreviewed detector checkpoint")
    if plan["generator"]["id"] != "Qwen/Qwen3-8B-Base":
        raise ValueError("wrong saved generator")
    expected_detection = dict(raw_completion_ids=True, prompt_prefix=False, special_token_prefix=False,
                              first_coordinate_abstention=True, cache="static", dtype="bfloat16",
                              tf32=False, trace_dtype="float32", score_dtype="float64",
                              generation_probabilities_used=False, max_memory_fraction=.85)
    if plan["detection"] != expected_detection:
        raise ValueError("unreviewed replay semantics")
    if (plan["batch_size"], plan["batch_count"], plan["max_concurrent_gpus"], plan["model_loads"]) != (50, 10, 10, 10):
        raise ValueError("unreviewed batching or concurrency")
    if plan["worker_batch_indices"] != [[i] for i in range(10)]:
        raise ValueError("workers must own one disjoint batch each")
    if (plan["conservative_total_allowance_usd"] != 35
            or {s: plan["stages"][s]["allowance_usd"] for s in STAGES} != {
                "prepare": .15, "pilot": 4.6, "replay": 30.2, "score": .05}):
        raise ValueError("unreviewed stage allowances")
    if (plan["stages"]["prepare"]["work_timeout_seconds"] != 600
            or plan["stages"]["pilot"]["work_timeout_seconds_per_worker"] != 3300
            or plan["stages"]["replay"]["work_timeout_seconds_per_worker"] != 2390
            or plan["stages"]["score"]["work_timeout_seconds"] != 300):
        raise ValueError("unreviewed deadlines")
    for stage, memory in (("prepare", 16384), ("score", 8192)):
        spec = plan["stages"][stage]
        if (spec["gpu"], spec["workers"], spec["cpu"], spec["memory_mib"]) != (None, 1, 4, memory):
            raise ValueError("unreviewed CPU resources")
    for stage, workers in (("pilot", 1), ("replay", 9)):
        spec = plan["stages"][stage]
        if (spec["gpu"], spec["workers"], spec["cpu_per_worker"], spec["memory_mib_per_worker"],
                spec["batches_per_worker"]) != ("H200", workers, 4, 16384, 1):
            raise ValueError("unreviewed GPU resources")
    if (plan["function_timeout_margin_seconds"], plan["startup_timeout_seconds"],
            plan["scaledown_window_seconds"]) != (30, 60, 2):
        raise ValueError("unreviewed worker lifecycle")
    records = plan["source_records"]
    if [(r["source"], r["prompt_idx"]) for r in records] != [("wm", i) for i in range(500)]:
        raise ValueError("source records must cover each watermarked prompt exactly once")
    if any(r["file"]["volume"] != "data" or r["file"]["path"] !=
           f'{plan["source_tag"]}/wm/wm_{i:04d}.pt' for i, r in enumerate(records)):
        raise ValueError("source paths differ from the original cohort")


def verify_prepared(plan, prepared):
    case = prepared["run"]["case"]
    if (prepared["run"]["protocol"] != plan["protocol"] or prepared["run"]["model"] != plan["detector"]
            or prepared["run"]["execution"] != plan["execution"]
            or case != make_case(plan) or prepared["partition_sha256"] != plan["partition_sha256"]):
        raise ValueError("prepared run differs from reviewed setup")
    batches = prepared["batches"]
    if len(batches) != 10:
        raise ValueError("require exactly 10 batches")
    for i, batch in enumerate(batches):
        identity = batch["identity"]
        if (identity["start"], identity["count"], identity["length"], identity["cache"], identity["protocol"]) != (
                50*i, 50, 14336, "static", plan["protocol"]):
            raise ValueError("batch coverage/shape/protocol changed")
        if batch["root"] != f'{prepared["root"]}/batches/{50*i:06d}':
            raise ValueError("batch location differs")


def make_case(plan):
    return {"id": RUN, "generation_model": "Qwen3-8B-Base", "construction": "online",
            "artifact": plan["source_artifact"], "lengths": [14336], "weights": ["map", "entropy"],
            "fpr": .001, "fpr_policy": "one_shot", "null_policy": "not_evaluated",
            "batch_size": 50, "cache": "static", "records": plan["source_records"]}


def cached_overlap(plan, run):
    """Conservative metadata filter; any overlap requires a reuse decision."""
    case = run.get("case", {})
    if (run.get("model", {}).get("id") != plan["detector"]["id"]
            or case.get("artifact", {}).get("sha256") != plan["source_artifact"]["sha256"]
            or max(case.get("lengths") or [0]) < 14336):
        return False
    expected = {(r["source"], r["prompt_idx"], r["tokens_sha256"]) for r in plan["source_records"]}
    return any((r.get("source"), r.get("prompt_idx"), r.get("tokens_sha256")) in expected
               for r in case.get("records", []))


def reject_other_traces(plan, own_root=None):
    base = Path("/results") / plan["protocol"] / "integrated"
    for path in base.glob("*/manifest.json"):
        if own_root and path.parent == Path("/results") / own_root:
            continue
        if cached_overlap(plan, json.loads(path.read_text())) and any(path.parent.glob("batches/*/trace.pt")):
            raise ValueError("overlapping saved 0.6B traces found; resolve reuse before GPU dispatch")


def frozen_source_prepared(plan):
    ref = plan["source_prepared"]
    path = Path("/results") / ref["path"]
    if path.stat().st_size != ref["bytes"] or rt._redetect_sha(path) != ref["sha256"]:
        raise ValueError("original combined source manifest changed")
    value = json.loads(path.read_text())
    if (value["run"]["case"]["records"] != plan["source_records"]
            or value["run"]["case"]["artifact"] != plan["source_artifact"]
            or value["partition_sha256"] != plan["partition_sha256"]):
        raise ValueError("original source identity differs")
    return value


def prepare(plan):
    from detectors import tensor_sha256
    from online_prc import OnlinePRCKey
    frozen_source_prepared(plan)
    reject_other_traces(plan)
    rt._verify_redetection_checkpoint(plan["detector"])
    for name, digest in plan["detector"]["metadata_sha256"].items():
        if rt._redetect_sha(Path("/cache") / plan["detector"]["cache_directory"] / name) != digest:
            raise ValueError("pinned detector metadata changed; no download permitted")
    artifact = rt._redetect_source(plan["source_artifact"], {"data": "/data"})
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if (artifact["T"], artifact["n"], artifact["generation_model"], artifact["experiment_seed"],
            key.check_weight, key.noise_rate, key.row_rate_numerator, key.row_rate_denominator,
            key.fingerprint, artifact["artifact_fingerprint"]) != (
            14336, 14336, "Qwen3-8B-Base", 12345, 3, .2, 99, 100,
            plan["online_key_sha256"], plan["artifact_fingerprint"]):
        raise ValueError("original key or generation settings changed")
    if tensor_sha256(artifact["partition"]) != plan["partition_sha256"]:
        raise ValueError("original partition changed")
    prepared = rt._prepare_redetection(make_case(plan), plan["detector"], plan["execution"],
                                       {"data": "/data"}, "/results")
    verify_prepared(plan, prepared)
    root = Path("/results") / prepared["root"]
    write_json(root / "prepared.json", prepared)
    write_json(Path("/results") / RUN / "prepared.json", prepared)
    refs = [file_ref(root / name, "results") for name in ("prepared.json", "manifest.json", "artifact.pt")]
    refs += [file_ref(Path("/results") / b["root"] / "inputs.pt", "results") for b in prepared["batches"]]
    refs.append(file_ref(Path("/results") / RUN / "prepared.json", "results"))
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": refs}


def child(stage, payload):
    # Finish native/scientific imports before the pickle compatibility aliases.
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
    torch.set_num_threads(1 if stage in ("pilot", "replay") else 4)
    rt.redetect_results.reload()
    if stage == "prepare":
        rt.data_vol.reload()
        rt.hf_cache.reload()
        return prepare(plan)
    prepared = payload["prepared"]
    if json.loads((Path("/results") / RUN / "prepared.json").read_text()) != prepared:
        raise ValueError("saved prepared inputs changed")
    verify_prepared(plan, prepared)
    if stage == "score":
        from online_8b_to_0p6b import score
        return score(prepared)
    if stage not in ("pilot", "replay"):
        raise ValueError("unknown stage")
    worker_id = payload["worker_id"]
    if type(worker_id) is not int or worker_id not in ([0] if stage == "pilot" else range(1, 10)):
        raise ValueError("invalid worker assignment")
    if "H200" not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory < 130*1024**3:
        raise ValueError("H200 required")
    reject_other_traces(plan, prepared["root"])
    batches = [prepared["batches"][i] for i in plan["worker_batch_indices"][worker_id]]
    # Load exactly once per worker, and only if a primary trace is absent.
    model = None if all((Path("/results") / b["root"] / "trace.pt").exists() for b in batches) else load_model(plan["detector"])
    results, refs = [], []
    for batch in batches:
        path = Path("/results") / batch["root"] / "trace.pt"
        if path.exists():
            rt._redetect_inputs(batch, "/results")
            rt._redetect_trace(path, batch["identity"])
            result = {"root": batch["root"], "cached": True}
        else:
            try:
                result = rt._recover_redetection_batch(model.model, batch, "/results", validate=False,
                                                       max_memory_fraction=.85)
            finally:
                # Keep completed primary evidence even if the memory guard fails.
                rt.redetect_results.commit()
        trace = rt._redetect_trace(path, batch["identity"])
        result["trace_metadata"] = {k: trace[k] for k in (
            "seconds", "peak_allocated_bytes", "peak_reserved_bytes", "total_memory_bytes",
            "memory_limit_fraction", "full_validation")}
        results.append(result)
        refs.append(file_ref(path, "results"))
        print(json.dumps({"worker": worker_id, **result}), flush=True)
    return {"worker_id": worker_id, "batches": results, "files": refs}


def bounded(stage, payload):
    plan = payload["plan"]
    verify_plan(plan)
    check_code(plan)
    rt.redetect_results.reload()
    spec = plan["stages"][stage]
    limit = spec["work_timeout_seconds_per_worker"] if stage in ("pilot", "replay") else spec["work_timeout_seconds"]
    task_id = f'{payload["worker_id"]:02d}' if stage in ("pilot", "replay") else "cpu"
    folder = Path("/results") / RUN / "attempts" / stage / task_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()  # Never repeat paid work after a provider restart.
    started = time.monotonic()
    log_path = Path("/tmp") / f"{RUN}-{stage}-{task_id}-{os.getpid()}.log"
    try:
        with log_path.open("x") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", stage,
                                     str(folder / "request.json"), str(folder / "response.json")],
                                    stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
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
        write_json(folder / "timing.json", {"stage": stage, "task_id": task_id,
                    "wall_seconds": time.monotonic() - started, "work_timeout_seconds": limit})
        rt.redetect_results.commit()
    result["files"] += [file_ref(folder / name, "results") for name in
                        ("request.json", "response.json", "worker.log", "timing.json")]
    return result


@app.function(cpu=(4, 4), memory=16384, timeout=630, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES)
def cpu_stage(stage, payload):
    if stage not in ("prepare", "score"):
        raise ValueError("invalid CPU stage")
    return bounded(stage, payload)


@app.function(gpu="H200", cpu=(4, 4), memory=16384, timeout=2420, startup_timeout=60,
              retries=0, max_containers=9, scaledown_window=2, volumes=VOLUMES)
def gpu_replay(payload):
    return bounded("replay", payload)


@app.function(gpu="H200", cpu=(4, 4), memory=16384, timeout=3330, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES)
def gpu_pilot(payload):
    return bounded("pilot", payload)


def assess_pilot(plan, result, timing):
    """Small metadata arithmetic only; the first 50 are production evidence."""
    if result["worker_id"] != 0 or len(result["batches"]) != 1:
        raise ValueError("require the first production batch only")
    replay = result["batches"][0]
    if replay.get("cached"):
        raise ValueError("a cached read cannot establish GPU throughput")
    metadata = replay["trace_metadata"]
    rate = .001261 + 4*.0000131 + 16*.00000222
    work_seconds = timing["wall_seconds"] * 1.15
    overhead = 60 + 30 + 2
    remaining_cost = 9 * (work_seconds + overhead) * rate
    fraction = metadata["peak_allocated_bytes"] / metadata["total_memory_bytes"]
    within = (work_seconds <= plan["stages"]["replay"]["work_timeout_seconds_per_worker"]
              and remaining_cost <= plan["stages"]["replay"]["allowance_usd"]
              and fraction < .85 and metadata["full_validation"] is False)
    return {"pilot_method_seconds": metadata["seconds"], "pilot_worker_wall_seconds": timing["wall_seconds"],
            "pilot_upper_resource_estimate_usd": (timing["wall_seconds"] + overhead) * rate,
            "peak_allocated_bytes": metadata["peak_allocated_bytes"], "peak_allocated_fraction": fraction,
            "remaining_worker_seconds_with_15pct_margin": work_seconds,
            "remaining450_estimate_usd_with_margin": remaining_cost,
            "total_estimate_with_pilot_allowance_reserved_usd": .15 + 4.6 + remaining_cost + .05,
            "fits_reviewed_remaining_allowance": within, "auto_continue": False,
            "decision": "request separate approval" if within else "stop and revise estimate; do not launch remaining450"}


def collect(stage, result, worker_id=None):
    for ref in result["files"]:
        if ref["volume"] != "results":
            raise ValueError("unexpected output volume")
        path = OUT / "cache/results" / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]:
            continue
        data = b"".join(rt.redetect_results.read_file(ref["path"]))
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("output transfer checksum differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
    name = f"collected_{stage}.json" if worker_id is None else f"workers/collected_{worker_id:02d}.json"
    write_json(OUT / name, {"files": result["files"]})


def authorize(stage, approval_reference):
    """Local checks run before app.run(), so rejection launches no paid job."""
    if stage not in STAGES or not approval_reference.strip():
        raise ValueError("require a named stage and explicit approval reference")
    plan = json.loads((OUT / "setup.json").read_text())
    verify_plan(plan)
    check_code(plan)
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark" or subprocess.check_output(
            ["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("profile or branch mismatch")
    plan_sha = rt._redetect_sha(OUT / "setup.json")
    approval_path = OUT / f"approval_{stage}.json"
    if not approval_path.exists():
        raise ValueError("explicit approval for this paid stage has not been recorded")
    approval = json.loads(approval_path.read_text())
    if (approval.get("plan_sha256") != plan_sha or approval.get("stage") != stage
            or approval.get("approval_reference") != approval_reference
            or approval.get("explicit_user_approval") is not True
            or approval.get("authorized_spend_usd", 0) < plan["stages"][stage]["allowance_usd"]
            or approval.get("confirmed_available_budget_usd", 0) < plan["stages"][stage]["allowance_usd"]):
        raise ValueError("approval must bind this exact setup, stage and updated budget")
    if (OUT / f"attempt_{stage}.json").exists():
        raise ValueError("stage already attempted; collect saved work, no automatic retry")
    for prior in STAGES[:STAGES.index(stage)]:
        attempt = OUT / f"attempt_{prior}.json"
        if not (OUT / f"collected_{prior}.json").exists() or not attempt.exists():
            raise ValueError("collect preceding stage first")
        if json.loads(attempt.read_text())["plan_sha256"] != plan_sha:
            raise ValueError("setup changed after a preceding stage")
    payload = {"plan": plan, "approval_reference": approval_reference}
    if stage != "prepare":
        payload["prepared"] = json.loads((OUT / "result_prepare.json").read_text())["prepared"]
        verify_prepared(plan, payload["prepared"])
    if stage == "replay":
        pilot = json.loads((OUT / "result_pilot.json").read_text())
        timing_ref = next(r for r in pilot["files"] if r["path"].endswith("/timing.json"))
        timing_path = OUT / "cache/results" / timing_ref["path"]
        if rt._redetect_sha(timing_path) != timing_ref["sha256"]:
            raise ValueError("pilot timing evidence changed")
        assessment = assess_pilot(plan, pilot, json.loads(timing_path.read_text()))
        if not assessment["fits_reviewed_remaining_allowance"]:
            raise ValueError("measured pilot does not support remaining450 within the reviewed budget/deadline")
        if approval.get("pilot_result_sha256") != rt._redetect_sha(OUT / "result_pilot.json"):
            raise ValueError("remaining450 approval must explicitly reference the measured pilot")
    return payload, plan_sha


def run(stage, approval_reference):
    payload, plan_sha = authorize(stage, approval_reference)
    # The local attempt marker also prevents accidental duplicate submission.
    attempt_path = OUT / f"attempt_{stage}.json"
    with attempt_path.open("x") as handle:
        json.dump({"plan_sha256": plan_sha, "approval_reference": approval_reference,
                   "stage": stage, "created_unix": time.time()}, handle, indent=2)
    with app.run():
        if stage == "replay":
            results, failures = {}, []
            with ThreadPoolExecutor(max_workers=9) as pool:
                futures = {pool.submit(gpu_replay.remote, {**payload, "worker_id": i}): i for i in range(1, 10)}
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        response = future.result()
                        write_json(OUT / f"workers/result_{worker_id:02d}.json", response)
                        collect(stage, response, worker_id)
                        results[worker_id] = response
                        print(json.dumps({"saved_worker": worker_id, "prompt_ids": [50*worker_id, 50*worker_id+49]}), flush=True)
                    except Exception as exc:
                        failures.append({"worker_id": worker_id, "error": str(exc)})
            if failures:
                write_json(OUT / "failures_replay.json", failures)
                raise RuntimeError("incomplete replay; completed batches preserved; no retry")
            result = {"worker_count": 9, "batch_count": 9,
                      "files": [ref for i in range(1, 10) for ref in results[i]["files"]]}
        elif stage == "pilot":
            result = gpu_pilot.remote({**payload, "worker_id": 0})
        else:
            spec = payload["plan"]["stages"][stage]
            result = cpu_stage.with_options(memory=spec["memory_mib"],
                         timeout=spec["work_timeout_seconds"] + 30).remote(stage, payload)
        write_json(OUT / f"result_{stage}.json", result)
        collect(stage, result)
    if stage == "pilot":
        timing_ref = next(r for r in result["files"] if r["path"].endswith("/timing.json"))
        timing = json.loads((OUT / "cache/results" / timing_ref["path"]).read_text())
        write_json(OUT / "pilot_assessment.json", assess_pilot(payload["plan"], result, timing))
    if stage == "score":
        prepared = payload["prepared"]
        report = json.loads((OUT / "cache/results" / prepared["root"] / "full.json").read_text())
        rt._append_redetection_csv(prepared, report, payload["plan"]["csv"])
        write_json(OUT / "native_comparison.json", {"N": 500, "T": 14336,
                   "native_8B": payload["plan"]["baseline_native_T14336"],
                   "detector_0p6b": report["counts"]["14336"], "empirical_fpr": "not_evaluated"})


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        write_json(sys.argv[4], child(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--stage", choices=STAGES)
        parser.add_argument("--approval-reference", default="")
        args = parser.parse_args()
        if args.stage:
            run(args.stage, args.approval_reference)
        else:
            plan = json.loads((OUT / "setup.json").read_text())
            verify_plan(plan)
            check_code(plan)
            print(json.dumps({"status": "ready_for_review", "N": 500, "T": 14336,
                              "gpu": "H200", "pilot_workers": 1, "remaining_workers": 9, "batch": 50,
                              "paid_compute_launched": False}))
