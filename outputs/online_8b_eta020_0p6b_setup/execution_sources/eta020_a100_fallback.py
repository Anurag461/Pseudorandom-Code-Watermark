"""Budgeted A100/batch25 continuation; reuse the successful H200 first50.

Only the remaining450 receive model execution. Cloud CPU reshards frozen inputs
and scores all500 together, retaining each original trace's execution identity.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import modal
import modal_run as rt
import online_8b_eta020_0p6b as primary
from fixed_4b_comparison import check_code, file_ref, load_model, write_json

OUT = primary.OUT / "a100_fallback"
RUN = primary.RUN + "_a100_remainder_v1"
RATE = .000694 + 4*.0000131 + 16*.00000222
app = modal.App("prc-eta020-0p6b-a100-remainder", image=primary.app.image.add_local_python_source(
    "eta020_a100_fallback"))


def budget(spent_upper):
    # Reserve CPU preparation, scoring and a further ten cents of headroom.
    limit = min(4800, math.floor((35 - spent_upper - .05 - .05 - .10) / (9*RATE)) - 92)
    expected = 3652.6696193927924  # Original A100/two-batch25 estimate.
    return {"spent_upper_usd": spent_upper, "worker_deadline_seconds": limit,
            "expected_worker_seconds": expected,
            "estimated_total_usd": spent_upper + .10 + 9*(expected + 92)*RATE,
            "deadline_total_resource_envelope_usd": spent_upper + .10 + 9*(limit + 92)*RATE,
            "fits": limit >= math.ceil(expected*1.10) and spent_upper >= 0}


def verify(payload):
    primary.verify_plan(payload["plan"])
    check_code(payload["plan"])
    if rt._redetect_sha(Path(__file__)) != payload["fallback_source_sha256"]:
        raise ValueError("fallback driver changed")
    if (payload["approval_reference"] != "user-20260922-continue-total-under35-else-original-a100"
            or not payload["budget"]["fits"] or payload["budget"] != budget(payload["budget"]["spent_upper_usd"])):
        raise ValueError("fallback not covered by the conditional budget authorization")
    primary.verify_prepared(payload["plan"], payload["original"])
    if payload["pilot_trace"]["path"] != payload["original"]["batches"][0]["root"] + "/trace.pt":
        raise ValueError("only original first50 can be reused")


def verify_combined(payload, prepared):
    original = payload["original"]
    if (prepared["run"]["case"]["records"] != payload["plan"]["source_records"]
            or prepared["run"]["model"] != original["run"]["model"]
            or prepared["artifact_sha256"] != original["artifact_sha256"]
            or prepared["partition_sha256"] != original["partition_sha256"]
            or prepared["batches"][0] != original["batches"][0]):
        raise ValueError("combined source/model/first50 identity changed")
    expected = [(0, 50)] + [(i, 25) for i in range(50, 500, 25)]
    if [(b["identity"]["start"], b["identity"]["count"]) for b in prepared["batches"]] != expected:
        raise ValueError("combined traces must cover500 once, preserving first50")
    if any(b["identity"]["length"] != 14336 or b["identity"]["cache"] != "static"
           or b["identity"]["protocol"] != rt.REDETECT_PROTOCOL for b in prepared["batches"]):
        raise ValueError("trace protocol changed")


def prepare(payload):
    from detectors import semantic_sha256
    original = payload["original"]
    ref = payload["pilot_trace"]
    pilot = Path("/results") / ref["path"]
    if rt._redetect_sha(pilot) != ref["sha256"] or pilot.stat().st_size != ref["bytes"]:
        raise ValueError("saved first50 trace changed")
    rt._redetect_trace(pilot, original["batches"][0]["identity"])
    rt._redetect_inputs(original["batches"][0], "/results")
    if any((Path("/results") / b["root"] / "trace.pt").exists() for b in original["batches"][1:]):
        raise ValueError("H200 continuation already produced traces; do not duplicate work")
    case = {**original["run"]["case"], "id": RUN, "batch_size": 25}
    execution = {**original["run"]["execution"], "gpu": "H200 first50; A100-80GB remaining450",
                 "fallback_source_sha256": payload["fallback_source_sha256"],
                 "reused_first50_trace": ref, "batching": "first50 batch50; remaining450 batch25"}
    run = {**original["run"], "case": case, "execution": execution}
    root = Path("/results") / rt.REDETECT_PROTOCOL / "integrated" / semantic_sha256(run)[:24]
    root.mkdir(parents=True, exist_ok=False)
    source_artifact = Path("/results") / original["root"] / "artifact.pt"
    if rt._redetect_sha(source_artifact) != original["artifact_sha256"]:
        raise ValueError("original artifact changed")
    shutil.copyfile(source_artifact, root / "artifact.pt")
    batches = [original["batches"][0]]
    for batch in original["batches"][1:]:
        inputs = rt._redetect_inputs(batch, "/results")
        for offset in (0, 25):
            start = batch["identity"]["start"] + offset
            values = {"tokens": inputs["tokens"][offset:offset+25].clone(), "partition": inputs["partition"]}
            directory = root / "batches" / f"{start:06d}"
            identity = {**batch["identity"], "run": root.name, "start": start, "count": 25,
                        "input_sha256": semantic_sha256(values)}
            rt._redetect_write(directory / "inputs.pt", values)
            batches.append({"root": str(directory.relative_to("/results")), "identity": identity})
    prepared = {**original, "root": str(root.relative_to("/results")), "run": run, "batches": batches}
    verify_combined(payload, prepared)
    write_json(root / "manifest.json", run)
    write_json(root / "prepared.json", prepared)
    write_json(Path("/results") / RUN / "prepared.json", prepared)
    refs = [file_ref(root / name, "results") for name in ("manifest.json", "prepared.json", "artifact.pt")]
    refs += [file_ref(Path("/results") / b["root"] / "inputs.pt", "results") for b in batches[1:]]
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": refs}


def child(stage, payload):
    import torch
    import scipy.special
    import galois
    import transformers
    import safetensors.torch
    import qwen
    import detectors
    verify(payload)
    torch.set_num_threads(1 if stage == "replay" else 4)
    rt.redetect_results.reload()
    if stage == "prepare":
        return prepare(payload)
    prepared = payload["prepared"]
    verify_combined(payload, prepared)
    if json.loads((Path("/results") / RUN / "prepared.json").read_text()) != prepared:
        raise ValueError("prepared continuation changed")
    if stage == "score":
        from online_8b_to_0p6b import score
        return score(prepared)
    worker = payload["worker_id"]
    if stage != "replay" or type(worker) is not int or worker not in range(9):
        raise ValueError("invalid worker")
    if "A100" not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory < 75*1024**3:
        raise ValueError("A10080GB required")
    model = load_model(payload["plan"]["detector"])
    results, refs = [], []
    for batch in prepared["batches"][1+2*worker:3+2*worker]:
        path = Path("/results") / batch["root"] / "trace.pt"
        if path.exists():
            raise ValueError("trace already exists; no duplicate GPU execution")
        try:
            result = rt._recover_redetection_batch(model.model, batch, "/results", validate=False,
                                                    max_memory_fraction=.85)
        finally:
            rt.redetect_results.commit()
        trace = rt._redetect_trace(path, batch["identity"])
        result["trace_metadata"] = {k: trace[k] for k in ("seconds", "peak_allocated_bytes",
                                  "total_memory_bytes", "full_validation")}
        results.append(result)
        refs.append(file_ref(path, "results"))
        print(json.dumps(result), flush=True)
    return {"worker_id": worker, "batches": results, "files": refs}


def bounded(stage, payload):
    verify(payload)
    rt.redetect_results.reload()
    task = f'{payload["worker_id"]:02d}' if stage == "replay" else "cpu"
    folder = Path("/results") / RUN / "attempts" / stage / task
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()
    limit = payload["budget"]["worker_deadline_seconds"] if stage == "replay" else 300
    started = time.monotonic()
    log_path = Path("/tmp") / f"{RUN}-{stage}-{task}.log"
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
                raise TimeoutError("budgeted deadline reached; no retry")
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
        write_json(folder / "timing.json", {"stage": stage, "task_id": task,
                   "wall_seconds": time.monotonic()-started, "work_timeout_seconds": limit})
        rt.redetect_results.commit()
    result["files"] += [file_ref(folder / n, "results") for n in ("request.json", "response.json", "worker.log", "timing.json")]
    return result


@app.function(cpu=(4, 4), memory=16384, timeout=330, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=primary.VOLUMES)
def cpu(stage, payload):
    if stage not in ("prepare", "score"):
        raise ValueError("invalid CPU stage")
    return bounded(stage, payload)


@app.function(gpu="A100-80GB", cpu=(4, 4), memory=16384, timeout=4830, startup_timeout=60,
              retries=0, max_containers=9, scaledown_window=2, volumes=primary.VOLUMES)
def gpu(payload):
    return bounded("replay", payload)


def run(stage, payload):
    verify(payload)
    if stage not in ("prepare", "replay", "score") or os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("invalid stage/profile")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("wrong branch")
    OUT.mkdir(parents=True, exist_ok=True)
    for prior in ("prepare", "replay", "score")[:("prepare", "replay", "score").index(stage)]:
        if not (OUT / f"collected_{prior}.json").exists():
            raise ValueError("collect preceding fallback stage first")
    with (OUT / f"attempt_{stage}.json").open("x") as handle:
        json.dump({"approval_reference": payload["approval_reference"], "budget": payload["budget"],
                   "fallback_source_sha256": payload["fallback_source_sha256"], "created_unix": time.time()}, handle, indent=2)
    if stage != "prepare":
        payload = {**payload, "prepared": json.loads((OUT / "result_prepare.json").read_text())["prepared"]}
    with app.run():
        write_json(OUT / f"app_{stage}.json", {"app_id": app.app_id})
        if stage == "replay":
            results, failures = {}, []
            with ThreadPoolExecutor(max_workers=9) as pool:
                futures = {pool.submit(gpu.with_options(timeout=payload["budget"]["worker_deadline_seconds"]+30).remote,
                                       {**payload, "worker_id": i}): i for i in range(9)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        response = future.result()
                        write_json(OUT / f"worker_{i:02d}.json", response)
                        primary.collect(f"fallback_worker_{i:02d}", response)
                        results[i] = response
                    except Exception as exc:
                        failures.append({"worker": i, "error": str(exc)})
            if failures:
                write_json(OUT / "failures.json", failures)
                raise RuntimeError("incomplete fallback; traces retained; no retry")
            result = {"files": [r for i in range(9) for r in results[i]["files"]]}
        else:
            result = cpu.with_options(memory=8192 if stage == "score" else 16384).remote(stage, payload)
        write_json(OUT / f"result_{stage}.json", result)
        primary.collect("fallback_"+stage, result)
        write_json(OUT / f"collected_{stage}.json", result["files"])
    if stage == "score":
        prepared = payload["prepared"]
        report = json.loads((primary.OUT / "cache/results" / prepared["root"] / "full.json").read_text())
        rt._append_redetection_csv(prepared, report, payload["plan"]["csv"])
        write_json(primary.OUT / "native_comparison.json", {"N": 500, "T": 14336,
                   "native_8B": payload["plan"]["baseline_native_T14336"],
                   "detector_0p6b": report["counts"]["14336"], "empirical_fpr": "not_evaluated"})


if __name__ == "__main__" and len(sys.argv) == 5 and sys.argv[1] == "--child":
    write_json(sys.argv[4], child(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
