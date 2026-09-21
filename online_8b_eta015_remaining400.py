"""Approval-gated continuation of prompts 100..499; reuse the completed pilot.

Only orchestration changes: existing generation, primary replay and CPU scoring.
No nulls, reference passes, benchmarks, automatic retries or local model work.
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

OUT = Path(os.environ.get("PRC_ETA015_SETUP_DIR", "outputs/online_8b_eta015_remaining400_setup"))
RUN = "online_8b_eta015_remaining400_v1"
STAGES = ("prepare", "generate", "freeze", "detect", "score")
app = modal.App("prc-online-8b-eta015-remaining400", image=rt.image.add_local_python_source(
    "online_8b_eta015_remaining400", "fixed_4b_comparison", "online_8b_to_0p6b",
    "online_prc_redetection"))
VOLUMES = {"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results}


def verify_plan(plan):
    if plan.get("attempt_namespace") not in (None, "prepare_retry1"):
        raise ValueError("unreviewed attempt namespace")
    if (plan["run_id"], plan["branch"], plan["eta"], plan["T"], plan["seed"], plan["t"]) != (
            RUN, "redetection", .15, 6144, 12345, 3):
        raise ValueError("unreviewed experiment")
    if (plan["prompt_indices"] != list(range(100, 500)) or plan["reuse_indices"] != list(range(100))
            or plan["null_count"] != 0 or plan["reporting_lengths"] != [6144]
            or plan["protocol"] != "completion_only_raw_abstain_v1"
            or plan["automatic_retries"] != 0 or plan["reference_replays"] != 0):
        raise ValueError("cohort, protocol, null or retry scope changed")
    if plan["generation_batches"] != [list(range(i, i + 100)) for i in range(100, 500, 100)]:
        raise ValueError("require four disjoint batches of 100")
    if [r["prompt_idx"] for r in plan["source_records"]] != plan["prompt_indices"]:
        raise ValueError("source records differ")
    if plan["detector_groups"] != {"8B": [[0], [1], [2], [3]],
                                   "0.6B": [[0, 1], [2, 3], [4], [5], [6], [7]]}:
        raise ValueError("require four native and six small-model workers, ten total")
    revisions = {"8B": "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
                 "0.6B": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"}
    for size, revision in revisions.items():
        spec = plan["models"][size]
        if (spec["id"], spec["revision"], spec["dtype"]) != (f"Qwen/Qwen3-{size}-Base", revision, "bfloat16"):
            raise ValueError("model changed")


def approved_plan(stage, approval_reference):
    if stage not in STAGES or not approval_reference.strip():
        raise ValueError("require an explicitly approved named stage")
    plan_path = OUT / "setup.json"
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("wrong Modal profile")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("wrong branch")
    check_code(plan)
    approval = json.loads((OUT / f"approval_{stage}.json").read_text())
    if (approval.get("stage") != stage or approval.get("plan_sha256") != rt._redetect_sha(plan_path)
            or approval.get("approval_reference") != approval_reference
            or not approval.get("explicit_user_approval")
            or approval.get("authorized_spend_usd", 0) < plan["stages"][stage]["allowance_usd"]):
        raise ValueError("approval must match this stage, setup hash and allowance")
    if (OUT / f"attempt_{stage}.json").exists():
        raise ValueError("stage already attempted; collect saved work, no automatic retry")
    for prior in STAGES[:STAGES.index(stage)]:
        if not (OUT / f"collected_{prior}.json").exists():
            raise ValueError(f"collect {prior} before {stage}")
        if json.loads((OUT / f"attempt_{prior}.json").read_text())["plan_sha256"] != rt._redetect_sha(plan_path):
            raise ValueError("setup changed since previous stage")
    return plan


def frozen_json(ref):
    path = Path("/results") / ref["path"]
    if rt._redetect_sha(path) != ref["sha256"]:
        raise ValueError("cached pilot report changed")
    return json.loads(path.read_text())


def prepare(plan):
    import torch
    from detectors import tensor_sha256
    source = rt._redetect_source(plan["source_artifact"], {"data": "/data"})
    target = rt._redetect_source(plan["target_artifact"], {"data": "/data"})
    if rt.artifact_compatibility_error(target, source):
        raise ValueError("source/target key, partition, prompts or sampler config differ")
    key = target["online_key"]
    if (source["T"], target["T"], target["num_prompts"], target["experiment_seed"],
            key["noise_rate"], key["check_weight"], key["row_rate_numerator"], key["row_rate_denominator"]) != (
            4096, 6144, 500, 12345, .15, 3, 99, 100):
        raise ValueError("frozen online settings differ")
    if tensor_sha256(target["partition"]) != plan["partition_sha256"]:
        raise ValueError("partition changed")
    for spec in plan["models"].values():
        rt._verify_redetection_checkpoint(spec)
        for name, sha in spec["metadata_sha256"].items():
            if rt._redetect_sha(Path("/cache") / spec["cache_directory"] / name) != sha:
                raise ValueError("checkpoint metadata changed; no download authorized")
    for ref in plan["source_records"]:
        record = rt._redetect_source(ref["file"], {"data": "/data"})
        rt.validate_online_watermarked_record(record, source, ref["prompt_idx"])
        if hashlib.sha256(record["tokens"].to(torch.int64).contiguous().numpy().tobytes()).hexdigest() != ref["tokens_sha256"]:
            raise ValueError("source tokens changed")
    target_dir = Path("/data") / plan["target_tag"] / "wm"
    if sorted(int(p.stem.split("_")[-1]) for p in target_dir.glob("wm_*.pt")) != list(range(100)):
        raise ValueError("target cache changed; revise missing-only setup before generation")
    for size, pilot in plan["pilot"].items():
        report = frozen_json(pilot["report"])
        if report["counts"] != pilot["counts"] or [r["prompt_idx"] for r in report["records"]] != list(range(100)):
            raise ValueError("completed pilot coverage changed")
        for batch in pilot["prepared"]["batches"]:
            path = Path("/results") / batch["root"] / "trace.pt"
            if rt._redetect_sha(path) != pilot["trace_sha256"][batch["root"]]:
                raise ValueError("completed pilot trace changed")
        for ref in pilot["prepared"]["run"]["case"]["records"]:
            record = rt._redetect_source(ref["file"], {"data": "/data"})
            rt.validate_online_watermarked_record(record, target, ref["prompt_idx"])
    destination = Path("/results") / RUN / "preparation.json"
    write_json(destination, {"passed": True, "remaining": 400, "pilot_reused": 100,
                             "source_artifact": plan["source_artifact"], "target_artifact": plan["target_artifact"]})
    rt.redetect_results.commit()
    return {"files": [file_ref(destination, "results"), plan["source_artifact"], plan["target_artifact"]]}


def generate(plan, indices):
    import torch
    if indices not in plan["generation_batches"]:
        raise ValueError("unreviewed generation batch")
    target_dir = Path("/data") / plan["target_tag"] / "wm"
    if any((target_dir / f"wm_{i:04d}.pt").exists() for i in indices):
        raise ValueError("target output exists; do not regenerate any record")
    for artifact in (plan["source_artifact"], plan["target_artifact"]):
        if rt._redetect_sha(Path("/data") / artifact["path"]) != artifact["sha256"]:
            raise ValueError("generation artifact changed")
    for ref in plan["source_records"]:
        if ref["prompt_idx"] in indices and rt._redetect_sha(Path("/data") / ref["file"]["path"]) != ref["file"]["sha256"]:
            raise ValueError("continuation source changed")
    rt._verify_redetection_checkpoint(plan["models"]["8B"])
    os.environ["PRC_MODEL_REVISION"] = plan["models"]["8B"]["revision"]
    torch.backends.cuda.matmul.allow_tf32 = False
    worker = rt.OnlineGenerationModel(tag=plan["target_tag"], model_size="8B",
        code_fingerprint_sha256=rt._local_code_fingerprint(), kv_cache_implementation="concat",
        null_kv_cache_implementation="static")
    worker.ready.local()
    result = worker.generate_wm.local({"prompt_indices": indices, "resume_source_tag": plan["source_tag"]})
    if (result["generated"], result["batch"], result["resume_prefix_T"], result["suffix_tokens_generated"]) != (100, 100, 4096, 204800):
        raise ValueError("generation scope differs; preserve outputs and stop")
    return {"generation": result, "files": [file_ref(target_dir / f"wm_{i:04d}.pt", "data") for i in indices]}


def freeze(plan):
    import numpy as np
    import torch
    from online_prc import OnlinePRCKey, OnlinePRCEncoder, derive_document_seed
    target = rt._redetect_source(plan["target_artifact"], {"data": "/data"})
    key = OnlinePRCKey.from_dict(target["online_key"])
    records = []
    for ref in plan["source_records"]:
        i = ref["prompt_idx"]
        before = rt._redetect_source(ref["file"], {"data": "/data"})
        path = Path("/data") / plan["target_tag"] / "wm" / f"wm_{i:04d}.pt"
        after = rt._redetect_load(path)
        rt.validate_online_watermarked_record(after, target, i)
        for field in ("tokens", "p_trace", "base_lm_entropy", "base_token_logprob", "prc_codeword_bits"):
            if not np.array_equal(np.asarray(before[field]), np.asarray(after[field])[:4096]):
                raise ValueError(f"original prefix changed: {i}/{field}")
        bits = OnlinePRCEncoder(key, [derive_document_seed(12345, i)]).encode_to_length(6144)[0]
        if not np.array_equal(bits, after["prc_codeword_bits"]):
            raise ValueError("continued PRC bitstream differs")
        records.append({"source": "wm", "prompt_idx": i, "file": file_ref(path, "data"),
                        "tokens_sha256": hashlib.sha256(after["tokens"].to(torch.int64).contiguous().numpy().tobytes()).hexdigest()})
    prepared, files = {}, []
    for size, batch, gpu in (("8B", 100, "H200"), ("0.6B", 50, "A100-80GB")):
        case = {"id": RUN + "_" + size, "generation_model": "Qwen3-8B-Base", "construction": "online",
                "artifact": plan["target_artifact"], "lengths": [6144], "weights": ["map", "entropy"],
                "fpr": .001, "fpr_policy": "one_shot", "null_policy": "not_evaluated",
                "batch_size": batch, "cache": "static", "records": records}
        execution = {"git_commit": plan["head"], "files": plan["runtime_source_sha256"],
                     "gpu": gpu, "source_hashes_authoritative": True}
        p = rt._prepare_redetection(case, plan["models"][size], execution, {"data": "/data"}, "/results")
        prepared[size] = p
        root = Path("/results") / p["root"]
        write_json(root / "prepared.json", p)
        files += [file_ref(root / name, "results") for name in ("manifest.json", "prepared.json", "artifact.pt")]
        files += [file_ref(Path("/results") / b["root"] / "inputs.pt", "results") for b in p["batches"]]
    rt.redetect_results.commit()
    return {"prepared": prepared, "prefixes_verified": 400, "files": files}


def detect(plan, payload):
    prepared, size, selected = payload["prepared"], payload["size"], payload["batch_indices"]
    expected = plan["detector_groups"][size]
    if selected not in expected or prepared["run"]["model"] != plan["models"][size]:
        raise ValueError("detector group or checkpoint changed")
    refs = prepared["run"]["case"]["records"]
    if [r["prompt_idx"] for r in refs] != list(range(100, 500)):
        raise ValueError("must replay only the remaining 400 records")
    batches = [prepared["batches"][i] for i in selected]
    for b in batches:
        if (Path("/results") / b["root"] / "trace.pt").exists():
            raise ValueError("primary trace exists; collect it, do not replay")
    we = load_model(plan["models"][size])
    results = []
    for b in batches:
        results.append(rt._recover_redetection_batch(we.model, b, "/results", validate=False))
        rt.redetect_results.commit()
    return {"batches": results, "files": [file_ref(Path("/results") / b["root"] / "trace.pt", "results") for b in batches]}


def score(plan, prepared):
    from online_8b_to_0p6b import score as existing_score
    files, combined = [], {}
    for size, new in prepared.items():
        result = existing_score(new)
        files += result["files"]
        report = json.loads((Path("/results") / new["root"] / "full.json").read_text())
        pilot = plan["pilot"][size]
        old = frozen_json(pilot["report"])
        records = old["records"] + report["records"]
        if [r["prompt_idx"] for r in records] != list(range(500)) or any(r["source"] != "wm" for r in records):
            raise ValueError("combined cohort must contain each prompt exactly once")
        if old["counts"] != pilot["counts"]:
            raise ValueError("pilot results changed")
        counts = {"6144": {w: {"wm": {"detected": sum(r["scores"]["6144"][w]["decision"] for r in records),
                                     "count": 500}, "null": {"detected": 0, "count": 0}} for w in ("map", "entropy")}}
        merged = {**report, "records": records, "counts": counts,
                  "trace_shard_sha256": {**pilot["trace_sha256"], **report["trace_shard_sha256"]},
                  "reuse": {"pilot": 100, "new": 400, "pilot_replayed": False, "pilot_rescored": False}}
        p = copy.deepcopy(new)
        p["root"] = RUN + "/combined/" + size
        p["run"]["case"]["records"] = pilot["prepared"]["run"]["case"]["records"] + new["run"]["case"]["records"]
        # Component batch offsets belong to their original manifests, not the
        # combined 500-record ordering. This manifest is for aggregation only.
        p["batches"] = []
        p["aggregate_only"] = True
        p["component_prepared"] = [pilot["prepared"], new]
        root = Path("/results") / p["root"]
        for name, value in (("prepared.json", p), ("manifest.json", p["run"]), ("full.json", merged),
                            ("summary.json", {k: v for k, v in merged.items() if k != "records"})):
            write_json(root / name, value)
            files.append(file_ref(root / name, "results"))
        combined[size] = {"root": p["root"], "counts": counts}
    rt.redetect_results.commit()
    return {"combined": combined, "files": files}


def child(stage, payload):
    # Corrected loading order: native/scientific modules before pickle aliases.
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
    torch.set_num_threads(1 if stage in ("generate", "detect") else 4)
    rt.data_vol.reload()
    rt.redetect_results.reload()
    if stage in ("generate", "detect"):
        expected = "H200" if stage == "generate" or payload["size"] == "8B" else "A100"
        if expected not in torch.cuda.get_device_name():
            raise ValueError("unexpected GPU")
    if stage == "prepare": return prepare(plan)
    if stage == "generate": return generate(plan, payload["indices"])
    if stage == "freeze": return freeze(plan)
    if stage == "detect": return detect(plan, payload)
    if stage == "score": return score(plan, payload["prepared"])
    raise ValueError("unknown stage")


def bounded(stage, payload, task_id, limit):
    verify_plan(payload["plan"])
    check_code(payload["plan"])
    rt.redetect_results.reload()
    attempts = Path("/results") / RUN / "attempts"
    if payload["plan"].get("attempt_namespace"):
        attempts = attempts / payload["plan"]["attempt_namespace"]
    folder = attempts / stage / task_id
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()  # Durable marker before any native/model work.
    started = time.monotonic()
    result = None
    # The child reloads mounted volumes; keep its open log off those volumes.
    log_path = Path("/tmp") / f"{RUN}-{stage}-{task_id}-{os.getpid()}.log"
    try:
        with log_path.open("x") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", stage,
                                     str(folder / "request.json"), str(folder / "response.json")],
                                    stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise TimeoutError("approved worker deadline exceeded; no retry")
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
def cpu_stage(stage, payload, limit):
    if stage not in ("prepare", "freeze", "score"): raise ValueError("invalid CPU stage")
    return bounded(stage, payload, "cpu", limit)


@app.function(gpu="H200", cpu=(4, 4), memory=65536, timeout=2730, startup_timeout=60,
              retries=0, max_containers=4, scaledown_window=2, volumes=VOLUMES)
def large_gpu(stage, payload, task_id, limit):
    if stage not in ("generate", "detect"): raise ValueError("invalid GPU stage")
    return bounded(stage, payload, task_id, limit)


@app.function(gpu="A100-80GB", cpu=(4, 4), memory=16384, timeout=2030, startup_timeout=60,
              retries=0, max_containers=6, scaledown_window=2, volumes=VOLUMES)
def small_gpu(payload, task_id, limit):
    return bounded("detect", payload, task_id, limit)


def download_files(refs):
    for ref in refs:
        path = OUT / "cache" / ref["volume"] / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]: continue
        volume = rt.data_vol if ref["volume"] == "data" else rt.redetect_results
        data = b"".join(volume.read_file(ref["path"]))
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("download checksum differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def collect(stage):
    results = [json.loads(p.read_text()) for p in sorted((OUT / "tasks" / stage).glob("*.json"))]
    expected = {"prepare": 1, "generate": 4, "freeze": 1, "detect": 10, "score": 1}[stage]
    for result in results: download_files(result["files"])
    if len(results) != expected: raise ValueError("incomplete stage; saved batches preserved; no retry")
    if stage == "score":
        for data in results[0]["combined"].values():
            folder = OUT / "cache/results" / data["root"]
            rt._append_redetection_csv(json.loads((folder / "prepared.json").read_text()),
                                      json.loads((folder / "full.json").read_text()), rt.REDETECT_CSV)
    write_json(OUT / f"collected_{stage}.json", {"tasks": len(results), "files": [r for x in results for r in x["files"]]})


@app.local_entrypoint()
def run(stage: str, approval_reference: str):
    plan = approved_plan(stage, approval_reference)
    base = {"plan": plan, "approval_reference": approval_reference}
    jobs = []
    if stage in ("prepare", "freeze", "score"):
        if stage == "score": base["prepared"] = json.loads((OUT / "tasks/freeze/cpu.json").read_text())["prepared"]
        spec = plan["stages"][stage]
        jobs = [("cpu", lambda: cpu_stage.with_options(memory=spec["memory_mib"], timeout=spec["work_timeout_seconds"] + 30).remote(
            stage, base, spec["work_timeout_seconds"]))]
    elif stage == "generate":
        limit = plan["stages"][stage]["work_timeout_seconds_per_worker"]
        jobs = [(str(ids[0]), lambda ids=ids: large_gpu.with_options(timeout=limit + 30).remote(
                    stage, {**base, "indices": ids}, str(ids[0]), limit))
                for ids in plan["generation_batches"]]
    else:
        prepared = json.loads((OUT / "tasks/freeze/cpu.json").read_text())["prepared"]
        for size, groups in plan["detector_groups"].items():
            for group in groups:
                task = {**base, "size": size, "batch_indices": group, "prepared": prepared[size]}
                task_id = size + "_" + str(group[0])
                if size == "8B":
                    limit = plan["stages"][stage]["native"]["work_timeout_seconds_per_worker"]
                    jobs.append((task_id, lambda task=task, task_id=task_id, limit=limit: large_gpu.with_options(timeout=limit + 30).remote(
                        "detect", task, task_id, limit)))
                else:
                    limit = plan["stages"][stage]["small"]["work_timeout_seconds_per_batch"] * len(group)
                    jobs.append((task_id, lambda task=task, task_id=task_id, limit=limit: small_gpu.with_options(timeout=limit + 30).remote(
                        task, task_id, limit)))
    write_json(OUT / f"attempt_{stage}.json", {"plan_sha256": rt._redetect_sha(OUT / "setup.json"),
               "stage": stage, "approval_reference": approval_reference, "tasks": [j[0] for j in jobs]})
    failures = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(call): task_id for task_id, call in jobs}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                write_json(OUT / "tasks" / stage / f"{task_id}.json", result)
                download_files(result["files"])
                print(json.dumps({"saved_stage": stage, "task_id": task_id}), flush=True)
            except Exception as exc:
                failures.append({"task_id": task_id, "error": str(exc)})
    if failures:
        write_json(OUT / f"failures_{stage}.json", failures)
        raise RuntimeError("stage failed; successful outputs retained; no automatic retry")
    collect(stage)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        write_json(sys.argv[4], child(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
    elif len(sys.argv) == 3 and sys.argv[1] == "collect":
        collect(sys.argv[2])
    else:
        print((OUT / "PLAN.md").read_text())
