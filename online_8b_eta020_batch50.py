"""One approval-gated H200 batch of 50 cached eta=.20 completions.

Existing preparation, completion-only replay and cloud CPU scoring routines.
No generation, null cohort, 0.6B inference, benchmark or automatic retry.
"""
from __future__ import annotations

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

OUT = Path("outputs/online_8b_eta020_batch50_setup")
RUN = "online_8b_eta020_T14336_N50_v1"
STAGES = ("prepare", "replay", "score")
app = modal.App("prc-online-8b-eta020-batch50", image=rt.image.add_local_python_source(
    "online_8b_eta020_batch50", "fixed_4b_comparison", "online_8b_to_0p6b", "online_prc_redetection"))
VOLUMES = {"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results}


def verify_plan(plan):
    if (plan["run_id"], plan["branch"], plan["eta"], plan["T"], plan["N"], plan["batch_size"]) != (
            RUN, "redetection", .2, 14336, 50, 50):
        raise ValueError("unreviewed experiment")
    if (plan["prompt_indices"] != list(range(50)) or plan["reporting_lengths"] != [14336]
            or plan["protocol"] != "completion_only_raw_abstain_v1"
            or any(plan[k] for k in ("new_generations", "null_count", "small_detector_records",
                                    "reference_replays", "automatic_retries"))):
        raise ValueError("unreviewed cohort or extra work")
    if (plan["model"]["id"], plan["model"]["revision"], plan["model"]["dtype"]) != (
            "Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", "bfloat16"):
        raise ValueError("unreviewed checkpoint")
    if plan["max_memory_fraction"] != .95 or plan["allowance_usd"] != 9.5:
        raise ValueError("unreviewed memory or spending allowance")
    if {k: v["work_timeout_seconds"] for k, v in plan["stages"].items()} != {
            "prepare": 300, "replay": 6000, "score": 180}:
        raise ValueError("unreviewed work deadlines")


def prepared_manifest():
    return json.loads((Path("/results") / RUN / "prepared.json").read_text())


def prepare(plan):
    import torch
    from online_prc import OnlinePRCKey
    rt._verify_redetection_checkpoint(plan["model"])
    artifact = rt._redetect_source(plan["artifact"], {"data": "/data"})
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if (artifact["T"], artifact["n"], artifact["generation_model"], artifact["experiment_seed"],
            artifact["artifact_fingerprint"], key.check_weight, key.noise_rate,
            key.row_rate_numerator, key.row_rate_denominator, key.fingerprint) != (
            14336, 14336, "Qwen3-8B-Base", 12345, plan["artifact_fingerprint"], 3, .2,
            99, 100, plan["online_key_sha256"]):
        raise ValueError("source artifact/settings differ")
    records = []
    for idx, ref in zip(plan["prompt_indices"], plan["source_files"]):
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
    if len(records) != 50:
        raise ValueError("require exactly 50 source records")
    case = {"id": RUN, "generation_model": "Qwen3-8B-Base", "construction": "online",
            "artifact": plan["artifact"], "lengths": [14336], "weights": ["map", "entropy"],
            "fpr": .001, "fpr_policy": "one_shot", "null_policy": "not_evaluated",
            "batch_size": 50, "cache": "static", "records": records}
    prepared = rt._prepare_redetection(case, plan["model"], plan["execution"],
                                       {"data": "/data"}, "/results")
    if len(prepared["batches"]) != 1 or prepared["batches"][0]["identity"]["count"] != 50:
        raise ValueError("require one batch of 50")
    root = Path("/results") / RUN
    write_json(root / "prepared.json", prepared)
    files = [file_ref(root / "prepared.json", "results")]
    for name in ("manifest.json", "artifact.pt"):
        files.append(file_ref(Path("/results") / prepared["root"] / name, "results"))
    files.append(file_ref(Path("/results") / prepared["batches"][0]["root"] / "inputs.pt", "results"))
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": files}


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
        model = load_model(plan["model"])
        result = rt._recover_redetection_batch(model.model, prepared["batches"][0], "/results",
                                               validate=False, max_memory_fraction=.95)
        rt.redetect_results.commit()
        path = Path("/results") / prepared["batches"][0]["root"] / "trace.pt"
        return {"replay": result, "files": [file_ref(path, "results")]}
    if stage == "score":
        from online_8b_to_0p6b import score as existing_score
        return existing_score(prepared)
    raise ValueError("unknown stage")


def bounded(stage, payload):
    verify_plan(payload["plan"])
    check_code(payload["plan"])
    rt.redetect_results.reload()
    folder = Path("/results") / RUN / "attempts" / stage
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


@app.function(cpu=(4, 4), memory=16384, timeout=330, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES)
def cpu_stage(stage, payload):
    if stage not in ("prepare", "score"):
        raise ValueError("unreviewed CPU stage")
    return bounded(stage, payload)


@app.function(gpu="H200", cpu=(4, 4), memory=65536, timeout=6030, startup_timeout=60,
              retries=0, max_containers=1, scaledown_window=2, volumes=VOLUMES)
def gpu_replay(payload):
    return bounded("replay", payload)


def collect(stage, result):
    for ref in result["files"]:
        path = OUT / "cache" / ref["volume"] / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]:
            continue
        content = b"".join(rt.redetect_results.read_file(ref["path"]))
        if len(content) != ref["bytes"] or hashlib.sha256(content).hexdigest() != ref["sha256"]:
            raise ValueError("download changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(OUT / f"collected_{stage}.json", {"files": result["files"]})


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
        result = gpu_replay.remote(payload)
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
