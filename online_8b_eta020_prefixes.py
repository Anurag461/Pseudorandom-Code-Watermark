"""Approved CPU-only prefix scoring of frozen T=14336 completion traces."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path("outputs/online_8b_eta020_prefixes_setup")
import modal
import modal_run as rt
from fixed_4b_comparison import check_code, file_ref, write_json

RUN = "online_8b_eta020_N500_prefixes_v1"
app = modal.App("prc-online-8b-eta020-prefixes", image=rt.image.add_local_python_source(
    "online_8b_eta020_prefixes", "online_8b_redetection", "online_prc_redetection",
    "fixed_4b_comparison"))


def verify(payload):
    plan = payload["plan"]
    if (plan["run_id"], plan["N"], plan["eta"], plan["generation_T"], plan["branch"]) != (
            RUN, 500, .2, 14336, "redetection"):
        raise ValueError("unapproved cohort")
    if (plan["candidate_lengths"] != list(range(14320, 0, -16))
            or plan["detectors"] != ["8B"] or plan["weights"] != ["map", "entropy"]
            or any(plan[k] for k in ("null_N", "generation_count", "model_replay_count",
                                    "gpu_count", "automatic_retries"))):
        raise ValueError("scope differs from approved prefix scoring")
    if plan["resources"] != {"workers": 1, "cpu": 4, "memory_mib": 8192, "gpu": None,
                             "work_timeout_seconds": 600, "expected_minutes": [1, 3],
                             "estimated_usd": [.005, .02], "conservative_allowance_usd": .1}:
        raise ValueError("unapproved resources")
    approval = payload["approval"]
    if (not approval["explicit_user_approval"] or approval["authorized_spend_usd"] != .1
            or approval["plan_sha256"] != payload["plan_sha256"]):
        raise ValueError("missing matching approval")
    check_code(payload["execution"])


def read_json(path, expected_sha):
    if rt._redetect_sha(path) != expected_sha:
        raise ValueError(f"source changed: {path}")
    return json.loads(path.read_text())


def compact(scores):
    return {n: {weight: {key: (None if isinstance(info[key], float) and not math.isfinite(info[key])
                                 else info[key])
                         for key in ("decision", "statistic", "threshold", "V", "r", "status")}
                for weight, info in values.items()} for n, values in scores.items()}


def score(payload):
    # Preserve the proven native/scientific loading order before pickle aliases.
    import torch
    import scipy.special
    import galois
    import transformers
    import safetensors.torch
    import qwen
    import detectors
    from detectors import prepare_online_map_prefix_context
    from online_8b_redetection import prepared_weights, adaptive_scores
    from online_prc_redetection import prefix_scores
    verify(payload)
    torch.set_num_threads(4)
    rt.redetect_results.reload()
    plan, files, summaries = payload["plan"], [], {}
    reported = None
    for size in plan["detectors"]:
        source = plan["sources"][size]
        source_root = Path("/results") / source["cloud_combined_root"]
        prepared = read_json(source_root / "prepared.json", source["prepared"]["sha256"])
        baseline = read_json(source_root / "full.json", source["report"]["sha256"])
        if prepared["component_prepared"] != source["component_prepared"]:
            raise ValueError("component manifests changed")
        if baseline["counts"]["14336"] != source["reused_T14336_counts"]:
            raise ValueError("cached 14336 scores changed")
        records, traces, hashes = [], [], {}
        first_artifact = None
        for component in prepared["component_prepared"]:
            path = Path("/results") / component["root"] / "artifact.pt"
            if rt._redetect_sha(path) != component["artifact_sha256"]:
                raise ValueError("component artifact changed")
            artifact = rt._redetect_load(path)
            if first_artifact is None:
                first_artifact = artifact
                context = prepare_online_map_prefix_context(artifact["online_key"], 14336)
            elif artifact["online_key"] != first_artifact["online_key"] or not torch.equal(
                    artifact["partition"], first_artifact["partition"]):
                raise ValueError("component keys/partitions differ")
            for batch in component["batches"]:
                path = Path("/results") / batch["root"] / "trace.pt"
                digest = rt._redetect_sha(path)
                if digest != source["trace_shard_sha256"][batch["root"]]:
                    raise ValueError("primary trace changed")
                hashes[batch["root"]] = digest
                inputs = rt._redetect_inputs(batch, "/results")
                trace = rt._redetect_trace(path, batch["identity"])
                for row, probabilities in enumerate(trace["probabilities_2_to_T"].numpy()):
                    ref = component["run"]["case"]["records"][batch["identity"]["start"] + row]
                    record = {k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")}
                    if size == "8B":
                        traces.append(prepared_weights(context, inputs["tokens"][row], probabilities,
                                                       inputs["partition"]))
                    else:
                        record["scores"] = compact(prefix_scores(context, inputs["tokens"][row], probabilities,
                                                inputs["partition"], reported, completion_only=True))
                    records.append(record)
        if hashes != source["trace_shard_sha256"] or len(records) != 500:
            raise ValueError("incomplete primary traces")
        ids = [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")} for r in records]
        if ids != [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")}
                   for r in baseline["records"]] or [r["prompt_idx"] for r in records] != list(range(500)):
            raise ValueError("cohort coverage differs")
        if size == "8B":
            counts, reported = adaptive_scores(records, traces, plan["candidate_lengths"])
            for r in records:
                r["scores"] = compact(r["scores"])
            del traces
        else:
            counts = {str(n): {w: {"wm": {"detected": sum(r["scores"][str(n)][w]["decision"]
                                                             for r in records), "count": 500},
                                   "null": {"detected": 0, "count": 0}}
                              for w in plan["weights"]} for n in reported}
        result_root = Path("/results") / RUN / size
        output_prepared = copy.deepcopy(prepared)
        output_prepared["root"] = str(result_root.relative_to("/results"))
        output_prepared["run"]["case"]["lengths"] = reported
        report = {"passed": True, "protocol": plan["protocol"], "counts": counts,
                  "reported_lengths": reported, "stop_rule": plan["stop_rule"],
                  "settings": baseline["settings"], "trace_shard_sha256": hashes,
                  "source_report_sha256": source["report"]["sha256"],
                  "reuse": {"T14336_scores_recomputed": False, "primary_traces_replayed": False},
                  "records": records}
        for name, value in (("full.json", report), ("prepared.json", output_prepared),
                            ("summary.json", {k: v for k, v in report.items() if k != "records"})):
            write_json(result_root / name, value)
            files.append(file_ref(result_root / name, "results"))
        rt.redetect_results.commit()
        summaries[size] = {k: report[k] for k in ("counts", "reported_lengths")}
        print(json.dumps({"saved": size, "lengths": len(reported), "last": reported[-1]}), flush=True)
    return {"files": files, "summaries": summaries}


@app.function(cpu=(4, 4), memory=8192, timeout=630, startup_timeout=60,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/results": rt.redetect_results})
def cpu_score(payload):
    verify(payload)
    rt.redetect_results.reload()
    folder = Path("/results") / RUN / "attempt"
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()
    started = time.monotonic()
    log_path = Path("/tmp") / f"{RUN}-{os.getpid()}.log"
    try:
        with log_path.open("x") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", str(folder / "request.json"),
                                     str(folder / "response.json")], stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
                raise TimeoutError("CPU scoring deadline exceeded; no retry")
        if status:
            raise RuntimeError(f"CPU scoring failed ({status}); no retry")
        result = json.loads((folder / "response.json").read_text())
    except Exception as exc:
        write_json(folder / "failure.json", {"error": str(exc)})
        raise
    finally:
        if log_path.exists():
            (folder / "worker.log").write_bytes(log_path.read_bytes())
            log_path.unlink()
        write_json(folder / "timing.json", {"wall_seconds": time.monotonic() - started,
                                           "cpu": 4, "memory_mib": 8192, "gpu": None})
        rt.redetect_results.commit()
    result["files"] += [file_ref(folder / name, "results") for name in
                        ("request.json", "response.json", "worker.log", "timing.json")]
    return result


def collect(result):
    for ref in result["files"]:
        path = OUT / "cache/results" / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]:
            continue
        content = b"".join(rt.redetect_results.read_file(ref["path"]))
        if len(content) != ref["bytes"] or hashlib.sha256(content).hexdigest() != ref["sha256"]:
            raise ValueError("download changed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(OUT / "collected.json", {"files": result["files"]})


@app.local_entrypoint()
def run(approval_reference: str):
    plan = json.loads((OUT / "setup.json").read_text())
    approval = json.loads((OUT / "approval.json").read_text())
    execution = json.loads((OUT / "execution_manifest.json").read_text())
    payload = {"plan": plan, "plan_sha256": rt._redetect_sha(OUT / "setup.json"),
               "approval": approval, "execution": execution}
    verify(payload)
    if (approval_reference != approval["approval_reference"]
            or os.environ.get("MODAL_PROFILE") != "new-prc-watermark"
            or subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection"):
        raise ValueError("approval, profile, or branch mismatch")
    if (OUT / "attempt.json").exists():
        raise ValueError("already attempted; collect saved results, never repeat automatically")
    write_json(OUT / "attempt.json", {"plan_sha256": payload["plan_sha256"],
                                     "approval_reference": approval_reference})
    call = cpu_score.spawn(payload)
    write_json(OUT / "call.json", {"function_call_id": call.object_id})
    result = call.get()
    write_json(OUT / "result.json", result)
    collect(result)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        write_json(sys.argv[3], score(json.loads(Path(sys.argv[2]).read_text())))
    else:
        print((OUT / "PLAN.md").read_text())
