"""CPU-only scoring of cached 0.6B traces at missing native-8B lengths."""
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

import modal
import modal_run as rt

OUT = Path("outputs/online_8b_0p6b_missing_prefixes_setup")
RUN = "online_8b_0p6b_missing_prefixes_20260924_v1"
RESOURCES = {"workers": 1, "cpu": 4, "memory_mib": 8192, "gpu": None,
             "work_timeout_seconds": 600, "expected_minutes": [1, 3],
             "estimated_usd": [.005, .02], "conservative_allowance_usd": .1}
app = modal.App("prc-8b-0p6b-missing-prefixes", image=rt.image.add_local_python_source(
    "online_8b_0p6b_missing_prefixes", "online_prc_redetection"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_ref(path):
    return {"volume": "results", "path": str(path.relative_to("/results")),
            "sha256": rt._redetect_sha(path), "bytes": path.stat().st_size}


def verify_scope(plan):
    if (plan["run_id"], plan["N"], plan["branch"], plan["protocol"]) != (
            RUN, 500, "main", "completion_only_raw_abstain_v1"):
        raise ValueError("unexpected run, cohort, branch, or protocol")
    if (plan["detectors"] != ["0.6B"] or plan["weights"] != ["map", "entropy"]
            or plan["resources"] != RESOURCES
            or any(plan[k] for k in ("null_N", "generation_count", "model_replay_count",
                                    "gpu_count", "automatic_retries"))):
        raise ValueError("scope/resources differ from the proposed CPU-only run")
    families = [(f["id"], f["eta"], f["generation_T"], f["lengths"]) for f in plan["families"]]
    priority_lengths = [11856] + [n for n in range(11840, 14336, 16) if n != 11856]
    if families != [("eta020", .2, 14336, priority_lengths), ("eta015", .15, 6144, [4096])]:
        raise ValueError("unexpected prefix scope or priority")


def verify(payload):
    verify_scope(payload["plan"])
    approval = payload["approval"]
    if (approval.get("explicit_user_approval") is not True
            or approval.get("authorized_spend_usd") != .1
            or approval.get("plan_sha256") != payload["plan_sha256"]):
        raise ValueError("explicit approval of this exact plan and allowance is required")
    encoded = (json.dumps(payload["plan"], indent=2, allow_nan=False) + "\n").encode()
    if hashlib.sha256(encoded).hexdigest() != payload["plan_sha256"]:
        raise ValueError("plan hash differs")
    for name, digest in payload["execution"]["runtime_source_sha256"].items():
        if rt._redetect_sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"reviewed source changed: {name}")


def read_json(path, digest):
    if rt._redetect_sha(path) != digest:
        raise ValueError(f"cached source changed: {path}")
    return json.loads(path.read_text())


def compact(scores):
    return {n: {w: {k: (None if isinstance(info[k], float) and not math.isfinite(info[k])
                         else info[k])
                     for k in ("decision", "statistic", "threshold", "V", "r", "status")}
                for w, info in weights.items()} for n, weights in scores.items()}


def score(payload):
    # Complete scientific imports before the established NumPy pickle aliases.
    import torch
    import scipy.special
    import galois
    import transformers
    import safetensors.torch
    import qwen
    import detectors
    from detectors import prepare_online_map_prefix_context
    from online_prc_redetection import prefix_scores

    verify(payload)
    torch.set_num_threads(4)
    rt.redetect_results.reload()
    plan, files, summaries = payload["plan"], [], {}
    for family in plan["families"]:
        source = family["source"]
        root = Path("/results") / source["root"]
        prepared = read_json(root / "prepared.json", source["prepared_sha256"])
        baseline = read_json(root / "full.json", source["report_sha256"])
        if (baseline["counts"] != source["existing_counts"]
                or baseline["settings"]["eta"] != family["eta"]
                or prepared["run"]["model"]["id"] != "Qwen/Qwen3-0.6B-Base"):
            raise ValueError("cached detector/cohort differs")
        components = prepared.get("component_prepared", [prepared])
        records, hashes, first_artifact = [], {}, None
        for component in components:
            path = Path("/results") / component["root"] / "artifact.pt"
            if rt._redetect_sha(path) != component["artifact_sha256"]:
                raise ValueError("cached artifact changed")
            artifact = rt._redetect_load(path)
            if first_artifact is None:
                first_artifact = artifact
                context = prepare_online_map_prefix_context(artifact["online_key"], family["generation_T"])
            elif (artifact["online_key"] != first_artifact["online_key"]
                  or not torch.equal(artifact["partition"], first_artifact["partition"])):
                raise ValueError("component keys or partitions differ")
            for batch in component["batches"]:
                if batch["identity"]["count"] != 50 or batch["identity"]["length"] != family["generation_T"]:
                    raise ValueError("unexpected cached batch")
                path = Path("/results") / batch["root"] / "trace.pt"
                digest = rt._redetect_sha(path)
                if digest != source["trace_shard_sha256"][batch["root"]]:
                    raise ValueError("cached trace changed")
                hashes[batch["root"]] = digest
                inputs = rt._redetect_inputs(batch, "/results")
                if not torch.equal(inputs["partition"], artifact["partition"]):
                    raise ValueError("batch partition differs from artifact")
                trace = rt._redetect_trace(path, batch["identity"])
                for row, probabilities in enumerate(trace["probabilities_2_to_T"].numpy()):
                    ref = component["run"]["case"]["records"][batch["identity"]["start"] + row]
                    record = {k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")}
                    record["scores"] = compact(prefix_scores(
                        context, inputs["tokens"][row], probabilities, inputs["partition"],
                        family["lengths"], completion_only=True))
                    records.append(record)
        ids = [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")} for r in records]
        expected = [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")}
                    for r in baseline["records"]]
        if (hashes != source["trace_shard_sha256"] or ids != expected
                or [r["prompt_idx"] for r in records] != list(range(500))
                or any(r["source"] != "wm" for r in records)):
            raise ValueError("incomplete or changed cohort")
        counts = {str(n): {w: {"wm": {"detected": sum(r["scores"][str(n)][w]["decision"]
                                                        for r in records), "count": 500},
                              "null": {"detected": 0, "count": 0}}
                          for w in plan["weights"]} for n in family["lengths"]}
        result_root = Path("/results") / RUN / family["id"]
        output_prepared = copy.deepcopy(prepared)
        output_prepared["root"] = str(result_root.relative_to("/results"))
        output_prepared["run"]["case"]["lengths"] = family["lengths"]
        report = {"passed": True, "protocol": plan["protocol"], "counts": counts,
                  "reported_lengths": family["lengths"], "settings": baseline["settings"],
                  "trace_shard_sha256": hashes, "source_report_sha256": source["report_sha256"],
                  "reuse": {"existing_lengths_recomputed": False, "primary_traces_replayed": False},
                  "execution": payload["execution"], "records": records}
        for name, value in (("full.json", report), ("prepared.json", output_prepared),
                            ("summary.json", {k: v for k, v in report.items() if k != "records"})):
            write_json(result_root / name, value)
            files.append(file_ref(result_root / name))
        if family["id"] == "eta020":
            priority = {"eta": .2, "T": 11856, "counts": counts["11856"]}
            write_json(result_root / "priority_T11856.json", priority)
            files.append(file_ref(result_root / "priority_T11856.json"))
            print(json.dumps({"priority_result": priority}), flush=True)
        rt.redetect_results.commit()
        summaries[family["id"]] = counts
        print(json.dumps({"saved": family["id"], "lengths": len(counts)}), flush=True)
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
                proc.kill()
                proc.wait()
                raise TimeoutError("CPU deadline exceeded; no retry")
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
    result["files"] += [file_ref(folder / name) for name in
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
    payload = {"plan": json.loads((OUT / "setup.json").read_text()),
               "plan_sha256": rt._redetect_sha(OUT / "setup.json"),
               "approval": json.loads((OUT / "approval.json").read_text()),
               "execution": json.loads((OUT / "execution_manifest.json").read_text())}
    verify(payload)
    if (approval_reference != payload["approval"]["approval_reference"]
            or os.environ.get("MODAL_PROFILE") != "new-prc-watermark"
            or subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "main"):
        raise ValueError("approval, profile, or branch mismatch")
    if (OUT / "attempt.json").exists():
        raise ValueError("already attempted; collect saved results, never retry automatically")
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
