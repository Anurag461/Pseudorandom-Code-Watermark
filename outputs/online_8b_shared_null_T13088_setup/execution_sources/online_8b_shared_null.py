"""Prepare-only by default: one native 8B null replay, two frozen PRC keys.

Paid stages require their own setup-bound approval. No generation, benchmarks,
reference replay, automatic retry, or laptop model execution is included.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from fixed_4b_comparison import check_code, file_ref, load_model, write_json

OUT = Path("outputs/online_8b_shared_null_T13088_setup")
RUN = "online_8b_shared_null_T13088_N500_v1"
STAGES = ("prepare", "replay", "score")
app = modal.App("prc-native-8b-shared-null", image=rt.image.add_local_python_source(
    "online_8b_shared_null", "fixed_4b_comparison", "online_prc_redetection"))
VOLUMES = {"/data": rt.data_vol, "/cache": rt.hf_cache, "/results": rt.redetect_results}


def verify_plan(plan):
    if (plan["run_id"], plan["branch"], plan["profile"], plan["N"], plan["T"], plan["batch_size"]) != (
            RUN, "redetection", "new-prc-watermark", 500, 13088, 50):
        raise ValueError("unreviewed shared-null scope")
    if (plan["protocol"] != rt.REDETECT_PROTOCOL or plan["weights"] != ["map", "entropy"]
            or plan["target_fpr"] != .001 or plan["fpr_policy"] != "one_shot"
            or plan["prompt_indices"] != list(range(500))
            or any(plan[k] for k in ("generation_count", "watermarked_replay_count",
                                     "reference_replays", "benchmarks", "automatic_retries"))):
        raise ValueError("unreviewed scoring protocol or extra work")
    if (plan["model"]["id"], plan["model"]["revision"], plan["model"]["dtype"]) != (
            "Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4", "bfloat16"):
        raise ValueError("wrong detector checkpoint")
    if [k["id"] for k in plan["keys"]] != ["eta015", "eta020"]:
        raise ValueError("require the two original keys")
    for spec, eta, lengths in zip(plan["keys"], (.15, .2),
                                  (list(range(6144, 4655, -16)), list(range(13088, 11839, -16)))):
        if spec["eta"] != eta or spec["lengths"] != lengths:
            raise ValueError("frozen reporting grid changed")
    if plan["detection"] != {"raw_completion_ids": True, "prompt_prefix": False,
            "special_token_prefix": False, "first_coordinate_abstention": True,
            "dtype": "bfloat16", "tf32": False, "cache": "static", "trace_dtype": "float32",
            "score_dtype": "float64", "max_memory_fraction": .95}:
        raise ValueError("detector semantics changed")
    if (plan["stages"], plan["proposed_total_allowance_usd"]) != ({
            "prepare": {"gpu": None, "workers": 1, "cpu": 4, "memory_mib": 16384,
                        "work_timeout_seconds": 600, "allowance_usd": .15},
            "replay": {"gpu": "H200", "workers": 10, "cpu": 4, "memory_mib": 65536,
                       "work_timeout_seconds": 4350, "allowance_usd": 64.7},
            "score": {"gpu": None, "workers": 1, "cpu": 4, "memory_mib": 8192,
                      "work_timeout_seconds": 600, "allowance_usd": .15}}, 65):
        raise ValueError("review proposed resources/deadlines before changing them")
    if [r["path"] for r in plan["source_file_refs"]] != [
            f"_nulls/qwen3_8b_base/T13088/null_{i:04d}.pt" for i in range(500)]:
        raise ValueError("source bank must contain all500 original nulls once")
    if any(r["volume"] != "data" or len(r["sha256"]) != 64 for r in plan["source_file_refs"]):
        raise ValueError("require frozen source-file identities")


def frozen_json(ref):
    path = Path("/"+ref["volume"]) / ref["path"]
    if path.stat().st_size != ref["bytes"] or rt._redetect_sha(path) != ref["sha256"]:
        raise ValueError(f"frozen metadata changed: {ref['path']}")
    return json.loads(path.read_text())


def verify_prepared(plan, prepared):
    run = prepared["run"]
    if (run["protocol"] != plan["protocol"] or run["model"] != plan["model"]
            or run["execution"] != plan["execution"]
            or prepared["partition_sha256"] != plan["partition_sha256"]
            or run["shared_keys"] != plan["keys"]):
        raise ValueError("prepared source/model/key identity changed")
    records = run["case"]["records"]
    if ([(r["source"], r["prompt_idx"]) for r in records] != [("null", i) for i in range(500)]
            or [r["file"] for r in records] != plan["source_file_refs"]):
        raise ValueError("prepared cohort changed")
    if len(prepared["batches"]) != 10:
        raise ValueError("require ten disjoint batches")
    for i, b in enumerate(prepared["batches"]):
        v = b["identity"]
        if (v["start"], v["count"], v["length"], v["cache"], v["protocol"]) != (
                i*50, 50, 13088, "static", plan["protocol"]):
            raise ValueError("invalid batch coverage")


def reject_duplicate_traces(plan, own_root=None):
    paths = {r["path"] for r in plan["source_file_refs"]}
    for path in (Path("/results") / plan["protocol"] / "integrated").glob("*/manifest.json"):
        if own_root and path.parent == Path("/results") / own_root:
            continue
        run = json.loads(path.read_text())
        case = run.get("case", {})
        if (run.get("model", {}).get("id") == plan["model"]["id"]
                and max(case.get("lengths") or [0]) >= plan["T"]
                and any(r.get("source") == "null" and r.get("file", {}).get("path") in paths
                        for r in case.get("records", []))
                and any(path.parent.glob("batches/*/trace.pt"))):
            raise ValueError("overlapping completed native-null traces found; resolve reuse before dispatch")


def watermarked_records(plan, spec):
    """Reuse frozen per-record decisions, without recomputing a watermarked score."""
    result = None
    expected_model = plan["model"]
    for source in spec["watermarked_sources"]:
        prepared, report = frozen_json(source["prepared"]), frozen_json(source["report"])
        model = prepared["run"]["model"]
        if (any(model[k] != expected_model[k] for k in ("id", "revision", "dtype", "tokenizer_sha256"))
                or prepared["partition_sha256"] != plan["partition_sha256"]
                or prepared["run"]["case"]["artifact"] != spec["artifact"]
                or prepared["run"]["protocol"] != plan["protocol"]
                or not report["passed"] or report["protocol"] != plan["protocol"]
                or report["settings"]["eta"] != spec["eta"]):
            raise ValueError("watermarked evidence belongs to different model/protocol/key")
        rows = report["records"]
        if [(r["source"], r["prompt_idx"]) for r in rows] != [("wm", i) for i in range(500)]:
            raise ValueError("watermarked evidence must cover500 once")
        if result is None:
            result = [{k: r[k] for k in ("source", "prompt_idx", "tokens_sha256")} | {"scores": {}} for r in rows]
        for target, row in zip(result, rows):
            if target["tokens_sha256"] != row["tokens_sha256"]:
                raise ValueError("watermarked sources differ")
            for n in spec["lengths"]:
                value = row["scores"].get(str(n))
                if value is None:
                    continue
                if str(n) in target["scores"] and target["scores"][str(n)] != value:
                    raise ValueError("conflicting saved scores")
                target["scores"][str(n)] = value
    if result is None or any(set(r["scores"]) != {str(n) for n in spec["lengths"]} for r in result):
        raise ValueError("saved watermarked scores do not cover the frozen grid")
    return result


def prepare(plan):
    import torch
    from detectors import semantic_sha256, tensor_sha256
    from online_prc import OnlinePRCKey
    reject_duplicate_traces(plan)
    manifest = frozen_json(plan["source_manifest_ref"])
    rt._verify_redetection_checkpoint(plan["model"])
    for name, sha in plan["model"]["metadata_sha256"].items():
        if rt._redetect_sha(Path("/cache") / plan["model"]["cache_directory"] / name) != sha:
            raise ValueError("pinned checkpoint metadata changed; no download")
    artifacts = {}
    for spec in plan["keys"]:
        artifact = rt._redetect_source(spec["artifact"], {"data": "/data"})
        key = OnlinePRCKey.from_dict(artifact["online_key"])
        if (key.noise_rate, key.check_weight, key.row_rate_numerator, key.row_rate_denominator,
                artifact["experiment_seed"], artifact["generation_model"], artifact["T"]) != (
                spec["eta"], 3, 99, 100, 12345, "Qwen3-8B-Base", spec["generation_T"]):
            raise ValueError("original key settings changed")
        error = rt.null_cache_manifest_compatibility_error(manifest, artifact, 13088, "static")
        if error or tensor_sha256(artifact["partition"]) != plan["partition_sha256"]:
            raise ValueError(f"null/key compatibility differs: {error}")
        artifacts[spec["id"]] = artifact
        watermarked_records(plan, spec)  # Verify needed evidence before GPU spending.
    first = artifacts["eta015"]
    partition = first["partition"].to(torch.bfloat16)
    if (partition.ndim != 2 or partition.shape[0] != 2 or not torch.all((partition == 0) | (partition == 1))
            or not torch.all(partition.sum(0) == 1)):
        raise ValueError("invalid binary partition")
    records, tokens = [], []
    for i, ref in enumerate(plan["source_file_refs"]):
        record = rt._redetect_source(ref, {"data": "/data"})
        rt.validate_online_null_record(record, first, i, 13088, source_length=13088,
                                      expected_kv_cache_implementation="static", require_provenance=True)
        value = torch.as_tensor(record["tokens"], dtype=torch.int64).contiguous()
        if value.shape != (13088,) or torch.any(value < 0) or torch.any(value >= partition.shape[1]):
            raise ValueError("invalid original completion tokens")
        records.append({"source": "null", "prompt_idx": i, "file": ref,
                        "tokens_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()})
        tokens.append(value)
    case = {"id": RUN, "generation_model": "Qwen3-8B-Base", "construction": "online",
            "artifact": plan["keys"][0]["artifact"], "lengths": [13088], "weights": ["map", "entropy"],
            "fpr": .001, "fpr_policy": "one_shot", "null_policy": "only",
            "batch_size": 50, "cache": "static", "records": records}
    run = {"protocol": plan["protocol"], "schema_version": 2, "case": case,
           "model": plan["model"], "execution": plan["execution"], "shared_keys": plan["keys"],
           "source_manifest": plan["source_manifest_ref"]}
    root = Path("/results") / plan["protocol"] / "integrated" / semantic_sha256(run)[:24]
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "manifest.json", run)
    files = []
    for spec in plan["keys"]:
        a = artifacts[spec["id"]]
        path = root / (spec["id"] + "_artifact.pt")
        rt._redetect_write(path, {"partition": a["partition"], "online_key": a["online_key"]})
        files.append(file_ref(path, "results"))
    batches = []
    for start in range(0, 500, 50):
        inputs = {"tokens": torch.stack(tokens[start:start+50]), "partition": partition}
        directory = root / "batches" / f"{start:06d}"
        identity = {"protocol": plan["protocol"], "run": root.name, "start": start,
                    "count": 50, "length": 13088, "cache": "static", "input_sha256": semantic_sha256(inputs)}
        rt._redetect_write(directory / "inputs.pt", inputs)
        batches.append({"root": str(directory.relative_to("/results")), "identity": identity})
        files.append(file_ref(directory / "inputs.pt", "results"))
    prepared = {"root": str(root.relative_to("/results")), "run": run, "batches": batches,
                "partition_sha256": tensor_sha256(partition),
                "key_artifacts": {s["id"]: file_ref(root / (s["id"]+"_artifact.pt"), "results") for s in plan["keys"]}}
    verify_prepared(plan, prepared)
    write_json(root / "prepared.json", prepared)
    write_json(Path("/results") / RUN / "prepared.json", prepared)
    files += [file_ref(root / n, "results") for n in ("manifest.json", "prepared.json")]
    rt.redetect_results.commit()
    return {"prepared": prepared, "files": files}


def merge_counts(wm, nulls, lengths):
    if [(r["source"], r["prompt_idx"]) for r in wm+nulls] != [
            (s, i) for s in ("wm", "null") for i in range(500)]:
        raise ValueError("require exactly500 watermarked and500 null candidates")
    return {str(n): {w: {source: {"detected": sum(bool(r["scores"][str(n)][w]["decision"]) for r in rows),
                                 "count": len(rows)} for source, rows in (("wm", wm), ("null", nulls))}
                          for w in ("map", "entropy")} for n in lengths}


def score(plan, prepared):
    from detectors import prepare_online_map_prefix_context
    from online_prc_redetection import prefix_scores
    files, summaries = [], {}
    for spec in plan["keys"]:
        artifact = rt._redetect_source(prepared["key_artifacts"][spec["id"]], {"results": "/results"})
        context = prepare_online_map_prefix_context(artifact["online_key"], max(spec["lengths"]))
        wm = watermarked_records(plan, spec)
        records, hashes = [], {}
        for batch in prepared["batches"]:
            inputs = rt._redetect_inputs(batch, "/results")
            path = Path("/results") / batch["root"] / "trace.pt"
            trace = rt._redetect_trace(path, batch["identity"])
            hashes[batch["root"]] = rt._redetect_sha(path)
            for row, probabilities in enumerate(trace["probabilities_2_to_T"].numpy()):
                scores = prefix_scores(context, inputs["tokens"][row], probabilities,
                                       inputs["partition"], spec["lengths"], completion_only=True)
                compact = {n: {w: {k: None if isinstance(s[k], float) and not math.isfinite(s[k]) else s[k]
                                   for k in ("decision", "statistic", "threshold", "V", "r", "status")}
                               for w, s in values.items()} for n, values in scores.items()}
                ref = prepared["run"]["case"]["records"][batch["identity"]["start"]+row]
                records.append({k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")} | {"scores": compact})
        counts = merge_counts(wm, records, spec["lengths"])
        root = Path("/results") / RUN / spec["id"]
        aggregate = copy.deepcopy(prepared)
        aggregate.update(root=str(root.relative_to("/results")), batches=[], aggregate_only=True,
                         shared_null_prepared_root=prepared["root"])
        aggregate["run"]["case"].update(id=RUN+"_"+spec["id"], artifact=spec["artifact"],
                    lengths=spec["lengths"], records=wm+records, null_policy="evaluate")
        report = {"passed": True, "protocol": plan["protocol"], "counts": counts,
                  "reported_lengths": spec["lengths"], "records": wm+records,
                  "stop_rule": "none; all frozen reporting lengths",
                  "settings": {"eta": spec["eta"], "t": 3, "r_setting": "causal round(0.99L), startup-clamped"},
                  "trace_shard_sha256": hashes, "null_source_manifest": plan["source_manifest_ref"],
                  "watermarked_sources": spec["watermarked_sources"],
                  "reuse": {"shared_null_N": 500, "replay_count_per_null": 1,
                            "watermarked_replayed": False, "watermarked_rescored": False}}
        for name, value in (("prepared.json", aggregate), ("full.json", report),
                            ("summary.json", {k: v for k, v in report.items() if k != "records"})):
            write_json(root / name, value)
            files.append(file_ref(root / name, "results"))
        summaries[spec["id"]] = {"root": str(root.relative_to("/results")), "counts": counts}
        rt.redetect_results.commit()
    return {"summaries": summaries, "files": files}


def child(stage, payload):
    # Keep native imports ahead of the existing pickle compatibility aliases.
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
        rt.hf_cache.reload()
        return prepare(plan)
    prepared = payload["prepared"]
    if json.loads((Path("/results") / RUN / "prepared.json").read_text()) != prepared:
        raise ValueError("prepared inputs changed")
    verify_prepared(plan, prepared)
    if stage == "score":
        return score(plan, prepared)
    worker = payload["worker_id"]
    if stage != "replay" or type(worker) is not int or worker not in range(10):
        raise ValueError("invalid GPU assignment")
    if "H200" not in torch.cuda.get_device_name() or torch.cuda.get_device_properties(0).total_memory < 130*1024**3:
        raise ValueError("H200 required")
    reject_duplicate_traces(plan, prepared["root"])
    batch = prepared["batches"][worker]
    path = Path("/results") / batch["root"] / "trace.pt"
    if path.exists():
        raise ValueError("primary trace already exists; resolve reuse without a repeated GPU launch")
    model = load_model(plan["model"])
    try:
        result = rt._recover_redetection_batch(model.model, batch, "/results", validate=False,
                                               max_memory_fraction=.95)
    finally:
        rt.redetect_results.commit()
    trace = rt._redetect_trace(path, batch["identity"])
    result["trace_metadata"] = {k: trace[k] for k in ("seconds", "peak_allocated_bytes", "peak_reserved_bytes",
                               "total_memory_bytes", "memory_limit_fraction", "full_validation")}
    return {"worker_id": worker, "replay": result, "files": [file_ref(path, "results")]}


def bounded(stage, payload):
    verify_plan(payload["plan"])
    check_code(payload["plan"])
    rt.redetect_results.reload()
    task = f'{payload["worker_id"]:02d}' if stage == "replay" else "cpu"
    folder = Path("/results") / RUN / "attempts" / stage / task
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    rt.redetect_results.commit()
    started = time.monotonic()
    limit = payload["plan"]["stages"][stage]["work_timeout_seconds"]
    log_path = Path("/tmp") / f"{RUN}-{stage}-{task}-{os.getpid()}.log"
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
                raise TimeoutError("budgeted work deadline reached; no retry")
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
        write_json(folder / "timing.json", {"wall_seconds": time.monotonic()-started,
                                           "stage": stage, "task_id": task, "work_timeout_seconds": limit})
        rt.redetect_results.commit()
    result["files"] += [file_ref(folder / n, "results") for n in ("request.json", "response.json", "worker.log", "timing.json")]
    return result


@app.function(cpu=(4, 4), memory=16384, timeout=630, startup_timeout=60,
              max_containers=1, retries=0, scaledown_window=2, volumes=VOLUMES)
def cpu_stage(stage, payload):
    if stage not in ("prepare", "score"):
        raise ValueError("invalid CPU stage")
    return bounded(stage, payload)


@app.function(gpu="H200", cpu=(4, 4), memory=65536, timeout=4380, startup_timeout=60,
              max_containers=10, retries=0, scaledown_window=2, volumes=VOLUMES)
def gpu_stage(payload):
    return bounded("replay", payload)


def authorize(stage, reference):
    plan = json.loads((OUT / "setup.json").read_text())
    verify_plan(plan)
    check_code(plan)
    if stage not in STAGES or not reference.strip():
        raise ValueError("require a named stage and explicit approval reference")
    if (os.environ.get("MODAL_PROFILE") != "new-prc-watermark" or
            subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection"):
        raise ValueError("wrong profile or branch")
    sha = rt._redetect_sha(OUT / "setup.json")
    approval_path = OUT / f"approval_{stage}.json"
    if not approval_path.exists():
        raise ValueError("setup only; this paid stage has no explicit approval")
    a = json.loads(approval_path.read_text())
    if (a.get("explicit_user_approval") is not True or a.get("approval_reference") != reference
            or a.get("plan_sha256") != sha or a.get("stage") != stage
            or a.get("authorized_spend_usd", 0) < plan["stages"][stage]["allowance_usd"]
            or a.get("confirmed_available_budget_usd", 0) < plan["stages"][stage]["allowance_usd"]
            or a.get("authorized_total_budget_usd", 0) < plan["proposed_total_allowance_usd"]):
        raise ValueError("require exact stage approval and a sufficient new total budget")
    if (OUT / f"attempt_{stage}.json").exists():
        raise ValueError("already attempted; do not repeat paid work")
    for prior in STAGES[:STAGES.index(stage)]:
        if not (OUT / f"collected_{prior}.json").exists():
            raise ValueError("collect preceding stage first")
        if json.loads((OUT / f"attempt_{prior}.json").read_text())["plan_sha256"] != sha:
            raise ValueError("setup changed after prior stage")
    payload = {"plan": plan, "approval_reference": reference}
    if stage != "prepare":
        payload["prepared"] = json.loads((OUT / "result_prepare.json").read_text())["prepared"]
        verify_prepared(plan, payload["prepared"])
    return payload, sha


def collect(result, name):
    for ref in result["files"]:
        if ref["volume"] != "results":
            raise ValueError("unexpected output volume")
        path = OUT / "cache/results" / ref["path"]
        if path.exists() and rt._redetect_sha(path) == ref["sha256"]:
            continue
        data = b"".join(rt.redetect_results.read_file(ref["path"]))
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("transfer checksum differs")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+".partial")
        tmp.write_bytes(data)
        tmp.replace(path)
    write_json(OUT / f"collected_{name}.json", result["files"])


def run(stage, reference):
    payload, sha = authorize(stage, reference)  # All authorization checks precede app.run.
    with (OUT / f"attempt_{stage}.json").open("x") as f:
        json.dump({"stage": stage, "plan_sha256": sha, "approval_reference": reference,
                   "created_unix": time.time()}, f, indent=2)
    with app.run():
        write_json(OUT / f"app_{stage}.json", {"app_id": app.app_id})
        if stage == "replay":
            completed, failures = {}, []
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = {pool.submit(gpu_stage.remote, {**payload, "worker_id": i}): i for i in range(10)}
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        write_json(OUT / f"workers/result_{i:02d}.json", result)
                        collect(result, f"worker_{i:02d}")
                        completed[i] = result
                    except Exception as exc:
                        failures.append({"worker": i, "error": str(exc)})
            if failures:
                write_json(OUT / "failures_replay.json", failures)
                raise RuntimeError("incomplete replay; completed traces saved; no automatic retry")
            result = {"worker_count": 10, "files": [r for i in range(10) for r in completed[i]["files"]]}
        else:
            memory = payload["plan"]["stages"][stage]["memory_mib"]
            result = cpu_stage.with_options(memory=memory).remote(stage, payload)
        write_json(OUT / f"result_{stage}.json", result)
        collect(result, stage)
    if stage == "score":
        for spec in payload["plan"]["keys"]:
            root = OUT / "cache/results" / result["summaries"][spec["id"]]["root"]
            rt._append_redetection_csv(json.loads((root / "prepared.json").read_text()),
                                     json.loads((root / "full.json").read_text()), payload["plan"]["csv"])


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
            print(json.dumps({"status": "setup_only", "null_N": 500, "T": 13088,
                              "keys": 2, "reporting_points": 173, "paid_compute_launched": False}))
