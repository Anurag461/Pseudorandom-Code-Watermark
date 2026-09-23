"""Replay four approved online cohorts, reuse null traces, and score saved prefixes."""
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from pathlib import Path

import modal
import modal_run as runtime

OUT = Path("outputs/online_0p6b_redetect_setup")
app = runtime.app
image = runtime.image.add_local_python_source("online_prc_redetection")
WEIGHTS = ("map", "entropy")


def save(path, value):
    runtime._redetect_write(path, value)


def prefix_scores(context, tokens, probabilities, partition, lengths, *, completion_only):
    """Reuse causal checks; preserve the scalar detector's summation order."""
    import numpy as np
    from detectors import _soft_tokens, tokens_to_bits, score_prepared_online_map_prefix
    maximum = context["maximum_length"]
    bits = tokens_to_bits(tokens[:maximum], partition)
    probs = np.asarray(probabilities, dtype=np.float64)[:maximum-int(completion_only)]
    if probs.shape != (maximum-int(completion_only),) or not np.isfinite(probs).all():
        raise ValueError("invalid probability coverage")
    result = {str(n): {} for n in lengths}
    for weight in WEIGHTS:
        soft = _soft_tokens(bits, probs, weight, completion_only)
        checks = np.prod(soft[context["supports"]], axis=1)
        prepared = {**context, "signed_check_values": context["otp_signs"]*checks,
                    "squared_check_values": checks**2}
        for n in lengths:
            info = score_prepared_online_map_prefix(prepared, n, fpr=.001, fpr_policy="one_shot")
            result[str(n)][weight] = {**info, "weight": weight}
    return result


def requests():
    plan = json.loads((OUT / "proposed_plan.json").read_text())
    completed = json.loads(Path("outputs/fixed_0p6b_redetect_setup/results_summary.json").read_text())
    fixed = {x["id"]: x for x in completed["settings"]}
    result = []
    for family in plan["families"]:
        request = dict(family)
        request["id"] = f"online_0p6b_eta{round(family['eta']*100):03d}_n{family['replay_length']}"
        path = Path(family["recorded_sweep"])
        if runtime._redetect_sha(path) != family["recorded_sweep_sha256"]:
            raise ValueError("historical sweep changed")
        sweep = json.loads(path.read_text())
        request["expected_map"] = {str(row["prompt_idx"]):
            {n: score["decision"] for n, score in row["map_scores"].items()}
            for row in sweep["grid_results"]}
        request["full_audits"] = {}
        if sweep.get("final_audit"):
            audit = sweep["final_audit"]
            request["full_audits"][str(audit["n"])] = audit["counts"]
        extra = []
        if family["eta"] == .05:
            extra = ["outputs/online_causal_n256_t3_eta0.05_prompts500_sampler-poscdf-v1_from_n400.json",
                     "outputs/online_causal_n400_t3_eta0.05_prompts500_sampler-poscdf-v1.json"]
        if family["eta"] == .20:
            extra = ["outputs/online_causal_n3104_t3_eta0.20_prompts500_sampler-poscdf-v1_kvcache-static-v1_from_n4096.json"]
        request["additional_evidence"] = []
        for filename in extra:
            audit = json.loads(Path(filename).read_text())
            request["full_audits"][str(audit["n"])] = audit["counts"]
            request["additional_evidence"].append({"path": filename, "sha256": runtime._redetect_sha(filename)})
        null_source = family["null_cache_reuse"]
        source_case = null_source["completed_case"]
        folder = Path("outputs/fixed_0p6b_redetect_setup/cases") / source_case
        request["null_prepared"] = json.loads((folder / "prepared.json").read_text())
        request["null_report_sha256"] = fixed[source_case]["result_sha256"]
        live = Path("outputs/fixed_0p6b_redetect_setup/cases") / null_source["live_source_hash_evidence_case"] / "manifest.json"
        request["null_refs"] = [r for r in json.loads(live.read_text())["cases"][0]["records"] if r["source"] == "null"]
        result.append(request)
    return result


def execution(gpu):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in runtime.EXECUTION_FILES:
        if subprocess.check_output(["git", "show", f"{commit}:{name}"]) != Path(name).read_bytes():
            raise ValueError(f"runtime must match the committed version: {name}")
    return {"git_commit": commit, "files": {n: runtime._redetect_sha(n) for n in runtime.EXECUTION_FILES},
            "gpu": gpu, "allocator": "expandable_segments:True",
            "orchestrator_sha256": runtime._redetect_sha(__file__)}


def assert_execution(expected):
    for name, sha in expected["files"].items():
        if runtime._redetect_sha(Path(runtime.__file__).parent / name) != sha:
            raise ValueError(f"runtime source changed: {name}")
    if runtime._redetect_sha(__file__) != expected["orchestrator_sha256"]:
        raise ValueError("online orchestration source changed")


def import_nulls(target, source, source_report_sha, destination):
    """Import only byte-verified response-only traces with identical inputs."""
    import torch
    root = Path(destination)
    actual_source = json.loads((root/source["root"]/"manifest.json").read_text())
    if actual_source != source["run"] or actual_source["model"] != target["run"]["model"]:
        raise ValueError("cached null model/provenance mismatch")
    if (actual_source["execution"]["files"] != target["run"]["execution"]["files"]
            or actual_source["execution"]["gpu"] != target["run"]["execution"]["gpu"]):
        raise ValueError("cached null execution mismatch")
    report_path = root/source["root"]/"full.json"
    if runtime._redetect_sha(report_path) != source_report_sha:
        raise ValueError("cached null report checksum mismatch")
    report = json.loads(report_path.read_text())
    if not report["passed"]:
        raise ValueError("cached source run did not pass")
    by_start = {b["identity"]["start"]: b for b in source["batches"]}
    imported = []
    for batch in target["batches"]:
        start, count = batch["identity"]["start"], batch["identity"]["count"]
        refs = target["run"]["case"]["records"][start:start+count]
        if all(r["source"] == "wm" for r in refs):
            continue
        if any(r["source"] != "null" for r in refs):
            raise ValueError("batch mixes watermarked and null candidates")
        old = by_start[start]
        old_refs = source["run"]["case"]["records"][start:start+count]
        if [(r["source"], r["prompt_idx"], r["file"]["sha256"]) for r in refs] != [
                (r["source"], r["prompt_idx"], r["file"]["sha256"]) for r in old_refs]:
            raise ValueError("cached null candidates differ")
        old_inputs, new_inputs = runtime._redetect_inputs(old, destination), runtime._redetect_inputs(batch, destination)
        if any(not torch.equal(old_inputs[k], new_inputs[k]) for k in ("tokens", "partition")):
            raise ValueError("cached null token/partition mismatch")
        path = root/old["root"]/"trace.pt"
        if runtime._redetect_sha(path) != report["trace_shard_sha256"][old["root"]]:
            raise ValueError("cached null trace checksum mismatch")
        original = runtime._redetect_trace(path, old["identity"])
        provenance = {"source_root": old["root"], "trace_sha256": runtime._redetect_sha(path),
                      "source_identity": old["identity"], "source_execution": actual_source["execution"],
                      "source_report_sha256": source_report_sha}
        copied = {"identity": batch["identity"], "probabilities_2_to_T": original["probabilities_2_to_T"],
                  "probabilities_sha256": original["probabilities_sha256"], "full_validation": False,
                  "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "seconds": 0.0,
                  "imported_from": provenance}
        output = root/batch["root"]/"trace.pt"
        if output.exists():
            existing = runtime._redetect_trace(output, batch["identity"])
            if existing.get("imported_from") != provenance or existing["probabilities_sha256"] != copied["probabilities_sha256"]:
                raise ValueError("conflicting cached imported trace")
        else:
            save(output, copied)
        runtime._redetect_trace(output, batch["identity"])
        imported.append({"target_root": batch["root"], **provenance})
    if sum(b["identity"]["count"] for b in target["batches"] if b["identity"]["start"] >= 500) != 500 or not imported:
        raise ValueError("incomplete null reuse")
    return imported


@app.function(image=image, cpu=4, memory=8192, timeout=3600, max_containers=4, retries=0,
              volumes={"/data": runtime.data_vol, "/results": runtime.redetect_results})
def freeze_remote(request, model, expected_execution):
    import concurrent.futures
    import hashlib
    import numpy as np
    import torch
    from detectors import prepare_online_map_prefix_context, semantic_sha256, tensor_sha256
    from online_prc import OnlinePRCKey, GENERATION_SAMPLER_VERSION
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    torch.set_num_threads(1)
    assert_execution(expected_execution)
    runtime.redetect_results.reload()
    maximum, lengths = request["replay_length"], request["reported_lengths"]
    artifact_ref = {"volume": "data", "path": request["source_tag"]+"/artifacts.pt",
                    "sha256": request["artifact"]["artifact_sha256"], "bytes": request["artifact"]["artifact_bytes"]}
    artifact = runtime._redetect_source(artifact_ref, {"data": "/data"})
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    if (key.noise_rate != request["eta"] or key.check_weight != 3 or artifact["T"] != maximum
            or artifact["experiment_seed"] != 12345 or key.row_rate_numerator != 99 or key.row_rate_denominator != 100
            or runtime.artifact_generation_model_size(artifact) != "0.6B"):
        raise ValueError("artifact differs from approved online configuration")
    if tensor_sha256(artifact["partition"]) != request["artifact"]["partition_sha256"]:
        raise ValueError("partition changed")
    context = prepare_online_map_prefix_context(key, maximum)
    def verify(source, i):
        if source == "wm":
            path = Path("/data")/request["source_tag"]/"wm"/f"wm_{i:04d}.pt"
            ref = {"volume": "data", "path": str(path.relative_to("/data")),
                   "sha256": runtime._redetect_sha(path), "bytes": path.stat().st_size}
        else:
            ref = request["null_refs"][i]["file"]
        record = runtime._redetect_source(ref, {"data": "/data"})
        if source == "wm":
            segments = runtime.validate_online_watermarked_record(record, artifact, i)
            if any(s["sampler_version"] != GENERATION_SAMPLER_VERSION for s in segments):
                raise ValueError("watermarked sampler differs from approved lineage")
            if request["eta"] == .05:
                old_path = Path("/data")/request["source_tag"].replace("n512_T512", "n400_T400")/"wm"/f"wm_{i:04d}.pt"
                old = runtime._redetect_load(old_path)
                if not torch.equal(record["tokens"][:400], old["tokens"]) or not np.array_equal(np.asarray(record["p_trace"])[:400], old["p_trace"]):
                    raise ValueError("canonical n400/n256 lineage does not match ceiling")
        else:
            runtime.validate_online_null_record(record, artifact, i, maximum, source_length=request["null_source_T"])
        if record["prompt_idx"] != i or record["watermark"] != (source == "wm"):
            raise ValueError("candidate identity mismatch")
        tokens = record["tokens"][:maximum].to(torch.int64).clone()
        scores = prefix_scores(context, tokens, record["p_trace"], artifact["partition"], lengths, completion_only=False)
        if source == "wm":
            for n, expected in request["expected_map"][str(i)].items():
                if scores[n]["map"]["decision"] != expected:
                    raise ValueError(f"historical MAP decision differs: prompt={i}, n={n}")
        frozen = {"source": source, "prompt_idx": i, "file": ref,
                  "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest()}
        return frozen, {n: {w: s["decision"] for w, s in weights.items()} for n, weights in scores.items()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(lambda pair: verify(*pair), [(s, i) for s in ("wm", "null") for i in range(500)]))
    counts = {str(n): {w: {s: {"detected": sum(scores[str(n)][w] for ref, scores in verified if ref["source"] == s), "count": 500}
                           for s in ("wm", "null")} for w in WEIGHTS} for n in lengths}
    for n, old in request["full_audits"].items():
        for w in WEIGHTS:
            if counts[n][w]["wm"]["detected"] != old[w]["tp"] or counts[n][w]["null"]["detected"] != old[w]["fp"]:
                raise ValueError(f"historical full audit counts differ: n={n}, weight={w}")
    audit = {"passed": True, "id": request["id"], "candidate_count": 1000,
             "historical_counts": counts, "historical_map_decisions_matched": sum(len(v) for v in request["expected_map"].values()),
             "full_audit_lengths_matched": sorted(map(int, request["full_audits"])),
             "request_sha256": semantic_sha256(request), "orchestrator_sha256": expected_execution["orchestrator_sha256"]}
    case = {"id": request["id"], "generation_model": "Qwen3-0.6B-Base", "construction": "online",
            "artifact": artifact_ref, "lengths": lengths, "fpr": .001, "fpr_policy": "one_shot",
            "weights": list(WEIGHTS), "batch_size": request["batch_size"], "cache": "static",
            "records": [ref for ref, _ in verified],
            "old_tpr": {"detector_model": model["id"], "source": request["recorded_sweep"]+"; frozen historical generation p_trace",
                        "evidence_sha256": semantic_sha256(audit),
                        "counts": {n: {w: value["wm"] for w, value in weights.items()} for n, weights in counts.items()}}}
    prepared = runtime._prepare_redetection(case, model, expected_execution, {"data": "/data"}, "/results")
    imports = import_nulls(prepared, request["null_prepared"], request["null_report_sha256"], "/results")
    result = {"status": "ready", "id": case["id"], "audit": audit, "prepared": prepared,
              "manifest": {"schema_version": 1, "protocol": runtime.REDETECT_PROTOCOL, "model": model, "cases": [case]},
              "imported_null_batches": imports}
    remote = "setups/online_0p6b/"+case["id"]+"/"+prepared["root"].split("/")[-1]
    result["remote_setup_root"] = remote
    save(Path("/results")/remote/"request.json", request)
    save(Path("/results")/remote/"setup.json", result)
    (Path("/results")/remote/"online_prc_redetection.py").write_bytes(Path(__file__).read_bytes())
    runtime.redetect_results.commit()
    return result


@app.function(image=image, cpu=4, memory=8192, timeout=3600, retries=0,
              volumes={"/results": runtime.redetect_results})
def finish_remote(prepared):
    import torch
    from detectors import prepare_online_map_prefix_context
    torch.set_num_threads(1)
    assert_execution(prepared["run"]["execution"])
    runtime.redetect_results.reload()
    root = Path("/results")/prepared["root"]
    if runtime._redetect_sha(root/"artifact.pt") != prepared["artifact_sha256"]:
        raise ValueError("scoring artifact changed")
    artifact = runtime._redetect_load(root/"artifact.pt")
    case = prepared["run"]["case"]
    context = prepare_online_map_prefix_context(artifact["online_key"], max(case["lengths"]))
    records, hashes, imports = [], {}, []
    for batch in prepared["batches"]:
        inputs = runtime._redetect_inputs(batch, "/results")
        path = Path("/results")/batch["root"]/"trace.pt"
        trace = runtime._redetect_trace(path, batch["identity"])
        hashes[batch["root"]] = runtime._redetect_sha(path)
        if "imported_from" in trace:
            imports.append({"target_root": batch["root"], **trace["imported_from"]})
        for row, probs in enumerate(trace["probabilities_2_to_T"].numpy()):
            ref = case["records"][batch["identity"]["start"]+row]
            scores = prefix_scores(context, inputs["tokens"][row], probs, inputs["partition"], case["lengths"], completion_only=True)
            compact = {n: {w: {k: (None if isinstance(s[k], float) and not math.isfinite(s[k]) else s[k])
                              for k in ("decision", "statistic", "threshold", "V", "r", "status")}
                           for w, s in weights.items()} for n, weights in scores.items()}
            records.append({k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")} | {"scores": compact})
    if [(r["source"], r["prompt_idx"], r["tokens_sha256"]) for r in records] != [
            (r["source"], r["prompt_idx"], r["tokens_sha256"]) for r in case["records"]]:
        raise ValueError("incomplete candidate coverage")
    counts = {str(n): {w: {s: {"detected": sum(r["scores"][str(n)][w]["decision"] for r in records if r["source"] == s), "count": 500}
                          for s in ("wm", "null")} for w in WEIGHTS} for n in case["lengths"]}
    key = context["online_key"]
    report = {"passed": True, "protocol": runtime.REDETECT_PROTOCOL, "counts": counts,
              "settings": {"eta": key.noise_rate, "t": key.check_weight, "r_setting": "causal round(0.99L), startup-clamped"},
              "score_metadata": {"method": "hoeffding_online_causal", "fpr": .001, "fpr_policy": "one_shot", "weights": list(WEIGHTS)},
              "trace_shard_sha256": hashes, "imported_null_batches": imports, "records": records}
    save(root/"full.json", report)
    save(root/"summary.json", {k: v for k, v in report.items() if k != "records"})
    runtime.redetect_results.commit()
    return {"root": prepared["root"], "length_points": len(counts), "status": "completed"}


def sort_csv(path):
    import fcntl
    from decimal import Decimal
    with Path(path).open("r+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = csv.DictReader(handle)
        rows, columns = list(reader), reader.fieldnames
        rows.sort(key=lambda r: (r["PRC Construction"] != "fixed_prc", Decimal(r["eta"]), int(r["T"])))
        handle.seek(0)
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())


@app.local_entrypoint()
def main(stage: str = "freeze", group: str = "all"):
    if stage not in ("freeze", "full") or group not in ("all", "a10g", "a100"):
        raise ValueError("choose freeze/full and all/a10g/a100")
    planned = requests()
    planned = [r for r in planned if group == "all" or (r["gpu"] == "A10G") == (group == "a10g")]
    model = json.loads(Path("outputs/fixed_0p6b_redetect_setup/manifests/main_a100.json").read_text())["model"]
    if stage == "freeze":
        for result in freeze_remote.starmap([(r, model, execution(r["gpu"])) for r in planned], order_outputs=False):
            folder = OUT/"cases"/result["id"]
            save(folder/"setup.json", result)
            for key in ("manifest", "prepared", "audit"):
                save(folder/(key+".json"), result[key])
            print(json.dumps({"id": result["id"], "status": result["status"], "null_batches_imported": len(result["imported_null_batches"])}), flush=True)
        return
    for request in planned:
        folder = OUT/"cases"/request["id"]
        result = json.loads((folder/"setup.json").read_text())
        prepared = result["prepared"]
        if result["status"] != "ready" or prepared["run"]["execution"] != execution(request["gpu"]):
            raise ValueError("frozen setup is stale")
        watermark_batches = [b for b in prepared["batches"] if b["identity"]["start"] < 500]
        worker = runtime.RedetectionModel.with_options(**{
            **runtime.model_cls_options("0.6B", request["gpu"], 10), "memory": 8192, "scaledown_window": 2,
        })(entropy_model_size="0.6B", generation_model_size="0.6B", trace_kv_cache_implementation="static",
           completion_model=json.dumps(model, sort_keys=True))
        print(json.dumps({"id": request["id"], "stage": "watermarked_replay", "batches": len(watermark_batches), "null_gpu_batches": 0}), flush=True)
        worker.redetect_batch.remote(watermark_batches[0], validate=True)
        for done in worker.redetect_batch.map(watermark_batches):
            print(json.dumps(done), flush=True)
        finished = finish_remote.remote(prepared)
        for name in ("full.json", "summary.json"):
            (folder/name).write_bytes(b"".join(runtime.redetect_results.read_file(prepared["root"]+"/"+name)))
        runtime._append_redetection_csv(prepared, json.loads((folder/"full.json").read_text()), runtime.REDETECT_CSV)
        sort_csv(runtime.REDETECT_CSV)
        save(folder/"completed.json", finished)
        print(json.dumps({"id": request["id"], **finished}), flush=True)
