"""Freeze and CPU-verify the remaining single-block 0.6B PRC inputs.

This module cannot launch generation or GPU inference. Its frozen manifests
are consumed by the existing modal_run.py::redetect command.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

import modal
import modal_run as runtime

OUT = Path("outputs/fixed_0p6b_redetect_setup")
MODEL = "Qwen3-0.6B-Base"
WEIGHTS = ("map", "entropy", "naive")
RATE_COLUMNS = {
    "map": ("Map TPR", "Map FPR"),
    "entropy": ("Entropy Aware TPR", "Entropy FPR"),
    "naive": ("Naive TPR", "Naive FPR"),
}
app = modal.App("prc-fixed-0p6b-redetection-setup")
setup_image = runtime.image.add_local_python_source("fixed_prc_redetection_setup")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def rate(value):
    match = re.fullmatch(r"(\d+)/(\d+) \([^)]*\)", value)
    if not match:
        raise ValueError(f"missing historical integer count: {value!r}")
    k, n = map(int, match.groups())
    if n != 500 or not 0 <= k <= n:
        raise ValueError("setup requires the original 500-candidate cohorts")
    return {"detected": k, "count": n}


def selected_rows():
    with Path("hoeffding_results_summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for line, row in enumerate(rows, 2):
        if not (row["PRC Construction"] == "fixed_prc"
                and row["Generation Model"] == row["Entropy Model"] == MODEL
                and row["T"] == row["n"]):
            continue
        if float(row["eta"]) == .05 and int(row["n"]) in (400, 448):
            continue  # Completed MAP/entropy settings are explicitly excluded.
        if float(row["eta"]) == .20 and int(row["n"]) == 8192:
            continue  # Deferred by the user on 2026-09-19.
        if int(row["t"]) != 3 or float(row["Target FPR"]) != .001:
            raise ValueError("unexpected single-block paper configuration")
        selected.append((line, row))
    if len(selected) != 18:
        raise ValueError(f"expected the reviewed 18 main settings, got {len(selected)}")
    return selected


def hardware(n):
    """Reuse n400 A10 batch100 and n3104 A100-80GB batch125 evidence."""
    if n <= 512:
        return {"gpu": "A10G", "batch_size": 100}
    if n <= 2048:
        return {"gpu": "A100-80GB", "batch_size": 125}
    if n == 4096:
        return {"gpu": "A100-80GB", "batch_size": 100}
    raise ValueError(f"no approved hardware policy for n={n}")


def _catalog_index(path):
    result = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["volume"] != "prc-data":
                continue
            key = (row["workspace"], row["relative_path"])
            if key in result and result[key]["sha256"] != row["sha256"]:
                raise ValueError(f"ambiguous archived source: {key}")
            result[key] = row
    return result


def source_reference(catalog, workspace, path):
    if workspace == "live":
        return {"volume": "data", "path": path}
    item = catalog[(workspace, path)]
    sha = item["sha256"]
    return {"volume": "archive", "path": f"objects/sha256/{sha[:2]}/{sha}",
            "sha256": sha, "bytes": int(item["bytes"]),
            "original_workspace": workspace, "original_path": path}


def build_requests(catalog_path):
    """Resolve original cohorts; never choose a null merely by available length."""
    catalog = _catalog_index(catalog_path)
    requests = []
    for line, row in selected_rows():
        n, eta = int(row["n"]), float(row["eta"])
        tag = f"n{n}_t3_eta{eta:.2f}_T{n}_r{row['r value']}"
        expected_records = {}
        if (eta, n) in ((.05, 416), (.10, 768), (.15, 1504)):
            workspaces = ["live"] * 500
            # The remote shard report supplies the actual null source length.
            null_lengths = None
        elif eta == .05:
            workspaces, null_lengths = ["watermark-prc"] * 500, [n] * 500
        elif eta == .10:
            if n in (256, 400):
                tag = f"n{n}_t3_eta0.10_r{row['r value']}"
                workspaces, null_lengths = ["tinghui2012"] * 500, [8192] * 500
            else:
                workspaces, null_lengths = ["new-prc-watermark"] * 500, [n] * 500
        elif eta == .15:
            # n1024 has two historical cohorts. The retained 385/500 result
            # is checked against the lower-n cohort, not selected by n alone.
            ws = "eta-0-15-workspace" if n == 2048 else "eta-0-15-lower-n"
            workspaces, null_lengths = [ws] * 500, [8192] * 500
        elif eta == .20:
            aggregate_path = Path(f"outputs/aggregates/eta0.2_n{n}_T{n}_r{row['r value']}_qwen3_0p6b_base_fpr-0.001.json")
            aggregate = json.loads(aggregate_path.read_text())
            workspaces, null_lengths = [None] * 500, [None] * 500
            for file in aggregate["local_shard_files"]:
                shard = json.loads(Path(file).read_text())
                ws = shard["workspace_label"]
                if ws == "eta01-new-prc-workspace-from-github-acc":
                    ws = "new-prc-watermark"
                for i in shard["prompt_indices"]:
                    if workspaces[i] is not None:
                        raise ValueError("overlapping original shard coverage")
                    workspaces[i], null_lengths[i] = ws, shard["null_cache_T"]
                for record in shard["records"]:
                    expected_records[f"{record['source']}:{record['prompt_idx']}"] = record
            if any(x is None for x in workspaces + null_lengths):
                raise ValueError("incomplete original shard coverage")
        else:
            raise ValueError(f"unreviewed eta: {eta}")
        request = {"id": f"fixed_0p6b_eta{round(eta*100):03d}_n{n}",
                   "group": "main", "n": n, "eta": eta, "t": 3,
                   "r": int(row["r value"]), "fpr": .001,
                   "tag": tag, "generation_model": MODEL,
                   **hardware(n),
                   "source_summary": "hoeffding_results_summary.csv", "source_line": line,
                   "source_summary_sha256": digest("hoeffding_results_summary.csv"),
                   "expected_counts": {w: {s: rate(row[col]) for s, col in zip(("wm", "null"), cols)}
                                       for w, cols in RATE_COLUMNS.items()},
                   "artifact": source_reference(catalog, workspaces[0], tag + "/artifacts.pt"),
                   "expected_records": expected_records,
                   "resolve_live_shard": null_lengths is None}
        request["records"] = [
            {"source": s, "prompt_idx": i,
             "file": source_reference(catalog, workspaces[i],
                (f"{tag}/gens/gen_{2*i + (s == 'null'):04d}.pt" if eta == .10 and n in (256, 400)
                 else f"{tag}/wm/wm_{i:04d}.pt" if s == "wm"
                 else f"_nulls/T{null_lengths[i]}/null_{i:04d}.pt"))}
            for s in ("wm", "null") for i in range(500)
            if s == "wm" or null_lengths is not None]
        requests.append(request)
    for seed in (54321, 67890):
        file = f"outputs/fixed_replicate_n256_t3_eta0.05_prompts500_seed{seed}.json"
        old = json.loads(Path(file).read_text())
        tag = old["tag"]
        requests.append({"id": f"fixed_0p6b_eta005_n256_seed{seed}", "group": "replicates",
            "n": 256, "eta": .05, "t": 3, "r": 253, "fpr": .001, "tag": tag,
            "generation_model": MODEL, **hardware(256),
            "source_summary": file, "source_summary_sha256": digest(file),
            "artifact": source_reference(catalog, "live", tag + "/artifacts.pt"),
            "expected_counts": {w: {s: {"detected": old["counts"][w]["tp" if s == "wm" else "fp"], "count": 500}
                                     for s in ("wm", "null")} for w in WEIGHTS},
            "expected_records": {}, "resolve_live_shard": False,
            "records": [{"source": s, "prompt_idx": i,
                         "file": source_reference(catalog, "live", f"{tag}/wm/wm_{i:04d}.pt" if s == "wm"
                                      else f"_nulls/T{old['null_cache_T']}/null_{i:04d}.pt")}
                        for s in ("wm", "null") for i in range(500)]})
    return requests


def safe_source(reference, roots):
    path = PurePosixPath(reference["path"])
    if path.is_absolute() or any(p in ("", ".", "..") for p in reference["path"].split("/")):
        raise ValueError("source must stay inside its volume")
    return Path(roots[reference["volume"]]) / path


def export_setup():
    """Publish only successfully frozen cases, grouped by selected hardware."""
    requests = build_requests(OUT / "catalog/provenance.tsv")
    groups, entries = {}, []
    for request in requests:
        record = json.loads((OUT / "cases" / request["id"] / "setup.json").read_text())
        if record["status"] != "ready" or not record["audit"]["passed"]:
            raise ValueError(f"CPU preflight did not pass: {request['id']}")
        prepared = record["prepared"]
        manifest = record["manifest"]
        case = manifest["cases"][0]
        if (case["batch_size"] != request["batch_size"]
                or prepared["run"]["execution"]["gpu"] != request["gpu"]
                or case != prepared["run"]["case"]):
            raise ValueError(f"stale hardware preflight: {request['id']}")
        name = request["group"] + ("_a10g" if request["gpu"] == "A10G" else "_a100")
        grouped = groups.setdefault(name, {**manifest, "cases": []})
        grouped["cases"].append(case)
        entries.append({"id": request["id"], "group": request["group"], "manifest": name,
                        "n": request["n"], "eta": request["eta"], **hardware(request["n"]),
                        "candidates": len(case["records"]), "batches": len(prepared["batches"]),
                        "remote_volume": "prc-completion-only", "remote_root": prepared["root"],
                        "remote_setup_file": record["remote_setup_file"],
                        "artifact_sha256": prepared["artifact_sha256"],
                        "audit": record["audit"]})
    manifests = {}
    for name, manifest in groups.items():
        path = OUT / "manifests" / (name + ".json")
        write_json(path, manifest)
        selected = [e for e in entries if e["manifest"] == name]
        manifests[name] = {"path": str(path), "sha256": digest(path), "group": selected[0]["group"],
                           "gpu": selected[0]["gpu"], "settings": len(selected)}
    index = {"schema_version": 1, "protocol": runtime.REDETECT_PROTOCOL,
             "cpu_preflight_passed": True, "csv_out": "outputs/redetection/redetection_results_summary.csv",
             "main_settings": 18, "replicate_settings": 2,
             "deferred": [{"eta": .2, "n": 8192, "reason": "user deferred on 2026-09-19"}],
             "manifests": manifests, "settings": entries}
    write_json(OUT / "index.json", index)
    print(json.dumps({"main_settings": 18, "replicate_settings": 2, "manifests": list(manifests)}))


def freeze_case(request, model, execution, roots, destination):
    """Read-only historical audit followed by the production CPU preflight."""
    import concurrent.futures
    import numpy as np
    import torch
    from detectors import detect_hoeffding, semantic_sha256

    torch.set_num_threads(1)
    n = request["n"]
    if request["generation_model"] != MODEL or request["t"] != 3 or request["fpr"] != .001:
        raise ValueError("only the reviewed same-model single-block settings are allowed")
    if model["id"] != "Qwen/" + MODEL:
        raise ValueError("generation and detection must both be 0.6B Base")

    def load(ref):
        path = safe_source(ref, roots)
        frozen = {"volume": ref["volume"], "path": ref["path"],
                  "sha256": digest(path), "bytes": path.stat().st_size}
        for field in ("sha256", "bytes"):
            if field in ref and ref[field] != frozen[field]:
                raise ValueError(f"archived source changed: {ref['path']}")
        # _redetect_source checks the frozen bytes again at the actual read.
        return frozen, runtime._redetect_source(frozen, roots)

    artifact_ref, raw = load(request["artifact"])
    key, partition = raw["decoding_key"], raw["partition"]
    if key[1].shape != (request["r"], n) or key[-1] != 3 or abs(key[4] - request["eta"]) > 1e-12:
        raise ValueError("original key does not match the requested n/r/t/eta")
    if key[0].shape[0] != n:
        raise ValueError("original generator matrix has different block length")
    runtime.fixed_validate_generation_record(raw.get("config_sig", raw), "0.6B", "artifact")
    requests = list(request["records"])
    expected_records = dict(request["expected_records"])
    evidence = []
    if request["resolve_live_shard"]:
        matches = []
        for path in sorted((Path(roots["data"]) / request["tag"] / "shard_results").rglob("*.json")):
            shard = json.loads(path.read_text())
            cfg = shard.get("config", {})
            if (cfg.get("entropy_model") == MODEL and cfg.get("generation_model") == MODEL
                    and cfg.get("n") == n and cfg.get("T") == n and cfg.get("eta") == request["eta"]
                    and cfg.get("target_fpr") == .001 and shard.get("prompt_indices") == list(range(500))):
                matches.append((path, shard))
        if not matches:
            raise ValueError("no complete original shard report identifies the null cohort")
        if len({json.dumps(x[1]["records"], sort_keys=True) for x in matches}) != 1:
            raise ValueError("ambiguous full original shard reports")
        path, shard = matches[0]
        evidence.append({"path": str(path.relative_to(roots["data"])), "sha256": digest(path)})
        expected_records = {f"{r['source']}:{r['prompt_idx']}": r for r in shard["records"]}
        requests += [{"source": "null", "prompt_idx": i,
                      "file": {"volume": "data", "path": f"_nulls/T{shard['null_cache_T']}/null_{i:04d}.pt"}}
                     for i in range(500)]
    expected_ids = [(s, i) for s in ("wm", "null") for i in range(500)]
    if [(r["source"], r["prompt_idx"]) for r in requests] != expected_ids:
        raise ValueError("require exact ordered WM/null prompt coverage 0..499")

    def verify(ref):
        frozen, record = load(ref["file"])
        source, i = ref["source"], ref["prompt_idx"]
        runtime.fixed_validate_generation_record(record, "0.6B", source, i)
        if record["prompt_idx"] != i or record["watermark"] != (source == "wm"):
            raise ValueError("candidate identity/label mismatch")
        tokens = record["tokens"]
        if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64) or len(tokens) < n:
            raise ValueError("invalid original completion shape/length/dtype")
        tokens = tokens[:n].to(torch.int64).clone()
        if torch.any(tokens < 0) or torch.any(tokens >= partition.shape[1]):
            raise ValueError("out-of-vocabulary completion token")
        token_hash = hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest()
        probs = np.asarray(record["p_trace"], dtype=np.float64)[:n]
        if probs.shape != (n,) or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
            raise ValueError("invalid historical probability trace")
        expected = expected_records.get(f"{source}:{i}")
        if expected and token_hash != expected["tokens_sha256"]:
            raise ValueError("candidate token prefix differs from original detection evidence")
        scores = {}
        for weight in WEIGHTS:
            decision, info = detect_hoeffding(key, tokens, probs, partition, fpr=.001,
                                             weight=weight, return_info=True, completion_only=False)
            if expected and bool(decision) != expected[f"decision_{weight}"]:
                raise ValueError("historical decision differs from original per-candidate evidence")
            scores[weight] = bool(decision)
        return ({"source": source, "prompt_idx": i, "file": frozen, "tokens_sha256": token_hash}, scores)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        verified = list(pool.map(verify, requests))
    counts = {w: {s: {"detected": sum(scores[w] for ref, scores in verified if ref["source"] == s), "count": 500}
                  for s in ("wm", "null")} for w in WEIGHTS}
    if counts != request["expected_counts"]:
        raise ValueError(f"historical counts mismatch: actual={counts}; expected={request['expected_counts']}")
    audit = {"passed": True, "id": request["id"], "group": request["group"],
             "original_counts_reproduced": counts, "verified_candidates": 1000,
             "token_hashes_matched_to_original_reports": len(expected_records),
             "key_partition_semantic_sha256": semantic_sha256({"key": key, "partition": partition}),
             "source_summary": request["source_summary"], "source_summary_sha256": request["source_summary_sha256"],
             "source_report_evidence": evidence, "source_request_sha256": semantic_sha256(request),
             "setup_script_sha256": digest(__file__), "gpu_jobs_launched": 0}
    case = {"id": request["id"], "generation_model": MODEL, "construction": "fixed",
            "artifact": artifact_ref, "lengths": [n], "fpr": .001,
            "fpr_policy": "block_or_bonferroni", "weights": ["map", "entropy"],
            "batch_size": request["batch_size"], "cache": "static",
            "records": [ref for ref, _ in verified],
            "old_tpr": {"detector_model": "Qwen/" + MODEL,
                        "source": request["source_summary"], "evidence_sha256": semantic_sha256(audit),
                        "counts": {str(n): {w: counts[w]["wm"] for w in ("map", "entropy")}}}}
    # Use the production clean-input path, without instantiating a model.
    prepared = runtime._prepare_redetection(case, model, execution, roots, destination)
    if any(b["identity"]["length"] != n for b in prepared["batches"]):
        raise ValueError("preflight exported an unexpected completion length")
    return {"manifest": {"schema_version": 1, "protocol": runtime.REDETECT_PROTOCOL,
                         "model": model, "cases": [case]}, "audit": audit, "prepared": prepared}


@app.function(image=setup_image, cpu=4, memory=8192, timeout=3600, max_containers=4, retries=0,
              volumes={"/data": runtime.data_vol, "/archive": runtime.redetect_archive,
                       "/results": runtime.redetect_results})
def freeze_remote(request, model, execution, previous=None):
    import os
    os.environ["NUMBA_DISABLE_JIT"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    for name, expected in execution["files"].items():
        if digest(Path(runtime.__file__).parent / name) != expected:
            raise ValueError(f"remote execution source differs: {name}")
    try:
        if previous is not None:
            # Reuse the persisted successful historical audit, but revalidate
            # every frozen source through the production preflight after tuning.
            old_path = safe_source({"volume": "results", "path": previous["remote_setup_file"]},
                                   {"results": "/results"})
            old = json.loads(old_path.read_text())
            if old != {k: v for k, v in previous.items() if k != "remote_setup_file"}:
                raise ValueError("saved historical audit differs from remote evidence")
            if old["status"] != "ready" or not old["audit"]["passed"] or old["id"] != request["id"]:
                raise ValueError("no successful historical audit to reuse")
            case = old["manifest"]["cases"][0]
            if (old["manifest"]["model"] != model or case["lengths"] != [request["n"]]
                    or old["audit"]["original_counts_reproduced"] != request["expected_counts"]
                    or old["audit"]["source_summary_sha256"] != request["source_summary_sha256"]):
                raise ValueError("historical audit does not match the requested setting")
            case["batch_size"] = request["batch_size"]
            prepared = runtime._prepare_redetection(case, model, execution,
                            {"data": "/data", "archive": "/archive"}, "/results")
            result = {"manifest": old["manifest"], "audit": old["audit"], "prepared": prepared,
                      "retuned_from": {"path": previous["remote_setup_file"], "sha256": digest(old_path)}}
        else:
            result = freeze_case(request, model, execution,
                                 {"data": "/data", "archive": "/archive"}, "/results")
        record = {"id": request["id"], "status": "ready", **result}
    except Exception as exc:
        record = {"id": request["id"], "status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
    run_hash = hashlib.sha256(json.dumps([request, model, execution], sort_keys=True).encode()).hexdigest()[:24]
    directory = Path("/results/setups/fixed_0p6b_single_block") / request["id"] / run_hash
    write_json(directory / "setup.json", record)
    runtime.redetect_results.commit()
    record["remote_setup_file"] = str((directory / "setup.json").relative_to("/results"))
    return record


@app.local_entrypoint()
def main(stage: str = "plan", group: str = "main", only: str = ""):
    """Plan or freeze+preflight inputs on CPU. No GPU launch command exists here."""
    if stage not in ("plan", "freeze", "retune") or group not in ("main", "replicates", "all"):
        raise ValueError("choose plan/freeze/retune and main/replicates/all")
    requests = build_requests(OUT / "catalog/provenance.tsv")
    requests = [r for r in requests if (group == "all" or r["group"] == group) and (not only or r["id"] == only)]
    if not requests:
        raise ValueError("no matching requested settings")
    model = json.loads(Path("outputs/redetection/.archive/manifests/same_0p6b_eta005_n448.json").read_text())["model"]
    runtime._redetect_model_spec(model)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for name in runtime.EXECUTION_FILES:
        if subprocess.check_output(["git", "show", f"{commit}:{name}"]) != Path(name).read_bytes():
            raise ValueError(f"execution code must match the committed runtime: {name}")
    execution = {"git_commit": commit, "files": {p: digest(p) for p in runtime.EXECUTION_FILES},
                 "gpu": "A100-80GB", "allocator": "expandable_segments:True"}
    write_json(OUT / "requests" / f"{group}.json", requests)
    if stage == "plan":
        print(json.dumps({"settings": len(requests), "group": group, "stage": stage}))
        return
    jobs = []
    for r in requests:
        previous = None
        path = OUT / "cases" / r["id"] / "setup.json"
        if stage == "retune" and path.exists():
            candidate = json.loads(path.read_text())
            if candidate["status"] == "ready":
                previous = candidate
        jobs.append((r, model, {**execution, "gpu": r["gpu"]}, previous))
    for result in freeze_remote.starmap(jobs, order_outputs=False):
        folder = OUT / "cases" / result["id"]
        write_json(folder / "setup.json", result)
        if result["status"] == "ready":
            for key in ("manifest", "audit", "prepared"):
                write_json(folder / f"{key}.json", result[key])
        print(json.dumps({k: result[k] for k in ("id", "status", "remote_setup_file")}
                         | ({"error": result["error"]} if "error" in result else {})), flush=True)
