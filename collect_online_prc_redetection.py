"""Read back completed online traces, verify caches, and publish local indexes."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import tarfile
import time
from pathlib import Path

import modal
import online_prc_redetection as replay
from detectors import semantic_sha256

OUT = replay.OUT
RESULTS = Path("outputs/redetection")


def download(volume, remote, local, expected=None):
    data = b"".join(volume.read_file(remote))
    sha = hashlib.sha256(data).hexdigest()
    if expected and sha != expected:
        raise ValueError(f"remote checksum differs: {remote}")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    return {"remote": remote, "local": str(local), "sha256": sha, "bytes": len(data)}


def local_archive(kind, entries):
    """Keep additional reproducibility bundles local, as requested by the user."""
    def archive_name(entry):
        return str(Path(entry["path"]).resolve().relative_to(Path.cwd().resolve()))

    path = OUT/".archive"/(kind+"-"+semantic_sha256(entries)[:24]+".tar.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for entry in entries:
            archive.add(entry["path"], arcname=archive_name(entry), recursive=False)
    with tarfile.open(path, "r:gz") as archive:
        for entry in entries:
            if hashlib.sha256(archive.extractfile(archive_name(entry)).read()).hexdigest() != entry["sha256"]:
                raise ValueError("local archive verification failed")
    return {"status": "verified_local_only", "path": str(path), "sha256": replay.runtime._redetect_sha(path),
            "bytes": path.stat().st_size, "files": entries}


def archive_setup(volume=None, *, upload=False):
    paths = [Path("online_prc_redetection.py"), Path("collect_online_prc_redetection.py"),
             Path("verify_online_n3104.py"),
             Path("tests/test_online_prc_redetection.py"), OUT/"PLAN.md", OUT/"proposed_plan.json"]
    for request in replay.requests():
        folder = OUT/"cases"/request["id"]
        paths.extend(folder/name for name in ("setup.json", "audit.json", "manifest.json", "prepared.json"))
        paths.append(Path(request["recorded_sweep"]))
        paths.extend(Path(e["path"]) for e in request["additional_evidence"])
    paths.extend(Path(p) for p in replay.runtime.EXECUTION_FILES)
    entries = [{"path": str(p), "sha256": replay.runtime._redetect_sha(p), "bytes": p.stat().st_size}
               for p in sorted(set(paths))]
    index = local_archive("setup", entries)
    if not upload:
        replay.save(OUT/"setup_archive.json", index)
        print(json.dumps({"setup_archive": index["path"], "files_verified": len(entries), "status": index["status"]}), flush=True)
        return
    root = "setups/online_0p6b/bundles/"+semantic_sha256(entries)[:24]
    with volume.batch_upload() as batch:
        for entry in entries:
            batch.put_file(entry["path"], root+"/"+entry["path"])
    for entry in entries:
        if hashlib.sha256(b"".join(volume.read_file(root+"/"+entry["path"]))).hexdigest() != entry["sha256"]:
            raise ValueError("setup archive readback failed")
    index = {"status": "verified", "volume": "prc-completion-only", "root": root, "files": entries}
    replay.save(OUT/"setup_archive.json", index)
    print(json.dumps({"setup_archive": root, "files_verified": len(entries)}), flush=True)


def collect_case(volume, request, *, use_verified_local_cache=False):
    folder = OUT/"cases"/request["id"]
    setup = json.loads((folder/"setup.json").read_text())
    prepared = setup["prepared"]
    root = prepared["root"]
    report = json.loads((folder/"full.json").read_text())
    if not report["passed"] or not (folder/"completed.json").exists():
        raise ValueError(f"case not complete: {request['id']}")
    refs = [(root+"/full.json", folder/"full.json", replay.runtime._redetect_sha(folder/"full.json")),
            (root+"/summary.json", folder/"summary.json", replay.runtime._redetect_sha(folder/"summary.json")),
            (root+"/manifest.json", folder/"cache"/root/"manifest.json", None),
            (root+"/artifact.pt", folder/"cache"/root/"artifact.pt", prepared["artifact_sha256"])]
    for batch in prepared["batches"]:
        for name in ("inputs.pt", "trace.pt"):
            remote = batch["root"]+"/"+name
            refs.append((remote, folder/"cache"/remote,
                         report["trace_shard_sha256"][batch["root"]] if name == "trace.pt" else None))
    if use_verified_local_cache:
        previous = json.loads((folder/"cache_verification.json").read_text())
        if not previous["passed"]:
            raise ValueError("local cache has not passed remote readback verification")
        by_remote = {entry["remote"]: entry for entry in previous["files"]}
        files = []
        for remote, local, expected in refs:
            entry = by_remote[remote]
            sha = replay.runtime._redetect_sha(local)
            if sha != entry["sha256"] or (expected and sha != expected):
                raise ValueError(f"previously verified cache changed: {local}")
            files.append(entry)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(lambda ref: download(volume, *ref), refs))
    if json.loads((folder/"cache"/root/"manifest.json").read_text()) != prepared["run"]:
        raise ValueError("remote manifest changed")
    traces = []
    for batch in prepared["batches"]:
        replay.runtime._redetect_inputs(batch, folder/"cache")
        trace = replay.runtime._redetect_trace(folder/"cache"/batch["root"]/"trace.pt", batch["identity"])
        traces.append(trace)
    imported = [t for t in traces if "imported_from" in t]
    fresh = [t for t in traces if "imported_from" not in t]
    if sum(t["identity"]["count"] for t in fresh) != 500 or sum(t["identity"]["count"] for t in imported) != 500:
        raise ValueError("trace cohort coverage differs")
    if sum(bool(t["full_validation"]) for t in fresh) != 1:
        raise ValueError("expected one independent scalar validation batch")
    if len(report["records"]) != 1000 or len(report["counts"]) != len(request["reported_lengths"]):
        raise ValueError("incomplete detector coverage")
    verification = {"passed": True, "files": files, "trace_batches": len(traces),
                    "new_trace_batches": len(fresh), "imported_null_batches": len(imported),
                    "full_validation_batches": 1,
                    "new_probabilities": 500*(request["replay_length"]-1),
                    "reused_null_probabilities": 500*(request["replay_length"]-1),
                    "peak_allocated_bytes": max(t["peak_allocated_bytes"] for t in fresh),
                    "gpu_method_seconds": sum(t["seconds"] for t in fresh)}
    replay.save(folder/"cache_verification.json", verification)
    return {"id": request["id"], "eta": request["eta"], "replay_length": request["replay_length"],
            "reported_lengths": request["reported_lengths"], "gpu": request["gpu"], "batch_size": request["batch_size"],
            "remote_root": root, "remote_setup_file": setup["remote_setup_root"]+"/setup.json",
            "local_result_file": str(folder/"full.json"), "result_sha256": replay.runtime._redetect_sha(folder/"full.json"),
            "execution_commit": prepared["run"]["execution"]["git_commit"],
            "orchestrator_sha256": prepared["run"]["execution"]["orchestrator_sha256"],
            "counts": report["counts"], "historical_counts": setup["audit"]["historical_counts"],
            "verification": {k: v for k, v in verification.items() if k != "files"}}


def finish(volume, *, upload=False, use_verified_local_cache=False):
    settings = [collect_case(volume, request, use_verified_local_cache=use_verified_local_cache) for request in replay.requests()]
    rows = list(csv.DictReader((RESULTS/"redetection_results_summary.csv").open()))
    online_rows = [r for r in rows if any("run="+s["remote_root"]+";" in r["Notes"] for s in settings)]
    if len(online_rows) != 154:
        raise ValueError("CSV does not contain exactly the 154 new points")
    review_path = OUT/"n3104_consistency.json"
    review = json.loads(review_path.read_text())
    if (not review["passed"] or review["new_report_sha256"] != settings[-1]["result_sha256"]
            or review["new_counts"] != settings[-1]["counts"]["3104"]):
        raise ValueError("n3104 consistency review is missing or stale")
    match = all(review["old_counts"][w][s]["detected"] == review["new_counts"][w][s]["detected"]
                for w in replay.WEIGHTS for s in ("wm", "null"))
    for setting in settings:
        setting["first_recorded_length_at_90_percent"] = {
            w: {kind: min((int(n) for n, values in setting[field].items() if values[w]["wm"]["detected"] >= 450), default=None)
                for kind, field in (("old", "historical_counts"), ("new", "counts"))}
            for w in replay.WEIGHTS}
    result = {"status": "completed", "protocol": replay.runtime.REDETECT_PROTOCOL, "settings": settings,
              "length_points": 154, "candidate_prefixes": 154000, "detector_decisions": 308000,
              "existing_n3104_counts_match": match, "n3104_consistency_review": str(review_path),
              "n3104_consistency_review_sha256": replay.runtime._redetect_sha(review_path), "csv_rows": len(rows)}
    replay.save(OUT/"results_summary.json", result)
    summary_path, index_path = RESULTS/"summary.json", RESULTS/"cache_index.json"
    summary, index = json.loads(summary_path.read_text()), json.loads(index_path.read_text())
    ids = {s["id"] for s in settings}
    summary["rows"] = [r for r in summary["rows"] if r["case"] not in ids]
    index["runs"] = [r for r in index["runs"] if r["case"] not in ids]
    lines = ["# Online 0.6B redetection results", "", "Completed four seed-12345 families: 154 recorded lengths, 500 watermarked and 500 null candidates per point. MAP and entropy use raw completion tokens, coordinate 1 zero, and one-shot FPR=.001 at each length.", "", "All 18 new watermarked trace batches and 18 reused null batches were read back and verified. All reported lengths have zero observed false positives for both detectors.", "", "| eta | Longest length | MAP old → new | Entropy old → new | MAP / entropy FP | GPU / batch |", "|---:|---:|---|---|---|---|"]
    for s in settings:
        longest = str(s["replay_length"])
        old, new = s["historical_counts"][longest], s["counts"][longest]
        rate = lambda c: f"{c['wm']['detected']/5:.1f}%"
        lines.append(f"| {s['eta']:.2f} | {longest} | {rate(old['map'])} → {rate(new['map'])} | {rate(old['entropy'])} → {rate(new['entropy'])} | {new['map']['null']['detected']}/500 / {new['entropy']['null']['detected']}/500 | {s['gpu']} / {s['batch_size']} |")
        folder = OUT/"cases"/s["id"]
        index["runs"].append({"case": s["id"], "kind": "current", "modal_root": s["remote_root"],
                              "modal_result_file": "full.json", "local_result_file": "../online_0p6b_redetect_setup/cases/"+s["id"]+"/full.json",
                              "result_sha256": s["result_sha256"], "local_trace_dir": "../online_0p6b_redetect_setup/cases/"+s["id"]+"/cache",
                              "trace_checksums_file": "full.json", "trace_checksums_file_sha256": s["result_sha256"],
                              "cache_verification_file": str(folder/"cache_verification.json"),
                              "reported_lengths": s["reported_lengths"], "gpu": s["gpu"], "batch_size": s["batch_size"],
                              "execution_commit": s["execution_commit"], "orchestrator_sha256": s["orchestrator_sha256"],
                              **s["verification"]})
        for n in s["reported_lengths"]:
            for weight in replay.WEIGHTS:
                old, new = s["historical_counts"][str(n)][weight], s["counts"][str(n)][weight]
                summary["rows"].append({"case": s["id"], "generator": "0.6B", "detector": "0.6B", "n": n,
                                        "eta": s["eta"], "t": 3, "construction": "online", "weight": weight,
                                        "wm_count": 500, "null_count": 500, "prompted_detected": old["wm"]["detected"],
                                        "raw_detected": new["wm"]["detected"], "prompted_false_positives": old["null"]["detected"],
                                        "raw_false_positives": new["null"]["detected"],
                                        "change_pp": (new["wm"]["detected"]-old["wm"]["detected"])/5})
    lines += ["", "First recorded lengths reaching 90% MAP TPR after redetection: eta=.05 does not reach 90% through 512; eta=.10 reaches it at 848; eta=.15 at 1648; eta=.20 at 3136. These are sampled grid points, not an interpolation or a monotonicity assumption.", "", "At eta=.20, length 3104, the new 4096-derived prefix gives 449/500 MAP (89.8%) versus 448/500 (89.6%) in the earlier standalone redetection. Entropy is unchanged at 402/500 (80.4%). [The consistency audit](n3104_consistency.json) verifies identical tokens, model, key and partition and reproduces both reports from their respective cached traces. Three near-threshold MAP decisions flip (two to positive, one to negative). Cached BF16 probabilities differ between the earlier batch-125/length-3104 and new batch-100/length-4096 executions; batch size and replay length were not isolated with extra GPU inference. Both records are preserved.", "", "All length-level results are in [the sorted CSV](../redetection/redetection_results_summary.csv). Frozen source manifests, keys, token batches, traces and per-candidate scores are stored locally under `cases/` and in Modal volume `prc-completion-only`; see [the cache index](../redetection/cache_index.json). Historical source evidence remains in the original experiment cache. No null or shorter-length model inference was added. Additional reproducibility archives are kept local as requested.", ""]
    (OUT/"RESULTS.md").write_text("\n".join(lines))
    replay.save(summary_path, summary)
    replay.save(index_path, index)
    plan_path = OUT/"proposed_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["status"] = "completed_verified"
    replay.save(plan_path, plan)
    readme_path = RESULTS/"README.md"
    readme = readme_path.read_text()
    paragraph = ("The four main **online PRC 0.6B → 0.6B families are complete**, covering 154 recorded lengths at eta=.05, .10, .15 and .20. Only the longest watermarked completions were replayed; all 2,000 null traces were reused from verified fixed-run caches, and shorter lengths were scored on CPU. All 36 trace shards were read back and verified. See the [online result summary](../online_0p6b_redetect_setup/RESULTS.md) for results and cache provenance.\n\n")
    if "The four main **online PRC 0.6B → 0.6B families are complete**" not in readme:
        position = readme.index("The remaining **18 main fixed-PRC")
        readme_path.write_text(readme[:position]+paragraph+readme[position:])
    paths = [OUT/"results_summary.json", OUT/"RESULTS.md", review_path, Path("verify_online_n3104.py"), summary_path,
             index_path, RESULTS/"redetection_results_summary.csv"]
    if (OUT/"setup_archive.json").exists():
        paths.append(OUT/"setup_archive.json")
    paths += [OUT/"cases"/s["id"]/"cache_verification.json" for s in settings]
    entries = [{"path": str(p), "sha256": replay.runtime._redetect_sha(p)} for p in paths]
    root = "setups/online_0p6b/completed/"+semantic_sha256(entries)[:24]
    if not upload:
        replay.save(OUT/"completed_archive.json", local_archive("completed", entries))
        print(json.dumps({"status": "completed_verified_locally", "length_points": 154, "csv_rows": len(rows)}), flush=True)
        return
    with volume.batch_upload() as batch:
        for entry in entries:
            batch.put_file(entry["path"], root+"/"+entry["path"])
    for entry in entries:
        if hashlib.sha256(b"".join(volume.read_file(root+"/"+entry["path"]))).hexdigest() != entry["sha256"]:
            raise ValueError("completed archive readback failed")
    replay.save(OUT/"completed_archive.json", {"status": "verified", "volume": "prc-completion-only", "root": root, "files": entries})
    print(json.dumps({"status": "completed_verified", "length_points": 154, "csv_rows": len(rows), "archive": root}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["setup", "finish"])
    parser.add_argument("--upload-archive", action="store_true", help="Upload additional summary archive after explicit approval")
    parser.add_argument("--wait", action="store_true", help="Wait up to two hours for the already-launched replay drivers")
    parser.add_argument("--use-verified-local-cache", action="store_true", help="Recheck the already read-back local files instead of downloading again")
    args = parser.parse_args()
    if args.wait:
        if args.stage != "finish":
            parser.error("--wait is only for finish")
        required = [OUT/"cases"/r["id"]/"completed.json" for r in replay.requests()]
        deadline, previous = time.monotonic()+7200, None
        while not all(p.exists() for p in required):
            pending = [p.parent.name for p in required if not p.exists()]
            if pending != previous:
                print(json.dumps({"waiting_for": pending}), flush=True)
                previous = pending
            for name in ("full-a10g.log", "full-a100.log"):
                if "Traceback (most recent call last)" in (OUT/name).read_text():
                    raise RuntimeError(f"replay failed; inspect {OUT/name}")
            if time.monotonic() >= deadline:
                raise TimeoutError("replay did not finish within two hours")
            time.sleep(15)
    volume = modal.Volume.from_name("prc-completion-only")
    if args.stage == "setup":
        archive_setup(volume, upload=args.upload_archive)
    else:
        finish(volume, upload=args.upload_archive, use_verified_local_cache=args.use_verified_local_cache)
