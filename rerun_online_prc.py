"""Fresh same-setting replay of the approved online eta=.05, T=512 cohort."""
import copy
import csv
import json
import re
from pathlib import Path

import online_prc_redetection as online

runtime = online.runtime
app = online.app
image = online.image.add_local_python_source("rerun_online_prc")
PARENT = "online_0p6b_eta005_n512"


@app.function(image=image, cpu=4, memory=8192, timeout=1800, retries=0,
              volumes={"/data": runtime.data_vol, "/results": runtime.redetect_results})
def prepare_repeat(parent, case, execution, parent_report_sha):
    online.assert_execution(execution)
    if runtime._redetect_sha(__file__) != execution["rerun_driver_sha256"]:
        raise ValueError("rerun driver changed")
    runtime.redetect_results.reload()
    prepared = runtime._prepare_redetection(case, parent["prepared"]["run"]["model"], execution,
                                            {"data": "/data"}, "/results")
    if prepared["root"] == parent["prepared"]["root"]:
        raise ValueError("rerun must have a fresh cache namespace")
    for batch in prepared["batches"]:
        if batch["identity"]["start"] < 500 and (Path("/results")/batch["root"]/"trace.pt").exists():
            raise ValueError("watermarked rerun cache already exists; select a new label")
    imports = online.import_nulls(prepared, parent["prepared"], parent_report_sha, "/results")
    audit = copy.deepcopy(parent["audit"])
    audit.update({"id": case["id"], "historical_counts": {"512": audit["historical_counts"]["512"]},
                  "parent_audit": PARENT})
    remote = "setups/online_0p6b/"+case["id"]+"/"+prepared["root"].split("/")[-1]
    result = {"status": "ready", "id": case["id"], "audit": audit, "prepared": prepared,
              "manifest": {"schema_version": 1, "protocol": runtime.REDETECT_PROTOCOL,
                           "model": prepared["run"]["model"], "cases": [case]},
              "remote_setup_root": remote, "imported_null_batches": imports}
    online.save(Path("/results")/remote/"setup.json", result)
    runtime.redetect_results.commit()
    return result


def annotate_csv(root, parent_root):
    import fcntl
    import os
    with Path(runtime.REDETECT_CSV).open("r+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames
        selected = [r for r in rows if "run="+root+";" in r["Notes"]]
        if len(selected) != 1:
            raise ValueError("expected exactly one rerun CSV row")
        selected[0]["Notes"] += "; independent watermarked rerun; reused verified nulls; parent="+parent_root
        handle.seek(0)
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    online.sort_csv(runtime.REDETECT_CSV)


@app.local_entrypoint()
def rerun(label: str = "rerun01"):
    import torch
    import collect_online_prc_redetection as collect
    if not re.fullmatch(r"[a-z0-9_]+", label):
        raise ValueError("use a simple lowercase rerun label")
    parent_folder = online.OUT/"cases"/PARENT
    parent = json.loads((parent_folder/"setup.json").read_text())
    parent_report_sha = runtime._redetect_sha(parent_folder/"full.json")
    case = copy.deepcopy(parent["prepared"]["run"]["case"])
    case.update({"id": PARENT+"_"+label, "lengths": [512],
                 "rerun": {"parent_root": parent["prepared"]["root"], "parent_report_sha256": parent_report_sha,
                           "fresh_watermarked_inference": True, "reuse_nulls": True}})
    folder = online.OUT/"cases"/case["id"]
    if folder.exists():
        raise ValueError("local rerun already exists; select a new label")
    execution = online.execution("A10G")
    execution["rerun_driver_sha256"] = runtime._redetect_sha(__file__)
    setup = prepare_repeat.remote(parent, case, execution, parent_report_sha)
    prepared = setup["prepared"]
    for name in ("setup", "prepared", "audit", "manifest"):
        online.save(folder/(name+".json"), setup if name == "setup" else setup[name])
    worker = runtime.RedetectionModel.with_options(
        **{**runtime.model_cls_options("0.6B", "A10G", 5), "memory": 8192, "scaledown_window": 2}
    )(entropy_model_size="0.6B", generation_model_size="0.6B", trace_kv_cache_implementation="static",
      completion_model=json.dumps(prepared["run"]["model"], sort_keys=True))
    batches = [b for b in prepared["batches"] if b["identity"]["start"] < 500]
    print(json.dumps({"id": case["id"], "root": prepared["root"], "fresh_watermarked_batches": len(batches), "null_gpu_batches": 0}), flush=True)
    first = worker.redetect_batch.remote(batches[0], validate=True)
    if first["cached"]:
        raise ValueError("representative rerun unexpectedly reused a watermarked trace")
    print(json.dumps({"representative": first}), flush=True)
    for done in worker.redetect_batch.map(batches[1:]):
        if done["cached"]:
            raise ValueError("rerun unexpectedly reused a watermarked trace")
        print(json.dumps(done), flush=True)
    finished = online.finish_remote.remote(prepared)
    for name in ("full.json", "summary.json"):
        (folder/name).write_bytes(b"".join(runtime.redetect_results.read_file(prepared["root"]+"/"+name)))
    online.save(folder/"completed.json", finished)
    request = {"id": case["id"], "eta": .05, "replay_length": 512, "reported_lengths": [512], "gpu": "A10G", "batch_size": 100}
    verified = collect.collect_case(runtime.redetect_results, request)
    old_report = json.loads((parent_folder/"full.json").read_text())
    new_report = json.loads((folder/"full.json").read_text())
    score_changes = []
    for old, new in zip(old_report["records"], new_report["records"]):
        assert all(old[k] == new[k] for k in ("source", "prompt_idx", "tokens_sha256"))
        for weight in online.WEIGHTS:
            if old["scores"]["512"][weight] != new["scores"]["512"][weight]:
                score_changes.append({"source": old["source"], "prompt_idx": old["prompt_idx"], "weight": weight,
                                      "old": old["scores"]["512"][weight], "new": new["scores"]["512"][weight]})
    trace_changes = []
    for old_batch, new_batch in zip(parent["prepared"]["batches"], prepared["batches"]):
        old_path = parent_folder/"cache"/old_batch["root"]/"trace.pt"
        assert runtime._redetect_sha(old_path) == old_report["trace_shard_sha256"][old_batch["root"]]
        old = runtime._redetect_trace(old_path, old_batch["identity"])["probabilities_2_to_T"]
        new = runtime._redetect_trace(folder/"cache"/new_batch["root"]/"trace.pt", new_batch["identity"])["probabilities_2_to_T"]
        delta = (old-new).abs()
        trace_changes.append({"start": new_batch["identity"]["start"], "bitwise_equal": torch.equal(old, new),
                              "changed_values": int((delta != 0).sum()), "maximum_absolute_difference": float(delta.max())})
    comparison = {"status": "completed_verified", "id": case["id"], "parent": PARENT,
                  "parent_report_sha256": parent_report_sha, "request": request,
                  "previous_counts": old_report["counts"]["512"], "rerun_counts": new_report["counts"]["512"],
                  "score_changes": score_changes, "trace_comparison": trace_changes, "result": verified}
    summary_path = online.OUT/"reruns"/(case["id"]+".json")
    online.save(summary_path, comparison)
    runtime._append_redetection_csv(prepared, new_report, runtime.REDETECT_CSV)
    annotate_csv(prepared["root"], parent["prepared"]["root"])
    index_path = collect.RESULTS/"cache_index.json"
    index = json.loads(index_path.read_text())
    index["runs"].append({"case": case["id"], "kind": "rerun", "parent_case": PARENT,
                          "modal_root": prepared["root"], "modal_result_file": "full.json",
                          "local_result_file": "../online_0p6b_redetect_setup/cases/"+case["id"]+"/full.json",
                          "result_sha256": verified["result_sha256"], "rerun_summary_file": str(summary_path),
                          "cache_verification_file": str(folder/"cache_verification.json"), **verified["verification"]})
    online.save(index_path, index)
    paths = [Path(__file__), summary_path, folder/"setup.json", folder/"full.json", folder/"cache_verification.json"]
    entries = [{"path": str(p), "sha256": runtime._redetect_sha(p)} for p in paths]
    online.save(online.OUT/"reruns"/(case["id"]+"_archive.json"), collect.local_archive(case["id"], entries))
    print(json.dumps({"status": "completed_verified", "counts": comparison["rerun_counts"],
                      "changed_scores": len(score_changes), "all_traces_bitwise_equal": all(t["bitwise_equal"] for t in trace_changes),
                      "summary": str(summary_path)}), flush=True)
