"""Two disjoint eta=.15 batches using the unchanged, corrected replay worker."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import modal
import online_8b_to_0p6b as base

CASE = "eta015_T6144_N100"
OUT = base.OUT
app = modal.App("prc-online-8b-to-0p6b-parallel", image=base.rt.image.add_local_python_source(
    "online_8b_to_0p6b_parallel", "online_8b_to_0p6b", "fixed_4b_comparison", "online_prc_redetection"))


def select_batch(payload, index):
    if payload["case_id"] != CASE or index not in (0, 1):
        raise ValueError("only the two reviewed eta=.15 batches are allowed")
    base.verify_plan(payload["plan"], CASE)
    prepared = payload["prepared"]
    if len(prepared["batches"]) != 2:
        raise ValueError("require exactly two disjoint batches")
    for i, batch in enumerate(prepared["batches"]):
        identity = batch["identity"]
        if (identity["start"], identity["count"], identity["length"]) != (i * 50, 50, 6144):
            raise ValueError("batch scope changed")
    return {**payload, "prepared": {**prepared, "batches": [prepared["batches"][index]]}}


@app.function(gpu="A100-80GB", cpu=(4, 4), memory=16384,
              timeout=1030, startup_timeout=30, retries=0, max_containers=2,
              scaledown_window=2,
              volumes={"/data": base.rt.data_vol, "/cache": base.rt.hf_cache,
                       "/results": base.rt.redetect_results})
def replay_one(task):
    payload, index, revision = task["payload"], task["index"], task["revision"]
    base.check_code(payload["plan"])
    if base.rt._redetect_sha(__file__) != revision["adapter_sha256"]:
        raise ValueError("parallel adapter changed")
    child = select_batch(payload, index)
    batch = child["prepared"]["batches"][0]
    if (Path("/results") / batch["root"] / "trace.pt").exists():
        raise ValueError("trace already exists; collect it without model execution")
    folder = Path("/results") / "online_8b_to_0p6b_v1" / CASE / "attempts/replay" / f"batch_{index * 50:06d}"
    folder.mkdir(parents=True, exist_ok=False)
    base.write_json(folder / "request.json", child)
    started = time.monotonic()
    started_unix = time.time()
    try:
        with (folder / "worker.log").open("w") as log:
            process = subprocess.Popen([sys.executable, base.__file__, "--child", "replay",
                                        str(folder / "request.json"), str(folder / "response.json")],
                                       stdout=log, stderr=log)
            try:
                status = process.wait(timeout=1000)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
                raise TimeoutError("batch exceeded 1000s; no retry")
        if status:
            raise RuntimeError(f"batch failed with exit {status}; no retry")
        result = json.loads((folder / "response.json").read_text())
    finally:
        timing = {"wall_seconds": time.monotonic() - started, "batch_index": index,
                  "started_unix": started_unix, "finished_unix": time.time(),
                  "resources": revision["per_worker_resources"]}
        base.write_json(folder / "timing.json", timing)
        base.rt.redetect_results.commit()
    return {"index": index, "result": result, "timing": timing,
            "evidence": [base.file_ref(folder / name, "results") for name in
                         ("request.json", "response.json", "worker.log", "timing.json")]}


def download(ref, path):
    import hashlib
    if path.exists() and path.stat().st_size == ref["bytes"] and hashlib.sha256(path.read_bytes()).hexdigest() == ref["sha256"]:
        return
    data = b"".join(base.rt.redetect_results.read_file(ref["path"]))
    if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
        raise ValueError("transfer checksum differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def collect():
    folder = base.case_folder(CASE)
    results = [json.loads((folder / f"parallel_result_{i}.json").read_text()) for i in (0, 1)]
    for result in results:
        for ref in result["result"]["files"]:
            download(ref, folder / "cache" / ref["volume"] / ref["path"])
        for ref in result["evidence"]:
            download(ref, folder / "evidence/replay" / f"batch_{result['index'] * 50:06d}" / Path(ref["path"]).name)
    merged = {"batches": [b for result in results for b in result["result"]["batches"]],
              "files": [f for result in results for f in result["result"]["files"]]}
    base.write_json(folder / "result_replay.json", merged)
    evidence = folder / "evidence/replay"
    base.write_json(evidence / "request.json", json.loads((folder / "attempt_replay.json").read_text()))
    base.write_json(evidence / "response.json", merged)
    base.write_json(evidence / "timing.json", {
        "stage": "replay", "case": CASE, "parallel_workers": 2,
        "wall_seconds": max(r["timing"]["wall_seconds"] for r in results),
        "sum_worker_seconds": sum(r["timing"]["wall_seconds"] for r in results),
        "per_batch": [r["timing"] for r in results],
        "resources": json.loads((OUT / "parallel_eta015_setup.json").read_text())})
    (evidence / "worker.log").write_text("\n".join(
        (evidence / f"batch_{i * 50:06d}" / "worker.log").read_text() for i in (0, 1)))
    base.write_json(folder / "collected_replay.json", merged["files"])


@app.local_entrypoint()
def run(approval_reference: str):
    folder = base.case_folder(CASE)
    revision_path = OUT / "parallel_eta015_setup.json"
    revision = json.loads(revision_path.read_text())
    approval = json.loads((OUT / "parallel_eta015_approval.json").read_text())
    if not approval_reference.strip() or approval["setup_sha256"] != base.rt._redetect_sha(revision_path):
        raise ValueError("require explicit approval of this exact parallel setup")
    if base.rt._redetect_sha(__file__) != revision["adapter_sha256"]:
        raise ValueError("parallel adapter changed")
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("wrong Modal profile")
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("wrong branch")
    plan = json.loads((OUT / "setup.json").read_text())
    base.check_code(plan)
    plan_sha = base.rt._redetect_sha(OUT / "setup.json")
    if json.loads((folder / "attempt_prepare.json").read_text())["plan_sha256"] != plan_sha:
        raise ValueError("preparation setup changed")
    if not (folder / "collected_prepare.json").exists() or (folder / "attempt_replay.json").exists():
        raise ValueError("require collected preparation and no previous replay attempt")
    if base.rt._redetect_sha(folder / "result_prepare.json") != revision["prepared_result_sha256"]:
        raise ValueError("prepared inputs changed")
    prepared = json.loads((folder / "result_prepare.json").read_text())["prepared"]
    payload = {"plan": plan, "case_id": CASE, "prepared": prepared,
               "approval_reference": approval_reference}
    tasks = [{"payload": payload, "index": i, "revision": revision} for i in (0, 1)]
    for task in tasks:
        select_batch(payload, task["index"])
    existing = list(base.rt.redetect_results.listdir(prepared["root"] + "/batches", recursive=True))
    if any(entry.path.endswith("/trace.pt") for entry in existing):
        raise ValueError("saved trace exists; do not launch any replacement GPU pass")
    base.write_json(folder / "attempt_replay.json", {
        "case": CASE, "stage": "replay", "plan_sha256": plan_sha,
        "parallel_setup_sha256": base.rt._redetect_sha(revision_path),
        "approval_reference": approval_reference, "resources": revision})
    failures = []
    for result in replay_one.map(tasks, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failures.append(str(result))
            continue
        base.write_json(folder / f"parallel_result_{result['index']}.json", result)
        for ref in result["result"]["files"]:
            download(ref, folder / "cache" / ref["volume"] / ref["path"])
        print(json.dumps({"saved_batch": result["index"], "seconds": result["timing"]["wall_seconds"]}), flush=True)
    if failures:
        base.write_json(folder / "parallel_failures.json", failures)
        raise RuntimeError("parallel batch failed; completed traces preserved; no retry")
    collect()


if __name__ == "__main__" and sys.argv[1:] == ["collect"]:
    collect()
