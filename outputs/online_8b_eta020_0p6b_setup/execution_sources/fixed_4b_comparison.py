"""One approved stage at a time for fixed PRC 4B -> {4B, 0.6B}, N=100.

No new sampler, PRC construction, model implementation, or scoring routine.
Read the local PLAN.md first. Never invoke a paid stage without user approval.
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

OUT = Path("outputs/fixed_4b_eta005_n1024_N100_setup")
RUN = "fixed_4b_eta005_n1024_N100_v1"
DATA = Path("/data") / RUN
RESULTS = Path("/results") / RUN
FILES = (*rt.EXECUTION_FILES, "fixed_4b_comparison.py")
# GPU, host RAM (MiB), work timeout (s). All stages reserve four physical cores.
RESOURCES = {"prepare": (None, 16384, 600), "generate": ("H100", 65536, 600),
             "freeze": (None, 16384, 180), "replay_4b": ("H100", 65536, 600),
             "replay_0p6b": ("A100-80GB", 16384, 360), "score": (None, 8192, 180)}
BASE_RESOURCES = RESOURCES.copy()
PREREQUISITES = {"prepare": (), "generate": ("prepare",), "freeze": ("generate",),
                 "replay_4b": ("freeze",), "replay_0p6b": ("freeze",),
                 "score": ("replay_4b", "replay_0p6b")}
app = modal.App("prc-fixed-4b-100", image=rt.image.add_local_python_source("fixed_4b_comparison"))


def configure(n):
    """Select an isolated fixed-length cohort without altering its numerics."""
    if type(n) is not int or n not in (256, 512, 1024):
        raise ValueError("reviewed fixed lengths are 256, 512 and 1024")
    global OUT, RUN, DATA, RESULTS, RESOURCES
    OUT = Path(f"outputs/fixed_4b_eta005_n{n}_N100_setup")
    RUN = f"fixed_4b_eta005_n{n}_N100_v1"
    DATA, RESULTS = Path("/data") / RUN, Path("/results") / RUN
    RESOURCES = BASE_RESOURCES.copy()
    if n < 1024:
        limits = {"prepare": 120, "generate": 240, "freeze": 120,
                  "replay_4b": 240, "replay_0p6b": 180, "score": 120}
        RESOURCES = {stage: (gpu, memory, limits[stage])
                     for stage, (gpu, memory, _) in RESOURCES.items()}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def file_ref(path, volume):
    path = Path(path)
    return {"volume": volume, "path": str(path.relative_to("/" + volume)),
            "sha256": rt._redetect_sha(path), "bytes": path.stat().st_size}


def check_code(plan):
    for name, digest in plan["runtime_source_sha256"].items():
        if rt._redetect_sha(Path(__file__).parent / name) != digest:
            raise ValueError(f"source differs from reviewed setup: {name}")


def load_model(spec):
    rt._verify_redetection_checkpoint(spec)
    for name, digest in spec["metadata_sha256"].items():
        if rt._redetect_sha(Path("/cache") / spec["cache_directory"] / name) != digest:
            raise ValueError(f"checkpoint metadata changed: {name}")
    os.environ["PRC_MODEL_REVISION"] = spec["revision"]
    we = rt.load_watermark_model(spec["size"])
    # Preserve the corrected order: finish native/model imports BEFORE aliases.
    rt._numpy_pickle_compat()
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    if next(we.model.parameters()).dtype != torch.bfloat16:
        raise ValueError("BF16 model required")
    we.model.eval().requires_grad_(False)
    return we


def prepare(plan):
    import shutil
    from huggingface_hub import hf_hub_download
    from detectors import tensor_sha256
    artifact = rt._redetect_source(plan["source_artifact"], {"archive": "/archive"})
    key, part = artifact["decoding_key"], artifact["partition"]
    n = plan["n"]
    if (artifact["seed"], artifact["n"], artifact["T"], key[1].shape, key[-1], key[4]) != (
            12345, n, n, (round(.99 * n), n), 3, .05):
        raise ValueError("cached key differs from reviewed setup")
    if rt._redetect_sha("/root/prompts.jsonl") != plan["prompts_file_sha256"]:
        raise ValueError("prompt cache changed")
    prompts = [json.loads(s)["prompt_tokens"] for s in Path("/root/prompts.jsonl").read_text().splitlines()][:100]
    if artifact["prompt_ids_list"][:100] != prompts or any(len(p) != 50 for p in prompts):
        raise ValueError("first 100 prompt IDs differ from source artifact")
    if tensor_sha256(part) != plan["partition_sha256"]:
        raise ValueError("original partition changed")
    if DATA.exists() or (RESULTS / "setup.json").exists():
        raise ValueError("experiment already has a setup; do not overwrite or retry")
    for spec in plan["models"].values():
        folder = Path("/cache") / spec["cache_directory"]
        hashes = {**rt._redetect_model_spec(spec), **spec["metadata_sha256"]}
        for filename, digest in hashes.items():
            path = folder / filename
            if not path.exists():
                if not plan.get("allow_checkpoint_download", True):
                    raise ValueError(f"expected checkpoint cache missing: {path}; request approval before downloading")
                hf_hub_download(spec["id"], filename, revision=spec["revision"], local_dir=folder)
            if rt._redetect_sha(path) != digest:
                raise ValueError(f"checkpoint conflict; refusing to replace {path}")
        rt._verify_redetection_checkpoint(spec)
    rt.hf_cache.commit()
    DATA.mkdir(parents=True)
    shutil.copyfile(Path("/archive") / plan["source_artifact"]["path"], DATA / "artifacts.pt")
    write_json(RESULTS / "setup.json", plan)
    rt.data_vol.commit()
    return {"files": [file_ref(DATA / "artifacts.pt", "data"), file_ref(RESULTS / "setup.json", "results")]}


def generate(plan):
    import numpy as np
    import torch
    if (DATA / "generation.pt").exists():
        raise ValueError("saved generation exists; never generate twice")
    we = load_model(plan["models"]["4B"])
    if rt._redetect_sha(DATA / "artifacts.pt") != plan["source_artifact"]["sha256"]:
        raise ValueError("generation key artifact changed")
    artifact = rt._redetect_load(DATA / "artifacts.pt")
    prompts = torch.tensor(artifact["prompt_ids_list"][:100], dtype=torch.long, device=we.device)
    # Keep the legacy RNG behavior. Record available states and actual codewords;
    # key seed 12345 alone does not reproduce legacy GF.Random draws.
    rng = {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
           "numpy": np.random.get_state()}
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    tokens, p_trace, details = we.generate_batch_and_collect(
        we.model, prompts, plan["T"], artifact["encoding_key"], artifact["partition"].to(we.device),
        watermark=True, return_trace_details=True)
    saved = {"prompts": prompts.cpu(), "tokens": tokens, "p_trace": p_trace, "details": details,
             "rng_before": rng, "model": plan["models"]["4B"], "N": 100, "T": plan["T"],
             "artifact_sha256": plan["source_artifact"]["sha256"], "execution": plan["runtime_source_sha256"],
             "seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(),
             "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    # Commit the primary generation before record conversion or any later check.
    rt._redetect_write(DATA / "generation.pt", saved)
    rt.data_vol.commit()
    return {"files": [file_ref(DATA / "generation.pt", "data")],
            **{k: saved[k] for k in ("N", "T", "seconds", "gpu", "peak_allocated_bytes")}}


def freeze(plan, payload):
    import torch
    from detectors import build_prc_generation_record
    ref = payload["generate"]["files"][0]
    batch = rt._redetect_source(ref, {"data": "/data"})
    artifact_ref = payload["prepare"]["files"][0]
    artifact = rt._redetect_source(artifact_ref, {"data": "/data"})
    if batch["model"] != plan["models"]["4B"] or tuple(batch["tokens"].shape) != (100, plan["T"]):
        raise ValueError("generation cohort/model changed")
    if batch["artifact_sha256"] != artifact_ref["sha256"] or batch["execution"] != plan["runtime_source_sha256"]:
        raise ValueError("generation key or execution provenance changed")
    expected_prompts = torch.tensor(artifact["prompt_ids_list"][:100], dtype=torch.long)
    if not torch.equal(batch["prompts"], expected_prompts):
        raise ValueError("saved generation prompt IDs changed")
    records, files = [], []
    for i in range(100):
        record = build_prc_generation_record(batch["prompts"][i], batch["tokens"][i], batch["p_trace"][i],
            artifact["partition"], plan["n"], True, encoding_key=artifact["encoding_key"],
            **{k: batch["details"][k][i] for k in ("prc_codeword_bits", "base_lm_entropy", "base_token_logprob")})
        record.update(prompt_idx=i, watermark=True, generation_model="Qwen3-4B-Base", generation_model_size="4B")
        path = DATA / "wm" / f"wm_{i:04d}.pt"
        rt._redetect_write(path, record)
        ref = file_ref(path, "data")
        files.append(ref)
        records.append({"source": "wm", "prompt_idx": i, "file": ref,
                        "tokens_sha256": hashlib.sha256(record["tokens"].to(torch.int64).contiguous().numpy().tobytes()).hexdigest()})
    rt.data_vol.commit()
    prepared = {}
    for size, model in plan["models"].items():
        case = {"id": RUN + "_detect_" + size, "construction": "fixed", "generation_model": "Qwen3-4B-Base",
                "artifact": artifact_ref, "lengths": [plan["n"]], "weights": ["map", "entropy"], "fpr": .001,
                "fpr_policy": "block_or_bonferroni", "null_policy": "not_evaluated", "batch_size": 100,
                "cache": "static", "records": records}
        execution = {"git_commit": plan["head"], "files": plan["runtime_source_sha256"],
                     "gpu": "H100" if size == "4B" else "A100-80GB", "source_hashes_authoritative": True}
        prepared[size] = rt._prepare_redetection(case, model, execution, {"data": "/data"}, "/results")
        root = Path("/results") / prepared[size]["root"]
        rt._redetect_write(root / "prepared.json", prepared[size])
        files.extend(file_ref(root / name, "results") for name in
                     ("manifest.json", "prepared.json", "artifact.pt", "batches/000000/inputs.pt"))
    return {"prepared": prepared, "files": files}


def execute(stage, payload):
    plan = payload["plan"]
    configure(plan["n"])
    check_code(plan)
    if stage == "prepare":
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    # CPU pickle loading must also follow scientific/native imports. GPU model
    # loading happens before _numpy_pickle_compat in load_model, above.
    import torch
    import scipy.special
    import galois
    import transformers
    import qwen
    import detectors
    torch.set_num_threads(4)
    if stage == "prepare":
        return prepare(plan)
    if stage == "generate":
        return generate(plan)
    if stage == "freeze":
        return freeze(plan, payload)
    frozen = payload["freeze"]["prepared"]
    if stage in ("replay_4b", "replay_0p6b"):
        size = "4B" if stage == "replay_4b" else "0.6B"
        prepared = frozen[size]
        if len(prepared["batches"]) != 1 or prepared["batches"][0]["identity"]["count"] != 100:
            raise ValueError("exactly one replay batch of 100 required")
        we = load_model(plan["models"][size])
        result = rt._recover_redetection_batch(we.model, prepared["batches"][0], "/results", validate=False)
        rt.redetect_results.commit()
        return {**result, "files": [file_ref(Path("/results") / result["root"] / "trace.pt", "results")]}
    if stage != "score":
        raise ValueError("unknown stage")
    files, counts = [], {}
    for size, prepared in frozen.items():
        # Usual cloud CPU scoring, with no LM import or GPU allocation.
        report = rt._score_redetection(prepared, "/results")
        counts[size] = report["counts"]
        files.extend(file_ref(Path("/results") / prepared["root"] / name, "results")
                     for name in ("full.json", "summary.json"))
        rt.redetect_results.commit()
    return {"counts": counts, "files": files}


@app.function(cpu=(4, 4), memory=16384, timeout=630, startup_timeout=30,
              retries=0, max_containers=1, scaledown_window=2,
              volumes={"/data": rt.data_vol, "/cache": rt.hf_cache,
                       "/archive": rt.redetect_archive, "/results": rt.redetect_results})
def paid_stage(stage, payload):
    """A bounded child turns native crashes into normal failures, without retry."""
    configure(payload["plan"]["n"])
    limit = RESOURCES[stage][2]
    folder = RESULTS / "attempts" / stage
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "request.json", payload)
    started = time.monotonic()
    try:
        with (folder / "worker.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, __file__, "--child", stage,
                                     str(folder / "request.json"), str(folder / "response.json")], stdout=log, stderr=log)
            try:
                status = proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise TimeoutError(f"{stage} exceeded {limit}s; no retry")
        if status:
            raise RuntimeError(f"{stage} failed with exit {status}; saved worker.log; no retry")
        return json.loads((folder / "response.json").read_text())
    finally:
        write_json(folder / "timing.json", {"stage": stage, "wall_seconds": time.monotonic() - started,
                                            "resources": RESOURCES[stage]})
        rt.redetect_results.commit()


def collect(stage):
    """Storage downloads and CSV formatting only; safe to repeat without compute."""
    result = json.loads((OUT / f"result_{stage}.json").read_text())
    volumes = {"data": rt.data_vol, "results": rt.redetect_results}
    for ref in result["files"]:
        target = OUT / "cache" / ref["volume"] / ref["path"]
        data = b"".join(volumes[ref["volume"]].read_file(ref["path"]))
        if len(data) != ref["bytes"] or hashlib.sha256(data).hexdigest() != ref["sha256"]:
            raise ValueError("download differs from committed output")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name in ("request.json", "response.json", "worker.log", "timing.json"):
        data = b"".join(rt.redetect_results.read_file(f"{RUN}/attempts/{stage}/{name}"))
        target = OUT / "evidence" / stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if stage == "score":
        frozen = json.loads((OUT / "result_freeze.json").read_text())["prepared"]
        for prepared in frozen.values():
            report = json.loads((OUT / "cache/results" / prepared["root"] / "full.json").read_text())
            rt._append_redetection_csv(prepared, report, rt.REDETECT_CSV)
    write_json(OUT / f"collected_{stage}.json", result["files"])


@app.local_entrypoint()
def run(stage: str, approval_reference: str, n: int = 1024):
    """Run ONLY the named stage after the user approved its workload and cost."""
    if stage not in RESOURCES or not approval_reference.strip():
        raise ValueError("name one stage and record its explicit user approval")
    configure(n)
    plan = json.loads((OUT / "setup.json").read_text())
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "redetection":
        raise ValueError("this experiment must run from redetection")
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("use the reviewed new-prc-watermark Modal profile")
    check_code(plan)
    if (plan["N"], plan["n"], plan["T"], plan["eta"], plan["t"], plan["r"], plan["fpr"], plan["null_count"]) != (
            100, n, n, .05, 3, round(.99 * n), .001, 0):
        raise ValueError("this runner only implements the reviewed N=100 watermarked-only experiment")
    if rt._redetect_sha("prompts.jsonl") != plan["prompts_file_sha256"]:
        raise ValueError("prompt cache changed")
    attempt = OUT / f"attempt_{stage}.json"
    if attempt.exists():
        raise ValueError("stage was already attempted; no automatic retry or regeneration")
    plan_sha256 = rt._redetect_sha(OUT / "setup.json")
    if stage != "prepare":
        first = json.loads((OUT / "attempt_prepare.json").read_text())
        if first["plan_sha256"] != plan_sha256:
            raise ValueError("setup changed after preparation; obtain a new review before proceeding")
    for required in PREREQUISITES[stage]:
        if not (OUT / f"collected_{required}.json").exists():
            raise ValueError(f"complete and retrieve {required} before starting {stage}")
    payload = {"plan": plan, "approval_reference": approval_reference}
    for previous in ("prepare", "generate", "freeze"):
        path = OUT / f"result_{previous}.json"
        if path.exists():
            payload[previous] = json.loads(path.read_text())
    write_json(attempt, {"stage": stage, "approval_reference": approval_reference,
                        "resources": RESOURCES[stage], "plan_sha256": plan_sha256})
    gpu, memory, limit = RESOURCES[stage]
    options = {"memory": memory, "timeout": limit + 30}
    if gpu:
        options["gpu"] = gpu
    result = paid_stage.with_options(**options).remote(stage, payload)
    write_json(OUT / f"result_{stage}.json", result)
    collect(stage)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        write_json(sys.argv[4], execute(sys.argv[2], json.loads(Path(sys.argv[3]).read_text())))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "collect":
        configure(int(sys.argv[3]) if len(sys.argv) == 4 else 1024)
        collect(sys.argv[2])
    else:
        print((OUT / "PLAN.md").read_text())
