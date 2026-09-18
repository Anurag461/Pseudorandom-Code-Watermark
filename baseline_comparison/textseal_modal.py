"""Explicitly launched TextSeal pilot/full replay; importing launches nothing."""
from __future__ import annotations

import json
import os
import platform
from pathlib import Path
import time

import modal

from baseline_comparison.config import TEXTSEAL_COMMIT, TEXTSEAL_REPOSITORY
from baseline_comparison.textseal_redetect import (
    REPO, canonical, digest, file_sha, load_request, record_identity,
    require_pilot, run_record, validate_request, write_json,
)

DEPENDENCIES = [line for line in (Path(__file__).with_name("requirements-textseal.txt")).read_text().splitlines()
                if line and not line.startswith("#")]
image = (
    modal.Image.debian_slim(python_version="3.11").apt_install("git")
    .pip_install(*DEPENDENCIES)
    .pip_install(f"git+{TEXTSEAL_REPOSITORY}.git@{TEXTSEAL_COMMIT}", extra_options="--no-deps")
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
          "OMP_NUM_THREADS": "4", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir("baseline_comparison", "/root/baseline_comparison", copy=True)
)
app = modal.App("textseal-completion-redetect", image=image)
model_volume = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
results_volume = modal.Volume.from_name("prc-completion-only", create_if_missing=False)


def runtime_identity(manifest):
    import importlib.metadata
    import torch
    versions = {}
    for requirement in manifest["runtime"]["dependencies"]:
        name, expected = requirement.split("==")
        actual = importlib.metadata.version(name)
        if actual.split("+")[0] != expected:
            raise ValueError(f"runtime dependency differs: {name}={actual}, expected {expected}")
        versions[name] = actual
    if not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name():
        raise ValueError("setup requires an H100 CUDA worker")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return {"versions": versions, "python": platform.python_version(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
            "dtype": "bfloat16", "attention": "eager", "use_cache": False,
            "tf32": False, "bf16_reduced_precision_reduction": False}


def load_model(manifest, cache_root):
    """Verify cached checkpoint bytes and revision before loading HF weights."""
    import torch
    from transformers import AutoModelForCausalLM
    spec = manifest["model"]
    root = Path(cache_root) / spec["cache_directory"]
    files = {**spec["weight_files"], "config.json": spec["config_sha256"],
             "model.safetensors.index.json": spec["index_sha256"], "tokenizer.json": spec["tokenizer_sha256"]}
    for name, expected in files.items():
        if file_sha(root / name) != expected:
            raise ValueError(f"cached checkpoint differs: {name}")
        metadata = root / ".cache/huggingface/download" / f"{name}.metadata"
        if metadata.read_text().splitlines()[0] != spec["revision"]:
            raise ValueError(f"checkpoint revision differs: {name}")
    index = json.loads((root / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(spec["weight_files"]):
        raise ValueError("checkpoint shard index differs")
    model = AutoModelForCausalLM.from_pretrained(
        str(root), torch_dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).eval().to("cuda")
    model.config.use_cache = False
    if model.config.model_type != "qwen3" or model.config.vocab_size != 151936:
        raise ValueError("unexpected model configuration")
    torch.cuda.synchronize()
    return model


def cached_record(path, identity, detector, record, lengths, prefix_strategy=None):
    """Reject stale or corrupt caches; rescore only verified completion entropy."""
    payload = json.loads(Path(path).read_text())
    data = payload["data"]
    if payload["identity"] != identity or payload["data_sha256"] != digest(data):
        raise ValueError("cached record identity or checksum differs")
    ids = record["token_ids"]
    strategy = data.get("prefix_strategy", "reuse")
    if prefix_strategy is not None and strategy != prefix_strategy:
        raise ValueError("cached prefix strategy differs")
    import math
    if (data["input"] != record_identity(record) or data["completion_sha256"] != digest(ids)
            or data["completion_length"] != len(ids)
            or data.get("actual_model_inputs_verified") is not True
            or set(data["results"]) != {str(n) for n in lengths}):
        raise ValueError("cached completion entropy contract differs")
    if strategy == "direct":
        entropy_map = data["entropies_by_prefix"]
        if set(entropy_map) != {str(n) for n in lengths}:
            raise ValueError("cached direct entropy prefix coverage differs")
    elif strategy == "reuse":
        entropy = data["entropies_2_to_T"]
        if len(entropy) != len(ids)-1:
            raise ValueError("cached completion entropy contract differs")
        entropy_map = {str(n): entropy[:n-1] for n in lengths}
    else:
        raise ValueError("unknown cached prefix strategy")
    for n in lengths:
        h = entropy_map[str(n)]
        if len(h) != n-1 or any(not math.isfinite(v) or v < 0 for v in h):
            raise ValueError("cached completion entropy contract differs")
        if detector._result(ids[:n], h, .001) != data["results"][str(n)]:
            raise ValueError("cached upstream result differs from verified entropy")
    return data


@app.function(gpu="H100", cpu=(4, 4), memory=(65536, 65536), timeout=3600,
              max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": model_volume, "/results": results_volume})
def replay(manifest: dict, records: list[dict], stage: str, approved_sha: str, full_budget_usd: float = 0):
    import torch
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    started = time.monotonic()
    validate_request(manifest, records, stage, approved_sha)
    runtime = runtime_identity(manifest)
    root = Path("/results/textseal_completion_redetect") / approved_sha
    results_volume.reload()
    pilot_path = root / "pilot.json"
    if stage == "full":
        pilot = json.loads(pilot_path.read_text()) if pilot_path.exists() else None
        require_pilot(pilot, approved_sha, runtime, manifest["pilot_ids"])
        if not 0 < full_budget_usd <= 25 or pilot["estimated_full_usd"] > full_budget_usd:
            raise ValueError("full replay requires a sufficient explicit budget (maximum $25)")
        for record_id, expected in pilot["record_sha256"].items():
            if file_sha(root / "records" / f"{record_id}.json") != expected:
                raise ValueError("validated pilot record changed")
    write_json(root / "manifest.json", manifest)
    model_started = time.monotonic()
    model = load_model(manifest, "/cache")
    detector = TextSealCompletionDetector(model)
    load_seconds = time.monotonic()-model_started
    torch.cuda.reset_peak_memory_stats()
    hashes, rows, production_times = {}, [], []
    cached = forwards = 0
    for record in records:
        input_identity = record_identity(record)
        record_id = input_identity["record_id"]
        path = root / "records" / f"{record_id}.json"
        identity = {"manifest_sha256": approved_sha, "runtime": runtime, "input": input_identity}
        if path.exists():
            data = cached_record(path, identity, detector, record, manifest["prefix_lengths"], manifest["prefix_strategy"])
            if stage == "pilot" and data["validation"]["performed"] is not True:
                raise ValueError("pilot cache lacks direct-prefix validation")
            cached += 1
        else:
            data = run_record(detector, record, manifest["prefix_lengths"], stage == "pilot", manifest["prefix_strategy"])
            write_json(path, {"identity": identity, "data_sha256": digest(data), "data": data})
            forwards += len(data["forward_lengths"])
        hashes[record_id] = file_sha(path)
        production_times.append(data["production_seconds"])
        rows.append({"record_id": record_id, "method": record["method"], "results": data["results"],
                     "validation": data["validation"]})
        if len(rows) % 25 == 0:
            results_volume.commit()
            print(f"Completed {len(rows)}/{len(records)} TextSeal replay records", flush=True)
        if stage == "pilot" and data["validation"]["passed"] is not True:
            break  # Save the failed check and stop; never switch protocols silently.
    passed = len(rows) == len(records) and (stage != "pilot" or all(row["validation"]["passed"] for row in rows))
    rate = manifest["cost"]["resource_usd_per_second"]
    estimate = (load_seconds + max(production_times, default=0)*990 + 60) * 1.25 * rate
    report = {
        "stage": stage, "passed": passed, "manifest_sha256": approved_sha, "runtime": runtime,
        "prefix_strategy": manifest["prefix_strategy"],
        "requested_records": len(records), "completed_records": len(rows), "cached_records": cached,
        "new_model_forwards": forwards, "record_sha256": hashes, "rows": rows,
        "load_seconds": load_seconds, "total_seconds": time.monotonic()-started,
        "max_production_seconds_per_record": max(production_times, default=0),
        "estimated_full_usd": estimate, "measured_resource_usd": (time.monotonic()-started)*rate,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "notes": "Full cost estimate uses the slowest pilot record, model load, 60 seconds persistence allowance, and 25% margin. Startup/image/storage overhead is not metered here.",
    }
    write_json(root / f"{stage}.json", report)
    results_volume.commit()
    return report


@app.local_entrypoint()
def run(manifest: str, stage: str, approved_manifest_sha256: str, full_budget_usd: float = 0):
    """Manual stage dispatch only; a pilot never launches the full run."""
    frozen, records = load_request(manifest, stage, approved_manifest_sha256)
    if os.environ.get("MODAL_PROFILE") != frozen["execution"]["profile"]:
        raise ValueError("set MODAL_PROFILE to the profile in the reviewed setup")
    if stage == "full" and not 0 < full_budget_usd <= 25:
        raise ValueError("full stage requires a separately approved --full-budget-usd")
    timeout = frozen["execution"][f"{stage}_timeout_seconds"]
    if stage == "full":
        # Reserve $0.25 for overhead outside the timed function. Retries are off.
        timeout = min(timeout, int((full_budget_usd-.25) / frozen["cost"]["resource_usd_per_second"]))
        if timeout < 60:
            raise ValueError("approved full-run budget leaves less than 60 seconds of runtime")
    result = replay.with_options(timeout=timeout).remote(
        frozen, records, stage, approved_manifest_sha256, full_budget_usd,
    )
    destination = Path(manifest).parent / f"{stage}_report.json"
    write_json(destination, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "record_sha256")}, indent=2))
    if not result["passed"]:
        raise RuntimeError(f"{stage} did not pass; saved report: {destination}")
