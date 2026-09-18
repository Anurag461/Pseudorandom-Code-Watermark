"""Explicit, bounded step-3 validation; never dispatches the study sweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

from baseline_comparison.modal_app import image as generation_image, hf_cache, data_volume
from baseline_comparison.textseal_modal import image as detector_image
from baseline_comparison.self_bleu_validation import save, sha, validate_manifest

# Copy the sampler as text, without importing the notebook's top-level model load.
generation_image = generation_image.add_local_file("watermark_expt.py", "/root/watermark_expt.py", copy=False)
for name in ("prompts.jsonl", "qwen.py", "prc.py", "online_prc.py", "detectors.py", "watermark_expt.py"):
    detector_image = detector_image.add_local_file(name, f"/root/{name}", copy=True)
results = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
app = modal.App("prc-self-bleu-validation")


def checkpoint(manifest):
    spec = manifest["model"]
    root = Path("/cache") / spec["cache_directory"]
    files = {**spec["weight_files"], "config.json": spec["config_sha256"],
             "model.safetensors.index.json": spec["index_sha256"], "tokenizer.json": spec["tokenizer_sha256"]}
    for name, expected in files.items():
        if sha(root / name) != expected:
            raise ValueError(f"checkpoint bytes differ: {name}")
        if (root / ".cache/huggingface/download" / f"{name}.metadata").read_text().splitlines()[0] != spec["revision"]:
            raise ValueError(f"checkpoint revision differs: {name}")


def prc_replay_check(model, batches, artifact):
    import numpy as np
    import torch
    from qwen import completion_only_partition_trace_batch, make_kv_cache
    from detectors import _soft_tokens
    # Fresh responses from both PRC seeds and ordinary sampling; no prompt/traces.
    records = [b["responses"][0] for b in batches]
    tokens = torch.tensor([r["token_ids"] for r in records], device="cuda")
    part = artifact["partition"][1].to("cuda")
    calls = []
    def observe(module, args, kwargs):
        if len(args) != 1 or set(kwargs) != {"cache"} or args[0].shape != (len(records), 1):
            raise ValueError("unexpected PRC replay input")
        if not torch.equal(args[0], tokens[:, len(calls):len(calls)+1]):
            raise ValueError("PRC replay input is not the raw completion")
        calls.append(len(calls))
    handle = model.register_forward_pre_hook(observe, with_kwargs=True)
    try:
        trace = completion_only_partition_trace_batch(model, tokens, part, "static")
    finally:
        handle.remove()
    cache = make_kv_cache("static", max_length=tokens.shape[1]-1)
    with torch.no_grad():
        reference = torch.stack([(model(tokens[:, i:i+1], cache=cache)[:, -1].softmax(-1)*part).sum(-1).cpu()
                                 for i in range(tokens.shape[1]-1)], dim=1).float()
    short = completion_only_partition_trace_batch(model, tokens.flip(0)[:, :65], part, "static")
    first_zero = all(_soft_tokens(part[tokens[i]].to(torch.int8).cpu().numpy(), trace[i].numpy(), "map")[0] == 0
                     for i in range(len(records)))
    checks = {"raw_completion_inputs": len(calls) == tokens.shape[1]-1,
              "independent_reference_exact": torch.equal(trace, reference),
              "prefix_and_order_exact": torch.equal(short.flip(0), trace[:, :64]),
              "first_coordinate_abstains": first_zero}
    return {"passed": all(checks.values()), **checks, "response_ids": [r["response_id"] for r in records],
            "probabilities_2_to_T": trace.tolist(), "forward_count": len(calls)}


@app.function(image=generation_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=3000, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/data": data_volume, "/results": results})
def generation(manifest):
    import faulthandler
    import importlib.metadata
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes, load_qwen3_8b, _numpy_pickle_compat
    from baseline_comparison.config import PINNED_DEPENDENCIES
    from baseline_comparison.self_bleu_config import StudySetting, pilot_settings
    from baseline_comparison.self_bleu_generation import generate_response_batch
    from baseline_comparison.self_bleu_validation import load_online_sampler, compare_replicates
    started = time.monotonic()
    validate_manifest(manifest, "/root")
    root = Path("/results/self_bleu_validation") / manifest["id"]
    results.reload()
    if (root / "generation_report.json").exists():
        raise FileExistsError("validation already ran; retrieve its immutable report")
    save(root / "manifest.json", manifest)
    report = {"manifest_id": manifest["id"], "passed": False, "settings": [], "parameter_smokes": [], "files": {}}
    try:
        faulthandler.enable()
        # Match the historical runner: import the evaluation stack before the
        # legacy NumPy pickle aliases or any CUDA initialization.
        preload_official_runtimes()
        versions = {r.split("==")[0]: importlib.metadata.version(r.split("==")[0]) for r in PINNED_DEPENDENCIES}
        for r in PINNED_DEPENDENCIES:
            name, expected = r.split("==")
            if versions[name].split("+")[0] != expected:
                raise ValueError(f"dependency differs: {name}")
        if "H100" not in torch.cuda.get_device_name():
            raise ValueError("expected H100")
        execution = {"versions": versions, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                     "model_revision": manifest["model"]["revision"], "dtype": "bfloat16",
                     "tf32": torch.backends.cuda.matmul.allow_tf32,
                     "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                     "modal_image_id": os.environ.get("MODAL_IMAGE_ID")}
        report["execution"] = execution
        print("[step3] verifying offline checkpoint", flush=True)
        checkpoint(manifest)
        data_volume.reload()
        artifact_path = Path("/data") / manifest["artifact"]["path"]
        if sha(artifact_path) != manifest["artifact"]["sha256"]:
            raise ValueError("PRC generation artifact changed")
        _numpy_pickle_compat()
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        model = load_qwen3_8b()
        torch.cuda.reset_peak_memory_stats()
        sampler = load_online_sampler("/root/watermark_expt.py", device="cuda")
        prompts = [json.loads(s)["prompt_tokens"] for s in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        for_replay = []
        def generate(setting, seed, response, length=1024):
            kwargs = {"prc_artifact": artifact, "online_sampler": sampler} if setting.method == "online_prc" else {}
            with torch.no_grad():
                return generate_response_batch(model, prompts, list(range(50)), setting=setting,
                                               sampling_seed=seed, response_index=response, execution=execution,
                                               max_new_tokens=length, **kwargs)
        def persist(batch):
            relative = f"batches/{batch['manifest']['batch_id']}.json"
            save(root / relative, batch)
            report["files"][relative] = sha(root / relative)
            results.commit()
        for setting in (*pilot_settings("A"), StudySetting("textseal", alpha=0)):
            print(f"[step3] {setting.identity()}", flush=True)
            first = generate(setting, 12345, 0)
            persist(first)
            second = generate(setting, 67890, 1)
            persist(second)
            replay = generate(setting, 12345, 0)
            check = compare_replicates(first, second, replay,
                                       deterministic=setting.method == "gumbel_max" or (setting.method == "textseal" and setting.alpha == 0))
            expected = manifest["source_audit"]["expected_completion_sha256"].get(setting.method)
            compatible_setting = setting.method != "textseal" or setting.alpha == .1
            matches = [r["completion_sha256"] == old for r, old in zip(first["responses"], expected)] if expected and compatible_setting else None
            row = {"setting": setting.identity(), **check, "historical_first_response_matches": matches,
                   "batches": [first["manifest"]["batch_id"], second["manifest"]["batch_id"]]}
            report["settings"].append(row)
            print(json.dumps(row), flush=True)
            if setting.method == "online_prc":
                for_replay.extend((first, second))
            elif setting.method == "null":
                for_replay.append(second)
            if not check["passed"]:
                raise ValueError(f"replicate check failed: {setting.identity()}")
        for setting in (*pilot_settings("B")[1:], *pilot_settings("depth30")):
            batch = generate(setting, 12345, 0, 128)
            persist(batch)
            ref = batch["telemetry"].get("synthid_official_smoke_reference", {})
            passed = setting.method != "synthid_text" or (ref["indices_equal"] and ref["max_abs_score_difference"] == 0)
            report["parameter_smokes"].append({"setting": setting.identity(), "batch": batch["manifest"]["batch_id"], "passed": passed, "official_reference": ref})
            if not passed:
                raise ValueError("SynthID official score update differs")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        replay_check = prc_replay_check(model, for_replay, artifact)
        save(root / "prc_replay.json", replay_check)
        report["files"]["prc_replay.json"] = sha(root / "prc_replay.json")
        report["prc_replay"] = {k: v for k, v in replay_check.items() if k != "probabilities_2_to_T"}
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["memory_headroom_passed"] = report["peak_allocated_bytes"] < .85 * torch.cuda.get_device_properties(0).total_memory
        report["passed"] = replay_check["passed"] and report["memory_headroom_passed"]
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        report["measured_resource_usd"] = report["seconds"] * manifest["resource_usd_per_second"]
        save(root / "generation_report.json", report)
        results.commit()
    return report


@app.function(image=generation_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/data": data_volume, "/results": results})
def prc_replay_repair(manifest):
    """Resume only the failed diagnostic; never regenerate successful pairs."""
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes, load_qwen3_8b, _numpy_pickle_compat
    started = time.monotonic()
    validate_manifest(manifest, "/root")
    previous = manifest["resume_from_manifest"]
    results.reload()
    source = Path("/results/self_bleu_validation") / previous["id"]
    root = Path("/results/self_bleu_validation") / manifest["id"]
    if json.loads((source / "manifest.json").read_text()) != previous:
        raise ValueError("source request changed")
    report = json.loads((source / "generation_report.json").read_text())
    if (report.get("error") != "TypeError: Got unsupported ScalarType BFloat16" or
            len(report["settings"]) != 6 or len(report["parameter_smokes"]) != 4 or
            not all(row["passed"] for row in report["settings"] + report["parameter_smokes"]) or
            len(report["files"]) != 16):
        raise ValueError("repair requires complete successful generation checks and the known conversion failure")
    if (root / "generation_report.json").exists():
        raise FileExistsError("repair already ran; retrieve its report")
    save(root / "manifest.json", manifest)
    batches = {}
    for relative, expected in report["files"].items():
        if not relative.startswith("batches/") or sha(source / relative) != expected:
            raise ValueError("source batch changed")
        batch = json.loads((source / relative).read_text())
        save(root / relative, batch)
        if sha(root / relative) != expected:
            raise ValueError("batch copy differs")
        batches[batch["manifest"]["batch_id"]] = batch
    results.commit()
    preload_official_runtimes()
    checkpoint(manifest)
    data_volume.reload()
    artifact_path = Path("/data") / manifest["artifact"]["path"]
    if sha(artifact_path) != manifest["artifact"]["sha256"]:
        raise ValueError("PRC artifact changed")
    _numpy_pickle_compat()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    model = load_qwen3_8b()
    torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prc = next(row for row in report["settings"] if row["setting"]["method"] == "online_prc")
    null = next(row for row in report["settings"] if row["setting"]["method"] == "null")
    check = prc_replay_check(model, [batches[i] for i in prc["batches"] + null["batches"][1:]], artifact)
    save(root / "prc_replay.json", check)
    report["files"]["prc_replay.json"] = sha(root / "prc_replay.json")
    report["source_manifest_id"] = previous["id"]
    report["source_error"] = report.pop("error")
    report["manifest_id"] = manifest["id"]
    report["prc_replay"] = {k: v for k, v in check.items() if k != "probabilities_2_to_T"}
    report["repair_seconds"] = time.monotonic()-started
    report["repair_measured_resource_usd"] = report["repair_seconds"] * manifest["resource_usd_per_second"]
    report["replay_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["replay_memory_headroom_passed"] = report["replay_peak_allocated_bytes"] < .85 * torch.cuda.get_device_properties(0).total_memory
    report["generation_peak_note"] = "Generation completed; its peak memory was not recorded before the diagnostic conversion failed. The measured peak covers replay only."
    report["passed"] = check["passed"] and report["replay_memory_headroom_passed"]
    save(root / "generation_report.json", report)
    results.commit()
    return report


@app.function(image=detector_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/results": results})
def textseal_replay(manifest):
    from baseline_comparison.textseal_modal import DEPENDENCIES, runtime_identity, load_model
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    from baseline_comparison.textseal_redetect import run_record
    started = time.monotonic()
    validate_manifest(manifest, "/root")
    results.reload()
    root = Path("/results/self_bleu_validation") / manifest["id"]
    source = json.loads((root / "generation_report.json").read_text())
    if not source["passed"] or (root / "textseal_report.json").exists():
        raise ValueError("requires passing generation and no previous TextSeal report")
    report = {"manifest_id": manifest["id"], "passed": False, "records": [], "files": {}}
    try:
        request = {"model": manifest["model"], "runtime": {"dependencies": DEPENDENCIES}}
        report["execution"] = runtime_identity(request)
        model = load_model(request, "/cache")
        rows = [r for r in source["settings"] if r["setting"]["method"] in ("textseal", "null")]
        rows += [{"setting": r["setting"], "batches": [r["batch"]]} for r in source["parameter_smokes"] if r["setting"]["method"] == "textseal"]
        for row in rows:
            alpha = row["setting"].get("alpha", .1)
            detector = TextSealCompletionDetector(model, alpha=alpha)
            for batch_id in row["batches"]:
                relative = f"batches/{batch_id}.json"
                if sha(root / relative) != source["files"][relative]:
                    raise ValueError("saved generation batch changed")
                record = json.loads((root / relative).read_text())["responses"][0]
                clean = {"method": row["setting"]["method"], "prompt_index": record["prompt_index"], "token_ids": record["token_ids"]}
                lengths = [n for n in manifest["prefix_lengths"] if n <= len(clean["token_ids"])]
                data = run_record(detector, clean, lengths, True, "direct")
                relative = f"textseal_replay/{record['response_id']}.json"
                save(root / relative, {"alpha": alpha, "response_id": record["response_id"], "data": data})
                report["files"][relative] = sha(root / relative)
                report["records"].append({"response_id": record["response_id"], "alpha": alpha, "method": clean["method"],
                                           "validation": data["validation"], "forward_lengths": data["forward_lengths"],
                                           "raw_completion_inputs": data["actual_model_inputs_verified"]})
                print(f"[step3] TextSeal raw replay alpha={alpha} {record['response_id']}: {data['validation']}", flush=True)
                if not data["validation"]["passed"]:
                    raise ValueError("TextSeal direct-prefix parity failed")
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        report["measured_resource_usd"] = report["seconds"] * manifest["resource_usd_per_second"]
        save(root / "textseal_report.json", report)
        results.commit()
    return report


@app.local_entrypoint()
def run(manifest: str, stage: str):
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("expected MODAL_PROFILE=new-prc-watermark")
    path = Path(manifest)
    frozen = json.loads(path.read_text())
    validate_manifest(frozen, Path(__file__).resolve().parents[1])
    if stage == "generation":
        if "resume_from_manifest" in frozen:
            raise ValueError("resume requests cannot generate responses")
        report = generation.remote(frozen)
    elif stage == "prc-replay-repair":
        report = prc_replay_repair.remote(frozen)
    elif stage == "textseal":
        report = textseal_replay.remote(frozen)
    else:
        raise ValueError("stage must be generation, prc-replay-repair or textseal")
    filename = "generation" if stage == "prc-replay-repair" else stage
    save(path.parent / f"{filename}_report.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("validation failed; do not launch the pilot")
