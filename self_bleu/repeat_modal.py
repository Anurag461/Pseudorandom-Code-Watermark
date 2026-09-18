"""Explicit staged dispatch for repeat-policy ablation; no automatic sweep."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import modal

from .validation_modal import generation_image, detector_image, hf_cache, results, checkpoint
from .validation import save, sha
from .config import digest
from .repeat import SETUP, validate, upstream_hashes

app = modal.App("prc-self-bleu-repeat-ablation")


def begin(manifest, stage):
    validate(manifest, "/root")
    results.reload()
    root = Path("/results/self_bleu_repeat")/manifest["id"]
    previous = {"other_generators": "synthid", "textseal_replay": "other_generators"}.get(stage)
    if previous:
        prior = json.loads((root/previous/"report.json").read_text())
        if not prior["passed"] or prior["manifest_id"] != manifest["id"]:
            raise ValueError("previous stage has not passed")
        for name, expected in prior["files"].items():
            if sha(root/previous/name) != expected:
                raise ValueError("previous stage artifact changed")
    target = root/stage
    if (target/"started.json").exists():
        raise FileExistsError("stage already attempted; retrieve artifacts and account for cost before preparing a retry")
    save(root/"manifest.json", manifest)
    save(target/"started.json", {"manifest_id": manifest["id"], "stage": stage})
    results.commit()
    return target


def finish(root, report, started, manifest):
    report["seconds"] = time.monotonic()-started
    report["measured_resource_usd"] = report["seconds"]*manifest["cost"]["resource_usd_per_second"]
    save(root/"report.json", report)
    results.commit()
    return report


def generate(manifest, stage):
    import importlib.metadata
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes, load_qwen3_8b
    from baseline_comparison.config import PINNED_DEPENDENCIES
    from .repeat import (RepeatSetting, arm_setting, generate_repeat_batch,
                                   check_synthid_policy, check_sampler_policy)
    started = time.monotonic()
    root = begin(manifest, stage)
    preload_official_runtimes()
    if upstream_hashes() != manifest["upstream_sha256"]:
        raise ValueError("pinned upstream sources differ")
    versions = {entry.split("==")[0]: importlib.metadata.version(entry.split("==")[0]) for entry in PINNED_DEPENDENCIES}
    execution = {"versions": versions, "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                 "model_revision": manifest["model"]["revision"], "dtype": "bfloat16",
                 "tf32": torch.backends.cuda.matmul.allow_tf32,
                 "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
    if execution != manifest["generation_runtime"] or "H100" not in execution["gpu"]:
        raise ValueError("generation runtime differs from reusable Stage A pairs")
    checkpoint(manifest)
    model = load_qwen3_8b()
    prompts = [json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
    report = {"manifest_id": manifest["id"], "stage": stage, "passed": False,
              "runtime": execution, "files": {}, "checks": {}, "batches": []}
    arms = ("synthid_off",) if stage == "synthid" else ("textseal_on", "gumbel_on")
    for arm in arms:
        setting = arm_setting(arm)
        report["checks"][arm] = (check_synthid_policy("cuda") if setting.method == "synthid_text"
                                  else check_sampler_policy(setting.method, "cuda"))
        for response, seed in enumerate(manifest["seeds"]):
            # Native-policy adapter must reproduce the saved original prefix
            # before its opposite-policy full response may be generated.
            native_setting = RepeatSetting(setting.method, repeat_fallback=not setting.repeat_fallback)
            control = generate_repeat_batch(model, prompts, manifest["prompt_indices"], setting=native_setting,
                       sampling_seed=seed, response_index=response, execution=execution, max_new_tokens=64)
            expected = manifest["references"][f"{setting.method}/{response}"]["prefix_sha256"]
            if [digest(r["token_ids"]) for r in control["responses"]] != expected:
                raise ValueError("native-policy prefix does not reproduce Stage A; stop rather than mixing runtimes")
            control_path = f"controls/{arm}_r{response}.json"
            save(root/control_path, control)
            report["files"][control_path] = sha(root/control_path)
            results.commit()
            batch = generate_repeat_batch(model, prompts, manifest["prompt_indices"], setting=setting,
                       sampling_seed=seed, response_index=response, execution=execution)
            if setting.method == "synthid_text":
                check = batch["telemetry"]["same_policy_single_row_reference"]
                if not check["indices_equal"] or check["max_abs_score_difference"] != 0:
                    raise ValueError("ablation batch/single-row score parity failed")
            name = f"batches/{arm}_r{response}.json"
            save(root/name, batch)
            report["files"][name] = sha(root/name)
            report["batches"].append({"arm": arm, "response_index": response, "path": name,
                                      "batch_id": batch["manifest"]["batch_id"]})
            results.commit()
            print(f"[repeat] saved {arm} seed {seed}: 50 responses", flush=True)
    report["passed"] = True
    return finish(root, report, started, manifest)


@app.function(image=generation_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/results": results})
def synthid(manifest):
    return generate(manifest, "synthid")


@app.function(image=generation_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=900, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/results": results})
def other_generators(manifest):
    return generate(manifest, "other_generators")


@app.function(image=detector_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=300, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/results": results})
def textseal_replay(manifest):
    from baseline_comparison.textseal_modal import DEPENDENCIES, runtime_identity, load_model
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    from baseline_comparison.textseal_redetect import run_record
    started = time.monotonic()
    root = begin(manifest, "textseal_replay")
    request = {"model": manifest["model"], "runtime": {"dependencies": DEPENDENCIES}}
    runtime = runtime_identity(request)
    if runtime != manifest["textseal_runtime"]:
        raise ValueError("TextSeal replay runtime changed")
    model = load_model(request, "/cache")
    detector = TextSealCompletionDetector(model, alpha=.1)
    report = {"manifest_id": manifest["id"], "stage": "textseal_replay", "passed": False,
              "runtime": runtime, "files": {}, "rows": []}
    for response in (0, 1):
        batch = json.loads((root.parent/"other_generators/batches"/f"textseal_on_r{response}.json").read_text())
        for row in batch["responses"]:
            if digest(row["token_ids"]) != row["completion_sha256"]:
                raise ValueError("TextSeal completion changed")
            # No generation diagnostics or prompt IDs enter the detector.
            clean = {"method": "textseal", "prompt_index": row["prompt_index"], "token_ids": row["token_ids"]}
            data = run_record(detector, clean, manifest["prefix_lengths"], len(report["rows"]) == 0, "direct")
            if not data["actual_model_inputs_verified"] or (data["validation"]["performed"] and not data["validation"]["passed"]):
                raise ValueError("TextSeal raw-input or upstream parity check failed")
            name = f"records/{row['response_id']}.json"
            save(root/name, data)
            report["files"][name] = sha(root/name)
            report["rows"].append({"response_id": row["response_id"], "completion_sha256": row["completion_sha256"],
                                   "results": data["results"]})
        results.commit()
    report["passed"] = len(report["rows"]) == 100
    return finish(root, report, started, manifest)


@app.local_entrypoint()
def run(stage: str, setup: str = str(SETUP)):
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("expected MODAL_PROFILE=new-prc-watermark")
    if stage not in {"synthid", "other_generators", "textseal_replay"}:
        raise ValueError("select one explicit stage")
    path = Path(setup)
    manifest = json.loads((path/"manifest.json").read_text())
    validate(manifest)
    report = {"synthid": synthid, "other_generators": other_generators,
              "textseal_replay": textseal_replay}[stage].remote(manifest)
    save(path/f"{stage}_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("files", "rows")}, indent=2))
