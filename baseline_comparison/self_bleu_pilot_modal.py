"""Two bounded replay stages for saved Stage A responses; no generation path."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import modal

from baseline_comparison.self_bleu_validation_modal import generation_image, detector_image, hf_cache, data_volume, results, checkpoint
from baseline_comparison.self_bleu_validation import save, sha
from baseline_comparison.self_bleu_config import digest
from baseline_comparison.self_bleu_pilot import validate_request, SETUP

app = modal.App("prc-self-bleu-stage-a")


def finish(root, report, started, manifest):
    report["seconds"] = time.monotonic()-started
    report["measured_resource_usd"] = report["seconds"] * manifest["cost"]["resource_usd_per_second"]
    save(root / "report.json", report)
    results.commit()
    return report


@app.function(image=generation_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/data": data_volume, "/results": results})
def replay_prc(manifest, records):
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes, load_qwen3_8b, _numpy_pickle_compat
    from qwen import completion_only_partition_trace_batch
    from detectors import (prepare_online_map_prefix_context, prepare_online_map_prefix_trace,
                           score_prepared_online_map_prefix, detect_online_hoeffding, _soft_tokens)
    started = time.monotonic()
    validate_request(manifest, records, "prc", "/root")
    results.reload()
    root = Path("/results/self_bleu_pilot") / manifest["id"] / "prc"
    if (root / "report.json").exists():
        raise FileExistsError("pilot stage exists; retrieve instead of redispatching")
    preload_official_runtimes()
    checkpoint(manifest)
    data_volume.reload()
    path = Path("/data") / manifest["artifact"]["path"]
    if sha(path) != manifest["artifact"]["sha256"]:
        raise ValueError("PRC key artifact changed")
    _numpy_pickle_compat()
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    model = load_qwen3_8b()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    part = artifact["partition"][1].to("cuda")
    context = prepare_online_map_prefix_context(artifact["online_key"], 1024)
    report = {"manifest_id": manifest["id"], "stage": "prc", "passed": False, "files": {}, "rows": [],
              "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                          "dtype": "bfloat16", "tf32": False, "batch_size": 50, "token_step": True,
                          "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}}
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(records), 50):
        group = records[start:start+50]
        tokens = torch.tensor([r["token_ids"] for r in group], device="cuda", dtype=torch.long)
        calls = []
        def observe(module, args, kwargs):
            pos = len(calls)
            if (len(args) != 1 or set(kwargs) != {"cache"} or
                    (pos == 0 and kwargs["cache"].get_seq_len() != 0) or
                    not torch.equal(args[0], tokens[:, pos:pos+1])):
                raise ValueError("PRC model input includes non-completion context or a reused cache")
            calls.append(pos)
        handle = model.register_forward_pre_hook(observe, with_kwargs=True)
        try:
            trace = completion_only_partition_trace_batch(model, tokens, part, "static")
        finally:
            handle.remove()
        if len(calls) != 1023 or trace.shape != (len(group), 1023):
            raise ValueError("PRC evidence alignment differs")
        if start == 0:
            short = completion_only_partition_trace_batch(model, tokens.flip(0)[:, :65], part, "static")
            report["batch50_prefix_order_exact"] = torch.equal(short.flip(0), trace[:, :64])
            if not report["batch50_prefix_order_exact"]:
                raise ValueError("PRC prefix/order check failed")
        scored = []
        for i, row in enumerate(group):
            ids, p = row["token_ids"], trace[i].numpy()
            bits = artifact["partition"][1, ids].to(torch.int8).numpy()
            if _soft_tokens(bits, p, "map")[0] != 0:
                raise ValueError("PRC first coordinate did not abstain")
            prepared = prepare_online_map_prefix_trace(artifact["online_key"], ids, p, artifact["partition"], 1024,
                                                       prepared_context=context, completion_only=True)
            scores = {}
            for length in manifest["prefix_lengths"]:
                info = score_prepared_online_map_prefix(prepared, length, fpr=.001)
                v, s = info["V"], info["statistic"]
                logp = -(s*s)/(2*v) if v > 1e-15 and s > 0 else 0.
                info.update(log_p_upper_bound=logp, p_value=math.exp(logp),
                            calibration_type="Hoeffding p-value upper bound")
                if not math.isfinite(info["threshold"]):
                    info["threshold"] = None
                scores[str(length)] = info
            if start == 0 and i == 0:
                for length in manifest["primary_lengths"]:
                    decision, ref = detect_online_hoeffding(artifact["online_key"], torch.tensor(ids[:length]), p[:length-1],
                                                          artifact["partition"], fpr=.001, return_info=True)
                    observed = scores[str(length)]
                    if decision != observed["decision"] or any(ref[k] != observed[k] for k in ("statistic", "V", "threshold")):
                        raise ValueError("optimized PRC score differs from direct API")
                report["direct_score_parity"] = True
            scored.append({k: row[k] for k in row if k != "token_ids"} | {"results": scores})
        relative = f"batch_{start//50:02d}.json"
        save(root / relative, {"manifest_id": manifest["id"], "inputs_sha256": digest(group), "rows": scored,
                               "probabilities_2_to_T": trace.tolist(), "raw_inputs_verified": True, "forward_count": len(calls)})
        report["files"][relative] = sha(root / relative)
        report["rows"].extend(scored)
        results.commit()
        print(f"[pilot] PRC completion-only scores {start+len(group)}/{len(records)}", flush=True)
    report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    report["passed"] = len(report["rows"]) == len(records)
    return finish(root, report, started, manifest)


@app.function(image=detector_image, gpu="H100", cpu=(4, 4), memory=(65536, 65536),
              timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": hf_cache, "/results": results})
def replay_textseal(manifest, records):
    from baseline_comparison.textseal_modal import DEPENDENCIES, runtime_identity, load_model
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    from baseline_comparison.textseal_redetect import run_record
    started = time.monotonic()
    validate_request(manifest, records, "textseal", "/root")
    results.reload()
    root = Path("/results/self_bleu_pilot") / manifest["id"] / "textseal"
    if (root / "report.json").exists():
        raise FileExistsError("pilot stage exists; retrieve instead of redispatching")
    request = {"model": manifest["model"], "runtime": {"dependencies": DEPENDENCIES}}
    runtime = runtime_identity(request)
    if runtime != manifest["textseal_runtime"]:
        raise ValueError("TextSeal execution differs from reused scores")
    model = load_model(request, "/cache")
    detector = TextSealCompletionDetector(model, alpha=.1)
    report = {"manifest_id": manifest["id"], "stage": "textseal", "passed": False, "runtime": runtime, "files": {}, "rows": []}
    for i, row in enumerate(records):
        clean = {k: row[k] for k in ("method", "prompt_index", "token_ids")}
        data = run_record(detector, clean, manifest["prefix_lengths"], i == 0, "direct")
        if i == 0 and not data["validation"]["passed"]:
            raise ValueError("TextSeal upstream parity failed")
        relative = f"records/{row['response_id']}.json"
        save(root / relative, {"manifest_id": manifest["id"], "response_id": row["response_id"], "data": data})
        report["files"][relative] = sha(root / relative)
        report["rows"].append({k: row[k] for k in row if k != "token_ids"} | {"results": data["results"]})
        if (i+1) % 25 == 0:
            results.commit()
            print(f"[pilot] TextSeal direct-prefix scores {i+1}/{len(records)}", flush=True)
    report["passed"] = len(report["rows"]) == len(records)
    return finish(root, report, started, manifest)


@app.local_entrypoint()
def run(stage: str, setup: str = str(SETUP)):
    if os.environ.get("MODAL_PROFILE") != "new-prc-watermark":
        raise ValueError("expected MODAL_PROFILE=new-prc-watermark")
    path = Path(setup)
    manifest = json.loads((path / "manifest.json").read_text())
    inputs = json.loads((path / "inputs.json").read_text())
    reused = json.loads((path / "reused_textseal.json").read_text())
    if digest(inputs) != manifest["input_sha256"] or digest(reused) != manifest["reused_textseal_sha256"]:
        raise ValueError("pilot inputs changed")
    if stage == "prc":
        records = [r for r in inputs if r["method"] in ("online_prc", "null")]
        worker = replay_prc
    elif stage == "textseal":
        have = {r["response_id"] for r in reused}
        records = [r for r in inputs if r["method"] in ("textseal", "null") and r["response_id"] not in have]
        worker = replay_textseal
    else:
        raise ValueError("stage must be prc or textseal")
    validate_request(manifest, records, stage, Path(__file__).resolve().parents[1])
    report = worker.remote(manifest, records)
    save(path / f"{stage}_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "files")}, indent=2))
    if not report["passed"]:
        raise RuntimeError("pilot replay failed")
