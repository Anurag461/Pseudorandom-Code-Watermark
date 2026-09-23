"""One bounded H100 run: SynthID depth 2/30, native repeat fallback on."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import time

import modal

from .config import StudySetting, digest
from .depth import SETUP, TIMEOUT, DEPTHS, validate, check_fallback
from .generation import generate_response_batch
from .repeat import upstream_hashes
from .validation import save, sha
from .validation_modal import generation_image, hf_cache, results, checkpoint

app = modal.App("prc-self-bleu-synthid-depth")


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),
              timeout=TIMEOUT,max_containers=1,retries=0,scaledown_window=2,
              volumes={"/cache":hf_cache,"/results":results})
def generation(manifest):
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes, load_qwen3_8b
    from baseline_comparison.config import PINNED_DEPENDENCIES
    started = time.monotonic()
    validate(manifest,"/root")
    results.reload()
    root = Path("/results/self_bleu_depth")/manifest["id"]
    if (root/"started.json").exists():
        raise FileExistsError("already attempted; retrieve saved batches and account for cost before retrying")
    save(root/"manifest.json",manifest)
    save(root/"started.json",{"manifest_id":manifest["id"]})
    results.commit()
    report = {"manifest_id":manifest["id"],"passed":False,"files":{},"checks":{},"batches":[]}
    try:
        preload_official_runtimes()
        if upstream_hashes()!=manifest["upstream_sha256"]:
            raise ValueError("upstream sources differ")
        execution = {"versions":{p.split("==")[0]:importlib.metadata.version(p.split("==")[0]) for p in PINNED_DEPENDENCIES},
            "gpu":torch.cuda.get_device_name(),"cuda":torch.version.cuda,"model_revision":manifest["model"]["revision"],
            "dtype":"bfloat16","tf32":torch.backends.cuda.matmul.allow_tf32,
            "bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction}
        if execution!=manifest["generation_runtime"] or "H100" not in execution["gpu"]:
            raise ValueError("generation runtime differs from saved comparison")
        report["runtime"] = execution
        checkpoint(manifest)
        model = load_qwen3_8b()
        prompts = [json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]

        def run(depth,response,length):
            with torch.no_grad():
                return generate_response_batch(model,prompts,manifest["prompt_indices"],setting=StudySetting("synthid_text",depth=depth),
                    sampling_seed=manifest["seeds"][response],response_index=response,execution=execution,max_new_tokens=length)

        def persist(name,batch):
            save(root/name,batch)
            report["files"][name] = sha(root/name)
            results.commit()

        for response in (0,1):
            batch = run(10,response,manifest["control_tokens"])
            if [digest(r["token_ids"]) for r in batch["responses"]]!=manifest["native_depth10_controls"][str(response)]:
                raise ValueError("depth-10 control does not reproduce saved prefixes")
            persist(f"controls/depth10_r{response}.json",batch)
        report["checks"]["depth10_native_prefixes"] = 100
        for depth in DEPTHS:
            report["checks"][f"depth{depth}_forced_repeat"] = check_fallback(depth,"cuda")
            for response in (0,1):
                batch = run(depth,response,1024)
                # Save expensive completed outputs before additional assertions.
                name = f"batches/depth{depth}_r{response}.json"
                persist(name,batch)
                parity = batch["telemetry"]["synthid_official_smoke_reference"]
                if not parity["indices_equal"] or parity["max_abs_score_difference"]!=0:
                    raise ValueError("official batch/single-row score parity failed")
                if response==0 and [digest(r["token_ids"][:128]) for r in batch["responses"]]!=manifest["parameter_smoke_prefixes"][str(depth)]:
                    raise ValueError("full responses do not reproduce saved parameter-smoke prefixes")
                report["batches"].append({"depth":depth,"response_index":response,"path":name,"batch_id":batch["manifest"]["batch_id"],"seconds":batch["telemetry"]["method_seconds"]})
                print(f"[depth] saved depth={depth} seed={manifest['seeds'][response]}: 50 responses",flush=True)
        report["checks"]["parameter_smoke_prefixes"] = 100
        report["passed"] = True
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.monotonic()-started
        report["measured_resource_usd"] = report["seconds"]*manifest["cost"]["resource_usd_per_second"]
        save(root/"report.json",report)
        results.commit()
    return report


@app.local_entrypoint()
def run(setup:str=str(SETUP)):
    if os.environ.get("MODAL_PROFILE")!="new-prc-watermark":
        raise ValueError("expected MODAL_PROFILE=new-prc-watermark")
    path = Path(setup)
    manifest = json.loads((path/"manifest.json").read_text())
    validate(manifest)
    report = generation.remote(manifest)
    save(path/"generation_report.json",report)
    print(json.dumps(report,indent=2))
