"""Explicit preflight/smoke/full stages for a committed prompt-free campaign."""
import json
from pathlib import Path
import subprocess

import modal

from prompt_free.manifest import SOURCE_FILES, digest_json, file_sha, plan, source_identity, validate

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = (ROOT/"prompt_free/requirements.txt").read_text().splitlines()
image = (modal.Image.debian_slim(python_version="3.11").pip_install(*DEPENDENCIES)
         .env({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
               "NUMBA_DISABLE_JIT": "1", "OMP_NUM_THREADS": "1"}))
# Deploy only the listed committed sources. Historical/untracked experiment
# packages are neither mounted nor imported by this production entrypoint.
for name in SOURCE_FILES:
    image = image.add_local_file(ROOT/name, "/root/"+name, copy=True)
app = modal.App("prc-prompt-free-redetection", image=image)
archive = modal.Volume.from_name("prc-research-archive", create_if_missing=False)
data = modal.Volume.from_name("prc-data", create_if_missing=False)
models = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
results = modal.Volume.from_name("prc-completion-only", create_if_missing=False)


@app.function(cpu=4, memory=8192, timeout=1800, retries=0, scaledown_window=2,
              include_source=False, volumes={"/archive": archive, "/data": data, "/results": results})
def preflight(manifest, source):
    import torch
    from prompt_free.storage import prepare_case
    torch.set_num_threads(1)
    validate(manifest)
    if source_identity("/root")["sha256"] != source["sha256"]:
        raise ValueError("deployed source differs from the committed source")
    results.reload()
    prepared = []
    for case in manifest["cases"]:
        prepared.append(prepare_case(case, manifest["model"], source,
                                     {"archive": "/archive", "data": "/data"}, "/results"))
        results.commit()
        print(f"Verified {case['id']}: {len(case['records'])} cached candidates", flush=True)
    return prepared


@app.cls(gpu="A10G", cpu=4, memory=8192, timeout=3600, retries=0,
         max_containers=1, scaledown_window=2, include_source=False,
         volumes={"/cache": models, "/results": results})
class Detector:
    model_json: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from safetensors.torch import load_file
        from qwen import Qwen3Model, return_qwen_config, load_weights_into_qwen
        self.spec = json.loads(self.model_json)
        if (self.spec["id"], self.spec["size"], self.spec["dtype"]) != ("Qwen/Qwen3-0.6B-Base", "0.6B", "bfloat16"):
            raise ValueError("unsupported detector configuration")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        cache = Path("/cache")/self.spec["cache_directory"]
        weights = cache/"model.safetensors"
        if file_sha(weights) != self.spec["weights_sha256"] or file_sha(cache/"tokenizer.json") != self.spec["tokenizer_sha256"]:
            raise ValueError("cached checkpoint/tokenizer differs from manifest")
        metadata = cache/".cache/huggingface/download/model.safetensors.metadata"
        if metadata.read_text().splitlines()[0] != self.spec["revision"]:
            raise ValueError("cached model revision differs")
        config = {**return_qwen_config("0.6B"), "dtype": torch.bfloat16}
        with torch.device("cuda"):
            self.model = Qwen3Model(config)
        load_weights_into_qwen(self.model, config, load_file(str(weights)))
        self.model.eval().requires_grad_(False)
        self.code_sha = source_identity("/root")["sha256"]
        self.validated = set()
        # Tokenizer identity is verified as bytes; no encoding API is used.

    @modal.method()
    def batch(self, descriptor):
        import time
        import torch
        from prompt_free.core import recover
        from prompt_free.storage import load_gpu_input, load_pt, trace_payload, validate_trace, json_write
        from qwen import teacher_force_partition_trace_batch
        identity = descriptor["identity"]
        if identity["model"] != self.spec or identity["code_sha256"] != self.code_sha:
            raise ValueError("GPU execution model/code differs from prepared batch")
        results.reload()
        output = Path("/results")/descriptor["root"]
        inputs = load_gpu_input(descriptor, "/results")
        path = output/"trace.pt"
        if path.exists():
            validate_trace(load_pt(path), identity)
            return {"root": descriptor["root"], "cached": True, "passed": True}
        tokens = inputs["tokens"].to("cuda")
        part = inputs["partition"][1].to("cuda")
        cache = identity["cache"]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        trace = recover(self.model, tokens, part, cache=cache)
        torch.cuda.synchronize()
        replay_seconds = time.perf_counter()-t0
        def check_memory_margin():
            used = torch.cuda.max_memory_reserved()
            total = torch.cuda.get_device_properties(0).total_memory
            if used >= .85*total:
                raise ValueError(f"batch reserved {used}/{total} GPU bytes; exceeds 85% memory margin; choose and audit a smaller explicit batch")
        # Reject an oversized replay before paying for independent validation.
        check_memory_margin()
        signature = (identity["run_id"], len(tokens), tokens.shape[1], cache, identity["partition_sha256"])
        validated_now = signature not in self.validated
        if validated_now and tokens.shape[1] > 1:
            independent = teacher_force_partition_trace_batch(
                self.model, tokens[:, :1], tokens[:, 1:], part,
                kv_cache_implementation=cache, chunk_size=1)
            if not torch.equal(trace, independent):
                raise ValueError("raw replay differs from independent token-step reference")
            length = min(65, tokens.shape[1])
            received = []
            hook = self.model.register_forward_pre_hook(lambda model, args: received.append(args[0].detach().cpu().clone()))
            try:
                prefix = recover(self.model, tokens[:, :length], part, cache=cache)
            finally:
                hook.remove()
            if not torch.equal(torch.cat(received, 1), tokens[:, :length-1].cpu()):
                raise ValueError("model inputs contain something other than raw completion tokens")
            reversed_rows = recover(self.model, tokens.flip(0)[:, :length], part, cache=cache)
            if not torch.equal(prefix, reversed_rows.flip(0)) or not torch.equal(prefix, trace[:, :length-1]):
                raise ValueError("batch-order/prefix consistency check failed")
            changed = tokens[:, :length].clone()
            cutoff = length//2
            changed[:, cutoff:] = (changed[:, cutoff:]+17) % part.numel()
            causal = recover(self.model, changed, part, cache=cache)
            if not torch.equal(prefix[:, :cutoff], causal[:, :cutoff]):
                raise ValueError("future tokens affected earlier predictions")
        check_memory_margin()
        self.validated.add(signature)
        value = trace_payload(trace, identity)
        temporary = path.with_suffix(".partial")
        torch.save(value, temporary)
        temporary.replace(path)
        validation = {"passed": True, "validation_run_on_this_batch": validated_now,
                      "validation_signature": list(signature), "raw_inputs_only": True,
                      "inference_dtype": "bfloat16", "replay_seconds": replay_seconds,
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "gpu": torch.cuda.get_device_name(), "trace_sha256": file_sha(path)}
        json_write(output/"validation.json", validation)
        results.commit()
        print(f"Saved {identity['case_id']} batch of {len(tokens)} raw-completion traces", flush=True)
        return {"root": descriptor["root"], **validation, "cached": False}


@app.function(cpu=4, memory=8192, timeout=3600, retries=0, scaledown_window=2,
              include_source=False, volumes={"/results": results})
def finish(prepared):
    import torch
    from prompt_free.storage import aggregate
    torch.set_num_threads(1)
    results.reload()
    result = aggregate(prepared, "/results")
    results.commit()
    return result


@app.local_entrypoint()
def main(manifest: str = "prompt_free/manifests/pilots.json", stage: str = "preflight", case: str = "", workers: int = 1):
    from prompt_free.storage import json_write
    if stage not in ("preflight", "smoke", "full"):
        raise ValueError("stage must be preflight, smoke or full")
    if not 1 <= workers <= 4:
        raise ValueError("workers must be in [1,4]")
    manifest_path = Path(manifest).resolve()
    content = json.loads(manifest_path.read_text())
    validate(content)
    source = source_identity(ROOT, require_commit=True)
    # Manifests must also be reviewable and committed before any remote stage.
    relative = manifest_path.relative_to(ROOT).as_posix()
    committed = subprocess.check_output(["git", "show", f"{source['git_commit']}:{relative}"], cwd=ROOT)
    if committed != manifest_path.read_bytes():
        raise ValueError("manifest changes must be committed before remote execution")
    if case:
        content["cases"] = [c for c in content["cases"] if c["id"] == case]
        if not content["cases"]:
            raise ValueError("case not found in manifest")
    print(json.dumps(plan(content), indent=2), flush=True)
    prepared = preflight.remote(content, source)
    local = ROOT/"outputs/prompt_free"/digest_json({"manifest": content, "source": source})[:24]
    json_write(local/"prepared.json", prepared)
    json_write(local/(stage+"_execution.json"), {"stage": stage, "maximum_gpu_workers": workers,
                                               "gpu": "A10G", "source": source})
    if stage == "preflight":
        print(f"Preflight complete; no GPU inference launched. Saved {local}", flush=True)
        return
    detector = Detector.with_options(max_containers=workers)(model_json=json.dumps(content["model"], sort_keys=True))
    jobs = [p["batches"][0] for p in prepared] if stage == "smoke" else [b for p in prepared for b in p["batches"]]
    # Each worker owns one model. Calls keep fixed batch membership, order and
    # fresh KV caches; worker scheduling cannot combine or resize batches.
    for done in detector.batch.map(jobs):
        json_write(local/"batches"/(digest_json(done["root"])+".json"), done)
    if stage == "smoke":
        print(f"Smoke passed for {len(prepared)} case(s); full campaign was not run.", flush=True)
        return
    for p in prepared:
        completed = finish.remote(p)
        for name in ("full.json", "summary.json"):
            path = local/p["identity"]["case"]["id"]/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"".join(results.read_file(completed["root"]+"/"+name)))
        print(json.dumps(completed), flush=True)
