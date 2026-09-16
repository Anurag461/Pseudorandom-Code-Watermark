"""Run the matched-length entropy-projection diagnostic on Modal caches.

The 8B and 14B null caches already contain full-vocabulary entropy.  The older
0.6B T=8192 cache predates that trace field, so its first 1,808 tokens are
teacher-forced once through Qwen3-0.6B-Base.  No text is generated.

Run:

    modal run modal_entropy_projection_diagnostic.py \
      --trace-t 1808 --num-prompts 500
"""

from __future__ import annotations

import os
import time

import modal


APP_NAME = "prc-entropy-projection-diagnostic"
DEFAULT_TRACE_T = 1808
DEFAULT_NUM_PROMPTS = 500
DEFAULT_CPU_SHARD_SIZE = 25
DEFAULT_GPU_BATCH = 50
DEFAULT_CPU_CONTAINERS = 10
DEFAULT_GPU_CONTAINERS = 5
DEFAULT_GPU_ESTIMATOR_CHUNK_SIZE = 64
DEFAULT_GPU_KV_CACHE_IMPLEMENTATION = "static"
ZERO_POINT_SIX_CACHE_T = 8192
ZERO_POINT_SIX_ARTIFACT_TAG = (
    "online_causal_prc_v1/qwen3_0p6b_base/"
    "n4096_T4096_t3_eta0.20_rr99of100_"
    "sampler-poscdf-v1_kvcache-static-v1"
)
SAVED_MODEL_SPECS = {
    "8B": {
        "cache_t": 13088,
        "directory": "/data/_nulls/qwen3_8b_base/T13088",
    },
    "14B": {
        "cache_t": 1808,
        "directory": "/data/_nulls/qwen3_14b_base/T1808",
    },
}


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.30.2",
        "scipy==1.14.1",
        "galois==0.4.2",
        "numba==0.59.1",
        "numpy==1.26.0",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PRC_MODEL_CACHE_DIR": "/cache/models",
        "PRC_MODEL_SIZE": "0.6B",
        "PRC_MODEL_VARIANT": "base",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl", copy=True)
    .add_local_python_source(
        "prc",
        "online_prc",
        "qwen",
        "constants",
        "detectors",
        "watermark_expt",
        "entropy_projection_analysis",
    )
)
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=False)
app = modal.App(APP_NAME, image=image)


def _numpy_pickle_compat() -> None:
    import sys

    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


@app.function(
    cpu=1.0,
    memory=2048,
    volumes={"/data": data_vol},
    timeout=1800,
    max_containers=DEFAULT_CPU_CONTAINERS,
)
def analyze_saved_entropy_shard(request: dict) -> dict:
    """Analyze saved same-model 8B or 14B entropy traces without a GPU."""
    import numpy as np
    import torch
    from entropy_projection_analysis import summarize_trace_batch

    _numpy_pickle_compat()
    started = time.time()
    data_vol.reload()
    model_size = str(request["model_size"])
    if model_size not in SAVED_MODEL_SPECS:
        raise ValueError(f"unsupported saved-trace model {model_size}")
    spec = SAVED_MODEL_SPECS[model_size]
    trace_t = int(request["trace_t"])
    indices = [int(index) for index in request["prompt_indices"]]
    if trace_t > int(spec["cache_t"]):
        raise ValueError(f"T={trace_t} exceeds {model_size} cache length")

    full_rows = []
    probability_rows = []
    partition_hash = None
    bytes_read = 0
    for index in indices:
        path = os.path.join(spec["directory"], f"null_{index:04d}.pt")
        bytes_read += os.path.getsize(path)
        record = torch.load(path, weights_only=False, map_location="cpu")
        if record.get("watermark") not in (False, None):
            raise ValueError(f"{model_size} record {index} is not a null")
        full = np.asarray(record.get("base_lm_entropy"), dtype=np.float64).reshape(-1)
        p1 = np.asarray(record.get("p_trace"), dtype=np.float64).reshape(-1)
        if full.size < trace_t or p1.size < trace_t:
            raise ValueError(f"{model_size} record {index} lacks T={trace_t} traces")
        observed_hash = str(record.get("partition_sha256") or "")
        if len(observed_hash) != 64:
            raise ValueError(f"{model_size} record {index} lacks partition provenance")
        if partition_hash is None:
            partition_hash = observed_hash
        elif observed_hash != partition_hash:
            raise ValueError(f"{model_size} shard mixes vocabulary partitions")
        full_rows.append(full[:trace_t])
        probability_rows.append(p1[:trace_t])

    payload = summarize_trace_batch(
        model_size=model_size,
        prompt_indices=indices,
        full_entropy_bits=np.stack(full_rows),
        partition_probability=np.stack(probability_rows),
        trace_t=trace_t,
        partition_sha256=partition_hash,
        source_cache_t=int(spec["cache_t"]),
        source_kind="saved_generation_trace",
    )
    payload["execution"] = {
        "seconds": time.time() - started,
        "bytes_read": int(bytes_read),
        "gpu": None,
        "model_forward_positions": 0,
    }
    return payload


@app.cls(
    gpu="A10G",
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=7200,
    max_containers=DEFAULT_GPU_CONTAINERS,
)
class ZeroPointSixEntropyReplay:
    """Recover missing full entropy for legacy 0.6B null records."""

    @modal.enter()
    def load(self):
        os.environ["PRC_MODEL_SIZE"] = "0.6B"
        os.environ["PRC_MODEL_VARIANT"] = "base"
        import watermark_expt as we

        self.we = we
        hf_cache.commit()

    @modal.method()
    def analyze(self, request: dict) -> dict:
        import numpy as np
        import torch
        from detectors import tensor_sha256
        from entropy_projection_analysis import summarize_trace_batch

        _numpy_pickle_compat()
        started = time.time()
        data_vol.reload()
        trace_t = int(request["trace_t"])
        estimator_chunk_size = int(request["estimator_chunk_size"])
        kv_cache_implementation = str(request["kv_cache_implementation"])
        indices = [int(index) for index in request["prompt_indices"]]
        if trace_t > ZERO_POINT_SIX_CACHE_T:
            raise ValueError("requested trace exceeds legacy 0.6B cache")

        artifact_path = f"/data/{ZERO_POINT_SIX_ARTIFACT_TAG}/artifacts.pt"
        artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
        partition = artifact["partition"]
        partition_hash = tensor_sha256(partition)
        token_rows = []
        reference_rows = []
        bytes_read = os.path.getsize(artifact_path)
        for index in indices:
            path = f"/data/_nulls/T{ZERO_POINT_SIX_CACHE_T}/null_{index:04d}.pt"
            bytes_read += os.path.getsize(path)
            record = torch.load(path, weights_only=False, map_location="cpu")
            if record.get("watermark") not in (False, None):
                raise ValueError(f"0.6B record {index} is not a null")
            tokens = torch.as_tensor(record["tokens"], dtype=torch.long)[:trace_t]
            reference = np.asarray(record["p_trace"], dtype=np.float64).reshape(-1)[
                :trace_t
            ]
            if tokens.numel() != trace_t or reference.size != trace_t:
                raise ValueError(f"0.6B record {index} is shorter than T={trace_t}")
            token_rows.append(tokens.contiguous())
            reference_rows.append(reference)

        prompt_batch = torch.tensor(
            [artifact["prompt_ids_list"][index] for index in indices],
            dtype=torch.long,
            device=self.we.device,
        )
        token_batch = torch.stack(token_rows).to(self.we.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        p1, full_entropy_nats = self.we.estimate_partition_entropy_trace_batch(
            self.we.model,
            prompt_batch,
            token_batch,
            partition,
            kv_cache_implementation=kv_cache_implementation,
            chunk_size=estimator_chunk_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        else:
            peak_allocated = peak_reserved = 0
        full_entropy_bits = np.asarray(full_entropy_nats, dtype=np.float64) / np.log(2.0)
        payload = summarize_trace_batch(
            model_size="0.6B",
            prompt_indices=indices,
            full_entropy_bits=full_entropy_bits,
            # Use the projection probabilities saved at generation time, as
            # the 8B/14B paths do.  The teacher-forced p1 values are retained
            # as a numerical-replay sensitivity check.
            partition_probability=np.stack(reference_rows),
            trace_t=trace_t,
            partition_sha256=partition_hash,
            source_cache_t=ZERO_POINT_SIX_CACHE_T,
            source_kind=(
                "saved_generation_projection_with_teacher_forced_full_entropy"
            ),
            reference_partition_probability=p1,
        )
        payload["execution"] = {
            "seconds": time.time() - started,
            "bytes_read": int(bytes_read),
            "gpu": "A10G",
            "teacher_forced_token_positions": len(indices) * trace_t,
            "model_forward_positions": len(indices) * (
                int(prompt_batch.shape[1]) + max(trace_t - 1, 0)
            ),
            "estimator_chunk_size": estimator_chunk_size,
            "kv_cache_implementation": kv_cache_implementation,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        }
        return payload


@app.local_entrypoint()
def main(
    trace_t: int = DEFAULT_TRACE_T,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    cpu_shard_size: int = DEFAULT_CPU_SHARD_SIZE,
    gpu_batch: int = DEFAULT_GPU_BATCH,
    cpu_max_containers: int = DEFAULT_CPU_CONTAINERS,
    gpu_max_containers: int = DEFAULT_GPU_CONTAINERS,
    gpu_estimator_chunk_size: int = DEFAULT_GPU_ESTIMATOR_CHUNK_SIZE,
    gpu_kv_cache_implementation: str = DEFAULT_GPU_KV_CACHE_IMPLEMENTATION,
    output_dir: str = "outputs/entropy_projection_diagnostic",
):
    """Run all three models and save matched-prefix comparison artifacts."""
    from datetime import datetime, timezone

    from entropy_projection_analysis import (
        build_comparison_payload,
        merge_model_shards,
        write_comparison_outputs,
    )

    trace_t = int(trace_t)
    num_prompts = int(num_prompts)
    cpu_shard_size = int(cpu_shard_size)
    gpu_batch = int(gpu_batch)
    gpu_estimator_chunk_size = int(gpu_estimator_chunk_size)
    gpu_kv_cache_implementation = str(gpu_kv_cache_implementation)
    if trace_t <= 0 or trace_t > DEFAULT_TRACE_T:
        raise ValueError(f"trace_t must be in [1, {DEFAULT_TRACE_T}]")
    if num_prompts <= 0 or num_prompts > DEFAULT_NUM_PROMPTS:
        raise ValueError(f"num_prompts must be in [1, {DEFAULT_NUM_PROMPTS}]")
    if cpu_shard_size <= 0 or gpu_batch <= 0 or gpu_estimator_chunk_size <= 0:
        raise ValueError("shard sizes must be positive")
    if gpu_kv_cache_implementation not in ("concat", "static"):
        raise ValueError("gpu kv cache implementation must be concat or static")
    indices = list(range(num_prompts))
    started = time.time()

    models = {}
    execution = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "generation_attempts": 0,
        "matched_prefix_t": trace_t,
        "models": {},
    }
    for model_size in ("8B", "14B"):
        requests = [
            {
                "model_size": model_size,
                "trace_t": trace_t,
                "prompt_indices": chunk,
            }
            for chunk in _chunks(indices, cpu_shard_size)
        ]
        worker = analyze_saved_entropy_shard.with_options(
            max_containers=min(int(cpu_max_containers), len(requests))
        )
        model_started = time.time()
        shards = list(worker.map(requests))
        models[model_size] = merge_model_shards(shards, num_prompts)
        execution["models"][model_size] = {
            "wall_seconds": time.time() - model_started,
            "aggregate_method_seconds": sum(
                float(item["execution"]["seconds"]) for item in shards
            ),
            "bytes_read": sum(int(item["execution"]["bytes_read"]) for item in shards),
            "gpu": None,
            "shard_count": len(shards),
        }

    replay_requests = [
        {
            "trace_t": trace_t,
            "prompt_indices": chunk,
            "estimator_chunk_size": gpu_estimator_chunk_size,
            "kv_cache_implementation": gpu_kv_cache_implementation,
        }
        for chunk in _chunks(indices, gpu_batch)
    ]
    replay = ZeroPointSixEntropyReplay.with_options(
        max_containers=min(int(gpu_max_containers), len(replay_requests))
    )()
    replay_started = time.time()
    replay_shards = list(replay.analyze.map(replay_requests))
    models["0.6B"] = merge_model_shards(replay_shards, num_prompts)
    execution["models"]["0.6B"] = {
        "wall_seconds": time.time() - replay_started,
        "aggregate_method_seconds": sum(
            float(item["execution"]["seconds"]) for item in replay_shards
        ),
        "bytes_read": sum(
            int(item["execution"]["bytes_read"]) for item in replay_shards
        ),
        "gpu": "A10G",
        "shard_count": len(replay_shards),
        "estimator_chunk_size": gpu_estimator_chunk_size,
        "kv_cache_implementation": gpu_kv_cache_implementation,
        "teacher_forced_token_positions": sum(
            int(item["execution"]["teacher_forced_token_positions"])
            for item in replay_shards
        ),
        "peak_cuda_allocated_bytes": max(
            int(item["execution"]["peak_cuda_allocated_bytes"])
            for item in replay_shards
        ),
        "peak_cuda_reserved_bytes": max(
            int(item["execution"]["peak_cuda_reserved_bytes"])
            for item in replay_shards
        ),
    }
    ordered_models = {name: models[name] for name in ("0.6B", "8B", "14B")}
    execution["local_end_to_end_wall_seconds"] = time.time() - started
    payload = build_comparison_payload(
        ordered_models, execution, trace_t, num_prompts
    )
    paths = write_comparison_outputs(payload, output_dir)

    print("[entropy-projection] matched comparison complete", flush=True)
    for model_size, model in ordered_models.items():
        stats = model["metric_statistics"]
        print(
            f"[entropy-projection] {model_size}: "
            f"H(token)={stats['full_entropy_bits']['mean']:.4f} bits; "
            f"H(bucket)={stats['projected_entropy_bits']['mean']:.4f} bits; "
            f"gap={stats['gap_bits']['mean']:.4f} bits; "
            f"retained={model['projection_retention_ratio_of_means']:.2%}",
            flush=True,
        )
    print(f"[entropy-projection] outputs: {paths}", flush=True)
