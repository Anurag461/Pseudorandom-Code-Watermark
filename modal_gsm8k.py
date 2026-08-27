"""
Run the gsm8k utility + watermark-detection eval on Modal (GPU).

This wraps the local `watermark_expt.chat_eval_benchmark_batched` path:
  - batched, left-padded PRC generation over gsm8k prompts (watermarked + null),
  - per-row prefix-column Hoeffding detection (detect_hoeffding, which delegates
    to the prefix-column detector for outputs shorter than n),
  - reports watermarked/unwatermarked task score, detection TPR (power), and the
    empirical FPR on the unwatermarked arm.

The heavy lifting lives in watermark_expt.py (unchanged); this file only adds the
Modal image, GPU function, and result persistence.

Usage:
    modal run modal_gsm8k.py
    modal run modal_gsm8k.py --limit 200 --batch-size 8 --fpr 1e-4 \
        --model-variant reasoning --max-new-tokens 1024
"""
import modal

MODEL_SIZE = "0.6B"
GPU = "A10G"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
        "scipy",
        "galois",
        "numpy",
        "ldpc",
        "datasets",
        "aiohttp",
        "tqdm",
        # IFEval verifier deps (benchmarks/ifeval_lib)
        "nltk",
        "langdetect",
        "immutabledict",
        "absl-py",
    )
    # Bake the nltk sentence-tokenizer data IFEval's verifier needs.
    .run_commands(
        "python -c \"import nltk; nltk.download('punkt', download_dir='/nltk_data'); "
        "nltk.download('punkt_tab', download_dir='/nltk_data')\""
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "NLTK_DATA": "/nltk_data",
            "PRC_MODEL_SIZE": MODEL_SIZE,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_python_source(
        "prc", "qwen", "constants", "detectors", "watermark_expt", "benchmarks"
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("prc-eval-results", create_if_missing=True)

# HF token for gated datasets (e.g. GPQA Diamond / Idavidrein/gpqa). Stored in
# Modal's secret store (never in the repo); injected as $HF_TOKEN at runtime.
# Create/update with: modal secret create huggingface HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("prc-gsm8k-eval", image=image)


@app.function(
    gpu=GPU,
    volumes={"/cache/hf": hf_cache, "/results": results_vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
)
def run_eval(
    benchmark: str = "gsm8k",
    limit: int = 200,
    batch_size: int = 8,
    fpr: float = 1e-4,
    model_variant: str = "reasoning",
    max_new_tokens: int = 1024,
    detect: bool = False,
    log_gen: bool = False,
    seed: int = 12345,
):
    # Set the model variant BEFORE importing watermark_expt: that module builds
    # and loads the model at import time from PRC_MODEL_VARIANT / PRC_MODEL_SIZE.
    import os
    os.environ["PRC_MODEL_VARIANT"] = model_variant

    import json
    from datetime import datetime, timezone

    import numpy as np
    import torch

    torch.manual_seed(seed)                       # reproducible partition + sampling
    import watermark_expt as wx                    # heavy: loads Qwen3 onto the GPU
    from benchmarks.registry import get_benchmark

    # n is fixed to 800 inside chat_eval_benchmark_batched's KeyGen call.
    bm = get_benchmark(benchmark, start=0, stop=None, step=1)
    hist_wm, scores_wm, recs_wm, _ = wx.chat_eval_benchmark_batched(
        bm, batch_size=batch_size, limit=limit, watermark=True,
        max_new_tokens=max_new_tokens, detect=detect, fpr=fpr, log_gen=log_gen,
    )

    bm_null = get_benchmark(benchmark, start=0, stop=None, step=1)
    hist_uw, scores_uw, recs_uw, _ = wx.chat_eval_benchmark_batched(
        bm_null, batch_size=batch_size, limit=limit, watermark=False,
        max_new_tokens=max_new_tokens, detect=detect, fpr=fpr, log_gen=log_gen,
    )

    def _rate(recs):
        vals = [r["detected"] for r in recs if r["detected"] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    def _len_stats(recs):
        # len(tokens) is the generated length truncated at the first EOS; a length
        # equal to max_new_tokens means the model never emitted EOS -> truncated.
        L = np.array([len(r["tokens"]) for r in recs], dtype=float)
        if L.size == 0:
            return {}
        return {
            "mean": float(L.mean()),
            "median": float(np.median(L)),
            "min": int(L.min()),
            "max": int(L.max()),
            "p90": float(np.percentile(L, 90)),
            "frac_truncated": float((L >= max_new_tokens).mean()),
        }

    result = {
        "benchmark": benchmark,
        "model_variant": model_variant,
        "model_size": MODEL_SIZE,
        "n": 800,
        "limit": limit,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "seed": seed,
        "num_examples": len(scores_wm),
        "wm_score": float(np.mean(scores_wm)),
        "unwm_score": float(np.mean(scores_uw)),
        "utility_gap": float(np.mean(scores_uw)) - float(np.mean(scores_wm)),
        "output_len_wm": _len_stats(recs_wm),
        "output_len_unwm": _len_stats(recs_uw),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if detect:
        result["fpr"] = fpr
        result["detection_tpr"] = _rate(recs_wm)   # power on watermarked outputs
        result["empirical_fpr"] = _rate(recs_uw)   # false positives on null outputs

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"/results/{benchmark}_{model_variant}_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()

    print(json.dumps(result, indent=2))
    # Return the raw histories too so short diagnostic runs can inspect outputs.
    return {"result": result, "history_wm": hist_wm, "history_unwm": hist_uw}


@app.local_entrypoint()
def main(
    benchmark: str = "gsm8k",
    limit: int = 200,
    batch_size: int = 8,
    fpr: float = 1e-4,
    model_variant: str = "reasoning",
    max_new_tokens: int = 1024,
    detect: bool = False,
    log_gen: bool = False,
):
    out = run_eval.remote(
        benchmark=benchmark,
        limit=limit,
        batch_size=batch_size,
        fpr=fpr,
        model_variant=model_variant,
        max_new_tokens=max_new_tokens,
        detect=detect,
        log_gen=log_gen,
    )
    result = out["result"]
    print(f"\n=== {result['benchmark']} eval result ===")
    for k, v in result.items():
        print(f"{k:>16}: {v}")


# ---- sharded full-benchmark eval (robust to preemption + client disconnect) ---
#
# The whole test set is split into `num_shards` strided index shards; each shard
# x arm (wm / unwm) runs as its own container. Modal retries any shard that gets
# preempted, and every shard commits its own {scores, lengths} JSON to the
# results volume, so partial progress survives a client disconnect and the final
# aggregate can be rebuilt from the volume even if the driver dies.

@app.function(volumes={"/cache/hf": hf_cache}, secrets=[hf_secret], timeout=600)
def count_examples(benchmark: str) -> int:
    from benchmarks.registry import get_benchmark
    return get_benchmark(benchmark, start=0, stop=None, step=1).num_examples()


@app.cls(
    gpu=GPU,
    volumes={"/cache/hf": hf_cache, "/results": results_vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 4,
    max_containers=24,
)
class Evaluator:
    model_variant: str = modal.parameter(default="reasoning")
    seed: int = modal.parameter(default=12345)

    @modal.enter()
    def _load(self):
        import os
        os.environ["PRC_MODEL_VARIANT"] = self.model_variant
        import torch
        torch.manual_seed(self.seed)          # same seeded partition in every container
        import watermark_expt as wx
        self.wx = wx

    @modal.method()
    def eval_shard(self, benchmark, indices, watermark, batch_size,
                   max_new_tokens, run_tag):
        import json
        import os
        from benchmarks.registry import get_benchmark

        arm = "wm" if watermark else "uw"
        path = f"/results/{run_tag}/shard_{arm}_{int(indices[0]):05d}.json"

        # Resume: a shard that already committed (e.g. before a prior timeout) is
        # reused as-is, so re-spawning only recomputes the missing shards.
        results_vol.reload()
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

        bm = get_benchmark(benchmark, start=0, stop=None, step=1)
        _, scores, recs, _ = self.wx.chat_eval_benchmark_batched(
            bm, batch_size=batch_size, watermark=watermark,
            max_new_tokens=max_new_tokens, detect=False, indices=list(indices),
        )
        out = {
            "benchmark": benchmark,
            "watermark": bool(watermark),
            "indices": [int(i) for i in indices],
            "scores": [int(s) for s in scores],
            "lengths": [int(len(r["tokens"])) for r in recs],
        }
        os.makedirs(f"/results/{run_tag}", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
        results_vol.commit()
        return out


@app.local_entrypoint()
def run_full(
    benchmark: str = "arc_easy",
    batch_size: int = 16,
    max_new_tokens: int = 1900,
    num_shards: int = 6,
    model_variant: str = "reasoning",
    run_tag: str = "full",
):
    # Orchestration runs SERVER-SIDE (orchestrate_full) so a flaky local client
    # can't kill the run: with --detach, the single dispatched orchestrator stays
    # alive and drives the shards + aggregation on Modal. It also writes
    # SUMMARY.json to the volume, so the result is recoverable even if the client
    # never sees the return value.
    summary = orchestrate_full.remote(
        benchmark, batch_size, max_new_tokens, num_shards, model_variant, run_tag
    )
    print(f"\n=== FULL {summary['benchmark']} "
          f"({summary['model_variant']}, max_new_tokens={summary['max_new_tokens']}) ===")
    for k in ("num_examples_wm", "num_examples_unwm", "unwm_score", "wm_score",
              "utility_gap", "len_wm", "len_unwm"):
        print(f"{k:>18}: {summary.get(k)}")
    print(f"(shards + SUMMARY.json under volume prc-eval-results:/{summary['tag']}/)")


@app.function(
    volumes={"/cache/hf": hf_cache, "/results": results_vol},
    secrets=[hf_secret],
    timeout=60 * 60 * 10,
)
def orchestrate_full(benchmark, batch_size, max_new_tokens, num_shards,
                     model_variant, run_tag, limit=None, sample=None,
                     sample_seed=0):
    """Server-side driver: count -> shard -> starmap eval_shard -> aggregate.

    Runs entirely on Modal so the local client only needs to survive dispatch.
    Writes SUMMARY.json to the results volume for out-of-band recovery.
    `limit` caps the examples from index 0 (subject-ordered smoke tests);
    `sample` instead draws a reproducible random subset of that size across the
    whole set (seeded by `sample_seed`), which is representative for e.g. MMLU.
    """
    import json
    import os
    import random

    import numpy as np
    from benchmarks.registry import get_benchmark

    n = get_benchmark(benchmark, start=0, stop=None, step=1).num_examples()
    idx = list(range(n))
    if sample is not None:
        idx = sorted(random.Random(sample_seed).sample(idx, min(int(sample), n)))
    elif limit is not None:
        idx = idx[:int(limit)]
    shards = [s for s in (idx[i::num_shards] for i in range(num_shards)) if s]
    tag = f"{benchmark}_{run_tag}"

    ev = Evaluator(model_variant=model_variant)
    tasks = (
        [(benchmark, s, True, batch_size, max_new_tokens, tag) for s in shards]
        + [(benchmark, s, False, batch_size, max_new_tokens, tag) for s in shards]
    )
    results = list(ev.eval_shard.starmap(tasks))

    wm_s, uw_s, wm_L, uw_L = [], [], [], []
    for r in results:
        (wm_s if r["watermark"] else uw_s).extend(r["scores"])
        (wm_L if r["watermark"] else uw_L).extend(r["lengths"])

    def stats(L):
        a = np.array(L, dtype=float)
        return {
            "mean": round(float(a.mean()), 1),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "max": int(a.max()),
            "frac_truncated": round(float((a >= max_new_tokens).mean()), 4),
        }

    summary = {
        "benchmark": benchmark,
        "model_variant": model_variant,
        "max_new_tokens": max_new_tokens,
        "tag": tag,
        "sample": (int(sample) if sample is not None else None),
        "sample_seed": sample_seed,
        "num_examples_wm": len(wm_s),
        "num_examples_unwm": len(uw_s),
        "unwm_score": round(float(np.mean(uw_s)), 4),
        "wm_score": round(float(np.mean(wm_s)), 4),
        "utility_gap": round(float(np.mean(uw_s) - np.mean(wm_s)), 4),
        "len_wm": stats(wm_L),
        "len_unwm": stats(uw_L),
    }
    os.makedirs(f"/results/{tag}", exist_ok=True)
    with open(f"/results/{tag}/SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    results_vol.commit()
    return summary
