# Qwen3-8B eta=0.1, n=768 Modal runbook

This run generates and detects with `Qwen3-8B-Base` using the standard
500-prompt experiment:

- `n = T = 768`
- `t = 3`
- `eta = 0.1`
- `r = round(0.99n) = 760`; all new runs enforce this setting
- target FPR `1e-3`
- 500 watermarked and 500 null generations
- generation-time 8B `p_trace` reused for map, entropy-aware, and naive
  detection; there is no separate entropy-model replay

## Cache layout

The 8B run does not read or write the legacy 0.6B generation caches.

- artifacts and watermarked generations:
  `/data/n768_t3_eta0.10_T768__gen-qwen3_8b_base_r760/`
- shared 8B null generations:
  `/data/_nulls/qwen3_8b_base/T768/`
- shared 8B null detection traces:
  `/data/_null_detection_traces/qwen3_8b_base/qwen3_8b_base/T768/`
- model weights: the existing `prc-hf-cache` Modal Volume

Before scaling generation, the runner loads one model container and commits
`/cache/models/Qwen3-8B-Base` to that Volume. The remaining containers read the
same persistent copy instead of downloading the 8B shards independently.

Each new generation record stores `generation_model_size=8B` and
`generation_model=Qwen3-8B-Base`. Detection rejects model-qualified records
whose metadata does not match the requested generation model.

Do not pass `--fresh` when resuming. The planner schedules only missing prompt
indices, and a complete longer 8B null cache can serve a shorter 8B run. A
0.6B null cache is never considered for an 8B request.

## Production-reusable smoke test

The smoke uses the real `n=768` configuration, so successful prompt outputs
are retained for the full run.

```bash
MODAL_PROFILE=YOUR_PROFILE modal run modal_run.py::main \
  --prompt-start 0 \
  --num-prompts 10 \
  --max-containers 2 \
  --n 768 \
  --t 3 \
  --eta 0.1 \
  --r-frac 0.99 \
  --fpr 1e-3 \
  --generation-model-size 8B \
  --gpu H100 \
  --batch 5 \
  --csv-out hoeffding_results_summary.csv
```

Verify that the log reports:

- `Qwen3-8B-Base` for both generation and entropy/detection
- `GPU=H100`
- an 8B-qualified configuration tag and null root
- 10 watermarked and 10 null outputs with 768 tokens each
- `cached_generation_p_trace`
- full parity rank `760/760`
- no authoritative CSV append for the partial shard

## Recommended full run: batch 100

Watermarked and null maps are dispatched concurrently. With 490–500 missing
prompts, batch 100 creates five watermarked plus five null calls and can use all
10 Starter GPUs.

```bash
MODAL_PROFILE=YOUR_PROFILE modal run modal_run.py::main \
  --prompt-start 0 \
  --num-prompts 500 \
  --max-containers 10 \
  --n 768 \
  --t 3 \
  --eta 0.1 \
  --r-frac 0.99 \
  --fpr 1e-3 \
  --generation-model-size 8B \
  --gpu H100 \
  --batch 100 \
  --csv-out hoeffding_results_summary.csv
```

## Cost-oriented alternative: batch 250

Batch 250 creates two watermarked plus two null calls, so it uses at most four
GPUs concurrently. It loads fewer model replicas but has coarser checkpoints
and must be validated against H100 memory usage.

```bash
MODAL_PROFILE=YOUR_PROFILE modal run modal_run.py::main \
  --prompt-start 0 \
  --num-prompts 500 \
  --max-containers 10 \
  --n 768 \
  --t 3 \
  --eta 0.1 \
  --r-frac 0.99 \
  --fpr 1e-3 \
  --generation-model-size 8B \
  --gpu H100 \
  --batch 250 \
  --csv-out hoeffding_results_summary.csv
```

Use `H100`, not `H100!`, so Modal may provide an H200 at the H100 price. If
batch 250 runs out of memory, rerun the same command with batch 100; completed
files remain cached.

## Completion checks

1. The planner reports zero missing files after rerunning the full command.
2. Exactly 500 watermarked and 500 null records are detected.
3. The shard config records generation and entropy model as
   `Qwen3-8B-Base` and trace source as `cached_generation_p_trace`.
4. The final CSV row has `Generation Model=Qwen3-8B-Base` and
   `Entropy Model=Qwen3-8B-Base`.
5. Preserve the generated 8B null cache for future eta/key experiments.
