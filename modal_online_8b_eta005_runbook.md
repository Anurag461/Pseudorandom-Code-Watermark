# Qwen3-8B online PRC, eta=0.05

Status: **completed 2026-08-22**. The earlier batch-50 smoke app
`ap-Iu1ukZEALSgCOM2NuOdUUp` was aborted before it reported completed
generation; the revised batch-125 campaign below completed successfully.

Final exact boundary: `n90=640`, with strict bracket `(639,640]`:

- `n=640`: MAP `452/500 (90.4%)`, passes;
- `n=639`: MAP `450/500 (90.0%)`, fails under strict `>90%`;
- selected full audit at `n=640`: entropy `359/500 (71.8%)`, naive
  `195/500 (39.0%)`, and observed FPR `0/500` for all three detectors;
- retained smoke: 131.1 H100 method-seconds for 125 records;
- production: 384.2 aggregate H100 method-seconds for three parallel batches
  of 125; peak CUDA reserved memory was 44,493,176,832 bytes per batch;
- refinement and authoritative audit: zero model token positions and no model
  worker;
- authoritative result is in both `online_causal_results_summary.csv` and
  `hoeffding_results_summary.csv`.

Completed apps: smoke `ap-OZ5vJFk3MXCbM0RGEFCEMP`, production
`ap-m3Z3BjXaWd9Cak3MCZpXR3`, refinement `ap-jsUsmTpKZQwGBxMIMBXrK1`, and
authoritative audit `ap-dFUcHfYvkNzpfiPfX5JErj`. Settled provider billing
before workspace credits was `$1.20659284`: `$1.12447738` H100,
`$0.07216837` CPU, and `$0.00994709` memory.

## Frozen experiment

| Setting | Value |
| --- | --- |
| Construction | `online_causal_prc_v1` |
| Generation/detection model | `Qwen3-8B-Base` |
| Prompts | canonical indices `0..499` |
| `t`, `eta`, nominal FPR | `3`, `0.05`, `1e-3` |
| Ceiling, floor, coarse step | `1280`, `416`, `16` |
| Crossing rule | strict MAP TPR `> 0.90` (`>=451/500`) |
| Generation | forced length, seed `12345`, sampler `poscdf-v1` |
| KV cache | `static` / `static-v1` |
| Generation batch | `125` |
| Generation layout | one retained smoke load + three parallel production loads |
| Production concurrency | three H100s |

The run generates each prompt once to `n=1280` and scores exact saved prefixes.
It does not generate separate text at each candidate length. The compatible
500-record Qwen3-8B static null cache at `T=13088` already exists and can serve
all requested prefixes, so the expected new text is watermarked text only.
Do not use `--fresh`: it would defeat checkpoint reuse.

Use `H100`, without `!`, so Modal may supply an H200 at the H100 price. Batch
125 requires four total model loads instead of ten batch-50 loads. It is the
best risk-adjusted setting for this deadline: 14B already fit batch 125 at the
same `n=1280`, while batch 250 is untested. Do not switch to A100-80GB without
a measured benchmark showing runtime below 1.58 times the H100 runtime.

Expected production cost remains **$2.5--4.5** and expected wall time is
**25--45 minutes**. The phase hard stop is **$6**. The smoke provides the
campaign-specific cost measurement before the remaining three loads launch.

## Stage 1: reusable 125-prompt smoke

This is one real production batch at the real ceiling. It intentionally skips
the final full audit, so it neither needs nor creates null text.

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n 1280 \
  --floor-n 416 \
  --step 16 \
  --target-map-tpr 0.90 \
  --num-prompts 125 \
  --t 3 \
  --eta 0.05 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 125 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 25 \
  --detection-max-containers 5 \
  --no-final-audit \
  --no-pin-floor-cache
```

Approve production only if all of these hold:

1. The app completes without an OOM or corrupt/quarantined record.
2. The log reports `Qwen3-8B-Base`, `kv_cache=static`, one batch of 125, and
   160,000 newly generated suffix token positions (less only if a compatible
   cache was already present).
3. All 125 prompt records are persisted and the prefix grid is written.
4. The projected four-load campaign cost, `4 * smoke_provider_cost`, is at
   most $6.

Do not accept or reject the experiment from the smoke TPR. With 125 prompts
the strict rule requires 113/125 and is too noisy to locate the 500-prompt
crossing.
Record the app ID and provider-billed cost in
`outputs/online_8b_eta005_cost_ledger.csv`.

Expected smoke outputs:

- `outputs/online_map_sweep_n1280_to416_step16_t3_eta0.05_prompts125_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1.json`
- the matching `.csv` and `_increments/` directory.

## Stage 2: 500-prompt production sweep

After a clean smoke and explicit approval, run:

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n 1280 \
  --floor-n 416 \
  --step 16 \
  --target-map-tpr 0.90 \
  --num-prompts 500 \
  --t 3 \
  --eta 0.05 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 125 \
  --max-containers 3 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --final-audit \
  --no-pin-floor-cache
```

With the smoke cached, the expected generation plan is 375 missing watermarked
records in three parallel batches of 125 and zero missing null records. Stop
the app if the planner unexpectedly requests null generation or a different
model/KV-cache namespace. Completed watermarked records are checkpointed
individually, so a retry must use the same command without `--fresh`.

Expected production outputs use this stem:

```text
outputs/online_map_sweep_n1280_to416_step16_t3_eta0.05_prompts500_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1
```

The JSON must contain the generation plan and cost, all evaluated prompt-level
MAP results, the coarse `last_passing_n_descending`, the next shorter failure,
and the selected-boundary MAP/entropy/naive audit.

Prefix evaluation is descending and has `stop_after_first_below=true`. It must
stop immediately at the first grid point that is not strictly above 90%.
Therefore `450/500` (exactly 90.0%) is a stopping failure and must not trigger
evaluation of any shorter prefix.

## Stage 3: close the crossing to one token (no new generation)

Read these two values from the production manifest:

- `COARSE_PASS_N = summary.last_passing_n_descending`
- `COARSE_FAIL_N = summary.next_shorter_n`

If both exist and differ by 16, substitute them below. The planner should use
the saved `n=1280` records via `prefix_from_longer`; no text generation should
occur.

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n COARSE_PASS_N \
  --floor-n COARSE_FAIL_N \
  --step 1 \
  --target-map-tpr 0.90 \
  --num-prompts 500 \
  --t 3 \
  --eta 0.05 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 125 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --final-audit \
  --no-pin-floor-cache
```

Before accepting this refinement, require `wm_missing=0`, `null_missing=0`,
`model token positions: 0`, and cache mode `prefix_from_longer` (or
`exact_cache` only if the refinement ceiling itself is 1280). The exact online
`n90` is its last passing length under the strict `>=451/500` rule.

Boundary contingencies:

- If `n=1280` is below threshold, report the censored result `n90 > 1280` and
  do not extend generation without a new cost estimate and approval.
- If `n=416` is still above threshold, rerun the 500-prompt sweep from the same
  `source-n=1280` with `floor-n=128` and `step=16`. This is prefix detection
  from the existing cache, not new text generation; then refine the resulting
  bracket at step 1.
- If the empirical curve has a monotonicity warning, preserve every evaluated
  prompt-level decision and report the warning instead of silently choosing a
  different crossing.

## Stage 4: authoritative audit and CSV publication

Substitute the exact selected length for `SELECTED_N`. This entrypoint has a
hard cache-only guard: it fails rather than launching a GPU if any generation
record is missing.

```bash
modal run modal_online_run.py::main \
  --generation-model-size 8B \
  --n SELECTED_N \
  --num-prompts 500 \
  --t 3 \
  --eta 0.05 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 125 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --cache-only \
  --csv-out online_causal_results_summary.csv
```

Require all of the following before publication:

- cache-only guard passed, all generation records were cached, and no model
  worker launched;
- exactly 500 watermarked and 500 null decisions for MAP, entropy, and naive;
- MAP count matches the exact refinement result;
- observed FPRs, source-cache tag/T, null-cache T, artifact fingerprint, and
  prompt-level results are saved;
- a second cache-only audit reproduces the counts and fingerprint; write that
  replay to a verification-only CSV, not the canonical online summary, to
  avoid a duplicate authoritative row.

Do **not** point `--csv-out` directly at `hoeffding_results_summary.csv`: the
online runner and the combined Hoeffding table have different schemas. After
review, normalize exactly one selected-boundary row into the combined table
with `PRC Construction=online_causal_prc`, both model fields set to
`Qwen3-8B-Base`, the three TPR/FPR counts from this audit, both log-Hoeffding
fields marked `skipped`, and `online_causal_prc_v1` plus source/null/fingerprint
provenance in `Notes`.
Back up the combined CSV first and do not publish the 50-prompt smoke, coarse
grid points, or duplicate cache-only replays as authoritative rows.

## Later fixed-versus-online comparisons

The `n=1280` source cache also covers the existing fixed-reference lengths
`n=416` and `n=749`. Score those exact online prefixes cache-only and preserve
prompt IDs/decisions for paired bootstrap and McNemar analysis; do not generate
new fixed text.
