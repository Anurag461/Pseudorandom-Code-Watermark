# Qwen3-8B online PRC, eta=0.10

Status: **completed 2026-08-22**.

Final exact descending boundary: `n90=1407`, with strict bracket
`(1406,1407]`:

- `n=1407`: MAP `451/500 (90.2%)`, passes;
- `n=1406`: MAP `450/500 (90.0%)`, fails under strict `>90%`;
- selected full audit at `n=1407`: entropy `394/500 (78.8%)`, naive
  `240/500 (48.0%)`, and observed FPR `0/500` for all three detectors;
- retained smoke: 280.0 H100 method-seconds for 50 records;
- production: 2,536.3 aggregate H100 method-seconds for nine parallel
  batches of 50; peak CUDA reserved memory was 42,331,013,120 bytes per batch;
- refinement and authoritative audit: zero model token positions and no model
  worker;
- authoritative result is in both `online_causal_results_summary.csv` and
  `hoeffding_results_summary.csv`.

Completed apps: smoke `ap-4OiqaTPx8LkuniTkNNL9HI`, production
`ap-gDJh4dTtuqFatFbfTHUr44`, refinement `ap-AXfaIOYG3SHA7ZIiDfTSyJ`, and
authoritative audit `ap-ExraUnESUNvGB7wfdgKONu`. Provider-billed costs still
need reconciliation from the Modal dashboard.

## Frozen experiment

| Setting | Value |
| --- | --- |
| Construction | `online_causal_prc_v1` |
| Generation/detection model | `Qwen3-8B-Base` |
| Prompts | canonical indices `0..499` |
| `t`, `eta`, nominal FPR | `3`, `0.10`, `1e-3` |
| Ceiling, floor, coarse step | `3072`, `768`, `16` |
| Crossing rule | strict MAP TPR `>0.90` (`>=451/500`) |
| Generation | forced length, seed `12345`, sampler `poscdf-v1` |
| KV cache | `static` / `static-v1` |
| Generation batch | `50` |
| Generation layout | one retained smoke load + nine parallel production loads |
| Production concurrency | nine H100s (within the 10-GPU limit) |

The run generates each prompt once to `n=3072` and scores exact prefixes of
those saved records. It must not generate separate watermarked text at every
candidate length. A complete compatible Qwen3-8B static null cache exists at
`T=13088` and can serve this campaign, so expected new text is watermarked text
only. Do not pass `--fresh`.

Use `H100`, without `!`, so Modal may supply an H200 at the H100 price. Batch
50 is already validated without OOM on Qwen3-8B at the longer `n=4096`; it is
the low-risk choice for the remaining deadline. The retained smoke plus nine
parallel production jobs require ten total model loads across the campaign.

Expected Modal cost is **$5.5--9** and expected wall time is **35--60
minutes**. The phase hard stop is **$12**. The cost basis is the measured
batch-50 `n=4096` result: 516.99 H100 method-seconds and $0.68394 total
provider cost for 50 records. The shorter `n=3072` run should cost less per
batch, but the smoke is the authoritative campaign-specific gate.

## Stage 1: retained 50-prompt smoke

This is one real production batch at the actual ceiling. It skips the final
full audit, so it neither needs nor creates null text.

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n 3072 \
  --floor-n 768 \
  --step 16 \
  --target-map-tpr 0.90 \
  --num-prompts 50 \
  --t 3 \
  --eta 0.10 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 50 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 10 \
  --detection-max-containers 5 \
  --no-final-audit \
  --no-pin-floor-cache
```

Approve production only if:

1. The planner reports exactly 50 missing watermarked records, no null work,
   model `Qwen3-8B-Base`, and static KV.
2. One batch of 50 completes without OOM, fallback GPU, corrupt record, or
   quarantine; expected new suffix token positions are 153,600.
3. All 50 records and the prefix grid are persisted.
4. The projected ten-load campaign cost, `10 * smoke_provider_cost`, is at
   most $12.

The smoke TPR is not a statistical acceptance criterion. With 50 prompts the
strict rule requires 46/50 and is too noisy to locate the 500-prompt crossing.
Record the app ID, measured GPU seconds, peak memory, and provider bill in
`outputs/online_8b_eta010_cost_ledger.csv`.

Expected smoke outputs:

- `outputs/online_map_sweep_n3072_to768_step16_t3_eta0.10_prompts50_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1.json`
- the matching `.csv` and `_increments/` directory.

## Stage 2: 500-prompt production sweep

After a clean smoke and approval, run:

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n 3072 \
  --floor-n 768 \
  --step 16 \
  --target-map-tpr 0.90 \
  --num-prompts 500 \
  --t 3 \
  --eta 0.10 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 50 \
  --max-containers 9 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --final-audit \
  --no-pin-floor-cache
```

With the smoke retained, the expected plan is 450 missing watermarked records
in nine parallel batches of 50 and zero missing null records. Stop the app if
the planner selects a different model/KV namespace or unexpectedly requests
null generation. Retry the identical command without `--fresh`; prompt records
are checkpointed individually.

Expected production outputs use this stem:

```text
outputs/online_map_sweep_n3072_to768_step16_t3_eta0.10_prompts500_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1
```

Prefix evaluation must be descending with `stop_after_first_below=true`. Stop
immediately at the first point that is not strictly above 90%: `450/500`
(exactly 90.0%) is a failure, and no shorter coarse prefixes may be evaluated.
Preserve any empirical monotonicity warnings.

## Stage 3: one-token refinement with no new generation

Read from the production manifest:

- `COARSE_PASS_N = summary.last_passing_n_descending`
- `COARSE_FAIL_N = summary.next_shorter_n`

If both exist and differ by 16, substitute them below:

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --generation-model-size 8B \
  --source-n COARSE_PASS_N \
  --floor-n COARSE_FAIL_N \
  --step 1 \
  --target-map-tpr 0.90 \
  --num-prompts 500 \
  --t 3 \
  --eta 0.10 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 50 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --final-audit \
  --no-pin-floor-cache
```

Require `wm_missing=0`, `null_missing=0`, cache mode `prefix_from_longer`, no
model worker, zero measured GPU seconds, and zero model token positions. The
exact descending `n90` is the last passing prefix immediately before the first
failure.

Boundary contingencies:

- If `n=3072` fails, report the censored result `n90 > 3072`; do not extend
  generation without a new estimate and approval.
- If `n=768` still passes, reuse the same `n=3072` cache in a second detection
  sweep down to `n=128`; do not regenerate text.
- If a monotonicity warning occurs, preserve and report it. Do not evaluate
  past the first failure merely to search for a later empirical rebound.

## Stage 4: authoritative cache-only audit and publication

Substitute the exact selected length for `SELECTED_N`:

```bash
modal run modal_online_run.py::main \
  --generation-model-size 8B \
  --n SELECTED_N \
  --num-prompts 500 \
  --t 3 \
  --eta 0.10 \
  --fpr 0.001 \
  --experiment-seed 12345 \
  --batch 50 \
  --max-containers 1 \
  --gpu H100 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static \
  --detection-shard-size 50 \
  --detection-max-containers 10 \
  --cache-only \
  --csv-out online_causal_results_summary.csv
```

The hard cache-only guard must pass, no model worker may launch, and the audit
must contain exactly 500 watermarked and 500 null decisions for MAP, entropy,
and naive detection. Its MAP count must reproduce the refinement result. Save
the observed FPRs, source/null cache provenance, prompt-level results, and
artifact fingerprint.

Do **not** point `--csv-out` at `hoeffding_results_summary.csv`; its schema is
different. After review, normalize exactly one selected-boundary row into the
combined table with `PRC Construction=online_causal_prc`, both model fields
`Qwen3-8B-Base`, all three TPR/FPR counts, both log-Hoeffding fields `skipped`,
and `online_causal_prc_v1` plus crossing/cache/fingerprint/app provenance in
`Notes`. Do not publish the smoke, coarse grid points, or replay duplicates as
authoritative rows.

## Fixed-versus-online reuse

The `n=3072` source cache covers the existing fixed-reference lengths
`n={768,1382,1625}`. Later, score those exact online prefixes cache-only and
preserve prompt IDs and decisions for paired bootstrap and McNemar analysis;
do not generate new fixed text.
