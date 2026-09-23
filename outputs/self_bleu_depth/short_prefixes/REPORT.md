# SynthID short-prefix detection: depths 2, 10 and 30

**Depth 10 provides a substantial short-prefix detection advantage over depth 2 on this cohort. Depth 30 provides no additional observed TPR gain over depth 10 at these cutoffs.** At 64 tokens, depths 10/30 each detect 100/100 responses versus 45/100 for depth 2. At 128, the counts are 100/100 versus 82/100. All three depths detect 100/100 by 256 tokens.

All scores use saved native-fallback-on generations, the fixed per-depth keys, completion-only inputs, the official context-repetition mask and the unchanged weighted-normal detector at nominal p < .001. No generation, model inference, GPU jobs or new null sampling was performed. The proposed depth-20 run was cancelled before launch; its unfinished runner edits were reverted.

## Watermarked detection

Each depth has 100 saved responses from the same 50 prompts and two seeds. Values below are TPR percentages with 95% paired prompt-bootstrap intervals.

| Depth | 64 tokens | 128 tokens | 256 tokens |
|---:|---:|---:|---:|
| 2 | 45.0 [35.0, 55.0] | 82.0 [75.0, 89.0] | 100.0 [100.0, 100.0] |
| 10 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |
| 30 | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] | 100.0 [100.0, 100.0] |

At this sample size, each percentage numerically equals the detected count out of 100. All-success bootstrap intervals collapse to [100, 100]; they do not establish perfect population detection.

## Direct paired depth differences

Differences are percentage points, left depth minus right depth. Each per-prompt detection fraction averages that prompt’s two response decisions before differencing and resampling.

| Contrast | 64 tokens: difference [95% interval] | 128 tokens: difference [95% interval] | 256 tokens: difference [95% interval] |
|---|---:|---:|---:|
| Depth 10 minus depth 2 | 55.0 [45.0, 65.0] | 18.0 [11.0, 25.0] | 0.0 [0.0, 0.0] |
| Depth 30 minus depth 2 | 55.0 [45.0, 65.0] | 18.0 [11.0, 25.0] | 0.0 [0.0, 0.0] |
| Depth 30 minus depth 10 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

At 64 tokens, each deeper setting detects all 45 depth-2 successes plus the other 55 paired response slots; at 128, it adds 18 to depth 2’s 82 successes. These contrasts preserve prompt/seed pairing and are not differences of marginal confidence endpoints.

## Saved nulls

The same saved null text is scored with each depth’s exact keys and weights. Keep the cohorts separate: the pilot contains 100 responses clustered within 50 prompts, while the historical corpus has one response for each of 500 prompts, including the pilot prompts.

| Depth | Tokens | Pilot false positives | Pilot FPR %, bootstrap [95% interval] | Historical false positives |
|---:|---:|---:|---:|---:|
| 2 | 64 | 0/100 | 0.0 [0.0, 0.0] | 1/500 |
| 2 | 128 | 0/100 | 0.0 [0.0, 0.0] | 1/500 |
| 2 | 256 | 1/100 | 1.0 [0.0, 3.0] | 1/500 |
| 10 | 64 | 0/100 | 0.0 [0.0, 0.0] | 0/500 |
| 10 | 128 | 0/100 | 0.0 [0.0, 0.0] | 1/500 |
| 10 | 256 | 0/100 | 0.0 [0.0, 0.0] | 1/500 |
| 30 | 64 | 0/100 | 0.0 [0.0, 0.0] | 1/500 |
| 30 | 128 | 0/100 | 0.0 [0.0, 0.0] | 0/500 |
| 30 | 256 | 0/100 | 0.0 [0.0, 0.0] | 0/500 |

The nonzero counts are depth 2’s 1/100 pilot false positive at 256 tokens, and historical counts of 1/500 at all three lengths for depth 2, 128/256 for depth 10, and 64 for depth 30. These are not matched empirical FPR results. The cohorts are too small to establish a calibrated 0.1% tail, and zero-count bootstrap intervals do not imply zero population FPR. No threshold was tuned on these nulls.

## Interpretation alongside the completed diversity comparison

The earlier 400/1,024-token comparison saturated detection for all three depths, hiding the short-prefix tradeoff. Depth 2 retained Self-BLEU close to ordinary sampling but now shows lower detection at 64/128 tokens. Depth 10 reaches the observed detection ceiling already at 64 tokens. Depth 30 had higher Self-BLEU than depth 10 at 400/1,024, while this follow-up finds no additional observed detection gain at 64/128/256. This supports depth 10 as a useful comparison setting on the current cohort; it does not establish equivalence of depths 10/30, performance below 64 tokens, robustness to attacks, or a Bayesian-detector result.

The prior diversity endpoints remain unchanged and are linked in the [depth-2/30 report](../depth2_30_v1/REPORT.md). This follow-up evaluates detection only; it does not add short-prefix Self-BLEU measurements.

## Protocol and verification

- Saved Qwen3-8B-Base generations: original prompts 0–49, seeds 12345/67890, native fallback on, temperature/top-p 1, and the same fixed predeclared per-depth keys. No model was loaded for this analysis.
- At length T, the official detector extracts eligible evidence from raw completion positions 3 through T−1. Every depth receives its own key list and depth-aware layer weights. Prompts and generation diagnostics are excluded.
- Compute TPR per prompt as the mean of its two binary decisions; resample 50 prompt clusters jointly for all depths, preserving both seed slots. Use 2,000 draws with seed 20260918 and percentile endpoints 2.5%/97.5%. Intervals are marginal, exploratory and unadjusted for multiplicity.
- Pilot-null FPR intervals use the same prompt-cluster draws. Paired pilot-null FPR contrasts and response-level discordance counts are retained in the machine-readable summary. Historical nulls are reported as counts without pooling them with pilot nulls.
- All 6,300 prefix scores were verified using the actual truncated completion tensor: evidence values and eligibility exactly match prefix extraction from the saved full response, ruling out dependence on later tokens.
- Independent weighted-normal calculations from direct-prefix evidence reproduced all 6,300 scores/decisions. The 1,400 previously saved depth-10 scores at 128/256 tokens were reproduced.
- An independent bootstrap implementation using prompt multiplicity weights reproduced all 36 absolute/paired intervals. Saved artifacts contain 900 paired prompt-level metric records and 6,300 response-level detector records.
- Bootstrap draw hash: `5ab115a4b5c632f81fd04e9cf02b5d4f67cf68dd60cf622759f99a04fb767c77`.

[Summary](summary.json) · [Prompt-level metrics](prompt_metrics.json) · [Frozen analysis manifest and source hashes](manifest.json) · [Reproduction script](analyze.py).

Reproduce locally with the pinned analysis environment and the saved source artifacts:

```sh
HF_HOME=/path/to/writable/hf-cache NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python outputs/self_bleu_depth/short_prefixes/analyze.py
```

Incremental Modal cost: **$0**. Cumulative planning charge remains **$7.71132** including previous allowances. No depth-20 job was launched.
