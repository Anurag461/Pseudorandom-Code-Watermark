# Completed: 8B→0.6B TPR, η=.20, T14336

All 500 saved watermarked completions were replayed and scored on the `redetection` branch. Completed September 22, 2026 at 17:28:41 UTC (18:28:41 London). All cloud apps are stopped.

| Detector weighting | Detected | TPR |
|---|---:|---:|
| Posterior (map) | 437/500 | 87.4% |
| Entropy | 418/500 | 83.6% |

H200, batch 50: first 50 followed by nine concurrent workers for the remaining 450. The first batch took 31.9 minutes including worker setup; peak allocated GPU memory was 59.6%. The measured budget gate passed, so the A100 fallback was not needed. Preparation through scoring took about 69 minutes elapsed.

Modal reported **$25.43796444 total**, checked September 22 at 18:52 UTC, leaving **$9.56203556 of the $35 budget**. Breakdown: preparation $0.01656486, first 50 $2.59398703, remaining 450 $22.82361461, CPU scoring $0.00379794. See [billing evidence](billing_final.json).

Pinned Qwen3-0.6B-Base BF16, raw saved Qwen3-8B-Base completion tokens, original key/partition, no prompt or special-token prefix, coordinate 1 abstention, static KV cache, TF32 disabled. Target FPR .001 with the original one-shot threshold. Empirical FPR and naive scoring were outside this TPR-only run.

All artifact checksums, ten saved trace shards, exact ordered coverage 0–499, and the single final CSV row were verified by local file/metadata inspection. No additional paid validation, retry, or generation was run. Saved native 8B comparison: posterior 452/500 (90.4%), entropy 433/500 (86.6%).

- [Per-record scores and counts](cache/results/completion_only_raw_abstain_v1/integrated/0263dfa535b7b6b4a3ee39a2/full.json)
- [Native 8B comparison](native_comparison.json)
- [Redetection results CSV](../redetection/redetection_results_summary.csv)
- [Execution status](progress.json)
