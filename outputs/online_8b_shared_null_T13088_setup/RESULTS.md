# Completed: native 8B FPR, η=.15/.20

All 500 saved nulls were replayed once through T13088 on ten H200s, batch 50. Both original keys were scored at all 173 frozen reporting points. Completed September 22, 2026 at 20:30:31 UTC (21:30:31 London), about 70 minutes after launch. All cloud apps are stopped.

At the longest evaluated length for each key:

| Key | Length | Posterior FPR | Entropy FPR | Reused posterior TPR | Reused entropy TPR |
|---|---:|---:|---:|---:|---:|
| η=.15 | 6144 | 0/500 (0%) | 0/500 (0%) | 465/500 (93.0%) | 436/500 (87.2%) |
| η=.20 | 13088 | 1/500 (0.2%) | 0/500 (0%) | 451/500 (90.2%) | 430/500 (86.0%) |

Across all 173 points, posterior empirical FPR ranges from 0/500 to 1/500; entropy FPR is 0/500 throughout. These are empirical sample rates at the original one-shot target FPR .001. The same 500 nulls are shared by both keys and all prefixes.

Modal reports **$55.74380810 total**, checked September 22 at 21:17 UTC, leaving **$9.25619190 of the $65 allocation**. Preparation $0.02199037; replay $55.71858245; scoring $0.00323528. [Billing evidence](billing_final.json).

All 78 downloaded files passed checksum checks. Exact ordered coverage of 500 nulls and 500 saved watermarked records per key, the identical ten shared trace hashes, all 173 CSV rows, and unchanged watermarked counts were verified from saved metadata. No additional model execution or numerical scoring was used for these local checks. [Verification evidence](completion_verification.json).

The local connection briefly logged a DNS/heartbeat error during replay. All ten workers completed and their outputs were collected; no paid retry or additional validation pass was launched.

Protocol: original saved completion token IDs, no prompt or special-token prefix, coordinate 1 abstention, pinned Qwen3-8B-Base BF16, TF32 disabled, static KV, float32 probability traces and float64 CPU scoring. No null generation or extension. FPR above T13088, including T14336, remains outside this run.

- [η=.15 per-record report](cache/results/online_8b_shared_null_T13088_N500_v1/eta015/full.json)
- [η=.20 per-record report](cache/results/online_8b_shared_null_T13088_N500_v1/eta020/full.json)
- [Redetection results CSV](../redetection/redetection_results_summary.csv)
- [Execution status](progress.json)
