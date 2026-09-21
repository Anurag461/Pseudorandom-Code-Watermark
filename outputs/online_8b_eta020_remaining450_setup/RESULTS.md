# Online 8B eta=0.20, T=14336: completed native detection, N=500

Completed September 21, 2026 on `redetection`. Nine H200 workers replayed the remaining 450 watermarked completions in parallel, batch size 50. The completed first 50 traces and scores were reused without replay or rescoring.

| Native 8B detector scoring | Detected | TPR |
|---|---:|---:|
| MAP | 452/500 | **90.4%** |
| Entropy | 433/500 | **86.6%** |

This is the complete prompt-ID cohort 0–499. The first-50 saved scores are unchanged. The completed cloud report contains ten trace-shard hashes, covering all 500 prompts exactly once.

Settings match the successful first batch: `Qwen/Qwen3-8B-Base`, model/tokenizer revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16, TF32 disabled, static KV, `completion_only_raw_abstain_v1`, raw completion token IDs, first-coordinate abstention, MAP and entropy. Online PRC eta=0.20, T=n=14336, t=3, seed=12345, row rate 99/100, r=14193, target FPR=0.001 with one-shot policy. The original generation key, partition and token IDs were reused; no generation-time probabilities were substituted for detector traces.

Null N=0; empirical FPR was not evaluated. No new generations, 0.6B detection, reference passes, benchmarks or paid retries were run.

## Actual runtime and spending

| Stage | Resources | Actual Modal cost (USD) |
|---|---|---:|
| Prepare 450 cached inputs | 4 CPU cores, 16 GiB | 0.02666520 |
| Nine native replay batches | **9 H200s concurrently**, batch 50; 4 CPU cores and 64 GiB host RAM each | 59.83484877 |
| Score new 450 and merge existing first-50 scores | 4 CPU cores, 8 GiB | 0.00154738 |
| **Additional total** | | **59.86306135** |
| Earlier completed first-50 batch | | 6.62260577 |
| **Full eta=0.20 native N=500 total** | | **66.48566712** |

The additional total is **$15.13693865 below the approved $75 allowance**. Charges include billed host resources and startup/shutdown. All three apps are stopped with zero tasks; see [billing_final.json](billing_final.json) and [apps_final.json](apps_final.json).

Elapsed time from the first CPU app's creation to the final app stopping was **83 minutes 58 seconds**. The parallel GPU stage occupied **78 minutes 12 seconds**. Primary inference per batch took 73.59–73.74 minutes. All nine batches peaked at 134.05 GB allocated GPU memory, with 150.11 GB reported by PyTorch. No OOM or restart occurred. [gpu_inference_telemetry.json](gpu_inference_telemetry.json) records all nine H200s at 99–100% utilization; the midway snapshot confirms the same nine containers were still active.

## Saved outputs and reuse

- [Results CSV](../redetection/redetection_results_summary.csv): one new row with **N=500**, null N=0 explicit. All 618 previous rows, including N=50, were preserved; there are now 619 result rows.
- New native trace component: `prc-completion-only:completion_only_raw_abstain_v1/integrated/ba61a52bdf2b3ae02c8256bd`, containing nine batches.
- Reused first-50 component: `prc-completion-only:completion_only_raw_abstain_v1/integrated/2b5c9fe9995098df6db69e1a`.
- Aggregate scores and manifests: `prc-completion-only:online_8b_eta020_T14336_remaining450_v1/combined/8B`, with local copies under this folder's `cache/results` directory.
- [cache_index.json](cache_index.json) records input/artifact/partition references, all trace hashes, report files and both local cache roots. Future shorter-prefix scoring can reuse the full traces without GPU replay. The aggregate manifest is for aggregation only; use its `component_prepared` manifests to access the actual trace batches.

Prepared inputs were saved and committed in `ee179b9dc1c8b8611e75478ce8dc6c5e1847682b`. All primary outputs were saved and committed in `7067d1eebaf4b65e268bf8dd453b2d0b38f29eb0` before CPU scoring started. Scores and the CSV row were committed in `737cfd1d1ce201736a4e592f4ef09707eab2bf40`.

Unrelated working-tree changes were preserved. Extra reproducibility source archives remain local, and nothing was pushed. Additional paid work requires a separate estimate and explicit approval.
