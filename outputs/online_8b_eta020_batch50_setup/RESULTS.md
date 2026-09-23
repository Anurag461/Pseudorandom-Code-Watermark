# Online PRC eta=0.20: first native 8B batch, N=50

Completed September 20, 2026 (Pacific time) on `redetection`. One H200 successfully replayed **50 cached watermarked completions at T=14336 in one batch**. This result covers prompt IDs 0–49, not the full 500-prompt cohort.

| Detector | Scoring | Detected | TPR |
|---|---|---:|---:|
| Qwen3-8B-Base | MAP | 48/50 | **96.0%** |
| Qwen3-8B-Base | Entropy | 47/50 | **94.0%** |

Target FPR was 0.001, one-shot. Null N=0; empirical FPR was not evaluated. New generations=0; 0.6B replays=0. These are completion-only results under `completion_only_raw_abstain_v1`, using raw saved tokens, coordinate-one abstention and the frozen source key/partition. The model/tokenizer revision, source fingerprints and sampling provenance are recorded in [setup.json](setup.json).

## Runtime, memory and actual spending

| Stage | Resources | Worker method time | Modal charge (USD) |
|---|---|---:|---:|
| Prepare | 4 CPU cores, 16 GiB | 52.90 seconds | 0.00563915 |
| Primary replay | One H200, batch 50; 4 CPU cores, 64 GiB host RAM | 4538.16 seconds | 6.61555999 |
| Score | 4 CPU cores, 8 GiB | 13.27 seconds | 0.00140663 |
| **Total** | All workers stopped | | **6.62260577** |

The detector replay itself took **4430.90 seconds (73.85 minutes)**. The GPU stage, including loading and I/O, took 75.64 minutes. Primary peak allocated GPU memory was **134.05 GB**, peak reserved memory 134.08 GB, and PyTorch-reported total memory 150.11 GB: about 89.3% allocated. Batch 50 completed without OOM or a paid retry. This establishes feasibility for this exact native 8B/T14336 workload; no larger batch was tested.

The total was **$2.87739423 below the approved $9.50 allowance**. Costs come from Modal billing, including billed host resources and startup/shutdown. All three apps are stopped with zero tasks; see [billing_final.json](billing_final.json) and [apps_final.json](apps_final.json). Combined with the earlier eta=0.15 expansion and prefix scoring ($28.68611404), the two tasks total $35.30871981. The eta=0.20 batch had its own explicit $9.50 approval.

## Saved results and reuse

- Primary trace: `prc-completion-only:completion_only_raw_abstain_v1/integrated/2b5c9fe9995098df6db69e1a/batches/000000/trace.pt`, also downloaded locally. SHA256: `fbce4378a69cc1e4cbb569812c9444c233665632e4ddfb31822df25591d3c507`.
- Full scores and summary: the same integrated root, `full.json` and `summary.json`. [cache_index.json](cache_index.json) points to the cached inputs, artifact/partition, trace and scores.
- The existing [results CSV](../redetection/redetection_results_summary.csv) has one new N=50 row; all 617 previous rows were preserved (618 total). [csv_verification.json](csv_verification.json) records the exact added row.
- Prepared inputs committed in `145f5961c9144619f1d0e1bde805a8b77e066cc8`; primary replay evidence committed before scoring in `f4dacdc9d0d4c3e7e89d08c80b9be2ce06aab38b`; scores/CSV committed in `f75e418ce0501b24a118d98738b18dbbc2caf09e`.

No new generation, null work, 0.6B replay, independent reference pass, benchmark or paid retry ran. Extra reproducibility source archives remain local. Unrelated working-tree changes were preserved; nothing was pushed. A local Git whitespace check was corrected before GPU dispatch, as recorded in `local_commit_recovery.json`; the old `pipeline_error.json` refers only to that resolved local issue.

The full trace can support later shorter-prefix CPU scoring and inclusion in a larger matched cohort without replaying these 50 records. Any additional paid work requires its own estimate and explicit approval.
