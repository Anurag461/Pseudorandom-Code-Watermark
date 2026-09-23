# Online 8B eta=0.20: completed shorter-prefix results

Completed on September 21, 2026, after explicit approval. Added **156 rows** to `outputs/redetection/redetection_results_summary.csv`: **T=14320 down to 11840 in steps of 16**, all with **N=500, null N=0**. All 619 existing result rows remain unchanged; the CSV now has 775 rows. The existing T=14336 result was reused without recomputing it.

| T | Native 8B MAP TPR | Native 8B entropy TPR | Role |
|---:|---:|---:|---|
| 14336 | 452/500 (90.4%) | 433/500 (86.6%) | Existing full-length result, retained |
| 11856 | 450/500 (90.0%) | 429/500 (85.8%) | Last visited point at or above 90% MAP |
| 11840 | 449/500 (89.8%) | 428/500 (85.6%) | First visited point strictly below 90% MAP |

The sweep stopped at the first below-90% MAP point on the descending 16-token grid. This is not a token-by-token boundary and does not assume TPR is monotone.

All scores use the ten saved completion-only native 8B trace shards for the same 500 T=14336 watermarked completions, with the original key and partition. Protocol: `completion_only_raw_abstain_v1`; raw completion token IDs, first-coordinate abstention, MAP and entropy; eta=0.20, t=3, seed=12345, row rate 99/100, one-shot target FPR=0.001. Generation-time probabilities were not substituted for detection traces.

**One CPU worker, 4 physical cores, 8 GiB RAM; 28.98 worker seconds.** Modal app lifetime was about 51 seconds. There were no new generations, GPU replays, 0.6B passes, nulls, independent validation jobs or retries. No experiment scoring ran on the laptop. Empirical FPR was not evaluated.

**Actual incremental cost: $0.00295399** (CPU $0.00220733; memory $0.00074666), below the approved $0.10 allowance. The earlier native N=500 work cost $66.48566712; including this sweep, that cohort totals **$66.48862111**. These are usage charges, not an account credit balance.

Modal app `ap-SpCXfDtlsiN1sBP6ETtXEN` is stopped with zero tasks. Billing and status are recorded in `billing_final.json` and `apps_final.json`.

Per-prompt scores, prepared provenance, source hashes and timing are saved both locally under `cache/results/online_8b_eta020_N500_prefixes_v1/` and in Modal volume `prc-completion-only` at the same run root. The source T=14336 scores and primary traces were reused. `collected.json` records hashes of the collected results; `csv_verification.json` records the append and preservation checks. The CSV retains the existing schema and its H200/batch50 source-trace provenance; the new prefix scoring itself was CPU-only.

Additional execution-source archives remain local. No additional archive was uploaded and nothing was pushed.
