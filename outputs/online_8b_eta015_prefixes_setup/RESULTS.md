# Online 8B eta=0.15 shorter-prefix results, N=500

Added 186 rows to the existing results CSV: 93 lengths per detector, T=6128 down to T=4656 in steps of 16. All 431 existing rows, including the T=6144 native and 0.6B rows, were preserved. The CSV now has 617 data rows.

The descending scan included the first native 8B MAP TPR strictly below 90%. Exactly 90% did not stop the scan. The 0.6B detector was scored at the same 93 lengths.

| T | Native MAP | Native entropy | 0.6B MAP | 0.6B entropy |
|---|---:|---:|---:|---:|
| 4688 | 451/500 (90.2%) | 429/500 (85.8%) | 430/500 (86.0%) | 406/500 (81.2%) |
| 4672 | 450/500 (90.0%) | 429/500 (85.8%) | 429/500 (85.8%) | 404/500 (80.8%) |
| 4656 | 448/500 (89.6%) | 429/500 (85.8%) | 428/500 (85.6%) | 403/500 (80.6%) |

Reused the same 500 saved T=6144 completions and all 15 completion-only trace shards (five native, ten small-model). No generation, model replay, GPU, null scoring, reference pass, benchmark, or retry. Existing T=6144 scores were not recomputed. Empirical FPR remains unmeasured, null N=0.

Scoring used the existing prepared_weights/adaptive_scores and prefix_scores routines, with raw completion tokens, first-coordinate abstention, MAP and entropy, the original online key and partition, t=3 and one-shot target FPR=0.001. Frozen successful source versions were used; the stopped eta=0.20 memory-limit patch was excluded.

One cloud worker used four CPU cores and 8 GiB RAM. The worker took 30.61 seconds; the app was active from 22:01:11 to 22:02:08 PDT on September 20, 2026. It is stopped with zero tasks. Final Modal-reported cost: **$0.00309468**, below the approved $0.10 allowance. Together with the completed eta=0.15 expansion, reported spending is **$28.68611404**.

Per-prompt scores, source/trace hashes, prepared manifests, timing, the user approval and billing evidence are cached locally and on the results volume. Extra source snapshots remain local. The CSV append used the existing repository writer and was checked for exact preservation of all original rows. No experiment scoring ran on the laptop.

See [results CSV](../redetection/redetection_results_summary.csv), [summary](results_summary.json), [CSV verification](csv_verification.json), and [billing](billing_final.json).
