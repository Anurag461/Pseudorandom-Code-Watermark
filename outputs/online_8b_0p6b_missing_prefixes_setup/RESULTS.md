# 0.6B prefix coverage completed

Completed September 24, 2026 on `main`. Added 157 posterior/entropy result rows: eta .20 at T11840–14320 in steps of 16, plus eta .15 at T4096. All native 8B lengths now have matching 0.6B results for the full N=500 cohorts. Eta .05 and .10 already had complete coverage. All 949 earlier rows were preserved exactly; the CSV now contains 1,106 rows.

The priority eta .20/T11856 result is posterior **430/500 (86.0%)** and entropy **411/500 (82.2%)**. At eta .15/T4096, posterior is **423/500 (84.6%)** and entropy **401/500 (80.2%)**.

At each eta's shortest recorded length where native 8B posterior TPR reaches at least 90%:

| Eta | T | Posterior: 8B → 0.6B | Drop (pp) | Entropy: 8B → 0.6B | Drop (pp) |
|---|---:|---|---:|---|---:|
| .05 | 864 | 90.0% → 83.2% | 6.8 | 79.4% → 76.0% | 3.4 |
| .10 | 1664 | 90.2% → 84.6% | 5.6 | 81.6% → 76.6% | 5.0 |
| .15 | 4672 | 90.0% → 85.8% | 4.2 | 85.8% → 80.8% | 5.0 |
| .20 | 11856 | 90.0% → 86.0% | 4.0 | 85.8% → 82.2% | 3.6 |

Posterior mean drop: **5.15 pp**; maximum: **6.8 pp**. Entropy mean drop: **4.25 pp**; maximum: **5.0 pp**, evaluated at those same posterior-defined lengths. These are descriptive averages across four eta settings, not uncertainty estimates or the lengths where entropy detection itself reaches 90%. Counts and calculation definitions are in `comparison_at_native_90.json`.

The comparison holds watermarked texts, online keys, partitions and nominal one-shot target FPR=0.001 fixed. It uses the saved completion-only probabilities and coordinate-one abstention. No model execution, GPU replay, generation, null generation, independent validation pass, benchmark or retry ran. Empirical posterior/entropy FPR is unmeasured for these new rows.

Two new rows also reuse existing standard-naive counts, without rescoring: eta .15/T4096 has 335/500 TPR and 0/500 FPR; eta .20/T13088 has 385/500 TPR and 0/500 FPR. Standard naive uses all partition bits including coordinate one and is independent of the detector model. The historical N=500 null cohorts remain explicitly identified separately from the unmeasured posterior/entropy FPR. See `naive_reuse.json` and the earlier naive source ledger. The user's decision to omit naive FPR above T13088 is preserved for all new rows.

Execution used one 4-core/8 GiB CPU worker, reading twenty cached batches of 50. Worker time was **35.8157 seconds**. All downloaded result hashes, cohort identities, reported decision counts, CSV preservation and final length coverage were checked using saved metadata. The app stopped with zero running tasks.

Provider-reported cost: **$0.00351665**, below the approved $0.10 allowance. Remaining from the $35 budget after the earlier $25.43796444 run and this scoring job: **$9.55851891**. The separate native-null budget was not pooled. Read-only billing evidence is in `billing_final.json`.

Modal app: https://modal.com/apps/new-prc-watermark/main/ap-8cdcmK0wsqAHqzZmad6Lw5

The frozen `setup.json` remains the original proposal so its approval hash stays valid. `approval.json`, `execution_manifest.json`, `collected.json`, `csv_verification.json` and the cached reports preserve execution and result provenance. Raw token/probability tensors remain on the existing Modal volume.
