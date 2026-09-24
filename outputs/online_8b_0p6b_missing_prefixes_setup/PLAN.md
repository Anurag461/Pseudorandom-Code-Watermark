# Match 0.6B detection to the native 8B prefix grid

Prepared and completed September 24, 2026 on `main`, following the user's “go ahead” approval of the $0.10 cap. **Completed once for $0.00351665.** See `RESULTS.md`, `approval.json`, and `billing_final.json`. The proposal below records the approved scope.

Score the existing completion-only 0.6B probability traces for 500 watermarked completions per eta, using posterior and entropy detection at target FPR 0.001 with the established one-shot threshold and coordinate-one abstention. Preserve all existing results.

| Eta | Missing lengths | New result rows |
|---|---|---:|
| .05 | None | 0 |
| .10 | None | 0 |
| .15 | T4096 | 1 |
| .20 | T11840 through T14320, step 16 | 156 |

Process eta .20 first, with T11856 first in its score list. Save its T11856 result before moving to eta .15. T11856 is the shortest recorded length where the native 8B posterior detector reaches 90% TPR. Use the full N=500 cohorts, with duplicate historical rows and smaller pilot subsets excluded from the coverage count.

All 20 required trace shards already have saved manifests and hashes: ten batches of 50 at T14336 for eta .20, and ten batches of 50 at T6144 for eta .15. The saved native and 0.6B reports have identical prompt IDs and full completion-token hashes for all 500 examples in each cohort. The completed eta .15 continuation runs previously checked preservation of the T4096 prefixes. No new model execution, probability replay, generation, GPU, null generation, benchmark, or separate validation pass is proposed. Empirical FPR remains unmeasured for these new rows.

One Modal CPU worker with **4 physical cores and 8 GiB RAM**, streaming the twenty existing batches of 50 sequentially. Reuse `online_prc_redetection.prefix_scores` and the existing cached-input checks. The scoring helper and detector implementation match the successful eta .15 prefix run byte for byte. The worker saves per-prompt decisions/statistics, counts, source hashes and timing. It has a 600-second work deadline, 630-second function timeout, and no automatic retries. Local and durable cloud attempt markers prevent accidental repeated dispatch.

**Expected time: 1–3 minutes. Estimated cost: $0.005–$0.02. Requested maximum allowance: $0.10.** The earlier eta .15 job scored 186 rows in 30.61 worker seconds for $0.00309468; the native eta .20 job scored 156 rows in 28.98 seconds for $0.00295399. These are existing measurements, not new benchmarks. The allowance includes the one invocation's cache checks, scoring, startup and shutdown. Any paid retry requires separate approval.

The read-only billing check at 2026-09-24 00:50 UTC confirms the previous eta .20 0.6B run cost $25.43796444, leaving **$9.56203556** of the $35 budget. Spending the entire $0.10 allowance would leave **$9.46203556**. The separately authorized native shared-null job cost $55.74380810 of its $65 cap; that separate allowance is not pooled here. The usage API does not expose available account credit.

After the approved run, append the 157 completed results to `outputs/redetection/redetection_results_summary.csv` on `main`, preserve the earlier naive backfill and skipped long-null FPR decisions, and recompute the four-eta comparison at each native detector's first recorded 90% TPR length. The original instruction was to keep changes local; the user subsequently authorized pushing these completed results on September 24, 2026.

The exact workload is frozen in `setup.json`; source hashes are in `execution_manifest.json`. `preflight.json` records local metadata, syntax and scope-guard checks only. `approval.template.json` is explicitly false and is not an authorization. Launch requires a separately recorded explicit user approval with the matching plan hash and $0.10 allowance.
