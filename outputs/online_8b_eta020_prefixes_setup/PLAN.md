# Online 8B eta=0.20, N=500: shorter-prefix scoring

Prepared and completed on `redetection`, September 21, 2026. The user explicitly approved this CPU-only workload and its $0.10 allowance. **Completed once; actual charge $0.00295399.** See `RESULTS.md` and `approval.json`. The proposal below records the approved scope.

Reuse all ten completed native 8B trace shards for the same 500 watermarked T=14336 completions (prompt IDs 0–499). The existing full-length result is MAP 452/500 (90.4%) and entropy 433/500 (86.6%). Its scores and CSV row will be retained without recomputation or duplication.

Start at **T=14320**, then **14304, 14288, ...**, in the established 16-token steps. Score MAP and entropy at each visited prefix. Stop immediately after the first native MAP result **strictly below 450/500 (90%)**, and include that point in the CSV. Exactly 450/500 continues the sweep. This is a boundary on the 16-token grid, not a token-by-token boundary or an assumption that TPR is monotone.

Use the established `completion_only_raw_abstain_v1` protocol: raw completion IDs, coordinate-one abstention, original online key and partition, eta=0.20, t=3, seed=12345, row rate 99/100 and one-shot target FPR=0.001. The cached traces come from `Qwen/Qwen3-8B-Base` revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16 with TF32 disabled. The CPU job loads saved inputs/traces and computes prefix statistics; it does not load or execute the model.

**Generation=0, GPU replay=0, GPU count=0, 0.6B detector work=0. Null N=0 and null cost=$0; empirical FPR is not evaluated.** No benchmark, independent validation pass or automatic retry is included. Generation-time probabilities are not used as detection traces.

One cloud CPU worker, **4 physical cores and 8 GiB RAM**, will use the existing `prepared_weights` and `adaptive_scores` routines. The successful eta=0.15 prefix wrapper is adapted only for this cohort, maximum length and native-only detector scope. Native/scientific imports still precede the NumPy compatibility aliases. Per-prompt scores, exact source hashes, timing and billing evidence will be saved.

Expected time: **1–3 minutes**. Estimated charge: **$0.005–$0.02**. Requested conservative allowance: **$0.10**. The prior N=500 eta=0.15 sweep covered 93 prefix lengths for native 8B plus 0.6B in 30.61 worker seconds and cost $0.00309468. This proposal allows for longer traces and cache I/O without a new benchmark. A 600-second work deadline, 630-second function timeout and 60-second startup timeout bound this one attempt; retries=0 and a durable attempt marker prevents automatic duplication.

The completed eta=0.20 native N=500 work cost **$66.48566712**. Spending this entire additional allowance would bring it to **$66.58566712**. Billing exposes usage, not the available account credit balance; unused allowances from earlier runs are not treated as approval for this new CPU run.

The existing CSV contains 619 result rows. Append one row per visited lower length through the first below-90% point, with **N=500 and null N=0 explicit**, using the existing schema and append routine. Preserve every prior row. Save and narrowly commit the completed scores, CSV update and cost evidence on `redetection`; keep additional reproducibility archives local and do not push.

Source roots, component manifests and hashes are frozen in `setup.json`. Local preflight performed only AST, hash and scope-guard checks; no model execution or experiment scoring ran locally. The exact CPU-only workload and $0.10 allowance were approved before launch, per `AGENTS.md`; see `approval.json`.
