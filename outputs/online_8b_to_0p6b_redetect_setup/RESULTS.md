Completed on branch `redetection`. Each saved 8B cohort was replayed once with Qwen3-0.6B-Base; all shorter results reuse the same completion-only traces. No new generations, nulls, reference passes, benchmarks or retries ran.

| eta | Longest T | N | 8B MAP | 0.6B MAP | 8B entropy | 0.6B entropy |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1280 | 500 | 96.2% | 92.6% | 90.0% | 86.8% |
| 0.10 | 3072 | 500 | 94.2% | 90.8% | 91.0% | 88.0% |
| 0.15 | 6144 | 100 | 94.0% | 92.0% | 90.0% | 84.0% |

N=500 for eta=0.05/T=1280 and eta=0.10/T=3072; N=100 for eta=0.15/T=6144. The eta=0.15 T=4096 cohort and reporting point were excluded as requested. The prior T6144 8B primary detection and scores are complete; its independent reference was previously stopped and was not resumed.

There are 124 reported points: eta=0.05 has 33 (1280 down to 848 by 16, plus 768/512/400/256/128), eta=0.10 has 90 (3072 down to 1648 by 16), eta=0.15 has only 6144. The original adaptive MAP<90% stop rule was disabled for this matched comparison; every frozen requested length was scored on cloud CPU using existing prefix_scores, with no further GPU replay.

Protocol: completion_only_raw_abstain_v1; raw completion IDs, no prompt or BOS/EOT prefix, first-coordinate abstention, MAP and entropy, BF16 model, TF32 disabled, static KV cache, float32 recovered probabilities and float64 CPU scores. Existing online keys and partition, seed 12345, t=3, row rate 99/100 with startup clamping, and target FPR 0.001 using one_shot per length were retained. Generation-time probabilities were not used for detection.

Pinned 0.6B model/tokenizer: Qwen/Qwen3-0.6B-Base revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd. Saved 8B comparison checkpoint: Qwen/Qwen3-8B-Base revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4. Their cached tokenizer.json files were verified byte-identical, and both vocabularies have 151936 rows. Exact weights, source manifests, tokens, keys and partition were verified by the approved preparation stages.

**Empirical FPR was not evaluated.** Null generation/replay/scoring count 0 and separate cost $0; generation cost $0. FPR CSV fields are skipped, not zero-percent estimates.

| Case / stage | Worker time (longer worker for parallel replay) | Actual provider cost |
|---|---:|---:|
| eta005_T1280_N500 / prepare | 72.28 s | $0.00704887 |
| eta005_T1280_N500 / replay | 353.27 s | $0.28172066 |
| eta005_T1280_N500 / score | 14.24 s | $0.00154735 |
| eta010_T3072_N500 / prepare | 42.07 s | $0.00500347 |
| eta010_T3072_N500 / replay | 1691.28 s | $1.33296191 |
| eta010_T3072_N500 / score | 17.62 s | $0.00182866 |
| eta015_T6144_N100 / prepare | 16.74 s | $0.00229098 |
| eta015_T6144_N100 / replay | 672.84 s | $1.06740567 |
| eta015_T6144_N100 / score | 9.59 s | $0.00098461 |

**Total actual cost: $2.70079218**, compared with the approved $4.30 conservative allowance. Estimated remaining balance: **$2.89**. Charges include GPU/CPU/memory before workspace credits. Remaining balance derives from the user's approximate starting balance and recorded provider usage. [Billing evidence](billing_final.json) retains all nine app IDs and charges. All nine apps are stopped with zero tasks.

The revised two-GPU replay plus final CPU scoring cost **$1.06839028**, below its $1.79 conservative estimate. The two GPU workers overlapped for 671.12 seconds (11.19 minutes).

GPU work used A100 80GB workers: the eta=.05 and eta=.10 cohorts ran sequential batches (4 × 125 at T=1280 and 5 × 100 at T=3072). Following the user’s scheduling correction and explicit revised approval, eta=.15 used two simultaneous workers, each processing a different batch of 50 at T=6144. This is nine paid stages/apps and ten paid worker executions in total. The [parallel timing evidence](parallel_execution.json) records actual overlap and total worker time. Each primary batch was committed to Modal immediately, then downloaded and narrowly committed to git before its cloud CPU scoring stage. Each cohort cache_index.json records saved tokens, original key artifacts, manifests, primary traces, score reports, checksums and timing evidence. Extra source snapshots remain local; no additional reproducibility archive was uploaded.

[eta=.05 cache index](cases/eta005_T1280_N500/cache_index.json) · [eta=.10 cache index](cases/eta010_T3072_N500/cache_index.json) · [eta=.15 cache index](cases/eta015_T6144_N100/cache_index.json)

The [main CSV](../redetection/redetection_results_summary.csv) has 429 rows: all 305 prior rows preserved plus 124 additions. N is explicit. The [paired comparison CSV](paired_comparison.csv) lists the original 8B and new 0.6B TPRs and differences in percentage points. Existing 8B scores are comparison references, not historical prompted 0.6B values. The main CSV old-TPR fields remain unavailable.

The [initial approved setup](PLAN.md), immutable setup.json, [initial approval](approval.json), [revised parallel setup](PARALLEL_PROPOSAL.md), [parallel approval](parallel_eta015_approval.json), [execution progress](progress.json), source hashes and [metadata verification](metadata_verification.json) preserve the full audit trail. All model execution and scoring ran in Modal. Local work only collected saved reports and metadata, verified transfer checksums and formatted results. Unrelated working-tree changes were preserved; nothing was pushed.

| eta | T | N | 8B MAP | 0.6B MAP | Δ MAP (pp) | 8B entropy | 0.6B entropy | Δ entropy (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 1280 | 500 | 96.2% | 92.6% | -3.6 | 90.0% | 86.8% | -3.2 |
| 0.05 | 1264 | 500 | 96.2% | 92.6% | -3.6 | 89.8% | 86.2% | -3.6 |
| 0.05 | 1248 | 500 | 96.0% | 92.2% | -3.8 | 88.8% | 85.4% | -3.4 |
| 0.05 | 1232 | 500 | 95.8% | 91.6% | -4.2 | 88.6% | 85.0% | -3.6 |
| 0.05 | 1216 | 500 | 95.8% | 91.4% | -4.4 | 88.6% | 85.0% | -3.6 |
| 0.05 | 1200 | 500 | 95.6% | 91.2% | -4.4 | 88.8% | 85.0% | -3.8 |
| 0.05 | 1184 | 500 | 95.6% | 91.0% | -4.6 | 88.4% | 83.8% | -4.6 |
| 0.05 | 1168 | 500 | 95.4% | 90.8% | -4.6 | 88.0% | 83.4% | -4.6 |
| 0.05 | 1152 | 500 | 95.2% | 90.6% | -4.6 | 87.8% | 83.2% | -4.6 |
| 0.05 | 1136 | 500 | 95.0% | 90.0% | -5.0 | 87.4% | 82.8% | -4.6 |
| 0.05 | 1120 | 500 | 94.8% | 90.2% | -4.6 | 88.2% | 83.4% | -4.8 |
| 0.05 | 1104 | 500 | 94.8% | 90.2% | -4.6 | 86.8% | 82.0% | -4.8 |
| 0.05 | 1088 | 500 | 94.4% | 90.2% | -4.2 | 86.8% | 81.4% | -5.4 |
| 0.05 | 1072 | 500 | 93.8% | 89.2% | -4.6 | 85.8% | 81.4% | -4.4 |
| 0.05 | 1056 | 500 | 93.8% | 89.2% | -4.6 | 86.0% | 80.6% | -5.4 |
| 0.05 | 1040 | 500 | 93.6% | 88.8% | -4.8 | 85.4% | 80.4% | -5.0 |
| 0.05 | 1024 | 500 | 93.2% | 88.0% | -5.2 | 84.4% | 79.2% | -5.2 |
| 0.05 | 1008 | 500 | 92.8% | 87.8% | -5.0 | 84.2% | 79.4% | -4.8 |
| 0.05 | 992 | 500 | 93.0% | 87.8% | -5.2 | 84.2% | 79.8% | -4.4 |
| 0.05 | 976 | 500 | 93.0% | 87.0% | -6.0 | 83.2% | 78.6% | -4.6 |
| 0.05 | 960 | 500 | 92.8% | 86.6% | -6.2 | 82.6% | 77.8% | -4.8 |
| 0.05 | 944 | 500 | 91.8% | 85.6% | -6.2 | 82.8% | 77.0% | -5.8 |
| 0.05 | 928 | 500 | 91.4% | 85.2% | -6.2 | 81.6% | 77.2% | -4.4 |
| 0.05 | 912 | 500 | 91.4% | 85.6% | -5.8 | 81.6% | 77.0% | -4.6 |
| 0.05 | 896 | 500 | 90.8% | 84.6% | -6.2 | 82.0% | 77.2% | -4.8 |
| 0.05 | 880 | 500 | 90.4% | 83.8% | -6.6 | 80.8% | 76.2% | -4.6 |
| 0.05 | 864 | 500 | 90.0% | 83.2% | -6.8 | 79.4% | 76.0% | -3.4 |
| 0.05 | 848 | 500 | 89.6% | 82.2% | -7.4 | 78.4% | 75.6% | -2.8 |
| 0.05 | 768 | 500 | 87.2% | 79.4% | -7.8 | 75.8% | 69.6% | -6.2 |
| 0.05 | 512 | 500 | 69.0% | 58.0% | -11.0 | 51.8% | 45.2% | -6.6 |
| 0.05 | 400 | 500 | 54.8% | 45.2% | -9.6 | 41.6% | 34.8% | -6.8 |
| 0.05 | 256 | 500 | 30.2% | 25.2% | -5.0 | 20.6% | 16.8% | -3.8 |
| 0.05 | 128 | 500 | 9.2% | 7.4% | -1.8 | 7.0% | 6.0% | -1.0 |
| 0.10 | 3072 | 500 | 94.2% | 90.8% | -3.4 | 91.0% | 88.0% | -3.0 |
| 0.10 | 3056 | 500 | 94.2% | 90.8% | -3.4 | 91.0% | 87.6% | -3.4 |
| 0.10 | 3040 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.2% | -3.6 |
| 0.10 | 3024 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 86.8% | -4.0 |
| 0.10 | 3008 | 500 | 94.2% | 90.8% | -3.4 | 91.0% | 86.6% | -4.4 |
| 0.10 | 2992 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.0% | -3.8 |
| 0.10 | 2976 | 500 | 94.2% | 90.6% | -3.6 | 90.8% | 87.0% | -3.8 |
| 0.10 | 2960 | 500 | 94.2% | 90.6% | -3.6 | 90.8% | 87.2% | -3.6 |
| 0.10 | 2944 | 500 | 94.2% | 90.6% | -3.6 | 91.0% | 87.2% | -3.8 |
| 0.10 | 2928 | 500 | 94.2% | 91.0% | -3.2 | 90.8% | 87.6% | -3.2 |
| 0.10 | 2912 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.4% | -3.4 |
| 0.10 | 2896 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.2% | -3.6 |
| 0.10 | 2880 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.2% | -3.6 |
| 0.10 | 2864 | 500 | 94.2% | 90.8% | -3.4 | 90.8% | 87.2% | -3.6 |
| 0.10 | 2848 | 500 | 94.2% | 90.8% | -3.4 | 90.6% | 87.2% | -3.4 |
| 0.10 | 2832 | 500 | 94.2% | 90.4% | -3.8 | 90.6% | 87.0% | -3.6 |
| 0.10 | 2816 | 500 | 94.2% | 90.4% | -3.8 | 90.6% | 87.0% | -3.6 |
| 0.10 | 2800 | 500 | 94.2% | 90.2% | -4.0 | 90.8% | 87.0% | -3.8 |
| 0.10 | 2784 | 500 | 94.0% | 90.2% | -3.8 | 90.6% | 87.2% | -3.4 |
| 0.10 | 2768 | 500 | 94.0% | 90.2% | -3.8 | 90.8% | 87.2% | -3.6 |
| 0.10 | 2752 | 500 | 93.8% | 90.0% | -3.8 | 90.6% | 87.4% | -3.2 |
| 0.10 | 2736 | 500 | 94.0% | 90.0% | -4.0 | 90.6% | 87.4% | -3.2 |
| 0.10 | 2720 | 500 | 94.2% | 90.4% | -3.8 | 90.6% | 87.2% | -3.4 |
| 0.10 | 2704 | 500 | 94.0% | 90.2% | -3.8 | 90.6% | 86.8% | -3.8 |
| 0.10 | 2688 | 500 | 94.2% | 90.2% | -4.0 | 89.8% | 86.8% | -3.0 |
| 0.10 | 2672 | 500 | 94.0% | 90.4% | -3.6 | 90.2% | 86.4% | -3.8 |
| 0.10 | 2656 | 500 | 93.8% | 90.6% | -3.2 | 90.2% | 86.8% | -3.4 |
| 0.10 | 2640 | 500 | 93.8% | 90.4% | -3.4 | 90.4% | 86.4% | -4.0 |
| 0.10 | 2624 | 500 | 93.6% | 89.8% | -3.8 | 90.2% | 86.6% | -3.6 |
| 0.10 | 2608 | 500 | 93.6% | 90.0% | -3.6 | 89.8% | 86.6% | -3.2 |
| 0.10 | 2592 | 500 | 93.6% | 90.2% | -3.4 | 90.0% | 86.8% | -3.2 |
| 0.10 | 2576 | 500 | 93.6% | 90.4% | -3.2 | 90.0% | 86.6% | -3.4 |
| 0.10 | 2560 | 500 | 93.6% | 90.0% | -3.6 | 89.4% | 86.2% | -3.2 |
| 0.10 | 2544 | 500 | 93.4% | 89.8% | -3.6 | 89.6% | 86.2% | -3.4 |
| 0.10 | 2528 | 500 | 93.4% | 89.8% | -3.6 | 89.6% | 86.0% | -3.6 |
| 0.10 | 2512 | 500 | 93.4% | 89.6% | -3.8 | 90.0% | 86.2% | -3.8 |
| 0.10 | 2496 | 500 | 93.4% | 90.0% | -3.4 | 89.8% | 86.2% | -3.6 |
| 0.10 | 2480 | 500 | 93.4% | 90.0% | -3.4 | 89.6% | 85.6% | -4.0 |
| 0.10 | 2464 | 500 | 93.4% | 90.0% | -3.4 | 89.4% | 85.8% | -3.6 |
| 0.10 | 2448 | 500 | 93.0% | 89.8% | -3.2 | 89.6% | 85.8% | -3.8 |
| 0.10 | 2432 | 500 | 93.2% | 89.6% | -3.6 | 90.0% | 86.0% | -4.0 |
| 0.10 | 2416 | 500 | 93.4% | 89.4% | -4.0 | 89.4% | 86.0% | -3.4 |
| 0.10 | 2400 | 500 | 93.6% | 89.6% | -4.0 | 89.4% | 85.8% | -3.6 |
| 0.10 | 2384 | 500 | 93.4% | 89.4% | -4.0 | 89.4% | 85.6% | -3.8 |
| 0.10 | 2368 | 500 | 93.6% | 89.4% | -4.2 | 89.4% | 85.8% | -3.6 |
| 0.10 | 2352 | 500 | 93.6% | 89.4% | -4.2 | 89.6% | 85.8% | -3.8 |
| 0.10 | 2336 | 500 | 93.6% | 89.4% | -4.2 | 89.6% | 85.6% | -4.0 |
| 0.10 | 2320 | 500 | 93.6% | 89.2% | -4.4 | 89.6% | 85.6% | -4.0 |
| 0.10 | 2304 | 500 | 93.0% | 89.2% | -3.8 | 89.8% | 85.6% | -4.2 |
| 0.10 | 2288 | 500 | 93.2% | 89.0% | -4.2 | 89.6% | 85.2% | -4.4 |
| 0.10 | 2272 | 500 | 93.0% | 89.2% | -3.8 | 89.8% | 85.4% | -4.4 |
| 0.10 | 2256 | 500 | 93.6% | 89.4% | -4.2 | 89.4% | 85.0% | -4.4 |
| 0.10 | 2240 | 500 | 93.2% | 89.4% | -3.8 | 89.4% | 85.2% | -4.2 |
| 0.10 | 2224 | 500 | 93.4% | 89.2% | -4.2 | 89.6% | 85.4% | -4.2 |
| 0.10 | 2208 | 500 | 93.2% | 89.0% | -4.2 | 89.2% | 84.4% | -4.8 |
| 0.10 | 2192 | 500 | 93.0% | 89.0% | -4.0 | 89.6% | 84.8% | -4.8 |
| 0.10 | 2176 | 500 | 93.0% | 89.0% | -4.0 | 89.4% | 84.8% | -4.6 |
| 0.10 | 2160 | 500 | 92.8% | 89.2% | -3.6 | 89.2% | 84.2% | -5.0 |
| 0.10 | 2144 | 500 | 93.2% | 89.4% | -3.8 | 89.2% | 84.4% | -4.8 |
| 0.10 | 2128 | 500 | 93.0% | 89.4% | -3.6 | 89.4% | 84.8% | -4.6 |
| 0.10 | 2112 | 500 | 92.4% | 89.2% | -3.2 | 89.2% | 84.2% | -5.0 |
| 0.10 | 2096 | 500 | 92.8% | 89.4% | -3.4 | 89.4% | 84.2% | -5.2 |
| 0.10 | 2080 | 500 | 92.8% | 89.0% | -3.8 | 88.6% | 83.8% | -4.8 |
| 0.10 | 2064 | 500 | 92.6% | 88.6% | -4.0 | 88.0% | 83.6% | -4.4 |
| 0.10 | 2048 | 500 | 92.8% | 89.0% | -3.8 | 88.0% | 83.4% | -4.6 |
| 0.10 | 2032 | 500 | 92.4% | 88.8% | -3.6 | 88.0% | 83.4% | -4.6 |
| 0.10 | 2016 | 500 | 92.8% | 88.4% | -4.4 | 87.6% | 83.4% | -4.2 |
| 0.10 | 2000 | 500 | 92.2% | 88.4% | -3.8 | 87.8% | 82.8% | -5.0 |
| 0.10 | 1984 | 500 | 92.0% | 88.0% | -4.0 | 87.8% | 82.8% | -5.0 |
| 0.10 | 1968 | 500 | 92.0% | 88.0% | -4.0 | 87.4% | 82.0% | -5.4 |
| 0.10 | 1952 | 500 | 91.6% | 87.6% | -4.0 | 86.8% | 81.4% | -5.4 |
| 0.10 | 1936 | 500 | 91.8% | 87.4% | -4.4 | 86.6% | 81.8% | -4.8 |
| 0.10 | 1920 | 500 | 91.6% | 87.6% | -4.0 | 86.6% | 81.8% | -4.8 |
| 0.10 | 1904 | 500 | 91.8% | 87.2% | -4.6 | 86.4% | 81.6% | -4.8 |
| 0.10 | 1888 | 500 | 91.8% | 87.2% | -4.6 | 86.0% | 81.8% | -4.2 |
| 0.10 | 1872 | 500 | 91.4% | 87.4% | -4.0 | 85.8% | 81.2% | -4.6 |
| 0.10 | 1856 | 500 | 91.6% | 87.4% | -4.2 | 85.6% | 80.6% | -5.0 |
| 0.10 | 1840 | 500 | 91.4% | 87.2% | -4.2 | 85.6% | 80.0% | -5.6 |
| 0.10 | 1824 | 500 | 91.2% | 86.8% | -4.4 | 84.8% | 80.0% | -4.8 |
| 0.10 | 1808 | 500 | 90.8% | 87.0% | -3.8 | 84.6% | 80.0% | -4.6 |
| 0.10 | 1792 | 500 | 90.8% | 86.8% | -4.0 | 84.6% | 80.0% | -4.6 |
| 0.10 | 1776 | 500 | 90.8% | 87.0% | -3.8 | 85.0% | 79.4% | -5.6 |
| 0.10 | 1760 | 500 | 91.4% | 87.2% | -4.2 | 84.8% | 79.2% | -5.6 |
| 0.10 | 1744 | 500 | 91.2% | 86.6% | -4.6 | 84.6% | 79.2% | -5.4 |
| 0.10 | 1728 | 500 | 90.8% | 86.2% | -4.6 | 84.4% | 78.4% | -6.0 |
| 0.10 | 1712 | 500 | 90.6% | 85.8% | -4.8 | 83.4% | 78.0% | -5.4 |
| 0.10 | 1696 | 500 | 90.4% | 85.2% | -5.2 | 83.8% | 77.6% | -6.2 |
| 0.10 | 1680 | 500 | 90.2% | 84.8% | -5.4 | 82.8% | 76.8% | -6.0 |
| 0.10 | 1664 | 500 | 90.2% | 84.6% | -5.6 | 81.6% | 76.6% | -5.0 |
| 0.10 | 1648 | 500 | 89.8% | 85.0% | -4.8 | 81.0% | 76.2% | -4.8 |
| 0.15 | 6144 | 100 | 94.0% | 92.0% | -2.0 | 90.0% | 84.0% | -6.0 |
