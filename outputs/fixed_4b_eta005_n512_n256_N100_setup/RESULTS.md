Completed on September 20, 2026, on branch `redetection`.

| n=T | Detector | MAP TPR | Entropy TPR | Watermarked N | Null N |
|---:|---|---:|---:|---:|---:|
| 512 | 4B | 83/100 (83%) | 67/100 (67%) | 100 | 0 |
| 512 | 0.6B | 78/100 (78%) | 64/100 (64%) | 100 | 0 |
| 256 | 4B | 46/100 (46%) | 27/100 (27%) | 100 | 0 |
| 256 | 0.6B | 35/100 (35%) | 25/100 (25%) | 100 | 0 |

Generated each cohort once with Qwen3-4B-Base and reused its exact saved token IDs for both detectors. The two cohorts use the same first 100 original 50-token prompts and each length's archived fixed key/partition. Fixed PRC eta=0.05, t=3, seed12345, one block, r=507 for n512 and r=253 for n256, analytical target FPR=0.001.

**No empirical FPR evaluation:** zero null generations/replays, separate null cost $0. No benchmark, reference replay, extra validation pass, regeneration or retry ran.

Pinned checkpoint/tokenizer revisions are Qwen/Qwen3-4B-Base `906bfd4b4dc7f14ee4320094d8b41684abff8539` and Qwen/Qwen3-0.6B-Base `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Both tokenizer.json files are byte-identical and both model output vocabularies have 151936 rows. Cached weights and metadata were verified in the approved CPU preparation stages.

Protocol: `completion_only_raw_abstain_v1`, raw completions without prompt or special prefix, first-coordinate abstention, MAP and entropy. BF16 inference, TF32 disabled, concat generation KV cache and static replay cache, float32 recovered probabilities and existing float64 cloud CPU scoring. Generation uses temperature1, unrestricted top-k/top-p, the original PRC bucket sampler, forced T and no EOS stop. Generation-time probabilities are retained solely as provenance; each detector recovered its own completion-only trace. Legacy generation does not seed every RNG, so actual codewords, tokens and available RNG states are cached; seed-only regeneration is not claimed.

| Stage | Hardware / RAM | n512 worker time | n512 actual cost | n256 worker time | n256 actual cost |
|---|---|---:|---:|---:|---:|
| prepare | 4 CPU / 16 GiB | 37.12 s | $0.00387691 | 35.45 s | $0.00370068 |
| generate | H100, batch100 / 64 GiB | 77.57 s | $0.11111212 | 71.22 s | $0.10393370 |
| freeze | 4 CPU / 16 GiB | 12.37 s | $0.00229085 | 12.08 s | $0.00229093 |
| replay_4b | H100, batch100 / 64 GiB | 86.05 s | $0.12752642 | 53.74 s | $0.07881648 |
| replay_0p6b | A100 80GB, batch100 / 16 GiB | 36.42 s | $0.03286784 | 37.72 s | $0.03443380 |
| score | 4 CPU / 8 GiB | 11.18 s | $0.00112528 | 12.50 s | $0.00140668 |

**Actual provider-reported total: $0.50338169**, versus the approved $2.25 conservative allowance. Cohort totals: n512 $0.27879942; n256 $0.22458227. Estimated remaining balance: **$5.59**. Costs include CPU, memory and GPU usage and are before workspace credits. The remaining balance derives from the user's approximate starting balance and provider usage. [Billing evidence](billing_final.json) retains all twelve app IDs and itemized charges.

All twelve stages completed with one worker at a time. The final read-only app listing confirms all twelve apps are stopped with zero active tasks. Both primary generations and all four detector traces were saved and narrowly committed before their respective CPU scoring stages. Full score reports, manifests, prompts, original keys/partitions, primary traces, codeword/RNG evidence and timing logs are saved in the cohort cache indexes: [n512](../fixed_4b_eta005_n512_N100_setup/cache_index.json) and [n256](../fixed_4b_eta005_n256_N100_setup/cache_index.json). Additional execution source snapshots remain local; no extra reproducibility archive was uploaded.

The [existing CSV](../redetection/redetection_results_summary.csv) now has 305 rows: all 301 prior rows and four additions. Each new row records N=100 explicitly and /100 denominators; FPR fields are skipped. Local checks read saved metadata and verified transfer checksums without model execution or experiment scoring. See [metadata verification](metadata_verification.json).

The reviewed setup manifests remain immutable, including their prelaunch status. [Approval](approval.json) and [execution progress](progress.json) record authorization and completion separately. Runtime source hashes are authoritative. Unrelated working-tree changes were preserved; nothing was pushed.
