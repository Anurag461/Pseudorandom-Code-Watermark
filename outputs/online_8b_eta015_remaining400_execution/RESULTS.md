# Online 8B eta=0.15, T=6144 — completed N=500

Completed the remaining 400 watermarked prompts and reused the original 100 completions and both detectors’ existing scores. Each new completion was generated once and replayed once with each detector. No nulls, benchmarks, independent reference passes, or duplicate GPU runs were added.

| Detector | MAP TPR | Entropy TPR |
|---|---:|---:|
| 8B | 465/500 (93.0%) | 436/500 (87.2%) |
| 0.6B | 438/500 (87.6%) | 416/500 (83.2%) |

The 0.6B detector is 5.4 percentage points below native MAP detection and 4.0 points below native entropy detection.

**Null count: 0. Null generation/replay cost: $0. Empirical FPR was not evaluated.** The nominal target FPR is 0.001, using the one-shot policy.

| Cohort | Native MAP / entropy detections | 0.6B MAP / entropy detections |
|---|---:|---:|
| Reused first 100 | 94/100; 90/100 | 92/100; 84/100 |
| New 400 | 371/400; 346/400 | 346/400; 332/400 |

**Execution and settings.** Canonical prompt IDs 100–499 continued from their cached 4096-token prefixes to 6144: 819,200 newly sampled tokens. All 400 prefixes and full PRC bitstreams passed the cloud CPU checks. Evaluation used only T=6144. The generator used the established position-addressed poscdf-v1 sampler, temperature 1, unrestricted vocabulary, forced length, concat KV, seed 12345, t=3 and the 99/100 causal row schedule (r=6083). Detection used raw completion token IDs, static KV, first-coordinate abstention, MAP and entropy; generation-time probabilities were not used for detection. Models used BF16 with TF32 disabled.

| Model | Pinned revision |
|---|---|
| Qwen/Qwen3-8B-Base | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Qwen/Qwen3-0.6B-Base | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

Both checkpoints use the verified shared tokenizer SHA256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`. Partition SHA256: `503d1cf93958f0d765606ed3e25aa87a1a777cd08d8f44436b0c6ae9716aa184`. Exact model/configuration hashes and frozen execution source hashes are in [setup.json](setup.json).

Generation used four concurrent H200s, batch 100, four CPU cores and 64 GiB RAM each. Detection started all ten workers together: four H200s at batch 100 and six A100 80GB workers processing eight batches of 50. Two A100 workers each processed two distinct batches without reloading their model. Completed workers shut down. CPU preparation and scoring used the existing cloud routines.

Generation launch through scoring completion took 84.9 minutes, including input checks and downloads. The 400-record prefix/PRC check took 556.9 seconds, longer than the original estimate. All apps are stopped.

**Actual Modal-reported spending**, including both initial CPU preparation attempts:

| Work | Cost (USD) |
|---|---:|
| Initial preparation: log-handling failure | $0.00425320 |
| Preparation retry: provider preemption | $0.01762228 |
| 400 generation continuations | $13.79725035 |
| Prefix checks and detector manifests | $0.05022344 |
| Both primary detectors | $14.81090510 |
| CPU scoring and N=500 aggregation | $0.00276499 |
| **Total** | **$28.68301936** |

Rounded total: **$28.68**, approximately $6.43 below the $35.11 target. [Final billing confirmation](billing_final_confirmation.json) and [stopped-app evidence](apps_final.json) are saved.

The first CPU preparation failed because an open results-volume log prevented volume reload; the log was moved to /tmp. Modal preempted the approved retry after the source/checkpoint checks. The durable attempt marker prevented duplicated work on the provider restart. Remaining pilot checks were completed through read-only file hashes and the saved successful pilot audit; no further paid preparation run was launched. Generation and detection subsequently completed without retries.

**Saved outputs.** Two N=500 rows were appended to the [existing results CSV](../redetection/redetection_results_summary.csv), preserving all 429 original rows, including the N=100 pilot. [results_summary.json](results_summary.json) contains the compact results. [cache_index.json](cache_index.json) indexes 615 locally cached files and their cloud paths/hashes. Raw token/trace binaries are cached locally and committed to Modal volumes; Git retains manifests, hashes, reports, timing/cost evidence and CSV rows, following the repository’s binary ignores. Additional execution-source copies remain local; no extra archives were uploaded.

Generation was committed as `9f609f9d74e73dabd89ae80453cbf2a92272f710`, primary traces/manifests as `c8035f484ca63162611cc64d7c6f6b14caaa930a`, and scores/CSV as `b61d7312230da576ec2e104535d0e0a464de18cf` on `redetection`. No push was performed; unrelated working-tree changes were preserved.
