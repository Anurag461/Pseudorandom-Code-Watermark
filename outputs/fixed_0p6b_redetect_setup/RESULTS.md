# Completed fixed 0.6B redetection

All 18 main settings and both seed replicates completed on September 19, 2026. Each setting contains 500 watermarked and 500 null candidates. Eta=.20, n8192 remains deferred. Previously completed eta=.05 n400/n448 were not rerun.

Results use raw completion tokens, BF16 Qwen3-0.6B-Base, coordinate 1 score zero, t=3, target FPR=.001, and the original fixed-PRC key, partition, parity checks and threshold policy. The old columns are the exact original prompted scores for the same candidates. MAP denotes posterior-mean weighting. Naive redetection was not run.

Across the 18 main settings, the unweighted mean change in TPR is -1.62 percentage points for MAP and -1.00 points for entropy weighting. MAP decreases in 14 settings, is unchanged in three, and increases by 0.2 points at eta=.15 n2048. The largest main-setting MAP decrease is 5.8 points at eta=.10 n400 (55.2% to 49.4%).

At the earlier approximately 90% MAP points, eta=.05 n416 moves from 91.2% to 88.2%, and eta=.10 n768 from 91.6% to 89.6%. Among tested lengths including prior completed runs, the first lengths with at least 90% corrected MAP TPR are n448 (eta=.05; 90.0%), n1024 (eta=.10; 95.2%), n1504 (eta=.15; 91.0%), and n4096 (eta=.20; 96.8%). These are tested settings, not interpolated thresholds.

## Main results

FP columns give detections out of 500 null candidates. All TPR columns are percentages.

| eta | n | Old MAP | New MAP | Change (pp) | Old entropy | New entropy | Change (pp) | MAP FP | Entropy FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 256 | 66.8% | 62.4% | -4.4 | 47.4% | 43.8% | -3.6 | 0/500 | 0/500 |
| 0.05 | 416 | 91.2% | 88.2% | -3.0 | 76.8% | 73.6% | -3.2 | 0/500 | 0/500 |
| 0.05 | 512 | 95.8% | 94.4% | -1.4 | 84.6% | 84.2% | -0.4 | 0/500 | 0/500 |
| 0.05 | 1024 | 99.2% | 99.0% | -0.2 | 98.0% | 97.8% | -0.2 | 0/500 | 1/500 |
| 0.05 | 2048 | 99.6% | 99.6% | +0.0 | 99.4% | 99.4% | +0.0 | 0/500 | 0/500 |
| 0.10 | 256 | 30.2% | 27.0% | -3.2 | 19.8% | 18.8% | -1.0 | 0/500 | 0/500 |
| 0.10 | 400 | 55.2% | 49.4% | -5.8 | 38.4% | 36.8% | -1.6 | 0/500 | 0/500 |
| 0.10 | 512 | 68.2% | 67.0% | -1.2 | 50.2% | 49.4% | -0.8 | 0/500 | 0/500 |
| 0.10 | 768 | 91.6% | 89.6% | -2.0 | 79.0% | 77.0% | -2.0 | 0/500 | 0/500 |
| 0.10 | 1024 | 96.2% | 95.2% | -1.0 | 89.6% | 89.2% | -0.4 | 0/500 | 0/500 |
| 0.15 | 256 | 10.0% | 9.4% | -0.6 | 8.4% | 7.4% | -1.0 | 0/500 | 0/500 |
| 0.15 | 400 | 18.8% | 17.6% | -1.2 | 9.8% | 9.8% | +0.0 | 0/500 | 1/500 |
| 0.15 | 512 | 32.0% | 29.6% | -2.4 | 22.4% | 22.0% | -0.4 | 0/500 | 0/500 |
| 0.15 | 1024 | 77.0% | 74.6% | -2.4 | 61.8% | 60.8% | -1.0 | 0/500 | 0/500 |
| 0.15 | 1504 | 91.0% | 91.0% | +0.0 | 84.8% | 83.4% | -1.4 | 0/500 | 0/500 |
| 0.15 | 2048 | 97.4% | 97.6% | +0.2 | 95.2% | 95.2% | +0.0 | 0/500 | 0/500 |
| 0.20 | 2048 | 72.8% | 72.2% | -0.6 | 60.2% | 59.4% | -0.8 | 0/500 | 0/500 |
| 0.20 | 4096 | 96.8% | 96.8% | +0.0 | 95.2% | 95.0% | -0.2 | 1/500 | 1/500 |

MAP has one false positive at eta=.20 n4096. Entropy has one false positive each at eta=.05 n1024, eta=.15 n400, and eta=.20 n4096. Each nonzero observed rate is 0.2%; every other setting/weight has 0/500. Some null cohorts are reused across settings, so these are reported per setting rather than as independent pooled observations.

## Seed replicates

Both are eta=.05, n256. These remain separate from the 18-setting main sweep.

| Seed | Old MAP | New MAP | Change (pp) | Old entropy | New entropy | Change (pp) | MAP FP | Entropy FP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 54321 | 67.8% | 63.4% | -4.4 | 47.4% | 44.8% | -2.6 | 0/500 | 0/500 |
| 67890 | 63.4% | 57.0% | -6.4 | 45.2% | 41.8% | -3.4 | 0/500 | 0/500 |

## Cache and verification

All 20,000 candidate records match the frozen token hashes and ordering. All 40,000 MAP/entropy decisions reproduce their report counts and the CSV. All 184 trace files, 40 full/summary reports, and 20 scoring artifacts were read back from the persistent Modal volume and checksum-verified. Every run has one complete representative validation batch. The cache contains 19,596,000 newly recovered probabilities.

The three previously present CSV rows are preserved; the file now contains 23 rows. Detailed reports, traces, and scoring artifacts are also cached locally under `outputs/redetection/.archive/runs/`. Exact completion inputs remain in each persistent Modal run cache. The setup manifests identify every original candidate and source hash.

| GPU | Settings including replicates | Batch sizes | Maximum live allocation |
|---|---:|---|---:|
| A10G | 11 | 100 | 7.93 GB |
| A100-80GB | 9 | 100, 125 | 51.64 GB |

[Results CSV](../redetection/redetection_results_summary.csv) · [Machine-readable results](results_summary.json) · [Completion verification](completion_verification.json) · [Frozen setup index](index.json) · [Launch record](launch.json)

Large tensors are retained in Modal volume `prc-completion-only` and the local ignored archive; Git stores the summaries, checksums, manifests, setup code, and cache locations. The completion bundle in `results_cache_bundle.json` preserves this report and the final CSV in Modal.
