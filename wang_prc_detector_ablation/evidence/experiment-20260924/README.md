# Full Qwen3-8B-Base detector comparison — September 24, 2026

Status: **complete**. All seven approved calls finished successfully; app stopped with zero tasks.

Run commit: `3b15d2e81271f3b0194a89fd109d888c961dacb2` on `cryptoanalysis-redetection`.
Modal app: `ap-JL2Co4bAS6fq92w1TTRCyF`.
Configuration: `5825fb18b52e035dfb352bae`.
Cloud cache: `prc-data:wang_prc_detector_ablation/qwen3_8b_base/5825fb18b52e035dfb352bae`.

## Results

**MAP improves the matched-FPR statistic comparison at T=1.0 and T=1.2.**
At T=1.0 the gain is real in this sample but absolute detection remains low.
At T=1.4 both matched detectors reach 100%, so the large primary-method gap
there reflects the operating thresholds; it does not show a further matched-TPR advantage.

TPR intervals below are the crossed group-by-prompt bootstrap 95% intervals.
Each temperature has 160 watermarked texts and 80 held-out null texts crossed
with 256 evaluation keys (20480 pairings, not 20480 independent texts).

### Primary: each method with its specified threshold

| T | Wang published TPR [95% CI] | MAP standard TPR [95% CI] | Wang null FPR | MAP null FPR |
| --- | --- | --- | --- | --- |
| 1.0 | 0.00% [0.00, 0.00] | 5.00% [0.62, 11.27] | 0.0000% | 0.0000% |
| 1.2 | 0.00% [0.00, 0.00] | 97.50% [92.50, 100.00] | 0.0000% | 0.0049% |
| 1.4 | 33.75% [21.88, 45.62] | 100.00% [100.00, 100.00] | 0.0000% | 0.0098% |
| 1.6 | 98.12% [92.50, 100.00] | 100.00% [100.00, 100.00] | 0.0000% | 0.0049% |
| 1.8 | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.0000% | 0.0000% |

### Secondary: globally calibrated at target FPR 0.1%

| T | Hard TPR [95% CI] | MAP TPR [95% CI] | Hard null FPR | MAP null FPR |
| --- | --- | --- | --- | --- |
| 1.0 | 0.00% [0.00, 0.00] | 11.25% [3.75, 20.62] | 0.1172% | 0.0293% |
| 1.2 | 89.38% [79.38, 96.87] | 98.12% [92.50, 100.00] | 0.1416% | 0.0684% |
| 1.4 | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.0684% | 0.1465% |
| 1.6 | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.1025% | 0.0781% |
| 1.8 | 100.00% [100.00, 100.00] | 100.00% [100.00, 100.00] | 0.0879% | 0.0928% |

Paired matched-FPR gains (MAP minus hard):

- T=1.0: +11.25 percentage points, 95% CI +3.75 to +20.63.
- T=1.2: +8.75 points, 95% CI +2.50 to +16.88.
- T=1.4: 0 points; both methods detect 160/160.

MAP also has lower realized FPR than hard at T=1.0 and T=1.2, so these gains
are not explained by a higher held-out false-positive rate. AUC is 0.685 versus
0.545 at T=1.0 and 0.998 versus 0.971 at T=1.2 (MAP versus hard). Both reach
AUC 1.000 at T=1.4. These results support a scoring-information advantage at
the two lowest temperatures, while the very large primary gain at T=1.2 also
includes Wang's conservative-threshold effect.

At T=1.0, mean hierarchical bit entropy is 0.131 bits and latent-bit agreement
is 53.7%, compared with 0.658 bits and 75.1% at T=1.2. These diagnostics are
consistent with the much weaker watermark evidence in the lowest-temperature samples.

Zero observed FPR and degenerate bootstrap intervals do not establish zero
population FPR. Matched thresholds have the same calibration target, not identical
held-out FPR. Raw summary CSV retains harmless floating-point endpoint roundoff;
the displayed intervals are rounded to percentages.

### Saved outputs

- [All detector results and FPR intervals](results_summary.csv)
- [Paired differences](paired_differences.csv)
- [Primary TPR figure](tpr_vs_temperature.pdf) and [matched-FPR TPR figure](tpr_matched_fpr_vs_temperature.pdf)
- [Score versus entropy](score_vs_entropy.pdf) and [ROC curves](roc_low_temperature.pdf)
- [Mechanism diagnostics](mechanism_summary.csv)

All 35 output files (114498926 bytes) were downloaded with size checks and SHA-256
transfer records in `download_manifest.json`. The full 205600-row per-example
CSV and cached null-score arrays are in the ignored local directory
`wang_prc_detector_ablation/downloaded_results/5825fb18b52e035dfb352bae/`.
The complete trace/key/codeword cache remains in the cloud location below.

The run took **41.28 minutes** end to end; CPU scoring took **27.15 minutes**.
Runtime-derived cost is **about $4.48**, or **$4.77** including earlier checks.
Approximately **$30.23** remains from the $35 experiment budget.
Provider billing has not posted for this app; these are labeled estimates based
on measured runtimes, not a final invoice. See `billing_reconciliation.json`.

## Experiment

Pinned `Qwen/Qwen3-8B-Base`, revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; BF16 generation/replay, raw Base
prompts, reasoning off. Ten Wang keys, sixteen prompts and five temperatures;
160 WM and 160 null completions per temperature, each exactly 1024 tokens.
Primary replay uses only preceding completion IDs and zero soft evidence for the
first generated token. All token IDs and generation/replay probabilities remain
in the cloud cache for CPU-only detector reruns.

Primary comparison: published Wang hard threshold versus our unchanged standard
posterior threshold. Secondary comparison: the same hard/posterior statistics
with globally calibrated, whole-tie, inclusive thresholds at nominal FPR 0.001.
The primary and matched-FPR comparisons must be interpreted separately.

## Frozen calibration

At `2026-09-24T00:34:25.592262+00:00`, the designated 400 calibration nulls crossed
with 256 independent calibration keys fixed the following cutoffs:

| Statistic | Decision | Calibration detections / pairings |
| --- | --- | --- |
| Hard, matched FPR | H <= 8551 | 102 / 102400 |
| Posterior, matched FPR | Z >= 3.0229717052487137 and V > 0 | 102 / 102400 |

The Wang published cutoff remains approximately 7232.824913450762. The standard
posterior rule remains S >= sqrt(2 V log 1000), rejecting V=0. Null evaluation
uses 400 held-out completions and 256 disjoint independent keys. No watermarked
sample sets either matched cutoff. Threshold content SHA-256:
`f5ebc433c526bb7d8fed9ce0592f8e4a342c007cf0ec2f30056bece9a7d26fcc`.

Intervals use 2000 crossed group-by-prompt bootstrap replicates. Cross-key pairs
are not independent observations; FPR inference is conditional on the fixed
held-out key pool, and matched-FPR TPR intervals condition on frozen thresholds.

## Reproduction and numerical limits

This is the requested controlled Wang-channel experiment on Qwen3-8B-Base,
not a replication of the DeepSeek model or Figure 5. The official complete
DeepSeek T=1.8 artifact validated the adapted hard detector on all 160 individual
counts/decisions, with 159/160 detections in both original and adapted code.

The earlier 2% BF16 cached/uncached check failed; the later controls showed exact
static/concatenating cache equality and close FP32 cached/uncached agreement on
one completion at prefix lengths 1, 4, 8 and 16. Maximum observed BF16 TV was
4.04%; FP32 TV was about 0.00115%. This evidence supported proceeding with the
unchanged BF16 implementation, but does not establish batch-80 or long-context
numerical equivalence. No additional short T=1.8 smoke was run. The user approved
proceeding directly to this full experiment using the completed controls.

## Execution and costs

See `approval.json`, `quote.json`, `launch.json`, per-temperature `timing_*.json`
and `gpu_containers.json`. The seven approved calls were one 4-core/16-GiB CPU
preparation, five H100 workers (4 cores/64 GiB host RAM each; batch 80), and one
8-core/16-GiB CPU scoring/reporting call. No retry or additional compute run.

Exact launch command, from the committed isolated worktree:

```bash
MODAL_PROFILE=new-prc-watermark PYTHONUNBUFFERED=1 \
python -m wang_prc_detector_ablation.launch launch --stage experiment \
  --approval wang_prc_detector_ablation/approvals/experiment-20260924.json
```

Full paid-run quote: $12–$25. Preparation took 145.05 seconds. The five completed
GPU allocations total 3284.82 seconds (54.75 GPU-minutes, 0.91245 GPU-hours),
corresponding to approximately $4.242 including allocated CPU/RAM at quoted rates.
Final CPU timings and the pending provider-billing reconciliation are attached. The run has stopped; no further work is scheduled.
