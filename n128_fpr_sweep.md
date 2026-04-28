# PRC watermark — n=128 FPR sweep

Saved p-traces from the n=128 calibration run (60 jobs: 30 prompts × {watermarked, unwatermarked}, 1024 tokens each = 8 codeword cycles per trace, multinomial sampling). FPR sweep recomputes the threshold from `null_mean + z * null_std` for several FPR targets without regenerating any traces.

## Setup

- Model: Qwen3-0.6B-Base, multinomial sampling
- Codeword length: n = 128
- Tokens per prompt: 1024 (8 codeword repeats)
- Calibration null: 2000 simulated nulls, codewords drawn uniformly and pushed through the same Bernoulli channel using the watermarked p-traces
- Threshold rule: `threshold = null_mean + norm.ppf(1 - FPR) * null_std`

## Statistic distributions

| | entropy fold | naive fold |
|---|---|---|
| null mean / std | 0.20 / 3.97 | 0.10 / 2.76 |
| watermarked range | [+4.60, +47.32] | [−0.24, +44.91] |
| watermarked mean | +21.73 | +13.22 |
| unwatermarked range | [−0.86, +1.37] | [−0.66, +1.65] |
| unwatermarked mean | +0.01 | +0.07 |

## FPR sweep

| FPR target | z | threshold (entropy) | TPR entropy | threshold (naive) | TPR naive | empirical FPR |
|---|---|---|---|---|---|---|
| 1e-9 | 5.998 | 24.02 | 11/30 | 16.68 | 10/30 | 0/30 |
| 1e-6 | 4.753 | 19.07 | 17/30 | 13.24 | 11/30 | 0/30 |
| 1e-3 | 3.090 | 12.47 | 26/30 | 8.64 | 20/30 | 0/30 |
| 1e-2 | 2.326 | 9.44 | 28/30 | 6.53 | 22/30 | 0/30 |
| 5e-2 | 1.645 | 6.73 | 28/30 | 4.65 | 25/30 | 0/30 |
| 1e-1 | 1.282 | 5.29 | 29/30 | 3.64 | 26/30 | 0/30 |

## Observations

1. **Entropy fold wins at every FPR target.** Naive's null is 1.4× tighter, but entropy's watermarked signal is 1.6× stronger; entropy nets +6 prompts at moderate FPR (1e-3 to 1e-2).
2. **Empirical FPR is 0/30 everywhere.** Unwatermarked stats top out at +1.65, far below even the most permissive threshold tested (3.64). The simulated null overestimates real unwatermarked variance because it's built from the watermarked p-traces, which have a different distribution than unconstrained multinomial sampling.
3. **Data is perfectly separable.** min(WM entropy)=+4.60 vs max(UW entropy)=+1.37; any threshold in (1.37, 4.60) gives TPR=30/30 and FPR=0/30 on this benchmark — but with only 30 unwatermarked samples you can't *prove* an FPR below ~3% (1/30).
4. **Outlier rescued by entropy weighting.** One prompt scores −0.24 with naive fold (below null mean) but +4.60 with entropy weighting — the deterministic low-entropy steps were dragging naive's average toward zero.

## Comparison to n=4096

For reference, the same setup with n=4096 and 16384 tokens (4 codeword repeats):

| n | tokens | repeats | TPR (entropy, FPR=1e-9) |
|---|---|---|---|
| 4096 | 16384 | 4 | 29/30 (96.7%) |
| 4096 | 4096 | 1 | 24/30 (80.0%) |
| 128 | 1024 | 8 | 11/30 (36.7%) |

Smaller n caps the test statistic's dynamic range — the parity-check matrix has fewer rows, so each fold contributes less information regardless of how many repeats are stacked.

## Files

- `calib_workdir/` — saved p-traces and tokens for n=128 run
- `qwen_threshold.json` — current saved threshold (n=128, entropy fold, FPR=1e-9)
- `calib_workdir_n4096/`, `qwen_threshold_n4096.json` — preserved n=4096 results
- `fpr_sweep.py` — entropy-fold sweep script
- `fpr_sweep_naive.py` — naive-fold sweep script
- `run_calibration_n128.log` — full run log
