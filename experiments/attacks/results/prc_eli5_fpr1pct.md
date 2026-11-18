# PRC on ELI5 under token substitution at an actual FPR near 1%

ELI5 prompts (`Question: {q}\nAnswer:`, 500 seeded questions from
`sentence-transformers/eli5`), Qwen3-0.6B-Base, temperature 1, forced length. PRC uses
the Table 15 setup: fixed construction with n = T, t = 3, eta = 0.05, posterior-mean
detector. The substitution attack is the Table 15 attack (int(rate * n) distinct
positions, uniform replacement tokens, seeded per (source, prompt)).

The detector statistic is unchanged; only the cutoff differs from Table 15. Instead of
the analytical Hoeffding threshold, the cutoff on z = S / sqrt(V) is the 99th percentile
of z over all unwatermarked responses at that substitution rate, pooled across the
8 lengths (4000 responses), so the observed FPR is close to 1%.
This is an empirical calibration, not the paper's provable FPR bound. TPR in % with
95% Wilson intervals; FP = false positives / 500 unwatermarked responses at that length.

## 10% substitution

Cutoff z = 2.34 (normal 1% quantile 2.33; Hoeffding at nominal 1%: 3.03). Pooled observed FPR: 40/4000.

| n | TPR | FP |
| --- | --- | --- |
| 400 | 51.0 [46.6, 55.4] | 7/500 |
| 448 | 55.2 [50.8, 59.5] | 9/500 |
| 512 | 62.0 [57.7, 66.1] | 3/500 |
| 640 | 65.6 [61.3, 69.6] | 4/500 |
| 768 | 74.4 [70.4, 78.0] | 4/500 |
| 1024 | 84.2 [80.7, 87.1] | 5/500 |
| 1536 | 90.8 [87.9, 93.0] | 4/500 |
| 2048 | 96.4 [94.4, 97.7] | 4/500 |

60% TPR is reached at ≈ 492 tokens (between 448 and 512, log-linear interpolation).

## 25% substitution

Cutoff z = 2.31 (normal 1% quantile 2.33; Hoeffding at nominal 1%: 3.03). Pooled observed FPR: 40/4000.

| n | TPR | FP |
| --- | --- | --- |
| 400 | 15.2 [12.3, 18.6] | 3/500 |
| 448 | 23.4 [19.9, 27.3] | 7/500 |
| 512 | 23.4 [19.9, 27.3] | 5/500 |
| 640 | 32.0 [28.1, 36.2] | 4/500 |
| 768 | 35.6 [31.5, 39.9] | 5/500 |
| 1024 | 50.8 [46.4, 55.2] | 5/500 |
| 1536 | 61.6 [57.3, 65.8] | 4/500 |
| 2048 | 75.4 [71.4, 79.0] | 7/500 |

60% TPR is reached at ≈ 1446 tokens (between 1024 and 1536, log-linear interpolation).
