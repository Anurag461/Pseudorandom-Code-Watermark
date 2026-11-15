# CGZ (bit-level seed) under random token substitution, cf. Table 15

Christ-Gunn-Zamir watermark, single-seed construction, on the PRC vocabulary partition
(one bit per token). The seed follows the paper's binary construction: the empirical
entropy counter adds -log2 P(bit) per emitted bit until it reaches lambda, and the PRF
is keyed on the bit prefix. Detection tries every seed length and reports the
Bonferroni-corrected exact Gamma p-value.

Qwen3-0.6B-Base, temperature 1, 500 prompts per cell (one watermarked and one
unwatermarked response each), nominal FPR 1e-3, analytical threshold. Substitution
positions and replacement tokens are matched to the Table 15 PRC runs per
(source, prompt). TPR in % with 95% Wilson intervals; FP = false positives / 500.
The unwatermarked texts are identical for both lambda values, so their FP counts
are one sample, not two.

## 400-token texts

| Substitution rate | lambda = 30 TPR | lambda = 30 FP | lambda = 64 TPR | lambda = 64 FP |
| --- | --- | --- | --- | --- |
| 0% | 99.4 [98.3, 99.8] | 0/500 | 99.8 [98.9, 100.0] | 0/500 |
| 5% | 38.6 [34.4, 42.9] | 0/500 | 11.6 [9.1, 14.7] | 0/500 |
| 10% | 13.2 [10.5, 16.4] | 1/500 | 0.4 [0.1, 1.4] | 1/500 |
| 15% | 5.8 [4.1, 8.2] | 0/500 | 0.2 [0.0, 1.1] | 0/500 |
| 20% | 2.4 [1.4, 4.1] | 0/500 | 0.0 [0.0, 0.8] | 0/500 |
| 25% | 0.8 [0.3, 2.0] | 0/500 | 0.0 [0.0, 0.8] | 0/500 |
| 30% | 0.2 [0.0, 1.1] | 0/500 | 0.2 [0.0, 1.1] | 0/500 |

## 4096-token texts

| Substitution rate | lambda = 30 TPR | lambda = 30 FP | lambda = 64 TPR | lambda = 64 FP |
| --- | --- | --- | --- | --- |
| 0% | 100.0 [99.2, 100.0] | 0/500 | 100.0 [99.2, 100.0] | 0/500 |
| 5% | 38.0 [33.9, 42.3] | 1/500 | 12.0 [9.4, 15.1] | 1/500 |
| 10% | 14.4 [11.6, 17.7] | 2/500 | 1.6 [0.8, 3.1] | 2/500 |
| 15% | 4.2 [2.8, 6.3] | 1/500 | 0.6 [0.2, 1.7] | 1/500 |
| 20% | 1.8 [0.9, 3.4] | 2/500 | 0.2 [0.0, 1.1] | 2/500 |
| 25% | 0.6 [0.2, 1.7] | 0/500 | 0.2 [0.0, 1.1] | 0/500 |
| 30% | 0.0 [0.0, 0.8] | 0/500 | 0.0 [0.0, 0.8] | 0/500 |

## Seed length (tokens, clean watermarked texts)

| lambda | 400 tokens: median / mean / max | 4096 tokens: median / mean / max |
| --- | --- | --- |
| 30 | 38.0 / 39.5 / 126 | 38.0 / 38.9 / 94 |
| 64 | 81.0 / 82.3 / 177 | 80.0 / 81.7 / 158 |

TPR is set by seed survival: a substituted seed token breaks detection when the
replacement lands in the other partition half (probability about 1/2), so
TPR ≈ (1 - rate/2)^(seed length). With the seed intact, detection is near-certain.
