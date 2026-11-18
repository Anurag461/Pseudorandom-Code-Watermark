# PRC detectors under random token substitution at n = 1024 and 2048, cf. Table 15

C4 RealNewsLike prompts (the 500 Table 15 prefixes), Qwen3-0.6B-Base, temperature 1.
PRC uses the Table 15 setup: fixed construction with n = T, t = 3, eta = 0.05, analytical
Hoeffding threshold at nominal FPR 1e-3. The three detectors differ only in the soft
weights: posterior mean (map), entropy weighted, and naive (unweighted signs). All
three are computed from one detector-model pass per response. The substitution attack
is the Table 15 attack (int(rate * n) distinct positions, uniform replacement tokens,
seeded per (source, prompt)). The 1024 and 2048 columns are new runs (500 watermarked
and 500 unwatermarked responses per cell); the 400 and 4096 columns are copied from
Table 15 (`substitution_results.csv`), which reports the posterior-mean and
entropy-weighted detectors only. TPR in % with 95% Wilson intervals; FP = false
positives / unwatermarked responses.

## Posterior mean

| Substitution rate | n = 400 (Table 15) TPR | FP | n = 1024 (new) TPR | FP | n = 2048 (new) TPR | FP | n = 4096 (Table 15) TPR | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | 86.0 [82.7, 88.8] | 0/500 | 99.4 [98.3, 99.8] | 0/500 | 99.2 [98.0, 99.7] | 0/500 | 100.0 [99.2, 100.0] | 0/500 |
| 10% | 41.6 [37.4, 46.0] | 0/500 | 92.6 [90.0, 94.6] | 0/500 | 97.8 [96.1, 98.8] | 0/500 | 98.8 [97.4, 99.4] | 0/500 |
| 20% | 13.2 [10.5, 16.4] | 0/500 | 61.8 [57.5, 66.0] | 0/500 | 93.2 [90.6, 95.1] | 0/500 | 98.2 [96.6, 99.1] | 0/500 |
| 30% | 2.6 [1.5, 4.4] | 0/500 | 20.8 [17.5, 24.6] | 0/500 | 68.4 [64.2, 72.3] | 0/500 | 94.6 [92.3, 96.3] | 0/500 |

## Entropy weighted

| Substitution rate | n = 400 (Table 15) TPR | FP | n = 1024 (new) TPR | FP | n = 2048 (new) TPR | FP | n = 4096 (Table 15) TPR | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | 70.8 [66.7, 74.6] | 0/500 | 98.4 [96.9, 99.2] | 0/500 | 99.0 [97.7, 99.6] | 0/500 | 99.6 [98.6, 99.9] | 0/500 |
| 10% | 33.4 [29.4, 37.6] | 0/500 | 89.6 [86.6, 92.0] | 0/500 | 96.8 [94.9, 98.0] | 0/500 | 98.8 [97.4, 99.4] | 0/500 |
| 20% | 11.2 [8.7, 14.3] | 0/500 | 55.0 [50.6, 59.3] | 0/500 | 91.8 [89.1, 93.9] | 0/500 | 97.2 [95.4, 98.3] | 0/500 |
| 30% | 1.4 [0.7, 2.9] | 0/500 | 18.8 [15.6, 22.5] | 0/500 | 65.0 [60.7, 69.1] | 0/500 | 94.0 [91.6, 95.8] | 0/500 |

## Naive

| Substitution rate | n = 400 (Table 15) TPR | FP | n = 1024 (new) TPR | FP | n = 2048 (new) TPR | FP | n = 4096 (Table 15) TPR | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | — | — | 94.2 [91.8, 95.9] | 0/500 | 97.0 [95.1, 98.2] | 0/500 | — | — |
| 10% | — | — | 79.6 [75.8, 82.9] | 0/500 | 95.2 [93.0, 96.8] | 0/500 | — | — |
| 20% | — | — | 49.4 [45.0, 53.8] | 1/500 | 89.8 [86.8, 92.2] | 0/500 | — | — |
| 30% | — | — | 17.2 [14.1, 20.8] | 0/500 | 62.6 [58.3, 66.7] | 0/500 | — | — |
