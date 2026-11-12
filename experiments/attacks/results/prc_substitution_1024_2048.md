# PRC under random token substitution at n = 1024 and 2048, cf. Table 15

C4 RealNewsLike prompts (the 500 Table 15 prefixes), Qwen3-0.6B-Base, temperature 1.
PRC uses the Table 15 setup: fixed construction with n = T, t = 3, eta = 0.05, and the
posterior-mean detector with the analytical Hoeffding threshold at nominal FPR 1e-3.
The substitution attack is the Table 15 attack (int(rate * n) distinct positions,
uniform replacement tokens, seeded per (source, prompt)). The 1024 and 2048 columns
are new runs (500 watermarked and 500 unwatermarked responses per cell); the 400 and
4096 columns are copied from Table 15 (`substitution_results.csv`, `prc_map`).
TPR in % with 95% Wilson intervals; FP = false positives / unwatermarked responses.

| Substitution rate | n = 400 (Table 15) TPR | FP | n = 1024 (new) TPR | FP | n = 2048 (new) TPR | FP | n = 4096 (Table 15) TPR | FP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0% | 86.0 [82.7, 88.8] | 0/500 | 99.4 [98.3, 99.8] | 0/500 | 99.2 [98.0, 99.7] | 0/500 | 100.0 [99.2, 100.0] | 0/500 |
| 10% | 41.6 [37.4, 46.0] | 0/500 | 92.6 [90.0, 94.6] | 0/500 | 97.8 [96.1, 98.8] | 0/500 | 98.8 [97.4, 99.4] | 0/500 |
| 20% | 13.2 [10.5, 16.4] | 0/500 | 61.8 [57.5, 66.0] | 0/500 | 93.2 [90.6, 95.1] | 0/500 | 98.2 [96.6, 99.1] | 0/500 |
| 30% | 2.6 [1.5, 4.4] | 0/500 | 20.8 [17.5, 24.6] | 0/500 | 68.4 [64.2, 72.3] | 0/500 | 94.6 [92.3, 96.3] | 0/500 |

Reproduction check from the same pipeline: n = 400 at 10% substitution gives 39.4 [35.2, 43.7] (Table 15: 41.6 [37.4, 46.0]).
