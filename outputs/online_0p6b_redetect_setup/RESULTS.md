# Online 0.6B redetection results

Completed four seed-12345 families: 154 recorded lengths, 500 watermarked and 500 null candidates per point. MAP and entropy use raw completion tokens, coordinate 1 zero, and one-shot FPR=.001 at each length.

All 18 new watermarked trace batches and 18 reused null batches were read back and verified. All reported lengths have zero observed false positives for both detectors.

| eta | Longest length | MAP old → new | Entropy old → new | MAP / entropy FP | GPU / batch |
|---:|---:|---|---|---|---|
| 0.05 | 512 | 93.6% → 89.4% | 85.0% → 79.0% | 0/500 / 0/500 | A10G / 100 |
| 0.10 | 1024 | 97.0% → 94.2% | 89.0% → 85.8% | 0/500 / 0/500 | A100-80GB / 125 |
| 0.15 | 2048 | 96.8% → 95.2% | 92.6% → 92.2% | 0/500 / 0/500 | A100-80GB / 125 |
| 0.20 | 4096 | 96.0% → 95.0% | 92.2% → 91.4% | 0/500 / 0/500 | A100-80GB / 100 |

First recorded lengths reaching 90% MAP TPR after redetection: eta=.05 does not reach 90% through 512; eta=.10 reaches it at 848; eta=.15 at 1648; eta=.20 at 3136. These are sampled grid points, not an interpolation or a monotonicity assumption.

At eta=.20, length 3104, the new 4096-derived prefix gives 449/500 MAP (89.8%) versus 448/500 (89.6%) in the earlier standalone redetection. Entropy is unchanged at 402/500 (80.4%). [The consistency audit](n3104_consistency.json) verifies identical tokens, model, key and partition and reproduces both reports from their respective cached traces. Three near-threshold MAP decisions flip (two to positive, one to negative). Cached BF16 probabilities differ between the earlier batch-125/length-3104 and new batch-100/length-4096 executions; batch size and replay length were not isolated with extra GPU inference. Both records are preserved.

All length-level results are in [the sorted CSV](../redetection/redetection_results_summary.csv). Frozen source manifests, keys, token batches, traces and per-candidate scores are stored locally under `cases/` and in Modal volume `prc-completion-only`; see [the cache index](../redetection/cache_index.json). Historical source evidence remains in the original experiment cache. No null or shorter-length model inference was added. Additional reproducibility archives are kept local as requested.

Independent rerun of eta=.05, T=512 (A10G, batch 100): MAP 447/500 (89.4%), entropy 395/500 (79.0%), both FP 0/500. All 500 freshly recomputed watermarked traces are bitwise identical to the first replay; all detector scores match. The verified null traces were reused. The separately labeled rerun is appended to the sorted CSV. [Rerun evidence](reruns/online_0p6b_eta005_n512_rerun01.json).
