# Completed redetection results — September 22, 2026

Two completed campaigns contributed 174 rows to the redetection result CSV.

| Run | Coverage | Key results | Billed cost |
|---|---|---|---:|
| 8B→0.6B TPR, η=.20 | 500 saved watermarked completions, T14336 | Posterior 437/500 (87.4%); entropy 418/500 (83.6%) | $25.43796444 |
| Native 8B shared-null FPR, η=.15/.20 | 500 saved nulls replayed once through T13088; 173 reporting points | At T6144/.15: posterior 0/500, entropy 0/500. At T13088/.20: posterior 1/500 (0.2%), entropy 0/500 | $55.74380810 |

Both used H200 / batch 50 and preserved the original completion-only protocol. Native FPR reused saved watermarked decisions unchanged. All workers stopped. Total billed across both runs: **$81.18177254**. No new text generation or automatic paid retries were performed.

- [8B→0.6B results](../../outputs/online_8b_eta020_0p6b_setup/RESULTS.md)
- [Native 8B FPR results](../../outputs/online_8b_shared_null_T13088_setup/RESULTS.md)
- [Result CSV](../../outputs/redetection/redetection_results_summary.csv)

Per-record reports, frozen execution sources, hashes and billing records are archived with each run. Tensor caches remain in the Modal result volume under the paths in those manifests.
