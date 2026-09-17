# Redetection results

Updated September 17, 2026. This folder collects all completed redetection settings and their earlier diagnostics from this work.

The current protocol uses BF16 Qwen3-0.6B-Base, raw completion tokens with no prepended special token, and coordinate 1 score zero. Original candidates, keys, partitions, PRC indices and threshold formula are preserved. Every setting has 500 watermarked and 500 null candidates; t=3 and target FPR=0.001.

| Generation → detection | η | n | Detector | Prompted TPR | Raw-completion TPR | Change | Raw FP |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.6B → 0.6B | 0.05 | 400 | Posterior mean | 89.2% (446/500) | 86.0% (430/500) | -3.2 pp | 0/500 |
| 0.6B → 0.6B | 0.05 | 400 | Entropy weighted | 73.0% (365/500) | 71.0% (355/500) | -2.0 pp | 0/500 |
| 8B → 0.6B | 0.05 | 640 | Posterior mean | 76.4% (382/500) | 72.8% (364/500) | -3.6 pp | 0/500 |
| 8B → 0.6B | 0.05 | 640 | Entropy weighted | 64.2% (321/500) | 60.0% (300/500) | -4.2 pp | 0/500 |
| 0.6B → 0.6B | 0.2 | 3104 | Posterior mean | 90.8% (454/500) | 89.6% (448/500) | -1.2 pp | 0/500 |
| 0.6B → 0.6B | 0.2 | 3104 | Entropy weighted | 80.8% (404/500) | 80.4% (402/500) | -0.4 pp | 0/500 |

All prompted controls also had 0/500 false positives. The n=640 prompted columns use the matched-execution BF16 control (76.4% / 64.2%); the historical saved values were 76.2% / 64.0%. The n=400 construction is fixed-block PRC; n=640 and n=3104 use online PRC.

The largest observed reduction is at 8B → 0.6B, n=640: 3.6 points for posterior mean and 4.2 for entropy weighting. At n=3104, the corresponding reductions are 1.2 and 0.4 points. These results cover three settings; other paper settings have not yet been redetected under this protocol.

Earlier EOT and coordinate-1 diagnostics (posterior mean; all false positives are 0/500):

| Setting | EOT proxy, BF16 MAP | EOT + coordinate-1 zero MAP | EOT + hard fallback MAP | Raw + coordinate-1 zero MAP |
|---|---:|---:|---:|---:|
| 0.6B → 0.6B, n=400, η=0.05 | 85.6% | 86.0% | 85.6% | 86.0% |
| 8B → 0.6B, n=640, η=0.05 | 71.4% | 72.6% | 71.2% | 72.8% |

Removing the EOT token after abstaining at coordinate 1 changed posterior-mean TPR by 0.0 points at n=400 and +0.2 at n=640. Coordinate 1 occurs in 2/396 and 10/634 parity checks, respectively. Earlier raw-completion reports also preserve the prompted-coordinate-1-zero and fixed-numerical-threshold diagnostic comparisons.

The preliminary FP32 EOT n=400 run gave 86.2% posterior-mean and 71.4% entropy-weighted TPR. It changed precision as well as context, so it is retained as a diagnostic. BF16 EOT-proxy entropy TPR was 71.2% at n=400 and 59.2% at n=640; EOT-abstain and raw-abstain entropy TPR were 71.0% and 60.0%, respectively.

The n=3104 execution used code commit `dad704f`, eight A100 80GB batches of 125, one full validation batch and parallel workers. Peak live GPU allocation was 48.98 GB. Across all three current settings, 4,141,000 response-only probabilities are cached.

[Machine-readable results](summary.json) · [Cache locations and SHA-256 checksums](cache_index.json)

Git contains this report, the summary data, the cache index and ignore rules. Detailed reports, per-candidate results, batch validation records and earlier diagnostics remain in the local archive and in Modal volume `prc-completion-only`. The full 93-file results archive was read back from Modal and its SHA-256 verified before the results commit was replaced. Raw tensors remain in the same local and Modal caches. The cache index records the archive and individual run locations.

To retrieve the detailed archive into a separate directory:

```sh
mkdir -p outputs/redetection/.archive/restored
MODAL_PROFILE=new-prc-watermark python -m modal volume get \
  prc-completion-only archives/redetection/detailed-results-e78f61f.tar.gz \
  outputs/redetection/.archive/detailed-results-e78f61f.tar.gz
tar -xzf outputs/redetection/.archive/detailed-results-e78f61f.tar.gz \
  -C outputs/redetection/.archive/restored
```

The archive preserves the original reports and their provenance. No results were recomputed during this cleanup.

The subsequent code refactor integrates replay into `qwen.py`, scoring into
`detectors.py`, and orchestration into `modal_online_run.py`. Local validation
passed 109 tests and reproduced all 6,000 saved detector decisions across the
three settings (3,000 candidates, two weights). Statistics and thresholds agreed
within 4.6e-13. This used the existing traces; no Modal inference was run.
The previous implementation and frozen manifests are preserved in the
implementation archive listed in the cache index.
