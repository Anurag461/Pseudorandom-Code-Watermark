# Redetection results

The **online PRC 8B → 0.6B comparison is complete** at eta=.05/.10/.15, T=1280/3072/6144 and N=500/500/100, respectively. MAP TPR is **92.6%, 90.8%, 92.0%**; entropy TPR is **86.8%, 88.0%, 84.0%**. All 124 agreed eta/length points are reported, with shorter lengths scored from the same saved traces. Each completion was replayed once; eta=.15 used two parallel A100 workers with disjoint batches of 50. No new generation, nulls, independent reference pass, benchmark or retry ran. All apps are stopped. Provider-reported spending was **$2.70079218**. See the [results, paired 8B comparison and cache provenance](../online_8b_to_0p6b_redetect_setup/RESULTS.md).

The **fixed PRC 4B → {4B, 0.6B} comparisons at eta=.05, n=T=512 and 256 are complete**, with N=100 at each length. At n512, MAP/entropy TPR is **83%/67%** with 4B detection and **78%/64%** with 0.6B. At n256, it is **46%/27%** with 4B and **35%/25%** with 0.6B. Each cohort was generated once and reused by both detectors. No nulls or empirical FPR were included. The twelve approved stages cost **$0.50338169**. See the [results and provenance](../fixed_4b_eta005_n512_n256_N100_setup/RESULTS.md).

The **fixed PRC 4B → {4B, 0.6B} comparison at eta=.05, n=T=1024 is complete**.
Exactly 100 watermarked completions were generated once and reused by both
detectors. For the 4B detector, MAP TPR is **98/100 (98%)** and entropy TPR is
**93/100 (93%)**; for the 0.6B detector, they are **93/100 (93%)** and
**90/100 (90%)**. No null cohort was generated, so empirical FPR is unmeasured.
All six approved stages cost **$0.40304877** in provider-reported charges. See
the [complete results and provenance](../fixed_4b_eta005_n1024_N100_setup/RESULTS.md).

The **online PRC 8B → 8B campaign at eta=.05, .10, .15 is complete**, with shorter recorded prefixes stopping after the first redetected MAP TPR below 90%. At the longest lengths (1280, 3072 and 4096), MAP TPR is 96.2%, 94.2% and 88.0%; entropy TPR is 90.0%, 91.0% and 84.0%, respectively. Each point uses 500 watermarked and 500 null candidates.

Updated September 20, 2026. This folder collects all completed redetection settings and their earlier diagnostics from this work. The CSV contains 429 rows: 22 fixed 0.6B rows, six fixed 4B-generation rows, 156 online 0.6B rows, 119 online 8B full-cohort rows, 125 8B→0.6B rows and the 100-prompt 6144-token pilot. Counts include recorded prefixes, seed replicates and separately labeled reruns; they are not independent generation-run counts. All 305 rows present before this online 8B→0.6B comparison were preserved; its 124 rows were appended with explicit N=500 or N=100. Source hashes and coverage checks are preserved locally.

The online 8B → 8B eta=.15, n=6144 pilot completed primary generation, replay and scoring for 100 watermarked prompts: MAP 94/100 (94%) and entropy 90/100 (90%). No null cohort was evaluated, and the additional independent reference pass was stopped by the user. The CSV labels this as a pilot with unmeasured FPR; the remaining 400 prompts were not run. Among the 298 earlier CSV rows with full null evaluation, 292 have zero false positives for both detectors; six prefix rows have 1/500 (0.2%) for at least one detector. No experiment or scoring was launched during this reconciliation.

The four main **online PRC 0.6B → 0.6B families are complete**, covering 154 recorded lengths at eta=.05, .10, .15 and .20. Only the longest watermarked completions were replayed; all 2,000 null traces were reused from verified fixed-run caches, and shorter lengths were scored on CPU. All 36 trace shards were read back and verified. See the [online result summary](../online_0p6b_redetect_setup/RESULTS.md) for results and cache provenance.

An independent eta=.05, T=512 rerun reproduced MAP 447/500 (89.4%) and entropy 395/500 (79.0%), with zero false positives. All 500 freshly recomputed watermarked traces and every detector score matched the first run exactly; verified null traces were reused. The CSV includes a separately labeled rerun row. Additional reproducibility archives remain local as requested.

The remaining **18 main fixed-PRC 0.6B → 0.6B single-block settings and two seed replicates are complete**. Eta=.20, n8192 remains deferred. Across the 18 main settings, mean TPR changes are −1.62 percentage points for posterior weighting and −1.00 for entropy weighting. The [full result table](../fixed_0p6b_redetect_setup/RESULTS.md) gives all old/new TPRs, false-positive counts, hardware choices and cache verification. These 20 settings are included in the CSV and JSON indexes below; the following table preserves the six earlier settings.

The current protocol uses BF16 Qwen3-Base detectors (0.6B, 4B and 8B), raw completion tokens with no prepended special token, and coordinate 1 score zero. Original candidates, keys, partitions, PRC indices and threshold formula are preserved. Earlier full-cohort self-detection settings have 500 watermarked and 500 null candidates; the new online 8B→0.6B comparison has N=500/500/100 watermarked candidates and no nulls; the 6144-token pilot and the paired fixed 4B comparison each have 100 watermarked and zero null candidates. All use t=3 and target FPR=0.001.

| Generation → detection | η | n | Detector | Prompted TPR | Raw-completion TPR | Change | Raw FP |
|---|---:|---:|---|---:|---:|---:|---:|
| 0.6B → 0.6B | 0.05 | 400 | Posterior mean | 89.2% (446/500) | 86.0% (430/500) | -3.2 pp | 0/500 |
| 0.6B → 0.6B | 0.05 | 400 | Entropy weighted | 73.0% (365/500) | 71.0% (355/500) | -2.0 pp | 0/500 |
| 0.6B → 0.6B | 0.05 | 448 | Posterior mean | 93.2% (466/500) | 90.0% (450/500) | -3.2 pp | 0/500 |
| 0.6B → 0.6B | 0.05 | 448 | Entropy weighted | 79.8% (399/500) | 79.4% (397/500) | -0.4 pp | 0/500 |
| 8B → 0.6B | 0.05 | 640 | Posterior mean | 76.4% (382/500) | 72.8% (364/500) | -3.6 pp | 0/500 |
| 8B → 0.6B | 0.05 | 640 | Entropy weighted | 64.2% (321/500) | 60.0% (300/500) | -4.2 pp | 0/500 |
| 0.6B → 0.6B | 0.2 | 3104 | Posterior mean | 90.8% (454/500) | 89.6% (448/500) | -1.2 pp | 0/500 |
| 0.6B → 0.6B | 0.2 | 3104 | Entropy weighted | 80.8% (404/500) | 80.4% (402/500) | -0.4 pp | 0/500 |
| 8B → 8B | 0.05 | 1024 | Posterior mean | 96.0% (480/500) | 93.2% (466/500) | -2.8 pp | 0/500 |
| 8B → 8B | 0.05 | 1024 | Entropy weighted | 88.2% (441/500) | 84.4% (422/500) | -3.8 pp | 0/500 |
| 8B → 8B | 0.05 | 1280 | Posterior mean | 97.8% (489/500) | 96.2% (481/500) | -1.6 pp | 0/500 |
| 8B → 8B | 0.05 | 1280 | Entropy weighted | 92.4% (462/500) | 90.0% (450/500) | -2.4 pp | 0/500 |

The prompted controls for the 0.6B detector settings had 0/500 false positives. The n=640 prompted columns use the matched-execution BF16 control (76.4% / 64.2%); the historical saved values were 76.2% / 64.0%. The n=400 and n=448 constructions are fixed-block PRC; n=640, n=1024, n=1280 and n=3104 use online PRC.

Within this earlier table, the largest observed reduction is at 8B → 0.6B, n=640: 3.6 points for posterior mean and 4.2 for entropy weighting. At n=3104, the corresponding reductions are 1.2 and 0.4 points. This earlier table covers six settings; the linked single-block report adds 20 completed settings, including two seed replicates. The earlier n=3104 execution (448/500 MAP) is retained separately from the later 4096-prefix replay (449/500 MAP); the [saved consistency audit](../online_0p6b_redetect_setup/n3104_consistency.json) verified execution variation on the same candidates, model, key and partition.

Earlier EOT and coordinate-1 diagnostics (posterior mean; all false positives are 0/500):

| Setting | EOT proxy, BF16 MAP | EOT + coordinate-1 zero MAP | EOT + hard fallback MAP | Raw + coordinate-1 zero MAP |
|---|---:|---:|---:|---:|
| 0.6B → 0.6B, n=400, η=0.05 | 85.6% | 86.0% | 85.6% | 86.0% |
| 8B → 0.6B, n=640, η=0.05 | 71.4% | 72.6% | 71.2% | 72.8% |

Removing the EOT token after abstaining at coordinate 1 changed posterior-mean TPR by 0.0 points at n=400 and +0.2 at n=640. Coordinate 1 occurs in 2/396 and 10/634 parity checks, respectively. Earlier raw-completion reports also preserve the prompted-coordinate-1-zero and fixed-numerical-threshold diagnostic comparisons.

The preliminary FP32 EOT n=400 run gave 86.2% posterior-mean and 71.4% entropy-weighted TPR. It changed precision as well as context, so it is retained as a diagnostic. BF16 EOT-proxy entropy TPR was 71.2% at n=400 and 59.2% at n=640; EOT-abstain and raw-abstain entropy TPR were 71.0% and 60.0%, respectively.

The n=3104 execution used code commit `dad704f`, eight A100 80GB batches of 125, one full validation batch and parallel workers. Peak live GPU allocation was 48.98 GB. The 8B → 8B n=1280 run used commit `e5fa510`, eight H100 batches of 125 and one full validation batch; peak live allocation was 42.73 GB. Across the five inference runs, 5,867,000 response-only probabilities are cached. The n=1024 result reuses 1,023,000 of the n=1280 probabilities and adds no inference.

[Redetection CSV](redetection_results_summary.csv) · [Machine-readable results](summary.json) · [Cache locations and SHA-256 checksums](cache_index.json)

Git contains this report, the summary data, the redetection CSV, the cache index and ignore rules. Detailed reports, per-candidate results, batch validation records and earlier diagnostics remain in the local archive and in Modal volume `prc-completion-only`. The full 93-file results archive was read back from Modal and its SHA-256 verified before the results commit was replaced. Raw tensors remain in the same local and Modal caches. The cache index records the archive and individual run locations.

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
`detectors.py`, and orchestration into `modal_run.py`. Local validation
passed 109 tests and reproduced all 6,000 saved detector decisions across the
three settings (3,000 candidates, two weights). Statistics and thresholds agreed
within 4.6e-13. This used the existing traces; no Modal inference was run.
The previous implementation and frozen manifests are preserved in the
implementation archive listed in the cache index.

For the 8B → 8B n=1280 run, the old TPR columns use the same 500 watermarked candidates and their cached generation probabilities. Posterior TPR reproduces the saved 489/500 result; entropy-weighted TPR is 462/500. These historical scores required no model inference. Old null FPR was not recomputed. The current run uses 500 cached 8B nulls from T=1382, truncated to 1280 tokens. All eight new trace shards were read back and checksum-verified, and exactly one batch carries full validation.

The 8B → 8B n=1024 result is a local CPU prefix rescore of the same n=1280 traces and candidates, with coordinate 1 zero and the original online one-shot FPR policy. It preserves all 1014 prefix parity checks (11 contain coordinate 1). Both old TPRs use cached generation probabilities; all 500 old posterior decisions match the saved n=1024 result. No new Modal job or model inference was run. Detailed prefix results and provenance are saved locally and under the parent Modal cache in `prefixes/n1024/`.

The fixed-PRC 0.6B → 0.6B n=448 run used commit `789f42b`, eight A100 80GB batches of 125 and one full validation batch. Peak live allocation was 8.68 GB. All 500 watermarked and 500 null token hashes match the original audit; nulls reuse its T=512 cache truncated to 448. The original key and partition survived the galois compatibility loader unchanged. Coordinate 1 occurs in 2/444 parity checks. All eight trace shards, token batches and result evidence are cached locally and in Modal; saved traces and evidence were read back and verified.
