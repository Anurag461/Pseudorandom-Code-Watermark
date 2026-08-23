# Cache-only Qwen3-0.6B proxy-detector report

Status: **complete and validated** on 2026-08-23. The analysis teacher-forced
existing Qwen3-8B texts through Qwen3-0.6B-Base and generated zero tokens. It
covers all 500 canonical prompts, all four PRC eta values, the selected
boundaries/ceiling, the six controlled prefixes, and the TextSeal entropy-weight
sensitivity. It does not constitute end-to-end 0.6B watermark generation.

## Frozen design

The generator and quality likelihood remain Qwen3-8B-Base revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. The proxy probability model is
Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, recovered from the existing
offline cache metadata; its weights etag is
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
The exact Python 3.11 image definition fingerprint is
`e62c9627f72c0448435665ec51304050838f65ef68b8d77766d6176a03745e48`.
The finalized integration code fingerprint is
`abe599dbdd5ec91348b508a90fb1f5515620cf3bcf9dad46d594878e0b44c44f`.
TextSeal remains pinned to
`c60d0d1da2e59f09a698438e218a07ee779b4616`; Google's SynthID-Text
reference remains pinned to `addb4a158143c7c6851a1308f78b89fceed59683`.

One legacy code oddity is recorded rather than hidden: importing
`watermark_expt.py` also constructs an `AutoTokenizer` from the non-Base
`Qwen/Qwen3-0.6B` repository. The proxy replay never calls `prompt_to_ids` and
never uses that object; its prompts and continuation IDs come directly from
the validated 8B caches, while model weights come from the pinned Base cache
above. It therefore did not affect any replay probability or score.

The PRC portability analysis uses eta `.05`, `.10`, `.15`, and `.20` at
`T=640`, `1407`, `4096`, and `13088`, respectively. The `.15` result remains
a **censored ceiling** (`n90 > 4096`), not an exact boundary. The common-length
analysis uses `T={128,256,400,512,768,1024}`. PRC and TextSeal are the only
detectors here that consume model probabilities or entropy weights. SynthID's
frequentist hash/g-value test and Gumbel's exact-Gamma test are model-independent
and are carried unchanged rather than relabeled as 0.6B results.

## PRC detector portability

At each selected MAP boundary/ceiling, replacing native 8B probabilities with
0.6B probabilities reduced detection power while leaving the observed null
decisions almost exactly unchanged.

| eta | T/status | Weight | Native TPR | Proxy TPR | Difference | WM agreement | Native/proxy FPR |
|---:|---|---|---:|---:|---:|---:|---:|
| .05 | 640, exact | MAP | 90.4% | 76.2% | -14.2 pp | 85.4% | 0/500, 0/500 |
| .05 | 640, exact | entropy | 71.8% | 64.0% | -7.8 pp | 88.6% | 0/500, 0/500 |
| .10 | 1407, exact | MAP | 90.2% | 83.0% | -7.2 pp | 92.0% | 0/500, 0/500 |
| .10 | 1407, exact | entropy | 78.8% | 72.4% | -6.4 pp | 93.2% | 0/500, 0/500 |
| .15 | 4096, censored ceiling | MAP | 89.6% | 86.4% | -3.2 pp | 96.4% | 0/500, 0/500 |
| .15 | 4096, censored ceiling | entropy | 84.6% | 81.2% | -3.4 pp | 96.2% | 0/500, 0/500 |
| .20 | 13088, exact | MAP | 90.2% | 86.6% | -3.6 pp | 96.0% | 1/500, 1/500 |
| .20 | 13088, exact | entropy | 86.2% | 83.8% | -2.4 pp | 97.2% | 0/500, 0/500 |

The proxy is therefore useful as a conservative sensitivity analysis, but it
is not a drop-in replacement for the native detector. At the common 1,024-token
horizon, proxy MAP TPR is `91.6%`, `70.2%`, `29.0%`, and `7.0%` for eta
`.05`, `.10`, `.15`, and `.20`; entropy-weighted TPR is `81.6%`, `55.0%`,
`22.0%`, and `3.0%`. All eight proxy null cells at T=1024 are `0/500`.

## Common-method sensitivity at T=1024

| Method/detector | TPR | Observed FPR | Median watermarked -log10(p) |
|---|---:|---:|---:|
| PRC eta=.05, native 8B MAP | 96.0% | 0/500 | 11.26 |
| PRC eta=.05, proxy 0.6B MAP | 91.6% | 0/500 | 8.79 |
| TextSeal alpha=.1, native 8B weights | 100% | 0/500 | 280.15 |
| TextSeal alpha=.1, proxy 0.6B weights | 100% | 0/500 | 271.45 |
| SynthID depth 10, model-independent | 100% | 0/500 | 202.08 |
| Gumbel-Max, model-independent | 100% | 0/500 | 142.20 |

TextSeal remained saturated at every prefix with both entropy models. Across
6,000 prompt/sample/prefix comparisons, the proxy TextSeal scorer agreed with
the pinned official implementation; the maximum absolute p-value difference
was `3.4802e-7`, within the predeclared tolerance. The native TextSeal false
positives at T=128 and 768 were `1/500` each; proxy weighting changed the T=128
cell to `0/500` and retained `1/500` at T=768. This finite-sample difference is
not evidence that either approximation is better calibrated.

## Quality and diversity remain native-8B measurements

Median T=1024 metrics from the unchanged 8B likelihood and cached outputs are:

| Output | NLL | Perplexity | Repetition | Distinct-2 | Distinct-3 |
|---|---:|---:|---:|---:|---:|
| Null | 3.004 | 20.159 | .014 | .888 | .967 |
| PRC eta=.05 | 2.999 | 20.057 | .014 | .887 | .968 |
| PRC eta=.10 | 3.076 | 21.669 | .012 | .891 | .971 |
| PRC eta=.15 | 3.064 | 21.408 | .012 | .893 | .972 |
| PRC eta=.20 | 3.028 | 20.650 | .014 | .888 | .968 |
| SynthID depth 10 | 3.060 | 21.326 | .012 | .893 | .972 |
| TextSeal alpha=.1 | 1.590 | 4.905 | .262 | .640 | .721 |
| Gumbel-Max | .890 | 2.436 | .633 | .334 | .361 |

The proxy analysis does not remove the controlled comparison's quality and
diversity finding: PRC across all eta values and SynthID remain close to the
shared null, while TextSeal and especially Gumbel show substantial looping.
The low NLL of the latter outputs means repeated tokens are individually likely
under the base model; it does not make their sequence-level diversity loss a
quality-matched result. PRC eta and TextSeal alpha are unrelated parameters and
must be compared only through observed detection-quality operating points.

## Validation, runtime, and cost

All 2,500 PRC/null cache records and all 500 TextSeal generation records passed
prompt, model, token, trace, and fingerprint validation. Chunked causal
teacher forcing was tested against the historical one-token path at
`rtol=1e-6, atol=1e-7`. The required workload was 16,863,500 token positions;
the production run processed 16,601,740 after the 261,760-position benchmark
was reused. The main full-run wall time was 2,644.1 seconds (44.1 minutes),
including 1,137.6 seconds of PRC replay, 38.1 seconds of TextSeal replay, and
513.6 seconds of CPU detection. Peak CUDA reserved memory was 18,614,321,152
bytes (17.34 GiB).

Exact provider spend was **$3.21783640**, below the approved $20 cap:
`$2.84171420` GPU, `$0.32920227` CPU, and `$0.04691993` memory. This total
includes planning/tests, a safely blocked sequential benchmark, the exact
chunked benchmark, full replay, official TextSeal scoring, and the final
CPU-only quality aggregation. No run attempted text generation.

Five hundred nulls cannot tightly validate nominal 0.1% FPR. The four p-value
calibrations remain non-equivalent, and this analysis measures detector-model
sensitivity on fixed 8B text—not generator-size effects.

Machine-readable results are in `outputs/proxy_8b_prc_summary.csv`,
`outputs/proxy_8b_prc_portability_summary.csv`,
`outputs/proxy_8b_common_method_summary.csv`,
`outputs/proxy_8b_quality_summary.csv`, `outputs/proxy_8b_cost_ledger.csv`, and
`outputs/proxy_8b_validation_manifest.json`. The large prompt-level files are
fingerprinted but intentionally excluded from git.
