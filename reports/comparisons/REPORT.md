# Watermark comparison results

## Main findings

- **PRC has lower Self-BLEU than the tested TextSeal and Gumbel-Max settings**, including with matched repeat handling, but lower detection rates.
- **A diversity advantage over shallow SynthID is not established at temperature 1.** The paired PRC-minus-depth-2 Self-BLEU intervals include zero in all three model/decoder settings.
- **Repeat handling substantially reduces the apparent repetition advantage.** With fallback enabled, PRC, SynthID, TextSeal and Gumbel have similar observed repeated-four-gram fractions on the 8B cohort.
- **Lower temperature exposes a substantial PRC detection loss.** At temperature 0.7 and 1,024 tokens, PRC detects 3/100 responses, versus 93/100 for SynthID depth 2 and 100/100 for depth 10.

## Evaluation protocol

The paired studies use 50 fixed prompts, sampling seeds 12345 and 67890, fixed watermark keys, and 1,024 generated tokens per response. The reference model is **Qwen3-8B-Base**, run in BF16 on H100, with temperature 1, top-p=1 and no top-k truncation. PRC uses η=.05, t=3 and row rate 99/100; TextSeal uses α=.1. SynthID depths 2, 10 and 30 use their exact fixed key lists. Model, decoding and repeat-policy changes are identified below.

**All detection is completion-only:** no original prompt, BOS/chat template or generation-time probability trace is supplied to the detector. PRC retains first-coordinate abstention. Detectors are PRC posterior MAP/Hoeffding, TextSeal entropy-weighted, SynthID weighted-normal, and Gumbel Gamma, at nominal p<.001.

Self-BLEU measures overlap between a prompt's two decoded responses; lower indicates less lexical overlap. It averages the two BLEU directions on a 0–1 scale using SacreBLEU 2.4.3, 13a tokenization, exponential smoothing and effective order. Repeated-four-gram fraction measures repetition within each response. Intervals are marginal 95% percentile intervals from **2,000 paired bootstrap resamples of the 50 prompts**, retaining both responses and all compared arms. Results at 1,024 tokens are emphasized; 400-token results are secondary. Full settings and metric definitions are in the [asset catalogue](README.md) and [source manifests](data/source_manifest.json).

## Main comparison: 8B, full vocabulary, temperature 1

Repeat fallback is **ON for SynthID, TextSeal and Gumbel** in this table. Detection columns give counts out of 100. Self-BLEU, its paired difference, and repetition refer to 1,024 tokens. Negative differences favor PRC.

| Method | Self-BLEU | PRC − method Self-BLEU [95% CI] | Detected, 400 tokens | Detected, 1,024 tokens | Repeated 4-grams |
| --- | ---: | --- | ---: | ---: | ---: |
| Ordinary sampling | 0.01928 | −0.00038 [−0.00369, +0.00297] | — | — | 2.92% |
| PRC | 0.01890 | — | 55 | 97 | 2.38% |
| SynthID depth 2 | 0.01987 | −0.00097 [−0.00459, +0.00265] | 100 | 100 | 3.28% |
| SynthID depth 10 | 0.02206 | −0.00316 [−0.00731, +0.00102] | 100 | 100 | 2.42% |
| SynthID depth 30 | 0.02890 | −0.01000 [−0.01546, −0.00506] | 100 | 100 | Not reported |
| TextSeal α=.1 | 0.04727 | −0.02837 [−0.03531, −0.02201] | 100 | 100 | 2.74% |
| Gumbel-Max | 0.20718 | −0.18829 [−0.21248, −0.16445] | 100 | 100 | 2.81% |

PRC's lower Self-BLEU relative to TextSeal and Gumbel persists after matching repeat handling. Against SynthID depth 2, the estimated difference is small and uncertain, while PRC detection is lower, especially at 400 tokens. With fallback ON, all available paired PRC-versus-baseline repetition intervals at 1,024 tokens include zero.

[Full 1,024-token metrics](tables/05_8b_full_1024.csv) · [400-token metrics](tables/05_8b_full_400.csv) · [Paired differences](tables/06_8b_contrasts_1024.csv)

## Repeat handling and SynthID depth

The repeat-policy ablation changes only whether repeated contexts fall back to ordinary sampling. These are paired results on the same 50 prompts, at 1,024 tokens.

| Method | Self-BLEU, OFF → ON | Repeated 4-grams, OFF → ON |
| --- | ---: | ---: |
| TextSeal | 0.03770 → 0.04727 | 32.17% → 2.74% |
| Gumbel-Max | 1.00000 → 0.20718 | 51.73% → 2.81% |
| SynthID depth 10 | 0.02138 → 0.02206 | 9.35% → 2.42% |

Both policies detect 100/100 responses for each baseline at 400 and 1,024 tokens. All 300 original/modified response pairs pass the check that divergence never precedes the first repeated context. SynthID's ON-minus-OFF Self-BLEU difference is +0.00067 [−0.00309, +0.00421]: this ablation does not support fallback as the main explanation for its between-response diversity. Native Gumbel's two seed slots are identical under fixed keys, explaining Self-BLEU=1.

**Depth matters at short lengths.** SynthID depth 2 detects 45/100, 82/100 and 100/100 responses at 64, 128 and 256 tokens; depths 10 and 30 detect 100/100 at all three lengths. Depth 30 has higher long-prefix Self-BLEU than depth 2, without an observed detection gain over depth 10 on these cutoffs. These are frequentist-detector results, not an evaluation of the Bayesian SynthID detector.

[Repeat-policy intervals](tables/07_repeat_policy.csv) · [Short-prefix detection and nulls](tables/09_short_prefixes.csv)

## Sensitivity results

Each setting has its own matched ordinary control and 100 responses per arm. The difference column is measured at 1,024 tokens; detection entries show **400 / 1,024 tokens**, each out of 100.

| Setting | PRC − SynthID depth 2 Self-BLEU [95% CI] | PRC detected | SynthID depth 2 detected |
| --- | --- | ---: | ---: |
| 8B, top-k=100, temperature 1 | +0.00095 [−0.00503, +0.00734] | 33 / 85 | 100 / 100 |
| 0.6B, full vocabulary, temperature 1 | +0.00051 [−0.00141, +0.00248] | 79 / 99 | 100 / 100 |
| 8B, full vocabulary, temperature 0.7 | −0.01098 [−0.01976, −0.00227] | 1 / 3 | 87 / 93 |

SynthID depth 10 detects 100/100 at both lengths in all three settings. The 0.6B comparison also includes TextSeal and Gumbel with fallback ON: PRC again has lower Self-BLEU, with 99/100 versus 100/100 detection at 1,024 tokens. The temperature-0.7 Self-BLEU advantage over SynthID accompanies a severe detection loss, not an improved overall tradeoff.

**Numerical paths differ across studies.** The original 8B pipeline uses BF16 PRC bucket probabilities and SynthID updates, followed by FP32 sampling; ordinary sampling uses FP32 softmax. The temperature-0.7 study preserves these paths and applies temperature once. Top-100 and 0.6B use FP32 probability arithmetic, so their comparisons with the original study do not isolate truncation or model size alone. Temperature-0.7 replay records eight bucket-endpoint contradictions in each of the PRC and ordinary cohorts; the detector is unchanged, and their causal contribution is unresolved.

[Top-100 results](tables/10_8b_topk_1024.csv) · [0.6B results](tables/10_0p6b_full_1024.csv) · [Temperature-0.7 results](tables/10_8b_t07_1024.csv)

## Larger detection cohort and false positives

The separate 500-response-per-method cohort provides broader detection evidence. Entries are **true positives / false positives**, each out of 500, after completion-only scoring.

| Tokens | PRC | TextSeal | SynthID depth 10 | Gumbel |
| --- | ---: | ---: | ---: | ---: |
| 128 | 46 / 0 | 500 / 0 | 500 / 1 | 500 / 1 |
| 256 | 151 / 0 | 500 / 0 | 500 / 0 | 500 / 1 |
| 400 | 274 / 0 | 500 / 0 | 499 / 0 | 500 / 2 |
| 512 | 345 / 0 | 500 / 0 | 499 / 0 | 500 / 0 |
| 768 | 436 / 0 | 500 / 1 | 500 / 0 | 500 / 0 |
| 1024 | 466 / 0 | 500 / 0 | 500 / 0 | 500 / 0 |

This cohort retains native TextSeal/Gumbel generation without fallback and SynthID's historical tuple-mask scorer. The paired studies instead use SynthID's official context mask. Consequently, the cohorts are reported separately; historical native-policy repetition is not evidence of an advantage under matched fallback.

In the paired studies, every 1,024-token pilot-null count is 0/100. At shorter lengths, the exceptions are SynthID depth 10 at 400 tokens in the 8B full-vocabulary and top-100 studies, and depth 2 at 256 tokens: 1/100 each. The nominal .001 thresholds were not tuned or empirically matched. Small null samples and boundary bootstrap intervals do not establish zero FPR or perfect detection.

## Interpretation and paper assets

These results support **configuration-specific lexical diversity/detection tradeoffs**. They do not establish a general PRC advantage, a matched-FPR frontier, or equivalence to unwatermarked text. PRC-versus-ordinary Self-BLEU intervals include zero in all four regimes at both lengths, but repetition is not uniformly unchanged: at 0.6B/400 tokens, PRC's repeated-four-gram fraction is higher by 0.49 percentage points [0.06, 1.01]. Self-BLEU and repetition do not measure semantic quality. Fixed keys, repeated use of the same 50 prompts, and post-pilot choices limit generalization; overlapping historical and pilot nulls must not be pooled.

- **Main comparison:** [tradeoff figure](figures/02_matched_policy_tradeoff.pdf), [paired SynthID-depth-2 contrasts](figures/03_paired_depth2.pdf), and [500-response detection table](tables/03_large_detection.tex).
- **Supporting results:** [repeat-policy figure](figures/04_repeat_policy.pdf), [depth figure](figures/05_synthid_depth.pdf), and [temperature figure](figures/08_temperature.pdf).
- **Complete data and reusable tables:** [absolute results](data/absolute_results.csv), [paired contrasts](data/paired_contrasts.csv), and the [LaTeX/CSV asset catalogue](README.md).
