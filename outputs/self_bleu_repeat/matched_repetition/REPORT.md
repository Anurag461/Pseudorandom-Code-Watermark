# Repetition metrics with matched repeat handling

**The large within-response repetition gap shrinks dramatically when TextSeal,
Gumbel-Max and SynthID all use repeated-context fallback.** At 1,024 tokens,
repeated-4-gram rates are 2.38% PRC, 2.42% SynthID, 2.74% TextSeal and 2.81%
Gumbel, versus 2.92% for ordinary sampling. This cohort does not establish a PRC
advantage over the fallback-on baselines for either repeated-4-gram rate or
distinct-3: all paired PRC differences have 95% intervals including zero at both
primary lengths. That is uncertainty, not proof of equivalence.

## Exact historical metrics and matched cohort

- **Repeated token 4-gram fraction (lower is better):**
  `1 − unique token 4-grams / (T − 3)`. Each occurrence after the first counts
  toward the repeated fraction.
- **Distinct-3 (higher is better):** `unique token trigrams / (T − 2)`.
  This measures how many three-token sequences within a response are unique.

These are the existing functions in `baseline_comparison/scoring.py`, applied
to saved raw completion tokens only. No prompt tokens, detector masks, decoding
or special-token removal enter either metric. These token n-grams need not
correspond to words. Fallback activation counts instead concern repeated preceding
three-token contexts, with native prompt/zero initialization; their counts are
not substitutes for the two quality metrics above.

The comparison retains Qwen3-8B-Base, the same 50 prompts, two sampling seeds,
fixed keys/default parameters, 1,024-token generation and 400/1,024-token
prefixes. Main tables use means, matching the Self-BLEU pilot's auxiliary
metrics: average two responses within each prompt, then average 50 prompts.
Because every prompt has two responses, this equals the mean of 100 response
slots. Intervals use the same 2,000 paired prompt-cluster bootstrap draws.

SynthID uses its native fallback-on responses. TextSeal and Gumbel use the new
fallback-on responses. PRC is position based and has no corresponding context
fallback; ordinary sampling already draws from the unwatermarked distribution.
Their saved responses are reused unchanged. This matches the fallback rule
across contextual baselines while retaining native context initialization and
RNGs; it is not complete implementation harmonization.

## Fallback-on comparison

Values are means with marginal 95% intervals. Both metrics are shown as
percentages; higher distinct-3 means fewer repeated trigrams.

### 1,024 tokens

| Method | Repeated token 4-grams ↓ | Distinct-3 ↑ |
|---|---:|---:|
| PRC η=.05 | 2.38% [1.68, 3.16] | 95.61% [94.62, 96.51] |
| SynthID depth 10, fallback on | 2.42% [1.83, 3.14] | 95.54% [94.63, 96.35] |
| TextSeal α=.1, fallback on | 2.74% [2.14, 3.35] | 94.64% [93.80, 95.51] |
| Gumbel-Max, fallback on | 2.81% [2.05, 3.71] | 94.56% [93.41, 95.57] |
| Ordinary sampling | 2.92% [1.96, 4.33] | 95.14% [93.79, 96.20] |

### 400 tokens

| Method | Repeated token 4-grams ↓ | Distinct-3 ↑ |
|---|---:|---:|
| PRC η=.05 | 1.09% [0.77, 1.45] | 97.47% [96.90, 97.98] |
| SynthID depth 10, fallback on | 1.01% [0.73, 1.29] | 97.65% [97.19, 98.10] |
| TextSeal α=.1, fallback on | 1.30% [0.95, 1.66] | 97.11% [96.59, 97.62] |
| Gumbel-Max, fallback on | 1.22% [0.97, 1.50] | 97.06% [96.56, 97.50] |
| Ordinary sampling | 1.29% [0.91, 1.73] | 97.34% [96.72, 97.89] |

## How enabling fallback changes each baseline

Same prompts, seeds, keys and parameter values; entries are **off → on** at
1,024 tokens. The SynthID-off arm was generated for the earlier ablation;
its on arm is native. TextSeal/Gumbel's off arms are native, with on arms from
the follow-ups. These complete both the matched-on and matched-off comparisons;
all values and paired policy-effect intervals are retained in the JSON summary.

| Method | Repeated token 4-grams ↓ | Distinct-3 ↑ |
|---|---:|---:|
| TextSeal | 32.17% → 2.74% | 66.03% → 94.64% |
| Gumbel-Max | 51.73% → 2.81% | 47.20% → 94.56% |
| SynthID | 9.35% → 2.42% | 88.80% → 95.54% |

The earlier mixed-policy pilot had TextSeal at 32.17% repeated 4-grams and Gumbel
at 51.73%, while SynthID already had fallback on. The within-method interventions
reduce those values to 2.74% and 2.81%. This shows that the earlier large
within-response repetition differences were highly sensitive to repeat handling.
A comparison with all fallbacks off is available as a sensitivity check, but
SynthID-off is a modified configuration and does not replace its native default.

## Paired PRC comparisons

At 1,024 tokens, values are **percentage points**, PRC minus comparator, with
paired 95% intervals. Negative favors PRC for repeated-4-grams; positive favors
PRC for distinct-3. These are paired intervals computed directly from prompt
contrasts, not an inference from overlap of separate marginal intervals.

| Contrast | Repeated-4-gram difference | Distinct-3 difference |
|---|---:|---:|
| PRC − Ordinary sampling | -0.54 [-2.11, +0.63] | +0.47 [-0.92, +2.09] |
| PRC − TextSeal α=.1, fallback on | -0.36 [-1.20, +0.49] | +0.97 [-0.09, +2.03] |
| PRC − Gumbel-Max, fallback on | -0.43 [-1.60, +0.65] | +1.05 [-0.29, +2.49] |
| PRC − SynthID depth 10, fallback on | -0.04 [-0.91, +0.80] | +0.08 [-0.97, +1.15] |

All intervals above, and the analogous 400-token intervals, include zero. The
study uses one 50-prompt cohort and one fixed key per method; these exploratory
intervals are marginal and do not correct for multiple comparisons. Point
estimates alone cannot establish that PRC is less repetitive than the
fallback-on baselines. No parameter-frontier or overall text-quality claim follows.

This result is compatible with the different Self-BLEU outcomes. Repeated-4-grams
and distinct-3 measure repetition **inside one response**. Self-BLEU measures
similarity **between the two responses to a prompt**. TextSeal's within-response
repetition improves substantially with fallback even though its 1,024-token
Self-BLEU increases. Gumbel-on can avoid loops while still sharing substantial
text across its two generations. The saved detection and Self-BLEU results are
unchanged by this analysis.

## Supplemental medians and reproducibility

Earlier baseline diagnostics also reported medians. For comparison, these are
medians over the same 100 response slots at 1,024 tokens; they are not mixed with
the means above or with earlier larger/different prompt cohorts.

| Method | Median repeated token 4-grams ↓ | Median distinct-3 ↑ |
|---|---:|---:|
| PRC η=.05 | 1.03% | 97.16% |
| SynthID depth 10, fallback on | 1.18% | 96.92% |
| TextSeal α=.1, fallback on | 1.71% | 95.50% |
| Gumbel-Max, fallback on | 1.81% | 95.94% |
| Ordinary sampling | 1.18% | 96.77% |

All **1,000 historical prompt-level metric values** (five native configurations,
50 prompts, two lengths and two metrics) were reproduced to within 1e-12 before
interpreting the new values. Source and completion hashes are verified. The
analysis covers 800 response slots across all native/modified configurations and
1,600 response-prefix records. Gumbel-off's two seed slots are identical; the
bootstrap keeps prompt clusters together rather than treating slots as independent.

[Summary, paired contrasts and provenance](summary.json) ·
[Per-response prefix metrics](response_metrics.json) ·
[Reproduction script](analyze.py).
The script imports existing scoring functions and bootstrap code; it does not
change the frozen generator or detector sources. Restore Stage A and the repeat
ablation artifacts using their existing archive/index records, then run locally:

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python outputs/self_bleu_repeat/matched_repetition/analyze.py
```

No model inference, generation or Modal dispatch was needed. Incremental Modal
cost is **$0**; the study's cumulative planning charge remains **$6.70584**.
