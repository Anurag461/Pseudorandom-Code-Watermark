# Paired PRC comparisons: Self-BLEU, repetition and null counts

**PRC has lower Self-BLEU than TextSeal and Gumbel at 400 and 1,024 tokens under
both native generation policies and the fallback-on comparison. Each of those
paired 95% intervals is below zero. The PRC–SynthID intervals include zero in
both views.** With fallback enabled, PRC's repeated-4-gram and distinct-3
differences from every baseline remain uncertain. The saved detection results
also show lower observed PRC TPR, especially at 400 tokens, so this is not a
claim of overall dominance or matched-FPR superiority.

All calculations use the saved 50-prompt cohort with two responses per prompt.
This report does not combine it with the 500-prompt watermarked cohort.
No new generation, inference, detector calls or Modal jobs were needed.

## Direct paired Self-BLEU differences

For each prompt i, compute `d_i = SelfBLEU(PRC_i) − SelfBLEU(baseline_i)`.
Each Self-BLEU is the symmetric sentence BLEU between that prompt's two responses.
Average the 50 differences. For each bootstrap draw, sample 50 prompt indices
with replacement and average the corresponding differences, preserving the
PRC/baseline pairing and both response slots. Report the 2.5th/97.5th percentiles
of 2,000 draws, seed 20260918, with the exact frozen draw hash.

These are **direct paired contrast intervals**. They are not differences of
marginal confidence limits, nor constructed by combining separate policy-effect
intervals. Negative Self-BLEU differences favor PRC. Self-BLEU remains on its
0–1 scale; it is not reported as percentage points.

Native generation policies mean TextSeal/Gumbel fallback off and SynthID fallback
on in the pinned comparison implementations. In the fallback-on view, all three
contextual baselines enable fallback. PRC is position based; it and ordinary
sampling are unchanged. Native context initialization and RNGs are retained.
SynthID therefore uses the same responses in both views; its repeated table
entry is not additional evidence. The SynthID-off diagnostic is not substituted
for its native policy in either view.

### Native generation policies

| PRC minus baseline | 400 tokens: difference [95% interval] | 1,024 tokens: difference [95% interval] |
|---|---:|---:|
| TextSeal α=.1, off | -0.03731 [-0.04821, -0.02716] | -0.01880 [-0.02560, -0.01205] |
| SynthID depth 10, on | -0.00286 [-0.00664, +0.00091] | -0.00316 [-0.00731, +0.00102] |
| Gumbel-Max, off | -0.98099 [-0.98360, -0.97818] | -0.98110 [-0.98419, -0.97773] |

### Fallback-on comparison

| PRC minus baseline | 400 tokens: difference [95% interval] | 1,024 tokens: difference [95% interval] |
|---|---:|---:|
| TextSeal α=.1, on | -0.03920 [-0.05065, -0.02844] | -0.02837 [-0.03531, -0.02201] |
| SynthID depth 10, on | -0.00286 [-0.00664, +0.00091] | -0.00316 [-0.00731, +0.00102] |
| Gumbel-Max, on | -0.37474 [-0.42993, -0.32084] | -0.18829 [-0.21248, -0.16445] |

## Consolidated absolute results

These are means over the same 50 prompt pairs, with within-response repetition
first averaged over the two responses. Self-BLEU and repeated-4-grams are lower-is-
better; distinct-3 is higher-is-better. Repeated-4-grams are `1 − unique_4/(T−3)`;
distinct-3 is `unique_3/(T−2)`, computed on completion token IDs only. These
metrics differ from generation-time repeated-context/fallback activation counts.
All marginal metric intervals are retained in `summary.json`.

### Native policies

| Tokens | Method/policy | Self-BLEU ↓ | Repeated 4-grams ↓ | Distinct-3 ↑ | Detected |
|---:|---|---:|---:|---:|---:|
| 400 | PRC η=.05 | 0.01901 | 1.09% | 97.47% | 55/100 |
| 400 | TextSeal α=.1, off | 0.05632 | 10.12% | 88.44% | 100/100 |
| 400 | SynthID depth 10, on | 0.02187 | 1.01% | 97.65% | 100/100 |
| 400 | Gumbel-Max, off | 1.00000 | 25.42% | 73.45% | 100/100 |
| 400 | Ordinary sampling | 0.01871 | 1.29% | 97.34% | — |
| 1,024 | PRC η=.05 | 0.01890 | 2.38% | 95.61% | 97/100 |
| 1,024 | TextSeal α=.1, off | 0.03770 | 32.17% | 66.03% | 100/100 |
| 1,024 | SynthID depth 10, on | 0.02206 | 2.42% | 95.54% | 100/100 |
| 1,024 | Gumbel-Max, off | 1.00000 | 51.73% | 47.20% | 100/100 |
| 1,024 | Ordinary sampling | 0.01928 | 2.92% | 95.14% | — |

### Fallback on

| Tokens | Method/policy | Self-BLEU ↓ | Repeated 4-grams ↓ | Distinct-3 ↑ | Detected |
|---:|---|---:|---:|---:|---:|
| 400 | PRC η=.05 | 0.01901 | 1.09% | 97.47% | 55/100 |
| 400 | TextSeal α=.1, on | 0.05821 | 1.30% | 97.11% | 100/100 |
| 400 | SynthID depth 10, on | 0.02187 | 1.01% | 97.65% | 100/100 |
| 400 | Gumbel-Max, on | 0.39375 | 1.22% | 97.06% | 100/100 |
| 400 | Ordinary sampling | 0.01871 | 1.29% | 97.34% | — |
| 1,024 | PRC η=.05 | 0.01890 | 2.38% | 95.61% | 97/100 |
| 1,024 | TextSeal α=.1, on | 0.04727 | 2.74% | 94.64% | 100/100 |
| 1,024 | SynthID depth 10, on | 0.02206 | 2.42% | 95.54% | 100/100 |
| 1,024 | Gumbel-Max, on | 0.20718 | 2.81% | 94.56% | 100/100 |
| 1,024 | Ordinary sampling | 0.01928 | 2.92% | 95.14% | — |

## Direct paired repetition contrasts

Differences below are **percentage points**, PRC minus baseline, with the same
paired prompt bootstrap. Negative favors PRC for repeated-4-grams; positive
favors PRC for distinct-3. The fallback-on contrasts reproduce the completed
repetition analysis; native-policy contrasts are included for comparison.

### Native policies

| Tokens | PRC minus baseline | Repeated-4-gram difference [95% interval], pp | Distinct-3 difference [95% interval], pp |
|---:|---|---:|---:|
| 400 | TextSeal α=.1, off | -9.03 [-11.72, -6.48] | +9.03 [+6.44, +11.65] |
| 400 | SynthID depth 10, on | +0.08 [-0.29, +0.44] | -0.18 [-0.75, +0.37] |
| 400 | Gumbel-Max, off | -24.34 [-31.51, -17.51] | +24.03 [+17.23, +31.15] |
| 1,024 | TextSeal α=.1, off | -29.79 [-34.57, -24.92] | +29.58 [+24.74, +34.33] |
| 1,024 | SynthID depth 10, on | -0.04 [-0.91, +0.80] | +0.08 [-0.97, +1.15] |
| 1,024 | Gumbel-Max, off | -49.35 [-57.08, -41.75] | +48.42 [+40.92, +56.05] |

### Fallback on

| Tokens | PRC minus baseline | Repeated-4-gram difference [95% interval], pp | Distinct-3 difference [95% interval], pp |
|---:|---|---:|---:|
| 400 | TextSeal α=.1, on | -0.22 [-0.67, +0.25] | +0.37 [-0.27, +1.00] |
| 400 | SynthID depth 10, on | +0.08 [-0.29, +0.44] | -0.18 [-0.75, +0.37] |
| 400 | Gumbel-Max, on | -0.13 [-0.54, +0.27] | +0.42 [-0.24, +1.07] |
| 1,024 | TextSeal α=.1, on | -0.36 [-1.20, +0.49] | +0.97 [-0.09, +2.03] |
| 1,024 | SynthID depth 10, on | -0.04 [-0.91, +0.80] | +0.08 [-0.97, +1.15] |
| 1,024 | Gumbel-Max, on | -0.43 [-1.60, +0.65] | +1.05 [-0.29, +2.49] |

The large native-policy within-response repetition differences for TextSeal and
Gumbel shrink with fallback on; all fallback-on repetition contrast intervals
include zero. However, their between-response Self-BLEU contrasts with PRC remain
below zero. These findings describe different aspects of diversity and do not
conflict. They do not establish better overall semantic quality.

## Detection and existing null counts

PRC detects 55/100 at 400 tokens and 97/100 at 1,024; each baseline configuration
shown detects 100/100. Consequently, every paired PRC-minus-baseline TPR contrast
is **−45 pp [−56, −35]** at 400 and **−3 pp [−7, 0]** at 1,024. All-success
bootstrap intervals for a baseline do not establish perfect population detection.

All detectors keep their existing completion-only rules and nominal p < .001
thresholds. SynthID uses the study's official repeated-context detector mask;
TextSeal and Gumbel retain their existing rules. Native in this report describes
generation policy, not a switch to the legacy 500-prompt SynthID tuple-mask
scorer. The null evidence below matches the study's detectors.

| Detector | Pilot nulls: 400 | Pilot nulls: 1,024 | Historical shared nulls: 400 | Historical shared nulls: 1,024 |
|---|---:|---:|---:|---:|
| PRC | 0/100 | 0/100 | 0/500 | 0/500 |
| TextSeal | 0/100 | 0/100 | 0/500 | 0/500 |
| SynthID | 1/100 | 0/100 | 0/500 | 0/500 |
| Gumbel-Max | 0/100 | 0/100 | 2/500 | 0/500 |

The same null counts apply in both policy views because only generation fallback
changed; detector configurations did not. They are reused evidence, not new null
runs. Pilot nulls are 100 responses from 50 prompt clusters. Historical shared
nulls comprise 500 responses and overlap those prompts; **do not pool them into
600 independent observations** or use them as a held-out calibrated test set.

**Count correction:** Gumbel's historical shared-null count at 400 is **2/500**,
not zero. This is confirmed by the corrected comparison CSV, saved token scores
and Stage A's machine-readable summary. It supersedes the earlier Stage A prose
claim that all historical counts were zero at both primary lengths. SynthID's
pilot count at 400 is **1/100**. All listed 1,024-token counts are zero.

No matched empirical FPR calibration was performed. Even 0/500 has an approximate
0.735% upper endpoint for a two-sided exact binomial 95% interval under independent
sampling assumptions; it does not certify 0.1% FPR. The fresh clustered cohort
requires its own dependence-aware treatment. Raw counts should accompany the
nominal-threshold detection comparison rather than be presented as equal FPR.

## Scope and verification

The model and generation settings remain Qwen3-8B-Base, original 50 prompts,
seeds 12345/67890, fixed keys, temperature/top-p 1, and 1,024 generated tokens.
Configurations are online PRC η=.05, TextSeal α=.1, SynthID depth 10 and Gumbel-Max.
Self-BLEU uses the original SacreBLEU 2.4.3 signature. The bootstrap preserves
50 prompt units; deterministic native Gumbel's duplicate seed slots do not
increase the number of independent observations. All intervals are exploratory,
marginal and unadjusted for multiple comparisons.

- Recomputed all 200 modified prompt-level Self-BLEU values from hash-verified
  TextSeal/Gumbel completions and the pinned tokenizer, confirming prompt pairing.
- Reproduced all six native Self-BLEU contrasts and existing detection contrasts.
- Verified all absolute means/intervals against saved pilot, ablation and
  repetition summaries, including the existing fallback-on repetition contrasts.
- Recounted fresh-null decisions from saved scores and verified the historical
  counts against the applicable score records/CSV. No detector was called again.
- Recorded source hashes, the exact bootstrap draw hash and 700 prompt-level
  metric rows across seven unique configurations and two lengths.

The result supports lower Self-BLEU for PRC than the evaluated TextSeal/Gumbel
settings under either generation-policy view, but no established Self-BLEU
advantage over native SynthID. It does not establish within-response repetition
superiority with fallback on, matched-FPR superiority, a parameter frontier,
robustness across keys/models, or a comparison to Bayesian SynthID.

[Consolidated results and source hashes](summary.json) ·
[Paired prompt-level metric records](prompt_metrics.json) ·
[Reproduction script](analyze.py) ·
[Previous repetition report](../matched_repetition/REPORT.md) ·
[Repeat-policy results](../setup_v4/FOLLOWUP_REPORT.md).

With the existing pinned CPU analysis environment and saved raw artifacts:

```sh
HF_HOME=/path/to/writable/hf-cache NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python outputs/self_bleu_repeat/paired_comparison/analyze.py
```

Incremental Modal cost: **$0**. No new generation or model inference; the Self-BLEU
study's cumulative planning charge remains approximately **$6.71**, including
previous allowances. The original experiment records remain unchanged.
