# Paper assets and reproduction

The campaign is closed. This package is generated offline from saved results.

- [Concise comparison report](REPORT.md): principal findings, essential tables and methodological limitations.
- [Detailed results PDF](comparison_report.pdf): the complete September 19 results and diagnostics, retained as supporting material.
- [LaTeX assembly](paper_assets.tex): figure/table fragments can also be included independently.
- `figures/`: vector PDF/SVG and 300-dpi PNG exports.
- `tables/`: matching LaTeX fragments and human-readable CSVs.
- `data/`: full-precision normalized values, all paired contrasts, source hashes and checks.

REPORT.md is maintained separately. The asset builder preserves it and regenerates only the detailed supporting material.

## Figure catalogue

1. **01_redetection**: Historical 500-response-per-setting comparison after completion-only correction. Left: each cutoff is a separate one-shot detection test; coincident baseline curves are retained. Right: 1,024-token repetition under the native mixed policies, with TextSeal/Gumbel fallback OFF. These repetition gaps are not the matched-policy result.

2. **02_matched_policy_tradeoff**: 8B T=1 comparison with fallback ON for all contextual baselines, at 400 and 1,024 tokens. Points and error bars show means and marginal 95% prompt-bootstrap intervals. Ordinary Self-BLEU is a gray vertical mean/band because ordinary text has no watermarked TPR. The logarithmic Self-BLEU axis retains Gumbel without hiding the near-ordinary settings. This is a set of tested configurations, not a fitted frontier or equal-FPR comparison.

3. **04_repeat_policy**: Repeat fallback at 1,024 tokens: open circles are OFF, filled circles ON; each line connects the same method under the two policies. Left: between-response Self-BLEU on a log scale. Right: within-response repeated-four-gram fraction. Marginal 95% intervals are shown; paired policy effects are reported in the tables. Reduced within-response repetition need not reduce between-response overlap.

4. **05_synthid_depth**: SynthID depth comparison on saved 8B full-vocabulary T=1 responses. Left: detection at 64/128/256 tokens with prompt-bootstrap intervals; depth-10 and depth-30 curves coincide. Right: Self-BLEU at the original 400/1,024-token endpoints. Depth 30 adds observed overlap without an observed detection gain over depth 10 on these measured cutoffs; no population equivalence or Bayesian-detector claim follows.

5. **03_paired_depth2**: Direct paired PRC-minus-SynthID-depth-2 contrasts at 1,024 tokens. Left: Self-BLEU differences, where negative favors PRC. Right: detection differences in percentage points, where positive favors PRC. Error bars are paired prompt-bootstrap 95% intervals. Rows are separate matched-control experiments, not a pure one-factor model-size/truncation sweep; probability arithmetic differs for top-100 and 0.6B. The T=.7 study preserves the original 8B paths.

6. **06_repetition_sensitivity**: Within-response repetition at 1,024 tokens across the four completed regimes, showing ordinary, PRC, and SynthID depths 2/10. Error bars are marginal 95% prompt-bootstrap intervals. Cross-regime numerical paths differ as documented. These metrics concern individual responses and should not be substituted for between-response Self-BLEU.

7. **08_temperature**: The precision-faithful 8B full-vocabulary temperature sensitivity at 1,024 tokens. T=1 uses saved original outputs; T=.7 uses new matched controls. Left: Self-BLEU rises for every arm at the lower temperature. Right: PRC detection falls from 97/100 to 3/100, depth 2 from 100/100 to 93/100, and depth 10 remains 100/100. Intervals use prompt resampling; lines connect the two evaluated temperatures and do not imply an unmeasured sweep.

8. **07_common_histories**: Means on 25 preselected shared histories. Collision probability is the sum of squared token probabilities; maximum probability is the largest token probability. Retained mass is on the ordinary top-100 support after watermarking. Both decoders use FP32 arithmetic, so this controlled distribution probe is distinct from reproducing the historical BF16 full-vocabulary pipeline. No causal claim about whole-response Self-BLEU is inferred from these descriptive means.

## Table catalogue

- [Repeat fallback on/off at 1,024 tokens](tables/repeat_fallback_1024.pdf): revised paper table, including depth-30 repetition metrics; [LaTeX](tables/repeat_fallback_1024.tex).

- [01_experiment_ledger](tables/01_experiment_ledger.tex): Completed experiments and their non-overlapping roles
- [02_protocols](tables/02_protocols.tex): Numerical and policy differences that must accompany cross-study comparisons
- [03_large_detection](tables/03_large_detection.tex): Completion-only detection and shared-null counts (500 responses per cell)
- [03b_prc_redetection_change](tables/03b_prc_redetection_change.tex): PRC detection correction on the same saved historical watermarked responses
- [04_large_repetition](tables/04_large_repetition.tex): Within-response repetition on the historical cohort at 400 and 1,024 tokens
- [05_8b_full_1024](tables/05_8b_full_1024.tex): Original 8B full-vocabulary paired comparison at 1,024 tokens, T=1
- [06_8b_contrasts_1024](tables/06_8b_contrasts_1024.tex): Direct paired PRC-minus-baseline differences at 1,024 tokens
- [05_8b_full_400](tables/05_8b_full_400.tex): Original 8B full-vocabulary paired comparison at 400 tokens, T=1
- [06_8b_contrasts_400](tables/06_8b_contrasts_400.tex): Direct paired PRC-minus-baseline differences at 400 tokens
- [07_repeat_policy](tables/07_repeat_policy.tex): Effect of enabling repeated-context fallback
- [08_trajectories](tables/08_trajectories.tex): Original-to-modified repeat trajectories
- [09_short_prefixes](tables/09_short_prefixes.tex): Saved SynthID generations scored at short prefixes
- [10_8b_topk_1024](tables/10_8b_topk_1024.tex): 8B / top-k=100 / T=1 at 1,024 tokens
- [10_8b_topk_400](tables/10_8b_topk_400.tex): 8B / top-k=100 / T=1 at 400 tokens
- [10_0p6b_full_1024](tables/10_0p6b_full_1024.tex): 0.6B / full vocabulary / T=1 at 1,024 tokens
- [10_0p6b_full_400](tables/10_0p6b_full_400.tex): 0.6B / full vocabulary / T=1 at 400 tokens
- [10_8b_t07_1024](tables/10_8b_t07_1024.tex): 8B / full vocabulary / T=.7 at 1,024 tokens
- [10_8b_t07_400](tables/10_8b_t07_400.tex): 8B / full vocabulary / T=.7 at 400 tokens
- [11_primary_contrasts](tables/11_primary_contrasts.tex): PRC-minus-SynthID-depth-2 results across completed regimes
- [12_nulls_8b_full](tables/12_nulls_8b_full.tex): Null counts: 8B / full vocabulary / T=1
- [12_nulls_8b_topk](tables/12_nulls_8b_topk.tex): Null counts: 8B / top-k=100 / T=1
- [12_nulls_0p6b_full](tables/12_nulls_0p6b_full.tex): Null counts: 0.6B / full vocabulary / T=1
- [12_nulls_8b_t07](tables/12_nulls_8b_t07.tex): Null counts: 8B / full vocabulary / T=.7
- [13_replay_topk](tables/13_replay_topk.tex): Prefix-specific replay observations: 8B top-100
- [13_replay_small](tables/13_replay_small.tex): Prefix-specific replay observations: 0.6B full vocabulary
- [13_replay_temperature](tables/13_replay_temperature.tex): Prefix-specific replay observations: 8B T=.7
- [14_common_histories](tables/14_common_histories.tex): Distribution measurements on 25 preselected common histories
- [15_cost](tables/15_cost.tex): Self-BLEU campaign cumulative planning charges
- [16_all_contrasts_8b_full_1024](tables/16_all_contrasts_8b_full_1024.tex): All contrasts: 8B / full vocabulary / T=1, 1,024 tokens
- [16_all_contrasts_8b_full_400](tables/16_all_contrasts_8b_full_400.tex): All contrasts: 8B / full vocabulary / T=1, 400 tokens
- [16_all_contrasts_8b_topk_1024](tables/16_all_contrasts_8b_topk_1024.tex): All contrasts: 8B / top-k=100 / T=1, 1,024 tokens
- [16_all_contrasts_8b_topk_400](tables/16_all_contrasts_8b_topk_400.tex): All contrasts: 8B / top-k=100 / T=1, 400 tokens
- [16_all_contrasts_0p6b_full_1024](tables/16_all_contrasts_0p6b_full_1024.tex): All contrasts: 0.6B / full vocabulary / T=1, 1,024 tokens
- [16_all_contrasts_0p6b_full_400](tables/16_all_contrasts_0p6b_full_400.tex): All contrasts: 0.6B / full vocabulary / T=1, 400 tokens
- [16_all_contrasts_8b_t07_1024](tables/16_all_contrasts_8b_t07_1024.tex): All contrasts: 8B / full vocabulary / T=.7, 1,024 tokens
- [16_all_contrasts_8b_t07_400](tables/16_all_contrasts_8b_t07_400.tex): All contrasts: 8B / full vocabulary / T=.7, 400 tokens

Use figures 01-03 and the 500-prompt/matched-policy tables for the central comparison; the remaining figures document interventions and sensitivity. Keep the cohort/mask/precision captions when moving assets into a manuscript. Figure numbering in this package is an asset identifier, not a proposed final manuscript numbering.

## Offline build

```sh
MPLCONFIGDIR=/tmp/prc-comparison-matplotlib python reports/comparisons/build.py
python reports/comparisons/build.py --pdf
```

First command: Python, NumPy and Matplotlib. Second: ReportLab. It reads only saved aggregate data; there are no Modal imports or dispatch calls. Rebuild from the repository root. Run the two commands in environments providing those dependencies. The numerical tables are not rounded until formatting; CSV data exports retain full precision.

Every cited source is hashed in `data/source_manifest.json`. Repeated source cells and contrast means are checked. PDFs are rendered and visually inspected before delivery. Source reports remain untouched; historical prompted/proxy results are excluded from paper-ready numerical panels.

The export archive contains this entire report package, without raw completions, keys or model traces. It is built after visual verification.
