# Fixed versus online PRC: paired analysis

## Results

| Model | eta | n | Detector | Fixed TPR | Online TPR | Fixed - online (pp) | 95% paired CI (pp) | Holm p | Interpretation |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| Qwen3-0.6B-Base | 0.05 | 448 | map | 93.2% | 91.0% | 2.2 | [-1.2, 5.6] | 1 | inconclusive |
| Qwen3-0.6B-Base | 0.05 | 448 | entropy | 79.8% | 77.6% | 2.2 | [-2.8, 7.4] | 1 | inconclusive |
| Qwen3-0.6B-Base | 0.05 | 448 | naive | 58.2% | 52.0% | 6.2 | [0.2, 12.2] | 0.9709 | fixed_better |
| Qwen3-0.6B-Base | 0.15 | 1504 | map | 91.0% | 90.6% | 0.4 | [-3.0, 4.0] | 1 | practically_equivalent |
| Qwen3-0.6B-Base | 0.15 | 1504 | entropy | 84.8% | 79.6% | 5.2 | [0.8, 9.6] | 0.5834 | fixed_better |
| Qwen3-0.6B-Base | 0.15 | 1504 | naive | 68.2% | 55.8% | 12.4 | [6.4, 18.2] | 0.001186 | fixed_better |
| Qwen3-8B-Base | 0.05 | 416 | map | 69.6% | 68.2% | 1.4 | [-4.2, 7.0] | 1 | inconclusive |
| Qwen3-8B-Base | 0.05 | 416 | entropy | 44.8% | 48.0% | -3.2 | [-9.0, 2.8] | 1 | inconclusive |
| Qwen3-8B-Base | 0.05 | 416 | naive | 21.0% | 20.8% | 0.2 | [-4.6, 4.8] | 1 | practically_equivalent |
| Qwen3-8B-Base | 0.05 | 749 | map | 91.0% | 93.0% | -2.0 | [-5.2, 1.2] | 1 | inconclusive |
| Qwen3-8B-Base | 0.05 | 749 | entropy | 79.0% | 78.8% | 0.2 | [-4.6, 5.0] | 1 | practically_equivalent |
| Qwen3-8B-Base | 0.05 | 749 | naive | 49.0% | 46.6% | 2.4 | [-3.4, 8.2] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 768 | map | 66.4% | 68.6% | -2.2 | [-7.8, 3.6] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 768 | entropy | 46.0% | 45.4% | 0.6 | [-5.4, 6.6] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 768 | naive | 21.2% | 17.8% | 3.4 | [-1.4, 8.2] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1382 | map | 86.0% | 89.8% | -3.8 | [-7.4, 0.0] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1382 | entropy | 77.4% | 78.8% | -1.4 | [-6.4, 3.6] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1382 | naive | 51.8% | 46.2% | 5.6 | [-0.2, 11.4] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1625 | map | 89.2% | 92.4% | -3.2 | [-6.4, 0.0] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1625 | entropy | 82.6% | 84.0% | -1.4 | [-5.4, 2.8] | 1 | inconclusive |
| Qwen3-8B-Base | 0.10 | 1625 | naive | 60.0% | 54.6% | 5.4 | [-0.4, 11.2] | 1 | inconclusive |

## Conclusion

Across the predeclared family of 21 comparisons: fixed better: 3, inconclusive: 15, practically equivalent: 3.
For Qwen3-8B specifically, neither construction was called better in any of the 15 comparisons; 2 were practically equivalent and 13 were inconclusive. The data therefore do not establish a meaningful fixed-versus-online difference for 8B, but most intervals remain too wide to establish practical equivalence.
For MAP, no cell called either construction better. The 0.6B eta=0.15 cell was practically equivalent; the other six MAP cells were inconclusive.
Only 1 of 21 exact McNemar tests remained significant after Holm correction: 0.6B, eta=0.15, n=1504, naive (fixed - online = 12.4 points; adjusted p = 0.00119). The other two `fixed_better` labels follow the predeclared effect-plus-bootstrap-interval rule but do not survive familywise McNemar correction.
The row-level interpretations above apply the predeclared rules exactly; no conclusion is based on an unpaired aggregate comparison.

## Methods

Records were paired by prompt index (500 prompts per cell). Effects are fixed minus online TPR. The 95% percentile intervals use 10,000 deterministic paired prompt resamples with global analysis seed `20260822` and recorded comparison-specific seeds. Exact two-sided McNemar tests use discordant decisions; Holm correction is applied once across all 21 tests.

The smallest meaningful difference was predeclared as 5 percentage points. Practical equivalence requires the entire paired interval within [-5, +5] points. A method is called better only when the interval excludes zero and the estimated absolute effect is at least 5 points. All other results are inconclusive.

## Validation and provenance

All inputs passed exact prompt coverage, uniqueness, model, eta, length, nominal-FPR, experiment-seed, detector-completeness, statistic, threshold, and source-count checks before inference. The five new 8B inputs additionally require `execution_mode=cache_only`, prefix reuse, and a reconciled ledger showing zero generated tokens, no GPU worker, and total provider cost at or below $0.25.

Machine-readable records: [`outputs/fixed_vs_online_prompt_level.jsonl`](outputs/fixed_vs_online_prompt_level.jsonl) and [`outputs/fixed_vs_online_paired_summary.csv`](outputs/fixed_vs_online_paired_summary.csv). Validation/fingerprints: [`outputs/fixed_vs_online_validation.json`](outputs/fixed_vs_online_validation.json). Cost ledger: [`outputs/fixed_vs_online_cache_only_cost_ledger.csv`](outputs/fixed_vs_online_cache_only_cost_ledger.csv).

## Limitations

The analysis covers seven preselected model/eta/length cells and three detectors, not every possible operating point. Fixed and online keys are construction-specific, domain-separated derivations of the same experiment seed; they are not the same key object. Pairing therefore controls prompts, configuration, and seed, but not token-for-token watermark randomness. The online audit artifacts preserve detector scores and source fingerprints but not prompt text.
