# Multiplicity-adjusted response-diversity comparison

All 11 comparator cells previously flagged by unadjusted paired-bootstrap intervals remain significant after Holm correction across all 35 comparisons in the 1,024-token table. The means, displayed confidence intervals, caption findings, and bold cells are unchanged; the table notes now describe the corrected tests.

- SynthID depth 30 with repeat handling: Self-BLEU raw permutation p = 0.000480; Holm-adjusted p = 0.0120.
- The other ten retained comparisons: Monte Carlo raw p = 0.0000200 and Holm-adjusted p = 0.000700. The raw values are at the simulation resolution, not zero.
- Unwatermarked responses and SynthID depths 2 and 10 with repeat handling show no significant differences from PRC on any of the three diversity metrics. This does not establish equivalence or optimality.
- No 1,024-token detection comparison is significant. Its exact paired p-value is 0.25 before correction.
- Every adjusted decision is stable within the simultaneous 99.9% Monte Carlo precision bounds recorded in the results.

## Paper asset

Use [the table with uncertainty](../tables/repeat_fallback_1024_uncertainty.tex). It requires `booktabs` and is a standalone `table*` fragment. Existing paired-bootstrap 95% confidence bounds remain unadjusted; comparator bolding uses Holm-adjusted tests instead of those bounds. PRC estimates remain bold as the reference row.

The earlier report and aggregate exports retain their original unadjusted analyses. This directory is the corrected inference source for this specific table, not a correction of every analysis in the repository.

## Statistical protocol

The family includes 27 diversity comparisons (nine comparator settings times three metrics) and eight detection comparisons with PRC, all at 1,024 tokens. All settings use the same 50 prompts with two responses each. Repeated-4-gram fraction, Distinct-3 and detection are averaged over the two response slots before testing; Self-BLEU already provides one value per prompt. Each test therefore has 50 paired prompt-level differences, not 100 independent observations.

Two-sided paired permutation tests use the mean PRC-minus-comparator difference. Method labels are swapped within prompts, keeping both response slots together. The 27 diversity tests use 100,000 random assignments, seed 20260925, in batches of 2,000. The eight detection tests enumerate all eight effective assignments of their three nonzero prompt differences. Simulated p-values use twice the smaller one-sided `(count+1)/(100000+1)` tail, capped at one. Holm adjustment is applied jointly at family-wise alpha 0.05.

Holm allows dependence between tests, including their shared PRC reference and correlated metrics. Validity of the underlying paired tests requires method-label exchangeability under the null and independent prompt clusters. These are not assumption-free tests of equality of means. Results condition on the fixed keys, model and sampling settings. The protocol was fixed before computing the new test results.

The original bootstrap decision, new raw permutation decision, and Holm decision are recorded separately. All three flag the same 11 cells. Clopper–Pearson intervals with Bonferroni allocation over all simulated tails provide simultaneous 99.9% Monte Carlo precision bounds. Applying Holm to their lower and upper p-values leaves every decision unchanged. These bounds describe resampling precision, not sampling uncertainty in the population.

The 400-token detection comparison, the separate 864-token length sweep, and direct handling-on versus handling-off tests are outside this family. No new responses, token metrics, detector scores or models were computed for this analysis.

## Data and reproduction

- [paired_tests_holm.csv](paired_tests_holm.csv): compact results for all 35 comparisons.
- [results.json](results.json): raw and adjusted p-values, exact versus simulated tests, original flags, numerical-precision checks and the complete statistical protocol.
- [paired_inputs.json](paired_inputs.json): all paired scalar inputs and existing marginal intervals, without prompts, completions, watermark keys or traces.
- [analyze.py](analyze.py): the exact analysis source used in the successful run.
- [provenance.json](provenance.json): original source hashes, payload and code hashes, source-assembly notes and input checks. All 39 displayed point estimates were reproduced from the saved inputs.

The completed run used Python 3.12, NumPy 1.26.4 and SciPy 1.13.1. Re-execution performs statistical resampling; use an approved CPU environment. From this directory:

```python
import json
from pathlib import Path
from analyze import analyze

inputs = json.loads(Path("paired_inputs.json").read_text())
saved = json.loads(Path("results.json").read_text())
recomputed = analyze(inputs, saved["protocol"])
assert recomputed["rows"] == saved["rows"]
```

No cloud runner, billing records, authorization records, failed-run logs, duplicate source snapshots or rendered preview files are included in this package.

References: [SciPy paired permutation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html); [Holm adjustment](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html).
