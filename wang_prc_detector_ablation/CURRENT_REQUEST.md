# Current request — September 23, 2026

User instruction accompanying this attachment:

> Ignore the previous official-Qwen-model correction and return to Qwen3-8B-Base, reasoning off. If the initial sanity check is estimated to cost more than $5, skip it.

The text below is the latest detector/threshold specification and overrides conflicting earlier requirements.

## Important correction: preserve Wang et al.'s published detector as the primary baseline

Supersede the previous instructions that made the empirically recalibrated Wang hard detector the primary baseline.

We want to distinguish two separate questions:

1. **How does our detector compare against Wang et al.'s actual published detection method?**
2. **Does posterior scoring contain more useful detection information than Wang's hard statistic after controlling for false-positive rate?**

Therefore report both a **faithful-method comparison** and a **matched-FPR ablation**.

---

# 1. Primary comparison: use both methods exactly as specified

## Wang detector

Use Wang et al.'s hard detector exactly as implemented in their released code.

Let

$$
V_{\mathrm{hard}}
$$

be the number of violated parity checks.

Their published rule is:

$$
V_{\mathrm{hard}}
\le
\left(\frac12-r^{-1/4}\right)r
=
\frac r2-r^{3/4}.
$$

Use their inclusive `<=` decision rule exactly.

Do not empirically alter this threshold for the primary Wang result.

Record:

* `V_hard`;
* Wang's published threshold;
* Wang's published binary decision.

Label this detector in all tables/figures as something like:

**Wang hard detector (published threshold)**

Do not call an empirically recalibrated version simply "Wang detector."

---

## Our posterior detector

Use our posterior detector exactly according to our existing standard redetection protocol.

Primary detection is:

* completion-only;
* no original prompt;
* no BOS/EOT/chat-template context;
* first generated token: abstain on all associated hierarchical coordinates by setting their soft evidence to zero;
* all subsequent probabilities reconstructed from preceding completion tokens only.

For each parity check, compute the soft parity evidence as already specified.

Maintain:

$$
S=\sum_w q_w
$$

and

$$
V=\sum_w
\left(\prod_{j\in w}s_j\right)^2.
$$

Use our standard threshold:

$$
\tau_{\mathrm{post}}
=
\sqrt{2V\log 1000}.
$$

Declare the sample watermarked when

$$
S\ge \tau_{\mathrm{post}}.
$$

Equivalently, if storing the normalized score

$$
Z=S/\sqrt{V},
$$

the threshold is

$$
Z\ge\sqrt{2\log 1000}.
$$

Use the same convention already used elsewhere in our paper/code. Do not empirically retune this threshold for the primary posterior result.

Label this detector:

**Posterior detector (standard threshold, FPR target \(10^{-3}\))**

---

# 2. Primary table

For each temperature, first report the two actual methods:

| Temperature |  N | Wang published TPR | Posterior standard TPR | Wang realized null FPR | Posterior realized null FPR |
| ----------: | -: | -----------------: | ---------------------: | ---------------------: | --------------------------: |
|         1.0 |    |                    |                        |                        |                             |
|         1.2 |    |                    |                        |                        |                             |
|         1.4 |    |                    |                        |                        |                             |
|         1.6 |    |                    |                        |                        |                             |
|         1.8 |    |                    |                        |                        |                             |

These are the primary results.

Do not force these two methods to have the same FPR. They are being shown as the methods are actually specified.

Report realized null FPR so that any difference in operating point is transparent.

---

# 3. Secondary analysis: matched-FPR detector-statistic ablation

Separately ask:

> Holding the target false-positive rate fixed, is posterior soft evidence more informative than Wang's hard parity-count statistic?

For this analysis, ignore the published decision thresholds and treat both detectors as continuous/raw scores.

Use:

* Wang raw hard statistic `V_hard`, where lower is more watermarked;
* posterior raw score `Z` (or equivalently `S` together with `V`), where higher is more watermarked.

Calibrate both on the **same designated null calibration split**.

Target:

$$
\mathrm{FPR}=10^{-3}.
$$

## Hard statistic calibration

Because `V_hard` is integer-valued, choose the most permissive integer cutoff \(c_{\rm hard}\) satisfying

$$
\widehat{\Pr}_{H_0}
\left[
V_{\rm hard}\le c_{\rm hard}
\right]
\le10^{-3}.
$$

Use the inclusive decision

$$
V_{\rm hard}\le c_{\rm hard}.
$$

Do not randomly split ties. If including the next complete integer score level would exceed the FPR target, exclude it.

Call this:

**Wang hard statistic @ matched FPR**

not "Wang detector."

## Posterior statistic calibration

Using the same null calibration data, choose the most permissive posterior cutoff \(c_{\rm post}\) satisfying

$$
\widehat{\Pr}_{H_0}
\left[
Z\ge c_{\rm post}
\right]
\le10^{-3}.
$$

Use:

$$
Z\ge c_{\rm post}.
$$

Call this:

**Posterior statistic @ matched FPR**

This is an auxiliary detector-quality ablation, not the primary method comparison.

---

# 4. Calibration/evaluation separation

Never use watermarked samples to choose matched-FPR thresholds.

Use only the predesignated unwatermarked/null calibration split.

Freeze `c_hard` and `c_post`, then evaluate:

* TPR on watermarked samples;
* realized FPR on a completely held-out null set.

If using the previously specified cross-key null augmentation, keep it, but:

* calibration and held-out evaluation keys must be disjoint;
* do not treat all key/text cross-products as statistically independent for confidence intervals;
* cluster/bootstrap at the completion/prompt/key-group level as previously specified.

Use a single global matched-FPR cutoff per statistic across temperatures for the primary matched-FPR ablation, rather than tuning separate thresholds at each temperature.

---

# 5. Secondary matched-FPR table

Produce a second table:

| Temperature | Hard statistic @ \(10^{-3}\) TPR | Posterior statistic @ \(10^{-3}\) TPR | Hard realized FPR | Posterior realized FPR |
| ----------: | -------------------------------: | ------------------------------------: | ----------------: | ---------------------: |
|         1.0 |                                  |                                       |                   |                        |
|         1.2 |                                  |                                       |                   |                        |
|         1.4 |                                  |                                       |                   |                        |
|         1.6 |                                  |                                       |                   |                        |
|         1.8 |                                  |                                       |                   |                        |

This table answers whether posterior scoring itself improves detection after controlling for false-positive rate.

---

# 6. ROC analysis

Continue to compute ROC curves from the raw continuous statistics:

* Wang hard statistic;
* posterior statistic.

ROC/AUC analysis is threshold-independent and therefore provides another clean comparison of the information contained in the two statistics.

Especially show the low-temperature cases:

* `T=1.0`
* `T=1.2`
* `T=1.4`

---

# 7. Interpretation rules

Keep the interpretations distinct.

### If posterior beats Wang's published detector

This establishes that our full detection method performs better than the detection method actually evaluated by Wang et al. under this experimental setup.

However, by itself it does **not** establish that posterior scoring is intrinsically more informative, because the two published methods may operate at different FPRs.

### If posterior also beats the recalibrated hard statistic at matched FPR

This is the stronger detector result:

> Posterior probability-aware scoring extracts more useful watermark evidence from the same generated text than hard parity scoring.

### If posterior beats Wang's published method but not the matched-FPR hard statistic

Then the improvement is primarily attributable to Wang's conservative decision threshold rather than to soft/posterior evidence.

This distinction must be explicit in the analysis.

---

# 8. Oracle handling

The main experiment remains entirely **prompt-free**.

Do not use the oracle detector in either of the two primary comparison tables.

If retained at all, oracle-context posterior detection is only a diagnostic:

* it may use saved generation-time prompt-conditioned probabilities;
* it may score the first generated token normally because those probabilities exist;
* label it clearly as `oracle-context diagnostic`;
* do not use it to support the main detection-performance claim.

For the actual prompt-free posterior detector, continue to abstain on all hierarchical coordinates belonging to the first generated token.

---

# 9. Final result organization

The final output should make the hierarchy unambiguous:

### Main result

**Actual method vs actual method**

* Wang hard detector with Wang's published threshold.
* Our posterior detector with our standard threshold.

### Fair detector-statistic ablation

**Same FPR**

* Wang hard statistic empirically calibrated to FPR \(10^{-3}\).
* Posterior statistic empirically calibrated to FPR \(10^{-3}\).

### Threshold-free supporting analysis

* ROC curves.
* AUC.

Do not replace Wang's published detector with the recalibrated version anywhere in the main result.
