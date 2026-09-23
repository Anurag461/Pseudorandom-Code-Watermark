Implement a controlled detector-ablation experiment comparing Wang et al.'s hard PRC detector against our probability-aware posterior detector.

## Scientific question

Test:

> On exactly the same PRC-watermarked generations, does posterior/soft detection extract more watermark signal than Wang et al.'s hard parity detector, especially in low-entropy generation regimes?

This is **not** an exact reproduction of Wang et al.'s DeepSeek experiment. We deliberately use our established Qwen3-8B inference stack so generation is efficient and consistent with the rest of our paper.

The primary comparison must hold fixed:

* model,
* prompt,
* PRC construction,
* key,
* latent PRC codeword,
* sampled text,

and change only the detector.

---

## 1. Model and generation setup

Use the exact **Qwen3-8B-Base checkpoint, tokenizer, precision, and efficient KV-cached generation backend already used by our existing 8B experiments**.

Do not introduce DeepSeek or a new model backend.

Use:

* sequence length: exactly 1,024 generated tokens;
* full-vocabulary sampling;
* `top_p = 1.0`;
* no top-k truncation;
* temperatures:

  * `1.0`
  * `1.2`
  * `1.4`
  * `1.6`
  * `1.8`;
* the exact 16 prompts used in Wang et al.'s LLM experiment.

Do not stop at EOS; generate exactly 1,024 tokens.

Reuse our existing efficient batched KV-cache implementation. Do not reproduce Wang et al.'s inefficient full-prefix forward pass at every generated token.

---

## 2. Watermark construction

Use **Wang et al.'s fixed-length PRC construction**, not our online PRC and not our fixed one-bit-per-token embedding.

Reimplement the relevant semantics from their official code:

* `KeyGen`;
* `Encode`;
* OTP handling;
* permutation/key handling;
* sparse parity checks;
* `int_to_bits`;
* the hierarchical `sample_token`;
* their original hard `Detect`.

Use their main experimental parameters:

* `t = 3`;
* `eta = 0.1`;
* `r = floor(0.95 n)`.

For Qwen3's vocabulary, use

`token_bits = ceil(log2(vocab_size))`

rather than hard-coding 18, although for this model it should evaluate to 18.

For a 1,024-token output, therefore,

`n = token_bits * 1024`.

Generate and save the full PRC key and OTP for every key group.

---

## 3. Efficient implementation of Wang's hierarchical sampler

At each LLM token position, consume `token_bits` consecutive PRC codeword bits.

Starting from the full token-ID range `[0, 2^token_bits)`, recursively bisect the current interval.

At each hierarchy level:

1. compute the LM probability mass of the lower and upper halves of the current interval;
2. derive the conditional upper-half probability `p`;
3. apply Wang et al.'s binary PRC-conditioned sampling rule using the corresponding latent PRC bit;
4. follow the selected branch;
5. repeat until a token ID is determined.

Token IDs outside the actual vocabulary have zero probability.

Use prefix sums / cumulative probabilities so interval masses are efficient to compute.

The implementation should preserve Wang et al.'s sampler distribution exactly; only the transformer inference implementation should be optimized.

---

## 4. Experimental grid

Initially use **10 independent PRC key groups**.

For every key group:

* generate a PRC key once;
* for every one of the 16 prompts, generate one noisy encoded PRC codeword;
* reuse the same key/prompt identity across temperatures so temperature comparisons are paired.

Run all five temperatures:

`T = {1.0, 1.2, 1.4, 1.6, 1.8}`.

Per temperature this gives:

`10 keys × 16 prompts = 160 watermarked completions`.

Also generate matched ordinary/unwatermarked completions for null calibration/evaluation.

Total initial watermarked sample count:

`5 × 160 = 800`.

Do not automatically scale beyond 10 key groups.

---

## 5. Save enough information to rerun detection without generation

For every sample save:

* temperature;
* key-group ID;
* prompt ID;
* generation seed;
* prompt token IDs;
* generated token IDs;
* decoded text;
* latent PRC codeword;
* OTP;
* parity-check/key information;
* model/checkpoint fingerprint;
* tokenizer fingerprint.

During generation also save, for every token and every hierarchical bit decision:

* observed token-ID bit;
* conditional branch probability `p`;
* latent PRC bit used at that coordinate.

These generation-time probabilities will be useful for validating posterior calculations and for an oracle-context diagnostic.

Detection and plotting must be rerunnable without regenerating text.

---

## 6. Detector A: Wang hard detector

Convert every generated token ID to its `token_bits`-bit MSB-first representation and concatenate them.

Using Wang et al.'s exact OTP/parity-check conventions, compute the hard parity violations exactly as their detector does.

Record:

* number of violated parity checks;
* normalized continuous hard score;
* their original published binary decision.

Use a score orientation where larger always means "more likely watermarked."

Do not use their published threshold as the primary comparison threshold. Preserve it only as an additional reference result.

---

## 7. Detector B: posterior/soft detector adapted to Wang's channel

Do **not** ignore Wang's hierarchical embedding.

The soft detector must infer evidence for each latent PRC coordinate using the same hierarchical binary channel that generated the token.

For each observed token, walk down its binary token-ID path. At hierarchy level `j`:

* recover the conditional probability `p_j` that Wang's sampler would assign to the upper branch;
* observe the actual branch bit `b_j`;
* convert `(b_j, p_j)` into our posterior soft value for the corresponding latent PRC bit.

Use the same `map_soft_token` posterior formula already implemented in our detector code.

For example, for an observed binary branch bit `b` and branch probability `p`, use the established posterior-mean mapping rather than treating `b` as a hard observation.

This yields one soft value per latent PRC coordinate, so a 1,024-token output gives `n` soft PRC-coordinate values.

Then apply Wang's exact PRC parity-check structure and OTP, replacing hard XOR evidence by soft parity evidence.

For parity check `w`, compute the OTP-corrected product of the corresponding soft coordinate values.

Aggregate using our normalized soft-parity statistic:

`S = sum_w q_w`

`V = sum_w magnitude_w^2`

`Z = S / sqrt(V + 1e-12)`.

Preserve `S`, `V`, and `Z`.

Use `Z` as the primary posterior detector score.

---

## 8. Primary re-detection protocol

The primary posterior detector should use our established **completion-only / prompt-free** protocol.

During detection:

* do not supply the original prompt;
* do not supply EOT/BOS/chat-template context;
* condition only on preceding completion tokens.

For the first generated token there is no completion-only preceding context, so abstain on all of its hierarchical PRC coordinates by setting their soft values to zero.

For token `i >= 2`, replay the model on preceding raw completion tokens and recover the full next-token probability distribution at the original generation temperature.

From that distribution, reconstruct all hierarchical conditional branch probabilities for the observed token.

---

## 9. Oracle-context diagnostic

Also compute a separate **oracle posterior detector** using the hierarchical probabilities saved at generation time.

This uses the actual embedding context, including the prompt.

Label this result clearly as diagnostic.

Its purpose is to distinguish:

* failure because watermark information was genuinely lost,
  from
* failure because prompt-free replay cannot accurately reconstruct the original embedding probabilities.

The oracle score is not the headline detector.

---

## 10. Matched-FPR comparison

The primary scientific comparison is:

**hard parity detector vs prompt-free posterior detector at the same false-positive rate.**

Target:

`FPR = 1e-3`.

Do not compare posterior at `1e-3` against Wang's original published threshold and call that a detector improvement.

Calibrate continuous hard and posterior scores separately on unwatermarked/null samples.

Use independent PRC decoding keys for null scoring so we can cheaply obtain many null key/text pairings after LM replay.

Use one global threshold per detector across temperatures for the primary result.

Freeze thresholds before evaluating the watermarked set.

Also report realized held-out FPR.

Do not treat all cross-key null scores as independent when computing confidence intervals.

---

## 11. Statistical analysis

For each temperature report:

* number of watermarked samples;
* Wang published-threshold TPR;
* calibrated hard-detector TPR at FPR `1e-3`;
* prompt-free posterior TPR at FPR `1e-3`;
* oracle posterior TPR at the same calibration target;
* realized held-out FPR.

Report 95% confidence intervals.

Because samples cross the same 10 key groups and 16 prompts, use clustered/hierarchical bootstrap over key groups and prompts rather than treating all 160 responses as fully independent.

Also compute ROC curves and AUC for hard vs posterior scores.

---

## 12. Mechanism diagnostics

For every temperature also compute:

* average next-token entropy;
* average hierarchical binary entropy;
* fraction of latent PRC bits matching the recovered hard token-ID bits;
* distribution of hard scores;
* distribution of posterior scores.

The point is to determine whether the posterior detector's advantage grows as the hierarchical branch probabilities become highly imbalanced.

---

## 13. Required outputs

Produce:

`results_summary.csv`

with one row per `(temperature, detector)` containing:

* sample count;
* threshold;
* TPR;
* 95% CI;
* realized FPR;
* AUC.

Produce:

`per_sample_results.csv`

with:

* key group;
* prompt;
* temperature;
* hard score;
* posterior score;
* oracle score;
* decisions;
* entropy statistics;
* bit-agreement statistics.

Produce figures:

1. **TPR vs temperature at matched FPR `1e-3`**

   * calibrated hard detector;
   * prompt-free posterior detector;
   * optionally oracle posterior as a dashed diagnostic curve.

2. **ROC curves**

   * hard vs posterior;
   * especially show `T=1.0`, `1.2`, `1.4`.

3. **Detector performance vs hierarchical-bit entropy**

   * to diagnose the low-entropy failure mode.

Also save exact experiment configuration, git commit, GPU-hours, runtime, and Modal cost.

---

## 14. Validation before production

Before launching the full 10-key experiment:

### A. Wang released-data validation

Use their released complete `T=1.8` examples that contain keys/OTPs/token IDs.

Verify our hard-detector implementation reproduces Wang's own decisions/statistics.

Do not attempt to posterior-redetect the released temperature sweep that lacks keys.

### B. Fast-sampler validation

Run:

* 1 key;
* 2 prompts;
* 64 generated tokens;
* `T=1.0` and `T=1.8`.

Check:

* all token IDs valid;
* fast hierarchical sampler matches Wang's original sampler distribution under fixed synthetic test logits/random variates;
* token-ID-to-bit conversion is correct;
* latent codeword/key/OTP alignment is correct;
* KV-cached logits match uncached logits to normal numerical tolerance;
* saved generation-time hierarchical probabilities reproduce the sampler decisions.

Only then launch the full grid.

---

## 15. Interpretation

The key comparison is:

`same Wang PRC + same Qwen3-8B sample + same secret key`

with only

`hard parity evidence`

versus

`posterior probability-aware parity evidence`.

If posterior substantially outperforms recalibrated hard detection, especially at low temperature, the result supports the claim that hard decoding discards useful probability information in low-entropy regimes.

If Wang's published threshold performs poorly but recalibrated hard detection is comparable to posterior detection, then the issue is primarily threshold calibration rather than posterior inference.

If oracle posterior works but prompt-free posterior does not, then the bottleneck is probability reconstruction without the prompt.

If hard, prompt-free posterior, and oracle posterior all fail, then the low-entropy channel itself contains too little usable watermark information.

Do not make a broad claim until these alternatives have been distinguished.

---

## 16. Stop condition

Run the validation tests and then the initial:

`10 keys × 16 prompts × 5 temperatures`.

Afterward, print:

* the result table;
* matched-FPR hard vs posterior comparison;
* ROC/AUC results;
* GPU-hours;
* Modal cost;
* any implementation discrepancies.

**Do not automatically scale to more keys or launch additional experiments.**
