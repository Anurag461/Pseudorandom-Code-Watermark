# Final two-day experiment plan

## Scope and decision rules

- Deadline: 48 hours.
- Modal budget: $550 credits, at most 10 GPUs. Choose concurrency from each
  validated baseline batch shape. Do not split the fleet across competing
  long jobs.
- Expected additional spend for the unfinished work: **$25--50**. Hard stop:
  **$85 additional spend** without revising this plan. The remaining credits
  are failure contingency, not a reason to expand scope.
- Primary model: **Qwen3-8B-Base**. New 14B generation and all Qwen3.5-27B
  work are deferred.
- Primary threshold: one-shot TPR at nominal FPR `1e-3`; strict 90% crossing
  means at least `451/500` detections.
- Every production run must first pass a reusable actual-ceiling smoke. Stop if
  its projected cost exceeds the phase cap below.

## Completed: useful 8B online grid and selected-boundary audits

All four planned 8B online cells are resolved. The eta 0.15 result remains
right-censored because extending its ceiling is outside the approved cost
gate.

| eta | Result | Selected/ceiling full audit | Status |
| ---: | --- | --- | --- |
| 0.05 | exact bracket `(639,640]`; `n90=640` | MAP `452/500`; entropy `359/500`; naive `195/500`; all FPRs `0/500` | complete and published |
| 0.10 | exact bracket `(1406,1407]`; `n90=1407` | MAP `451/500`; entropy `394/500`; naive `240/500`; all FPRs `0/500` | complete and published |
| 0.15 | `n90 > 4096`; MAP `448/500` at ceiling | ceiling audit: entropy `423/500`; naive `335/500`; all FPRs `0/500` | complete, censored |
| 0.20 | exact bracket `(13072,13088]`; `n90=13088` | MAP `451/500`; entropy `431/500`; naive `385/500`; MAP FPR `1/500`, other FPRs `0/500` | complete and published |

The eta 0.05 and eta 0.10 campaigns include reusable smokes, 500-prompt
sweeps, one-token cache-only refinements, authoritative prompt-sharded audits,
prompt-level watermarked/null decisions, and publication in both summary
CSVs. Full provenance is in `modal_online_8b_eta005_runbook.md` and
`modal_online_8b_eta010_runbook.md`. No additional selected-boundary audit is
scheduled; only provider-cost reconciliation remains for the two new cells.

### Explicitly not scheduled

- **8B eta 0.15 extension:** continuing from 4096 must replay 2,048,000 cached
  prefix token positions one at a time before generating a suffix. The measured
  4096-token batch-50 rate implies about $6.7 before suffix work; realistic
  total is **$8--12**, so it fails the requested $5 gate. Keep `n90 > 4096`.
- **14B eta 0.15/0.20:** new long generations remain deferred for cost and
  schedule risk.
- **Fixed PRC at every online crossing:** do not run; it is unnecessary for the
  focused statistical comparison below.

## Completed: cheap 14B matched-reference rows from cache

Both full cache-only audits completed on 2026-08-22 and their authoritative
`online_causal_prc` rows are published in `hoeffding_results_summary.csv` and
`online_causal_results_summary.csv`:

| Model | eta | T=n | MAP | Entropy | Naive | Observed FPRs | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Qwen3-14B | 0.05 | 448 | 326/500 (65.2%) | 212/500 (42.4%) | 74/500 (14.8%) | MAP/entropy/naive all 0/500 | complete and published |
| Qwen3-14B | 0.10 | 800 | 311/500 (62.2%) | 193/500 (38.6%) | 69/500 (13.8%) | MAP/entropy/naive all 0/500 | complete and published |

The audits reused exact watermarked prefixes from the existing `T=1280` and
`T=3072` source caches plus the shared `T=1808` null cache. They launched no
model or GPU worker and generated zero tokens. Prompt-level decisions and
scores are preserved in
`outputs/online_causal_n448_t3_eta0.05_prompts500_gen-qwen3_14b_base_sampler-poscdf-v1_kvcache-static-v1_from_n1280.json`
and
`outputs/online_causal_n800_t3_eta0.10_prompts500_gen-qwen3_14b_base_sampler-poscdf-v1_kvcache-static-v1_from_n3072.json`.
Only provider-cost reconciliation remains.

## Priority 0: determine whether fixed and online differ meaningfully

This analysis is approved and can be completed without new text generation or
any GPU. All seven fixed-PRC prompt-level datasets and both 0.6B online full
audits are already local. Run five CPU-only, cache-only Modal audits to score
the missing exact 8B online prefixes from the existing source and null caches:

| Model | eta | Matched lengths |
| --- | ---: | --- |
| 0.6B | 0.05 | 448 (already complete) |
| 0.6B | 0.15 | 1504 (already complete) |
| 8B | 0.05 | 416 and 749 |
| 8B | 0.10 | 768, 1382, and 1625 |

Pair records by prompt index. For MAP, entropy, and naive decisions:

1. Report the paired TPR difference `fixed - online` with a prompt bootstrap
   95% percentile interval using 10,000 deterministic paired resamples and a
   recorded analysis seed.
2. Run exact McNemar tests on discordant decisions and apply Holm correction
   as one conservative family across all 21 comparisons (seven matched cells
   times three detectors).
3. Predeclare **5 percentage points** as the smallest meaningful difference.
   Call methods practically equivalent only if the entire paired 95% interval
   is inside `[-5, +5]` points; call one better only if the interval excludes
   zero and the estimated effect is at least 5 points; otherwise call the
   result inconclusive.
4. Preserve prompt-level decisions and scores with prompt IDs so the analysis
   is reproducible. The two constructions use domain-separated keys derived
   from the same experiment seed; they cannot literally share one key object.

Required checks before inference:

- exactly 500 shared prompt indices per cell, with no gaps or duplicates;
- matching model, eta, length, nominal FPR, and experiment seed within each
  fixed/online pair;
- complete MAP, entropy, and naive decisions plus statistics and thresholds;
- cache-only execution with zero generated tokens and no model/GPU worker.

Deliverables:

- `outputs/fixed_vs_online_prompt_level.jsonl`: prompt ID, configuration,
  construction-specific decision/statistic/threshold, and source provenance;
- `outputs/fixed_vs_online_paired_summary.csv`: fixed and online TPRs,
  difference, bootstrap interval, discordant counts, exact and Holm-adjusted
  McNemar p-values, and the predeclared interpretation;
- `fixed_vs_online_analysis.md`: compact results table, methods, conclusions,
  limitations, and links to the machine-readable artifacts;
- an itemized cache-only cost ledger and a final prompt-coverage/fingerprint
  validation record.

Resources and limits: five cache-only Modal CPU audits, expected spend below
**$0.10** with a **$0.25 hard stop**, no GPUs, and no new model downloads.
Expected wall time is **30--60 minutes**, including validation and report
generation.

Completed 2026-08-22. All five 8B audits passed the fail-closed cache-only
guard, generated zero tokens, and incurred no GPU charge. Exact Modal billing
was $0.04505808 total. The paired analysis validated all seven cells and 3,500
prompt pairs. For 8B, none of the 15 detector comparisons called either
construction better; two were practically equivalent and 13 inconclusive.
Across all 21 tests, only the 0.6B eta 0.15, n=1504 naive comparison remained
significant after Holm correction. Results and fingerprints are in
`fixed_vs_online_analysis.md` and `outputs/fixed_vs_online_validation.json`.

## Priority 1: PRC versus TextSeal, SynthID-Text, and Gumbel-Max

### Controlled 8B head-to-head

Use the official TextSeal implementation pinned to an exact commit. It includes
TextSeal and the paper's Gumbel/SynthID comparison path. Isolate it in a Python
3.11 Modal image rather than changing the working PRC environment.

Frozen setup:

- Qwen3-8B-Base; the same canonical 500 project prompts.
- Forced 1024-token continuations, temperature 1.0, top-p 1.0, reasoning off.
  This matches the current PRC decoding distribution; it is intentionally not
  the TextSeal paper's 27B setup.
- Methods: online PRC (`t=3`, eta 0.05), TextSeal (`alpha=0.1`), SynthID-Text
  (depth 10, frequentist detector), and Gumbel-Max.
- Watermark context length 3 for TextSeal/SynthID/Gumbel; deduplicate repeated
  `(context, token)` tuples as in the TextSeal evaluation.
- Score exact prefixes `T={128,256,400,512,768,1024}` from each causal
  generation. Reuse the new PRC eta-0.05 cache and existing 8B null texts.
- Use each method's analytic/frequentist nominal `p<1e-3` rule. Report observed
  null FPs over 500 separately; 500 nulls cannot tightly validate a 0.1% FPR.
- Report TPR, observed FPR, median `-log10(p)`, and the full detection curve.
  Label PRC's Hoeffding value as a p-value upper bound, TextSeal's as a
  moment-matched Gamma approximation, SynthID's as a normal approximation,
  and Gumbel-Max's as an exact Gamma test; do not imply identical calibration.

Quality/diversity checks:

- Mean base-model token NLL/perplexity, output length, repetition rate, and
  distinct-2/distinct-3 on all 500 outputs.
- On 50 fixed prompts, generate five seeds per nondeterministic method and
  report pairwise token agreement plus Self-BLEU. Treat Gumbel-Max determinism
  as a result, not a generation failure.
- Do not equate PRC eta with TextSeal alpha. Interpret detectability jointly
  with these quality/diversity measurements.

Codex can own the complete integration and smoke phase. The implementation
scope is:

- read the uploaded TextSeal v2 source and extract the exact experimental
  configuration;
- pin the official TextSeal and SynthID repositories to recorded commits;
- build an isolated Python 3.11 Modal image without changing the PRC Python
  environment or `pyproject.toml`;
- implement adapters for TextSeal, frequentist SynthID-Text, and Gumbel-Max on
  the canonical prompts and Qwen3-8B-Base;
- implement exact-prefix scoring, `(context, token)` deduplication, p-values,
  and a shared prompt-level result schema;
- add tests for determinism, prefix equivalence, deduplication, scoring, and
  artifact completeness;
- run the five-prompt smoke and check SynthID generation/g-values against
  Google's official reference;
- record runtime, peak memory, cost, repository commits, model revision,
  seeds, prompt coverage, and any paper-versus-code discrepancy;
- produce a standalone runbook and handoff, then record final status and costs
  in this plan after concurrent fixed-versus-online work finishes.

Required resources are network access to the two official repositories and
pinned dependencies, existing Modal authentication/model-cache access,
permission to add new integration/test/runbook artifacts, and at most one H100
for the smoke. No manual researcher implementation is required; the user only
needs to review scientifically meaningful discrepancies and authorize the
smoke spend.

| Work | Codex execution time | User involvement | Modal wall time | Expected cost | Cap |
| --- | ---: | --- | ---: | ---: | ---: |
| Pin/integrate official code, tests, and 5-prompt smoke | 2--4 h | approve network/Modal access; review meaningful discrepancies | 20--40 min | $1--3 | $5 |
| Three new 500-prompt baseline generations to 1024 | 1 h supervision | approve scale-up after smoke | 1--2 h | $15--30 | $45 |
| 50-prompt x 5-seed diversity subset | 1 h | none after scale-up approval | 30--60 min | $2--5 | $8 |
| Prefix detection and final analysis | 2--4 h | review conclusions | 30--90 min | <$2 | $5 |

Completed integration and five-prompt smoke on 2026-08-22 (runs crossed
2026-08-23 UTC). TextSeal is pinned to
`c60d0d1da2e59f09a698438e218a07ee779b4616`; Google's SynthID-Text is pinned
to `addb4a158143c7c6851a1308f78b89fceed59683`; Qwen3-8B-Base and its tokenizer
are pinned to `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. The isolated image definition
fingerprint is
`36ef066d906bf9df05801790b9327b8f2a8854d516add87b3f36d47afcd40217`.

All four methods produced complete scores for prompt rows `0..4` at all six
prefixes. Official TextSeal/Google SynthID references, exact-prefix scoring,
deduplication, schema validation, fixed-seed replay, and cache-only PRC/null
guards passed. Every observed null FPR was `0/5`; this is not a calibration
claim. Online PRC smoke TPR was 0%, 60%, then 100% from `T=400` onward; the
three new baselines were 100% at every smoke prefix. TextSeal and Gumbel-Max
also showed severe repetition/low distinct-n on several prompts, so their
detection must be interpreted with quality/diversity. Gumbel-Max was
seed-deterministic at fixed batch shape, with batch-shape sensitivity later
localized to model-logit/numerical variation rather than the power formula.

Exact integration/smoke campaign spend was `1.39371804` dollars,
including fail-closed diagnostics and controlled seed checks; it remained well
below the 10-dollar smoke cap. Peak CUDA memory was 17,668,689,920 bytes
allocated and 17,702,060,032 bytes reserved. Measured primary batch-5 times
were 37.006 seconds for TextSeal, 40.504 seconds for SynthID, and 35.554 seconds
for Gumbel-Max. Scaling those timings gave a prior batch-5 upper-bound
projection of 14--18 dollars and 20--35 minutes with ten H100 shards. The
500-prompt comparison remains incomplete and requires separate explicit
approval. See `controlled_baseline_smoke_report.md` and
`controlled_baseline_full_run_runbook.md`.

Completed a user-approved five-prompt quality/diversity diagnostic on
2026-08-22 (2026-08-23 UTC) before requesting scale-up. Cache-only prefix
analysis generated no tokens. A second complete TextSeal seed reproduced the
collapse: median distinct-3/repeated-4-gram rate at `T=1024` changed from
`0.3503/0.6405` to `0.3033/0.6934`. Under the paper decoding regime
(`temperature=0.8`, `top_p=0.9`, `T=400`), TextSeal's median
distinct-3/repetition was `0.6910/0.2821` versus vanilla's `0.8920/0.0605`;
TextSeal was materially worse on three prompts, nearly equal on one, and
better on one. Repeated TextSeal 4-gram events occurred at lower conditional
entropy than novel events on all five prompts, an association rather than a
causal or cross-model result.

The exact log-space Gumbel argmax matched the released power form token for
token through 400 tokens at batch sizes 1 and 5. The earlier batch-shape
sensitivity is therefore not a power-form underflow bug; it reflects
batch-dependent model logits/numerical execution amplified by deterministic
autoregression. An offline project-Qwen/Hugging-Face check retained a strict
failure (`8.726e-4` maximum batch-5 JSD versus the predeclared `1e-4` limit),
while all top-1 tokens agreed and native Hugging Face itself showed larger
batch-1/batch-5 variation (`1.622e-3` maximum JSD). The discrepancy limits
bitwise portability but does not invalidate the internally controlled,
fixed-batch project-Qwen comparison.

Exact diagnostic spend was `0.17113157` dollars (`0.14751330` H100,
`0.00919542` CPU, `0.01442285` memory), below its 2-dollar cap. The bounded
generation worker ran `78.234` seconds and peaked at 17,668,689,920 bytes CUDA
allocated; the dual-model parity check peaked at 33,229,244,928 bytes. No full
run was launched. See `controlled_baseline_diagnostic_report.md` and the
diagnostic cost/artifact manifests in `outputs/`.

The production batching plan is batch 50: ten workers, 50 prompts per worker,
one batch per method. The standalone batch-50 validation completed successfully
on prompt indices `0..49` with all 150 TextSeal, SynthID, and Gumbel outputs
exactly 1,024 tokens. The H100 worker took 166.386 seconds, including an
8.204-second model load, and peaked at 27,839,692,800 bytes CUDA reserved
(25.93 GiB). SynthID's pinned-reference check was exact, all saved entropy and
log-probability values were finite, and the full PRC/null cache preflight made
zero generation attempts.

The finalized validation bill was `0.27034319` dollars (`0.21751641` H100,
`0.02631032` CPU, `0.02651646` memory), below its 3-dollar cap. Scaling the
complete bill across ten shards gives a conservative `2.70343190`-dollar
generation estimate; allow **$3--4** end to end including CPU scoring, with a
recommended **$5 hard cap**. With ten H100s available, the measured compute
floor is 2.77 minutes and practical end-to-end wall time is **8--15 minutes**,
excluding unusual queue delay. See
`controlled_baseline_batch50_validation_report.md` and the compact validation
and cost artifacts in `outputs/`. The batch-50 gate is cleared, but no
500-prompt generation has been launched; it still requires explicit approval.

CPU-only scoring of that saved batch is also complete: 2,400 schema-valid rows
cover 50 prompts, four methods, watermarked/shared-null samples, and all six
prefixes. Online PRC TPR was 18%, 42%, 78%, 80%, 96%, and 98% from T=128 to
1,024; TextSeal, SynthID, and Gumbel were 100% at every prefix. All methods had
zero false positives among 50 nulls, which is descriptive only and cannot
tightly validate a nominal 0.1% FPR. All exact-prefix checks passed, TextSeal
official/common decisions agreed, and PRC/null generation attempts stayed zero.

The 50-prompt quality result confirms a material method-by-setting tradeoff.
At 1,024 tokens, median distinct-3/repetition were 0.972/0.012 for online PRC,
0.969/0.012 for SynthID, 0.692/0.275 for TextSeal, and 0.463/0.532 for Gumbel.
TextSeal had repetition above 0.1 on 41/50 prompts; Gumbel did so on 49/50.
The loss grows with length and is already visible by T=400 for Gumbel. The
frozen full run remains valid only as a joint detection-quality-diversity
comparison, not as a quality-matched detector comparison. Exact CPU scoring
cost was `0.00604867` dollars with zero GPU cost. No remaining generation was
launched. See `controlled_baseline_batch50_eval_report.md`.

Dependency incompatibilities may extend the integration wall time. Stop after
the smoke rather than launching 500-prompt production automatically; scaling
requires review of correctness, memory, throughput, and projected cost.

Use the TextSeal paper's frequentist SynthID detector for the main table so all
methods have a nominal FPR. Validate its generation and g-values on a small
smoke against Google's official SynthID reference; do not use the trained
Bayesian detector.

### Lightweight detector comparison

After the 8B eta-0.05/0.10 boundaries are selected, teacher-force their cached
watermarked and null tokens through Qwen3-0.6B-Base and rerun MAP/entropy
detection. Compare native-8B versus proxy-0.6B TPR/FPR at both selected
boundaries. Expected cost **$1--3**, wall time **20--45 minutes**.

The 4B proxy is a stretch goal only if all Priority 0 work and the baseline
table are complete by the middle of day 2 (estimated $3--7 and 30--60 minutes).
Do not add Qwen3.5-0.8B in this sprint: it introduces a new architecture and a
tokenizer/provenance compatibility check for little incremental value.

## Deferred work

- A faithful TextSeal paper replication would require Qwen3.5-27B,
  temperature 0.8, top-p 0.9, 400 tokens, **1,000 ELI5 prompts and five
  seeds**. The earlier “exactly 500 ELI5 prompts” description is not the v2
  paper protocol. Porting PRC to Qwen3.5 plus this generation volume is outside
  the two-day critical path.
- If time remains, a 50-prompt Qwen3.5-27B compatibility smoke may estimate a
  later replication cost, but it must be labeled a smoke rather than a paper
  replication.
- No new 14B generation or detection work remains. The two approved
  cache-only matched-reference audits are complete; only their provider-cost
  reconciliation remains.

## Forty-eight-hour schedule

### Day 1

1. Morning: reconcile provider costs for the completed 8B eta 0.05/0.10
   campaigns and freeze their manifests; do not rerun generation or audits.
2. Run the paired fixed-versus-online analysis. Verify the already-published
   14B cache-only rows while reconciling their provider costs; do not rerun the
   audits.
3. In parallel on local CPU: pin TextSeal, build the isolated adapter and
   prompt/result schema, and add detector-equivalence tests.
4. Evening: run 5-prompt TextSeal/SynthID/Gumbel smokes and inspect exact costs,
   tokenizer IDs, context deduplication, p-values, and saved provenance.

### Day 2

1. Morning: run the three 8B baseline generation campaigns sequentially with
   all 10 H100s; run CPU prefix detection as earlier generations finish.
2. Midday: run the 0.6B proxy replay and finish any incomplete paired-analysis
   checks.
3. Afternoon: produce the final boundary table, fixed-versus-online effect
   table, baseline prefix curves, and quality/diversity table.
4. Final 4 hours: rerun only failed/corrupt shards, verify prompt coverage and
   fingerprints, freeze cost ledgers, and write conclusions. Do not start new
   model/configuration tracks in this window.

## Required final deliverables

- Verify the already-published 8B selected online boundaries, censored cell,
  and two 14B cache-only matched-reference rows in
  `hoeffding_results_summary.csv`.
- A compact online-boundary matrix with exact brackets or explicit lower bounds.
- A prompt-paired fixed-versus-online statistics artifact and one results table.
- A separate baseline results CSV/JSONL for PRC/TextSeal/SynthID/Gumbel with
  prompt-level scores, p-values, prefixes, quality metrics, code commit, model
  revision, keys/seeds, and costs.
- One plot of TPR versus prefix length at nominal FPR `1e-3`, plus one
  detectability-versus-diversity/quality table.
- A final cost ledger and a short limitations paragraph distinguishing analytic
  bounds, approximate p-values, 500-null empirical resolution, and censored
  boundaries.

## Cost basis

- Measured 8B batch-50 generation at `T=4096`: $0.68394 and 516.99 H100
  method-seconds for 50 records.
- Measured 14B matched-boundary cache-only MAP checks: about $0.008--0.010;
  prior full cache-only audits cost about $0.015.
- Existing 8B eta-0.15 production plus null generation/audit: $11.13; the new
  lower-eta runs reuse null caches and are substantially shorter.
- The 8B eta-0.05 and eta-0.10 generation, exact-boundary refinement, and
  selected-boundary audits are complete; their provider costs still require
  final dashboard reconciliation.
- The two 14B matched-reference audits completed cache-only using the existing
  source caches and shared `T=1808` null cache; no GPU or generated tokens were
  used, and provider-cost reconciliation remains.

References: [official TextSeal repository](https://github.com/facebookresearch/textseal),
[official SynthID-Text reference](https://github.com/google-deepmind/synthid-text),
and the uploaded TextSeal v2 arXiv source (`arXiv:2605.12456v2`).
