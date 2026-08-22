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

## Priority 0: complete cheap 14B rows already in cache

Run full cache-only audits and append authoritative `online_causal_prc` rows to
`hoeffding_results_summary.csv`:

| Model | eta | T=n | Existing MAP | New work | Cost | Time |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen3-14B | 0.05 | 448 | 326/500 (65.2%) | entropy, naive, and all FPRs | <$0.03 | 5--15 min |
| Qwen3-14B | 0.10 | 800 | 311/500 (62.2%) | entropy, naive, and all FPRs | <$0.03 | 5--15 min |

These reuse the existing 14B watermarked prefixes and shared `T=1808` null
cache. No model or GPU generation should launch.

## Priority 0: determine whether fixed and online differ meaningfully

Do not generate new fixed text. Use existing fixed prompt-level records, then
score exact online prefixes from the new 8B caches for less than $0.10 total:

| Model | eta | Matched lengths |
| --- | ---: | --- |
| 0.6B | 0.05 | 448 (already complete) |
| 0.6B | 0.15 | 1504 (already complete) |
| 8B | 0.05 | 416 and 749 |
| 8B | 0.10 | 768, 1382, and 1625 |

Pair records by prompt index. For MAP, entropy, and naive decisions:

1. Report the paired TPR difference `fixed - online` with a prompt bootstrap
   95% confidence interval.
2. Run exact McNemar tests on discordant decisions and apply Holm correction
   across the reported cells.
3. Predeclare **5 percentage points** as the smallest meaningful difference.
   Call methods practically equivalent only if the entire paired 95% interval
   is inside `[-5, +5]` points; call one better only if the interval excludes
   zero and the estimated effect is at least 5 points; otherwise call the
   result inconclusive.
4. Preserve prompt-level decisions and scores with prompt IDs so the analysis
   is reproducible. The two constructions use domain-separated keys derived
   from the same experiment seed; they cannot literally share one key object.

Expected work: 2--3 researcher hours, under 30 minutes compute, effectively
zero Modal cost.

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

| Work | Researcher time | Modal wall time | Expected cost | Cap |
| --- | ---: | ---: | ---: | ---: |
| Pin/integrate official code, unit tests, 5-prompt smoke | 4--6 h | 20--40 min | $1--3 | $5 |
| Three new 500-prompt baseline generations to 1024 | 1 h supervision | 1--2 h | $15--30 | $45 |
| 50-prompt x 5-seed diversity subset | 1 h | 30--60 min | $2--5 | $8 |
| Prefix detection and final analysis | 2--4 h | 30--90 min | <$2 | $5 |

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
- No new 14B generation. The only scheduled 14B work is the two cache-only
  matched-reference audits above.

## Forty-eight-hour schedule

### Day 1

1. Morning: reconcile provider costs for the completed 8B eta 0.05/0.10
   campaigns and freeze their manifests; do not rerun generation or audits.
2. Run the paired fixed-versus-online analysis and the two 14B cache-only
   audits; update both summary CSVs.
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

- Verify the already-published 8B selected online boundaries and censored cell
  in `hoeffding_results_summary.csv`; append the two cache-only 14B
  matched-reference rows.
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
- Existing 14B source caches and the `T=1808` shared null cache make the two
  planned 14B audits CPU-only.

References: [official TextSeal repository](https://github.com/facebookresearch/textseal),
[official SynthID-Text reference](https://github.com/google-deepmind/synthid-text),
and the uploaded TextSeal v2 arXiv source (`arXiv:2605.12456v2`).
