# Detectability versus Self-BLEU: proposed experiment

Prepared 2026-09-15; revised for `comparison-with-redetect` at `4696382`.

**Current scope (2026-09-18):** completed depth-2/30 and short-prefix results are
in the README. The next authorized work is the [single matched top-100 batch](topk.md):
500 new responses across ordinary, PRC eta .05, and native SynthID depths 2/10/30.
Commit and push setup first, then run validation, then generation only if it
passes. Use BF16 model execution and common FP32 probability arithmetic,
completion-only replay, a predeclared PRC-minus-depth-2 primary Self-BLEU contrast
at 1,024 tokens, matched pilot nulls and paired prompt intervals. Four-hundred-token
results are secondary. Starting cumulative planning charge: $7.71132 of $200.
The revised `matched_v2` adds prefix-specific replay support and contradictory
bucket-endpoint diagnostics without altering the detector or dropping tokens.
Report PRC and ordinary nulls separately, including positions 2–64 and 65 onward.
Version 1 ran validation only ($0.15581); no generation. The updated starting
charge is $7.86713 and is included in version 2's reservation.
Stop after this batch. This replaces the older next-step sweep suggestions below;
the remaining historical plan is retained for provenance.

Implementation update (2026-09-18): Self-BLEU workflows now live in the
`self_bleu/` package, in `validation.py`, `pilot.py`, and `repeat.py`, with
separate Modal workers and shared configuration/generation modules. See the
[CLI and source-history notes](README.md#source-history-and-immutable-results).
The current repeat request is `outputs/self_bleu_repeat/setup_v4/manifest.json`;
its experimental settings and budget match setup_v2 and setup_v1. Completed results remain
immutable. The legacy SynthID scorer now requires explicit keys and derives
its depth metadata from them, preventing a depth sweep from silently using the
historical ten-key detector. This does not change the depth-10 pilot findings.

Status: **steps 1–5 complete**, including all repeat-policy follow-ups.
See the [pilot results](../outputs/self_bleu_pilot/stage_a_v2/REPORT.md) and
[validation report](../outputs/self_bleu_validation/step3-v4/REPORT.md).
Stage A reused saved pairs and completed missing prompt-free detection.
The cumulative planning charge is **$6.70584 of the initial $10**.
PRC's diversity advantage over default SynthID is small and uncertain under
the evaluated implementations. **Repeat-handling ablation now precedes any
parameter expansion:** the current SynthID generator falls back to ordinary
sampling on repeated contexts, while TextSeal and Gumbel do not. The ablation
runbook is [repeat_handling_ablation.md](repeat_handling_ablation.md).
The [SynthID-off result](../outputs/self_bleu_repeat/setup_v4/REPORT.md) shows more
within-response repeats but no clear paired Self-BLEU or TPR degradation.
Native fallback-on SynthID remains the main comparison. The completed
[TextSeal/Gumbel follow-ups](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md)
show a large Self-BLEU improvement for Gumbel but an increase for TextSeal at
1,024 tokens, while within-response repeats decrease for both. Detection remained
100/100 at both primary lengths. The next decision is the optional parameter
pilot; no broader sweep has been dispatched.
Total incremental Modal budget: **$200**, including validation, CPU, memory,
generation, scoring, and retries.

**Original design (pilot now complete):** start with a 50-prompt, two-response pilot. If its results
justify a larger study, extend to 500 prompts × two responses, retain a single
plain Gumbel-Max point, sweep TextSeal alpha including zero, and restrict
SynthID's main sweep to 2–20, with depth 30 as an additional sensitivity check.
Keep the current generation distribution. Two responses
are sufficient for the pairwise Self-BLEU estimand; five were an optional
precision improvement, not a requirement. Defer the large null study and
additional detector/model/key experiments until the pilot has answered whether
there is a useful comparison to pursue. The $200 is a ceiling, not a spend target.
Use completion-only detection throughout the new comparison. Generation still
uses the original prompts. Generation and Self-BLEU do not need to wait for
the historical detection rerun; detection conclusions require validated
completion-only scoring for this study's 8B model and online PRC construction.
The branch now supplies that validated redetection implementation and the
completed default-setting results, including the original shared null cohort.

## Next steps and implementation boundary

1. **Freeze the completed reference.** `self_bleu/reference.json`
   pins commit `46963822e9d1f89558337c013b1ff0d47fcc0fb2`, source hashes,
   model/prompt identity, PRC artifact and partition hashes, and the completed
   comparison/replay provenance. Preserve the historical results and Modal
   namespaces. Reuse the integrated PRC sampler/scorer and upstream TextSeal
   detector; generation-time repeat handling stays unchanged.
2. **Add experiment controls.** Expose a PRC sampling seed independently of its
   key seed, using the existing online sampler's document-seed argument.
   Parameterize TextSeal alpha in both generation and detection, and SynthID's
   key list/depth in generation and evidence extraction. Preserve the original
   first ten SynthID keys and predeclare a 30-key bank. Add ordinary sampling,
   prompt/response IDs, configuration fingerprints, and an isolated output
   namespace. Historical default calls must retain their existing behavior.
3. **Validate two-response generation.** Check fixed-key identity and seed
   replay on the target GPU with equal execution geometry, plus Gumbel and
   TextSeal alpha-zero determinism. Verify prompt exclusion, token alignment,
   and direct per-length TextSeal parity. Audit existing first-response caches
   before counting on the 200-continuation reuse estimate.
4. **Stage A pilot.** Use 50 prompts, two responses, the five default/control
   configurations, and 400/1,024-token evaluation. Reuse compatible redetection
   results; recover only missing evidence for new texts. Initial allocation $10.
5. **Repeat-handling ablation (complete).** Disable only SynthID's
   generation fallback on the same 50 prompts and two seeds at depth 10.
   Inspect that paired result before enabling a per-response
   context fallback for TextSeal .1 and Gumbel. Keep all detector formulas and
   masks fixed within each method. Preserve PRC's position-based construction.
   Report paired changes and actual fallback/divergence diagnostics.
6. **Decision and optional Stage B.** Compare paired Self-BLEU and completion-only TPR,
   then test TextSeal alpha 0/.5 and SynthID depths 2/20 plus depth 30 before
   interpreting a default-setting advantage as a frontier advantage.
7. **Conditional expansion.** Increase prompt coverage before responses per
   prompt, then run the frozen full grid and independent null calibration only
   if justified. The total study ceiling remains $200.

Implemented entry points: `self_bleu.config.StudySetting`, `pilot_settings`,
`verify_reference`, and `self_bleu.generation.generate_response_batch` in
`self_bleu/`. The latter uses already-loaded models, independent
sampling seeds and per-response identities; it performs no dispatch or file
writes. The baseline generator now accepts explicit alpha/key-list controls
and ordinary sampling while retaining historical defaults. The TextSeal
detector accepts the same alpha. API examples and validation commands are in
[self_bleu/README.md](README.md). The bounded step-3 worker and source-cache
audit are in `validation_modal.py` and `validation.py`.
The GPU check uses the planned batch of 50 prompts at 1,024 tokens so exact
cache reproduction and saved pilot outputs have the intended execution shape.
It verifies two seeds plus an intervening-seed replay, includes the deterministic
alpha-zero control, and checks alpha .5 / SynthID 2, 20, 30 on 128-token outputs.
PRC and TextSeal replay receive raw completion IDs only. Their independent
reference/direct-prefix checks precede pilot detection claims.

Both GPU stages have one H100 worker, four CPU cores, 64 GiB host memory,
no application retries, and 3,000/600-second timeouts. The combined worker
reservation is $4.65, plus a $2 overhead allowance and failed-attempt costs,
within the original $10 allocation. Historical generation caches are read-only;
new immutable outputs live on `prc-completion-only/self_bleu_validation/`.
Successful full-length response pairs count toward Stage A/B later. Short
parameter checks do not count as full-length frontier results. Local fixtures
are not reported as 8B validation.

Local validation: 17 new control tests passed; the related comparison,
TextSeal, shared-null and prompt-free suites had 97 passes and one existing
serializer-fixture failure. The same failure
(`test_redetect_load_preserves_newer_galois_key_when_factory_is_missing`)
was reproduced on an isolated copy of the untouched reference commit
(42 passes, one failure in `test_prompt_free.py`). It concerns the local
older-galois serializer plus the compatibility shim, not these controls; the
test skips in a fresh process without that shim. No production replay or
historical result was changed. Source/artifact hashes and the historical
PRC key fingerprint were verified after the changes.

**Step 3 result:** all six full-length configurations passed fixed-key and
same-seed replay checks. PRC, TextSeal .1, SynthID 10, and ordinary sampling
changed all 50 responses under the second sampling seed. Gumbel and TextSeal
alpha zero changed none. Short alpha .5 / SynthID 2, 20, 30 checks passed;
the saved SynthID batches matched the official score update exactly.
PRC raw-input observation, independent probability replay, prefix/order check,
and first-coordinate abstention passed. Seven TextSeal/null records passed
upstream parity with direct per-length raw-completion forwards.

All 27 archived result files passed checksum verification: 600 full-length
response records (Stage A plus alpha zero), 200 short parameter-check records,
replay evidence and reports. Historical first-response token matches were
50/50 each for TextSeal .1, SynthID 10 and Gumbel, and 0/50 for PRC and ordinary
sampling. PRC's historical first-response watermark bits still matched for
all 50 × 1,024 coordinates; a text-cache mismatch is not evidence of rekeying.
Use the fresh pairs. Reuse historical detector evidence/decisions only after
checking the relevant configuration, masking rule and token identity; a
SynthID tuple-mask decision cannot stand in for its context-mask analysis.

Recorded worker resource time totals **$1.55505**, including generation,
the PRC validation repair and TextSeal parity. With $0.50 reserved for the
failed startup and $2 for image/startup/storage overhead, the planning charge
is **$4.05505 of the initial $10**, leaving $5.94495 for the pilot under those
allowances. This is not a settled invoice. The report records the setup/import
failures and a BF16-to-NumPy conversion bug in the validation check; the repair
reused every saved generation. These were the costs through step 3; Stage A
costs and results below supersede that running balance.

**Step 4 result:** at 1,024 tokens, PRC Self-BLEU is .0189 with 97/100 detections;
TextSeal .1 is .0377 with 100/100; SynthID 10 is .0221 with 100/100; Gumbel is
1.0000 with 100/100. Ordinary sampling is .0193. The paired PRC-minus-SynthID
Self-BLEU difference is −.0032 (95% prompt-bootstrap interval −.0073 to +.0010),
well below the provisional .02 practical margin. At 400 tokens, PRC detects
55/100 versus 100/100 for the baselines. This supports a diversity advantage
over default TextSeal/Gumbel, but no established advantage over SynthID.

The analysis used two responses per prompt, 2,000 paired prompt-bootstrap
draws, official SynthID context masking and completion-only model evidence.
All 153 replay files verified; 400 independent direct-PRC checks agreed exactly;
6,600 historical CPU score comparisons preserved decisions; 24 local tests
passed. Fresh-null false positives were 0/100 everywhere except SynthID at
400 tokens (1/100). Historical null counts were 0/500 at both endpoints; those
prompts overlap this pilot and do not establish matched 0.1% FPR.

Successful new replay resources totaled $0.68375. Including a $0.78 allowance
for a failed verification attempt and the prior allowances, the planning
charge is **$5.51880**, leaving **$4.48120** of the initial $10. The failed call
was repaired by passing a tensor to the direct detector; frozen inputs,
analysis choices and keys were unchanged. No additional generation ran.

**Experiment (5): repeat handling.** The pilot conclusion is specific to
the evaluated generation policies: native SynthID fallback on, native TextSeal
and Gumbel fallback off. Test this asymmetry before treating the result as an
algorithm comparison or deciding on a parameter sweep. The frozen setup adds
100 SynthID-off responses first, followed by 100 TextSeal-on and 100 Gumbel-on
responses. Existing complementary arms, PRC and ordinary-sampling pairs are
reused. New generation uses the same fixed keys, prompts, seeds, model, batch
geometry and decoding. A source/runtime check, forced-repeat check and native
64-token prefix reproduction gate precede each arm. The three stage timeouts
plus $0.50 extra overhead reserve total $2.83241, bringing the cumulative
reservation to $8.35121 of $10. This is a reservation, not incurred spending.

**SynthID-off result (2026-09-18):** all 100 full-length responses completed.
Self-BLEU off minus on was +.00202 at 400 tokens (95% paired interval −.00200
to +.00649), and −.00067 at 1,024 (−.00421 to +.00309). Both policies detected
100/100 at both endpoints. All 100 pairs passed the no-divergence-before-first-
repeat check, with both native-prefix controls and H100 forced-repeat checks
passing. Mean repeats at 1,024 increased from 45.53 to 114.21 per response;
fallback counts decreased from 45.53 to zero. Every response encountered a
repeat under both policies. The ablation supports an effect on within-response
repetition, but does not support fallback as the main explanation for SynthID's
between-response Self-BLEU diversity on this cohort. The main comparison keeps
native SynthID fallback on. See the [complete report](../outputs/self_bleu_repeat/setup_v4/REPORT.md)
and its per-response diagnostics for first repeat/divergence positions.

The worker took 174.895 seconds, with $0.22587 estimated resources and no retry.
Including the predeclared $0.50 overhead allowance, the cumulative planning
charge is now **$6.24467**, leaving **$3.75533** of the initial $10. This remains
an estimate with allowances, not a settled Modal invoice. TextSeal/Gumbel-on
generation and TextSeal replay were still pending at that point; the following
result supersedes this stage's balance and status.

**TextSeal/Gumbel follow-ups (complete):** 100 fallback-on responses per method
and the 100-response direct-prefix TextSeal replay passed. At 1,024 tokens,
TextSeal Self-BLEU changed from .03770 to .04727 (on minus off +.00957; 95% paired
interval +.00402 to +.01528), while Gumbel changed from 1.00000 to .20718
(−.79282; −.81657 to −.76903). At 400 tokens, TextSeal's change was small and
uncertain (+.00189; −.00273 to +.00675); Gumbel improved to .39375.
Both policies for both methods detected 100/100 at both endpoints.

Mean repeated contexts at 1,024 fell from 346.71 to 54.86 for TextSeal and
539.00 to 55.82 for Gumbel. All 200 pairs passed the no-early-divergence and trace
checks, and all 200 native 64-token controls reproduced the originals. Thus the
fallback materially changes Gumbel's deterministic diversity point and reduces
within-response repetition in both methods, but does not improve TextSeal's
between-response Self-BLEU in this cohort. Preserve and label both policy variants.
The conclusion about PRC versus native SynthID is unchanged.

The two follow-up workers used $0.46117 in estimated resources without retries.
The current cumulative planning charge is **$6.70584**, leaving **$3.29416** of
the initial $10, with the existing repeat-study overhead allowance counted once.
See the [follow-up report](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md)
for both-policy diagnostics and artifact retrieval. All repeat-policy stages
are now complete; the optional parameter pilot remains a separate decision.

**Matched-policy repetition analysis:** the earlier repeated-token-4-gram and
distinct-3 metrics were recomputed locally for all native/modified responses.
At 1,024 tokens with fallback on for the three contextual baselines, repeated
4-grams are 2.38% PRC, 2.42% SynthID, 2.74% TextSeal, 2.81% Gumbel and 2.92%
ordinary sampling. Distinct-3 is respectively 95.61%, 95.54%, 94.64%, 94.56% and
95.14%. Every paired PRC contrast with those baselines includes zero at both
primary lengths for both metrics. The prior large within-response repetition
gap is therefore highly sensitive to repeat policy; it cannot support a PRC
advantage under matched fallback-on handling on this cohort. See the
[matched repetition report](../outputs/self_bleu_repeat/matched_repetition/REPORT.md)
for means, medians, paired intervals and matched-off sensitivity. This reused
saved tokens, reproduced 1,000 historical metric values and cost no Modal credit.

**Consolidated direct paired contrasts:** the
[paired comparison report](../outputs/self_bleu_repeat/paired_comparison/REPORT.md)
now joins Self-BLEU, repetition and detection with the existing null counts.
Self-BLEU contrasts resample the actual per-prompt PRC-minus-baseline differences;
they do not subtract separate confidence limits. Both native and fallback-on
comparisons favor PRC over TextSeal/Gumbel on Self-BLEU at both primary lengths,
while PRC-minus-SynthID intervals include zero. At 1,024 tokens with fallback on,
the differences are −.02837 [−.03531, −.02201] against TextSeal and −.18829
[−.21248, −.16445] against Gumbel. Repetition contrasts under fallback on still
include zero; PRC's lower TPR and unmatched empirical FPR remain limitations.
The historical Gumbel null count at 400 is 2/500, correcting earlier prose that
said all historical counts were zero; pilot SynthID has 1/100 at 400. These
cohorts overlap in prompts and are retained separately, not pooled. This was
local analysis of saved outputs with no additional Modal cost.

Keep generation and detector repeat handling separate: the primary contrasts
change generation only. SynthID retains its context-mask detector; TextSeal and
Gumbel retain their existing detectors and tuple masks. Reuse existing null
counts and keep the nominal-FPR limitation. The common fallback rule is scoped
to repeated three-token contexts within a response; native context initialization
is retained (SynthID starts with zeros; TextSeal/Gumbel start from prompt suffixes).
PRC has no corresponding context-reuse mechanism and is not modified.

**Requested depth follow-up completed:** SynthID depths 2 and 30 with native repeat fallback
on: 50 prompts × two seeds × two depths, 200 new 1,024-token responses. The
frozen request is `outputs/self_bleu_depth/depth2_30_v1/manifest.json`. Reuse saved
depth-10 SynthID, PRC and ordinary-sampling pairs; compare 400/1,024-token
completion-only detection and Self-BLEU using the same 2,000 paired prompt
bootstrap draws. Generation and detection both receive the explicit per-depth
prefix of the predeclared key bank. Rescore existing nulls for each depth.
The single H100 run has no automatic retries, a 900-second timeout and a total
planning reservation of $8.37076 including earlier work and a new $0.50 allowance.
No broader parameter sweep, extra-prompt expansion, Bayesian training or new
null generation is included. The pilot report remains an unchanged record.

The [depth follow-up report](../outputs/self_bleu_depth/depth2_30_v1/REPORT.md)
records 100/100 detections at both lengths for depths 2, 10 and 30. Depth 2 has
no established Self-BLEU difference from PRC, depth 10 or ordinary sampling;
depth 30 has higher Self-BLEU than each of them at both lengths. At 1,024 tokens,
PRC-minus-depth-2 Self-BLEU is −.00097 [−.00459, +.00265], while
PRC-minus-depth-30 is −.01000 [−.01546, −.00506]. PRC remains at 55/100 detection
at 400 and 97/100 at 1,024, so this does not establish PRC superiority over the
evaluated SynthID settings. Depth-2/30 false-positive counts are 0/100 pilot and
0/500 historical nulls at both lengths; these remain separate, limited null
cohorts rather than matched-FPR calibration. All generation controls passed,
1,400 saved depth-10 scores and 100 prompt-level Self-BLEU values reproduced,
and independent bootstrap/score verification passed. No retries were needed.
New worker resource estimate: $0.50548; cumulative planning charge including
the new $0.50 allowance: **$7.71132**. No broader sweep is dispatched.

**Short-prefix detection completed:** the requested depth-20 generation was
cancelled before launch and its unfinished runner changes were reverted.
Instead, the [short-prefix report](../outputs/self_bleu_depth/short_prefixes/REPORT.md)
scores saved fallback-on depths 2/10/30 at 64/128/256 tokens with the unchanged
completion-only weighted-normal detector. Depth 2 detects 45/100, 82/100 and
100/100; depths 10/30 each detect 100/100 at all three lengths. Paired gains over
depth 2 are +55 pp [45, 65] at 64 and +18 pp [11, 25] at 128, with no observed
gain of depth 30 over 10. Thus the earlier long-prefix detection saturation hid
a useful depth-10 versus depth-2 tradeoff. The report retains pilot and historical
null counts separately, including the observed false positives; no threshold
calibration or new generation was performed. All 6,300 scores were checked on
the actual truncated inputs and by an independent score calculation, 1,400
saved depth-10 scores reproduced, and all 36 bootstrap intervals independently
verified. Additional Modal cost: $0; cumulative planning charge stays $7.71132.

## Evidence behind the recommendation

The completed comparison in `controlled_baseline_full_report.md` already uses
all four requested methods on the same 500 prompts. It fixes Qwen3-8B-Base,
temperature 1, top-p 1, and 1,024 generated tokens. However, it has one response
per prompt: its distinct-n and repeated-4-gram measurements are **within-response
repetition**, not the **between-response diversity** measured by Self-BLEU.

The completed native-8B raw-completion comparison now gives online PRC MAP
54.8% TPR at 400 tokens and 93.2% at 1,024; TextSeal alpha .1 gives 100% at both.
Their observed FPRs at these two lengths are 0/500. TextSeal has 1/500 false
positives at 768. SynthID/Gumbel completion-only token-score results are also
published in `outputs/comparison_redetect/baseline_comparisons.csv`.
The old PRC figures (67.8% and 96%) used prompted probability traces and must
not be used for the new decision. These results motivate retaining parameter
sweeps and shorter prefixes to resolve baseline saturation.
Existing repetition results are useful diagnostics, but they do not predict
the new Self-BLEU measurements.

The supplied TextSeal source, `sections/3-experiments.tex`, uses Qwen3.5-27B,
1,000 ELI5 prompts, five seeds, 400-token answers, temperature 0.8, top-p 0.9,
context length 3, TextSeal alpha 0.1, and SynthID depth 10. Its tradeoff varies
alpha from 0 to 0.5 and depth from 2 to 20. Its separate diversity ablation in
`sections/4-ablations.tex` instead uses two responses, temperature 1, top-p 0.95,
and a maximum length of 2,048. These are distinct experiments.

The supplied SynthID main paper's Experimental details use instruction-tuned
Gemma/Mistral, context length 4, within-response repeated-context masking, and
normally depth 30 with a Bayesian detector. Supplement C.3 uses **two responses
per prompt**, sweeps depth 1–30, and varies Gumbel's watermark application
probability from 0.1 to 1. Its detection metric uses empirical null thresholds.
The depth-30 setting deserves a sensitivity check even though it is outside
TextSeal's plotted sweep. SynthID supplement C.1 reports that mean/weighted-mean
detectors can lose power at larger depths, while its Bayesian detector performs
better and plateaus, motivating the choice of 30. Consequently, depth and
detector must be considered together. Omitting 30 alone does not demonstrate
unfairness, but reproducing a 2–20 weighted-mean sweep does not test SynthID's
original strongest configuration. Do not transfer either paper's numerical
results to our model.

For a claim of superiority over SynthID broadly, evaluate its Bayesian detector
on the same depth-10/20/30 generations using separate training/development data
and held-out null calibration, or state that the comparison is limited to the
frequentist variant. A depth-30 weighted-mean check alone does not resolve the
detector issue. Training-free detection and localization are legitimate narrower
comparison scopes; our initial clean, global-detection pilot does not require
every detector to have an intrinsic p-value. The official SynthID repository
currently provides Bayesian detector and training code, although this does not
establish availability of the original trained weights or exact reproducibility.

## Minimum useful pilot and decisions

**Stage A: five configurations, 50 prompts, two responses each.** Use online PRC
eta=0.05, TextSeal alpha=0.1, SynthID depth 10, plain Gumbel-Max, and ordinary
sampling. Use the same model/settings below and generate to 1,024 once; measure
400 and 1,024 tokens, with other prefix detection scores available cheaply.
PRC eta=0.05 is the strongest detection setting among the existing positive
noise rates, so test it before spending on weaker eta settings.

Choose one existing 50-prompt batch without looking at new Self-BLEU outcomes,
preserving the historical grouping and row order. The predeclared rule
`random.Random(20260916).randrange(10)` selects batch 0, prompt rows 0–49.
This is an exploratory subset, not a representative-sample guarantee. Keep the
same batch for every response and method. Extend to different prompt batches
when a result is uncertain or promising.

There are 500 logical response slots, or 450 generations after accounting for
Gumbel's verified determinism. The original **200-new-continuation** estimate
was conditional on all historical first responses matching. Step 3 found only
150/250 exact first-response matches, covering TextSeal, SynthID and Gumbel.
It nevertheless saved both responses for every Stage A configuration as part
of the controlled reproduction checks. **Stage A now needs no new generation.**
Use those saved pairs and recover missing detector evidence; do not combine
the mismatching historical PRC/null texts with the new second responses.

**Stage B: check the competing frontiers on the same 50 prompts.** Add TextSeal
alpha=0 and 0.5 and SynthID depth 2 and 20. This adds 350 unique continuations
after checking the deterministic alpha=0 endpoint. Together A+B cover nine
configurations, 900 logical slots, and at most **800 unique continuations** from
scratch. Stage A can reveal a large advantage or disadvantage; Stage B is
needed before dismissing competing diversity settings or interpreting a
default-only win as a promising frontier result. Add **depth 30 as a sensitivity
check** on these same 50 prompts and two responses: another 100 continuations,
making 900 unique continuations from scratch including the check. It can use
weighted-mean scoring for the pilot, but a favorable PRC result remains
provisional pending the stronger-detector check. No Gumbel skip sweep, proxy
model, or Bayesian-detector training is needed for this initial screen.

For the pilot, use the fixed nominal p<0.001 rules with completion-only model
traces, and rescore the existing shared null texts under the same protocol as
a gross calibration check. Do not fit a 0.1% empirical threshold to
only 50 prompts or claim the existing 500 nulls establish equal FPR. Defer the
20,000-null campaign until a confirmatory comparison is justified.

**What would be encouraging?** At the same length, PRC has Self-BLEU close to
ordinary sampling and meaningfully below the baseline settings that retain
useful detection. For example, use 90% TPR at 1,024 tokens as an application
screen, consistent with the project's existing n90 analyses. A provisional
practical margin is 0.02 on the 0–1 BLEU scale: look for PRC within 0.02 of
ordinary sampling and at least 0.02 below competitive baselines. This margin
is a stated judgment for triage, not a universal quality threshold or a
statistical significance cutoff. Report paired bootstrap intervals and the
actual detection rates alongside it.

Meeting a 90% detection requirement while having lower Self-BLEU is a useful
tradeoff; PRC at 93.2% TPR versus a baseline at 100% does not establish strict
Pareto dominance. A stronger eventual result would show lower Self-BLEU at
matched detection, or higher detection at matched Self-BLEU, with independent
FPR checks. A null-like PRC Self-BLEU alone is insufficient if SynthID is also
null-like and detects more reliably.

**What would discourage a larger search for a PRC advantage?** SynthID depth
2/10/20 or TextSeal alpha=0.5 matches or improves PRC's diversity while matching
or improving detection at both lengths, with reasonably tight paired
uncertainty. If Stage A shows only a small ambiguous difference, expand to
100–200 prompts at two responses rather than increasing to five responses on
the same 50. Absence of significance on 50 prompts is not evidence of equality.

Raw-completion scores show a detection disadvantage for online PRC eta=0.05:
54.8% TPR at 400 and 93.2% at 1,024, versus 99.8%/100% for default SynthID. SynthID
also preserved within-response repetition metrics. Thus the plausible new
advantage is **between-response diversity at useful detection**, most plausibly
at 1,024 tokens, rather than a prediction that PRC will win detection outright.
Reassess this expectation using completion-only pilot scores: neither a go nor
a no-go decision should combine new Self-BLEU with old prompt-conditioned TPR.
Do not compare PRC at 1,024 tokens against a baseline at 400 to claim dominance.

If proceeding, freeze the final grid and analysis before inspecting the other
450 prompts. Report the pilot separately and verify its suggested advantage on
those remaining prompts; also report the complete 500-prompt descriptive result
for consistency. Publish or retain negative pilot findings rather than tuning
keys, prompts, or decoding until PRC wins.

## Frozen common setting

| Item | Proposed choice | Reason |
|---|---|---|
| Generator | Qwen/Qwen3-8B-Base, BF16, existing project Qwen loader | Direct continuation of the completed four-method comparison; weights and adapters already exist |
| Revision | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` for model/tokenizer | Preserve reproducibility |
| Evaluation prompts | Pilot: 50 rows; confirmation: all 500 rows of `prompts.jsonl`, original order | Preserve the main evaluation population and paired comparisons |
| Prompt format | Existing 50 token IDs; no chat template or added instructions | Preserve the continuation task and exact conditioning |
| Detector context | Raw completion tokens only, no prepended token; PRC coordinate 1 abstains | Match the branch's validated `completion_only_raw_abstain_v1` protocol |
| Decoding | Temperature 1.0, top-p 1.0, no top-k truncation, no repetition penalty | Preserve the current PRC sampling distribution for every method |
| Length | Exactly 1,024 continuation steps; same existing treatment of EOS | Avoid method-specific survival/filtering bias and reuse prefixes |
| Headline lengths | 400 and 1,024 tokens | Bridge TextSeal's short evaluation and our existing long evaluation |
| Additional prefixes | 32, 64, 128, 256, 512, 768 | Resolve saturated baselines; scoring needs no extra generation |
| Replicates | Two sampling seeds per prompt and configuration | Directly estimates pairwise diversity; preserves budget for prompt coverage |
| Watermark context | Three preceding tokens for TextSeal, Gumbel and SynthID; SynthID `ngram_len=4` | Preserve the controlled comparison; PRC is position based, so this parameter does not apply to it |
| Hardware/execution | H100, batch 50, identical prompt grouping/order for every seed; at most 10 workers | Existing measured performance and numerical consistency |
| Primary PRC construction | Online causal PRC, t=3, row rate 99/100, posterior-mean/MAP detector | Matches the existing cross-method comparison and supports valid causal prefixes |

The older fixed-PRC detection figure uses Qwen3-0.6B-Base, so this is not a claim
that every previous experiment used 8B. The 8B choice aligns with the completed
baseline comparison. Include the fixed-PRC bridge below and label construction
explicitly; do not call online and fixed PRC interchangeable.

## Parameter grid

| Family | Generation configurations | Count |
|---|---|---:|
| Online PRC | eta = 0.05, 0.10, 0.15, 0.20; t=3 throughout | 4 |
| TextSeal | alpha = 0, 0.05, 0.10, 0.20, 0.30, 0.50 | 6 |
| SynthID-Text | depth = 2, 5, 10, 15, 20 | 5 |
| Gumbel-Max | Plain single-key Gumbel-Max; one point | 1 |
| Unwatermarked | Ordinary sampling from the same base distribution | 1 |
| **Total** | **500 prompts × two responses × 17 configurations** | **17,000 responses** |

The separate depth-30 sensitivity check adds 1,000 response slots for the full
500-prompt study, giving 18,000 logical slots and 17,000 unique continuations
including both deterministic endpoints. Keep its label separate to distinguish
the TextSeal-range sweep from the original SynthID default.

Include TextSeal alpha=0: it is the no-routing-randomness endpoint, necessary
to cover the requested 0–0.5 range. It is deterministic under the same conditions
as plain Gumbel. Verify both in the pilot, then generate one response per prompt
for each and represent the identical second request without redundant GPU work.
Their Self-BLEU is 1; their detection uncertainty still has 500 prompt units.
This reduces actual generation to at most **16,000 continuations / 16,384,000
tokens**, before any valid cache reuse. This is the conditional full experiment,
not the initial pilot.

TextSeal alpha=0 and Gumbel are the same sampling principle, but their plotted
points need not coincide: TextSeal uses entropy weighting, and the current key
assignments differ. In the released convention alpha=0 selects TextSeal key B
(12387), while existing Gumbel uses key 42. Keep these recorded assignments for
continuity. Share generation across the two only if active keys, PRFs, samplers
and resulting tokens have actually been matched; do not assume equivalence.

PRC eta, TextSeal alpha and SynthID depth have different meanings.
Compare the resulting measured coordinates, not equal numerical parameter
values. Higher eta may weaken detection without appreciably moving PRC's
diversity: overlapping points or a nearly vertical curve would be a legitimate
result. Do not introduce an artificial PRC diversity knob just to spread them.
The plain Gumbel point is a deterministic reference, not an exact alpha=0
TextSeal endpoint under the existing different key assignments and detectors.

## Required separation of keys and sampling randomness

Use fixed secret keys across both responses. Change ordinary randomness:
PRC free/codeword bits, encoding noise and token sampling; TextSeal routing;
SynthID sampling/ties. Plain Gumbel has no sampling-seed diversity.
Reset response-local history between responses and isolate it between batch
members. Never carry SynthID history across the two replicates.

**Implementation prerequisite:** `modal_run.py::online_build_artifacts` derives the PRC
key from `experiment_seed`; `OnlineGenerationModel.generate_wm` also derives
document sampling seeds from that field. Running two different legacy
`experiment_seed` values would change the secret key as well as sampling.
The study's `StudySetting.key_seed` and `generate_response_batch(sampling_seed=...)`
now separate these controls and record them in configuration/batch identities.
The wrapper passes the saved key and independently derived document seeds
directly to the existing online sampler, without changing historical jobs. Preserve
the existing support key, OTP, and vocabulary partition across replicates.
`OnlinePRCEncoder` already accepts the key separately from document seeds.
Apply the same separation to any fixed-PRC bridge.

**Impact on existing results (checked 2026-09-17):** this is a control needed
for the new repeated-response experiment, not evidence that previous main
detection runs changed keys for every prompt. `modal_run.py` builds and saves
one key per configuration; workers load it and `generate_batch_and_collect`
draws fresh codewords under that key. The online runner likewise loads one
key and derives distinct document randomness for each prompt. Historical
online seed-12345/54321/67890 artifacts have distinct recorded key hashes, so
those separate replicate runs measure joint key/sample variation. They should
not be relabelled as sampling-only replicates under a single key. This finding
does not require rerunning the main TPR/FPR or within-response quality results.
For fixed-key Self-BLEU, add new responses under the matching existing key;
reuse old first responses only after the usual provenance checks. A CPU-only
check confirmed the existing encoder accepts a fixed key with different
document seeds and produces different codewords without changing that key.

Keep the original key configuration for the primary sweep: PRC key seed 12345
and existing partition, TextSeal key pair 42/12387, Gumbel key 42, and the
existing ten SynthID keys. `self_bleu.config.SYNTHID_KEY_BANK` extends these to
30 distinct keys using the first 31 bits of SHA256 over the fixed domain
`prc-self-bleu/synthid-key-bank/v1/` plus the zero-based layer index, 10–29.
Use nested prefixes of this list across depths. The chosen
sampling seeds are 12345 and 67890. The first replicate can reuse
an existing artifact only if key, sampling stream, prompts, length, precision,
batch layout, code and model provenance all match. Otherwise generate it anew.

The existing Gumbel diagnostic found power-form and log-space sampling identical
through 400 tokens at batch sizes 1 and 5. It ruled out the suspected underflow
explanation in those tests; batch-dependent logits explained the sensitivity.
Freeze batch geometry and repeat the useful equivalence checks to the new
length. Do not characterize the historical repetition as an established bug.

## Self-BLEU and uncertainty

Primary diversity metric: **mean pairwise Self-BLEU-4 within each prompt**.
For two outputs A and B, compute `(BLEU(A | B) + BLEU(B | A))/2` per prompt.
Then average the prompt-level values (50 in the pilot, 500 in the full study). Report
on a 0–1 scale, lower being more diverse. Exclude prompt tokens. At each length
T, truncate continuation token IDs to T before decoding and calculating BLEU.

Freeze an explicit text metric: SacreBLEU 2.4.3, four equally weighted n-gram
orders, `tokenize='13a'`, case-sensitive, exponential smoothing, effective order
enabled, score divided by 100. Decode with the pinned tokenizer, special tokens
omitted and cleanup disabled; record these flags and counts of special tokens.
Do not remove repetitions or select only attractive samples.

Also report pairwise BLEU on token IDs for continuity. With exactly two outputs,
the existing leave-one-out token-ID helper in `baseline_comparison/scoring.py:260`
is already a symmetric pairwise metric; with five outputs it would instead use
four references, which changes the metric. Paper tokenization/smoothing is not established
by the manuscripts alone, so label this a controlled adaptation, not an exact
numerical reproduction of their Self-BLEU axis.

Use paired prompt-cluster bootstrap intervals (2,000 resamples), carrying all
both responses and all method settings together when a prompt is resampled.
There is one response pair per prompt, not two independent BLEU observations.
Two independent responses give an unbiased Monte Carlo estimate of the expected
symmetrized pairwise BLEU for that prompt; five would reduce sampling noise but
would not change the pairwise estimand. Do not compare a two-response pairwise
metric against a five-response multi-reference metric and attribute the change
to watermark diversity.
Report both mean Self-BLEU and its paired difference from ordinary sampling.
Retain repeated-4-gram rate, distinct-3, exact duplicate-response fraction and
base-model NLL as diagnostics. NLL alone cannot distinguish fluent output from
high-probability loops.

## Detection and false-positive comparability

**Required context protocol: `completion_only_raw_abstain_v1`.** The
[September 17 redetection plan](../textseal_prompt_free_redetection_plan.md)
supersedes this proposal's earlier EOT-seeded protocol. Keep prompts for
generation and prompt-paired analysis, but exclude them from every detector
input. Replay saved completion IDs with fresh model state and no prepended
token. PRC coordinate 1 has soft score zero; its observed token supplies context
for coordinate 2. Preserve all original coordinates. TextSeal uses the upstream
raw-completion entropy path and its native initial-position eligibility.
Never reuse generation-time probabilities, entropies or KV caches, even when
generator and detector share the same weights. Preserve original text, keys
and partitions.

| Method | Required change to the current comparison scorer |
|---|---|
| PRC MAP | Recover completion-only partition probabilities and recompute scores and Hoeffding thresholds with the original online key |
| TextSeal | Recover completion-only full-vocabulary entropy and rerun its weighted statistic and reference checks |
| Gumbel-Max and frequentist SynthID | Existing evidence paths use completion tokens; verify that initial context eligibility and PRF inputs never include the generation prompt |
| Self-BLEU and generated text | No change from this detector-context correction |

Apply this protocol to both watermarked and null responses, at every prefix.
Keep method-native eligibility masks; never fill missing keyed watermark
contexts with the original prompt. Recompute
any empirical thresholds from the new null scores. Label legacy results
separately; never mix their scores or cached traces into these plots.

Reuse the integrated raw-input replay in `qwen.py`, completion-only scoring in
`detectors.py`, and manifest/cache orchestration in `modal_run.py::redetect`.
The native-8B PRC comparison and its T13088 shared-null alignment are complete;
reuse their matching traces and original online supports. The integrated PRC
replay is BF16 token-step execution with coordinate-1 abstention. Older
`completion_only/` EOT/FP32 scripts are historical diagnostics, not production.

Use `TextSealCompletionDetector` for TextSeal. Its native-8B BF16 upstream
HF path is validated and the 500 watermarked + 500 shared-null cohort is
complete. Run each requested length directly: longest-trace slicing failed
exact upstream parity at 128 because BF16 linear projections depend on matrix
shape. Preserve batch size 1, eager attention and the pinned runtime for this
reference path. PRC may reuse validated causal traces across prefixes;
TextSeal must retain a separate entropy vector for each actual prefix length.
New configurations and new responses need their own identities and checks,
not a repeat of the entire completed historical redetection campaign.

Report two views on the same watermarked responses:

1. **Nominal operating point, completion-only:** TPR under each method's
   nominal p<0.001 rule, with
   independently observed FPR. Use PRC's Hoeffding upper bound, TextSeal's
   entropy-weighted Gamma approximation, SynthID's weighted frequentist score,
   and Gumbel's Gamma statistic. Keep approximation/bound labels visible.
2. **Common empirical operating point:** calibrate a separate threshold for
   each detector/configuration/length on shared null responses at target FPR
   0.001; freeze it and audit FPR on different null responses. Also show 0.01
   using the same scores, which has less uncertainty in the tail.

The calibrated view is necessary before interpreting differences as a ranking
at a matched false-positive rate. It does not replace PRC's analytical result.
Conversely, median -log10(p) is useful as a secondary TextSeal-style plot but
does not put a conservative bound and approximate p-values on an equivalent
statistical scale. Store log survival probabilities where possible; explicitly
mark any numerical clipping. Predeclare 400 and 1,024 as separate endpoints.
An 'any prefix detected' decision would require a multiple-testing correction.

For a confirmatory matched-FPR comparison after the pilot, plan **10,000
independent null prompts for calibration plus 10,000 disjoint
null prompts for audit**, one ordinary 1,024-token response per prompt. Exclude
all evaluation and detector-training prompts, deduplicate document IDs and
prompt hashes, and sample from the same C4 RealNewsLike source. Reuse these
same null texts for all detectors. Recover full token entropy for TextSeal and
partition probabilities for PRC by completion-only model replay. Reuse cached
evidence only with matching tokens, partition where relevant, model, runtime,
precision and execution shape. The validated PRC token-step and TextSeal
per-length HF paths are different numerical executions: budget separate replay
rather than assuming a shared forward pass. Distinct watermarked texts also
require their own replay. Generation-time traces may
be retained as diagnostics, but must not feed the completion-only detectors.

The file named `prompts_10k.jsonl` contains only **4,983 rows** locally. It is
not a sufficient null pool; extend the source loader and freeze the additional
prompt manifest before launching. Do not manufacture 20,000 independent
observations by rescoring 500 texts with many keys. If new prompts cannot be
obtained, disclose a smaller calibration or clustered repeated-prompt design
and weaken the FPR claim accordingly.

At a true 0.1% FPR, 10,000 audit texts yield only about ten false positives;
the resulting estimate is still imprecise (roughly 0.05–0.18% for ten observed
events). Report counts and exact binomial intervals, not 'verified exactly
0.1%'. Threshold uncertainty should be assessed by resampling calibration
prompts as well as evaluation prompts. A finite-sample conservative rank rule,
such as `(1 + number of null scores >= observed score)/(N + 1)`, avoids arbitrary
quantile interpolation and handles ties; audit its actual achieved FPR.

**Detector details matter:** preserve TextSeal's entropy-weighted statistic;
do not take an uncorrected minimum of its weighted and unweighted p-values.
Use the same 8B model weights for its completion-only entropy trace, giving it
the strong same-model detector. PRC uses the matching model's completion-only
partition probabilities.
Report that these detectors have model access, while basic Gumbel/SynthID do
not; CPU/GPU detection costs should be recorded separately.

The historical integration deduplicates `(context, token)` tuples for all
three baselines. SynthID's own supplement A.1 instead masks **repeated contexts**
to match its generation rule. Retain the historical tuple-mask result as a
reproduction column, but make the native context-mask SynthID detector the
scientific comparison and publish the difference. Do not infer eligibility
solely from the generator's private skip decisions. Record effective scored
token counts and the handling of the initial context for every detector.

## Optional follow-ups after the pilot

1. **Fixed-PRC bridge:** if the intended claim includes fixed PRC, run it on
   the same 8B prompts at eta=0.05, t=3, r=floor(0.99n), separately at n=T=400
   and n=T=1,024, with two responses. Overlay these as separately labelled
   points against the existing baseline generations. Do not detect a short
   prefix of a fixed n=1,024 code as though generated using n=400. An online
   PRC pilot does not rule out an advantage for the fixed construction.
2. **Key sensitivity:** check a promising comparison on a predetermined
   100-prompt subset with two responses and two additional independent key
   sets. Compute diversity within each key, never across keys. Keep the
   baseline and PRC configurations fixed rather than searching for lucky keys.

Stochastic-skip Gumbel, Bayesian training, proxy detectors, new models and new
decoding regimes are outside the initial pilot. Depth 30 is a small additional
sensitivity check. Until a properly trained and independently calibrated
Bayesian detector is evaluated, limit the claim to the tested frequentist
SynthID variant. Do not interpret failure of depth-30 weighted-mean scoring as
evidence that the depth-30 Bayesian configuration is uncompetitive.

## Cost and execution plan

Measured evidence is unusually favorable: the completed 500-prompt run generated
1,500 baseline continuations at 1,024 tokens and scored them for **$2.65877380**
total. Mean H100 times per 50-prompt method batch were 44.68 s (TextSeal),
68.70 s (SynthID depth 10), and 43.97 s (Gumbel), with 25.93 GiB peak reserved
CUDA memory. Source: the full-run runtime JSON and provider cost ledger.

Historical linear scaling gives **$0.80 for Stage A from scratch**, another
**$0.62 for Stage B**, or **$1.42 for both pilots**. These are compute/scoring
anchors, not promised bills: they exclude the newly required completion-only
model replay. Startup, implementation validation, changed PRC generation and
CPU BLEU work can dominate a tiny experiment. The earlier **$1–3 estimate**
applies only to work resembling the historical pipeline, not the complete
revised pilot. Retain an initial **$10 pilot allocation including validation,
replay and retries**, measure 8B replay cost first, and revise allocations
within the $200 ceiling if needed before dispatching further work. Valid token-cache reuse reduces
new generation further; do not depend on it before checking the artifacts.
The completed TextSeal six-length replay of 1,000 texts recorded $0.86 in
resource time, and PRC's 500 shared-null replay recorded $0.48 including its
first-batch validation. These exclude some startup/loading/storage charges
and are timing anchors, not new charges or future-bill guarantees. Reuse their
compatible results rather than paying for the same replay again.
The added 100-response depth-30 check gives a further $0.18 at the historical
average rate, or $1.60 combined; measure its actual throughput because deeper
generation can be slower.

For a later full experiment, the same scaling gives **$28.36 for 16,000 unique
continuations**, plus **$35.45 for 20,000 nulls**. Actual PRC, depth-20 and metric
costs, plus completion-only replay on watermarked and null texts, must be
measured in the pilot. The full budget below is conditional on
proceeding; there is no reason to commit it before seeing the pilot.
Including the separate 1,000-response depth-30 check changes the historical
generation/scoring anchor to $30.13. The existing $40 sweep allocation remains
a provisional allocation, subject to actual depth-30 and replay timing.

Current [Modal resource pricing](https://modal.com/pricing), checked 2026-09-15,
lists H100 at $0.001097/s ($3.9492/h), plus CPU at $0.0000131/core/s and host
memory at $0.00000222/GiB/s. L40S is cheaper per hour, but its performance on
these custom samplers is unmeasured. Use the already measured H100; batch
throughput matters more than the hourly GPU price alone. Avoid region premiums.

| Phase | Ceiling |
|---|---:|
| Integration checks and Stages A+B pilot | $10 |
| Conditional full 500 × two, 17-configuration sweep, including metrics | $40 |
| 10k calibration + 10k audit nulls and scoring | $40 |
| Optional fixed-PRC and key checks | $20 |
| Unallocated / estimation reserve | $90 |
| **Total maximum** | **$200** |

The earlier **$70–110** planning range did not include separate completion-only
replay and is not a validated total for this revision. Re-estimate after the
8B smoke; use the reserve as needed while preserving the $200 maximum.
Pilot outputs count toward the full-grid work where compatible, so do not
double-count the same GPU work in the ledger. The phase ceilings are separate
conservative allocations. Do not report an exact future bill from historical
default-setting performance alone.

Execution order: local correctness checks; Stage A; Stage B or more prompts if
needed to resolve the decision; then, only if justified, the full sweep and
disjoint null pool, with bounded follow-ups as relevant. Reuse
pilot outputs only under the identical frozen protocol. Count cold starts,
CPU/memory, and failed attempts in the ledger. Parameterize the existing
adapters rather than replacing the validated infrastructure.

**Parallel work and dependencies:** the shared raw-completion contract and
default native-8B results are frozen at the reference commit. Prepare fixed-key sampling replicates,
reuse or generate its 50-prompt response pairs, and compute Self-BLEU while
the other task rescores historical experiments. Its full historical sweep is
not a prerequisite. The prerequisite for this study's go/no-go decision is a
validation of the new controls with the existing 8B replay/scoring adapters,
followed by missing PRC/TextSeal and null scores on the pilot. Reuse existing
pilot-text scores where identities match. Use separate run namespaces and frozen code snapshots
or isolated worktrees during concurrent development. Reuse compatible traces
by complete model/context/precision/token/partition identity and avoid
duplicate replay jobs. Independent scientific work need not mean simultaneous
unbounded GPU workers; reserve costs for all active work charged to this
study's budget before dispatch.

Before every dispatch, reserve the worst-case outstanding batch charges using
measured per-configuration timeouts and actual resource rates. Stop dispatch
when spent + outstanding reservations + proposed work would exceed $200.
Use short, resumable shards, immutable new run IDs, fail-on-overwrite, no
unbounded retries, and separate CPU scoring jobs. Merely monitoring settled
billing is insufficient because active workers can still incur charges.

Keep two responses per stochastic configuration: reducing to one would make
Self-BLEU unavailable. If the full projection exceeds its allocation, examine
the measured bottleneck and use the unallocated reserve within the $200 ceiling
or predeclare a smaller common prompt sample. Do not selectively drop
inconvenient methods or change keys to improve the curve.

## Outputs and claims

Produce two main panels, T=400 and T=1,024, with Self-BLEU on the horizontal
axis and TPR on the vertical axis: the upper-left is preferable. Mark default
settings and show prompt-bootstrap intervals. Provide separate completion-only
nominal-threshold and empirically calibrated versions, a shorter-prefix
detection plot, and the supplementary median-log-p/repetition diagnostics.
Show the null Self-BLEU reference and each detector's observed null FPR.

Retain per-response tokens, decoded text, prompt IDs, key fingerprints,
sampling seeds, parameters, detector evidence, eligibility masks, thresholds,
entropy/probability traces, BLEU signature, model/code hashes and cost records.
Join rows by explicit IDs, not batch order. Validate key constancy across
replicates, seed replay, prefix agreement, native detector masks, and complete
coverage before plotting.

The supported claim will be about **diversity across repeated generations from
the same prompt versus completion-only detection under our fixed Qwen3-8B
continuation setting**. Self-BLEU is a lexical
diversity measure, not semantic quality. This experiment does not establish
cross-domain superiority, edit robustness, localization, radioactivity, or
cryptographic security. In particular, t=3 is the current empirical setting,
not a newly established security parameter choice.

## Source record

- Local `../relevant papers/textseal.tar.gz`: `sections/3-experiments.tex`,
  `sections/4-ablations.tex`, `sections/5-appendix.tex`; checked against the
  accompanying `textseal.pdf`, pages 8 and 11. Public paper:
  [TextSeal](https://arxiv.org/abs/2605.12456).
- Local `../relevant papers/synthid main text.pdf`, Experimental details
  (PDF page 8); `synthid supplement.pdf`, A.1–A.3 and C.3 (PDF page 13).
  [SynthID paper](https://doi.org/10.1038/s41586-024-08025-4).
- [Official SynthID implementation and detector guidance](https://github.com/google-deepmind/synthid-text).
  Keep pinned commit `addb4a158143c7c6851a1308f78b89fceed59683` for generation.
- Existing TextSeal source pin:
  `c60d0d1da2e59f09a698438e218a07ee779b4616`.
- Project `baseline_comparison/config.py`, `official.py`, `scoring.py`,
  `comparison_runner.py`, `pinned_sources_manifest.json`, `online_prc.py`,
  `modal_run.py`, `controlled_baseline_full_report.md`,
  `controlled_baseline_diagnostic_report.md`, `hoeffding_tpr_paper_result.tex`.
- Runtime and billing:
  `outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/controlled_baseline_full_runtime.json`
  and `controlled_baseline_full_cost_ledger.csv` in the same directory.
- Frozen current reference: `self_bleu/reference.json`;
  `baseline_comparison/README.md`, `textseal_prompt_free_redetection_plan.md`,
  `outputs/comparison_redetect/baseline_comparisons.provenance.json`, and
  `outputs/redetection/cache_index.json`.
- Prompt file SHA-256:
  `0bf0560438d9d4b7a85ebf8b7349d6d028aa02b11a381982742ee55b4430530c`.
