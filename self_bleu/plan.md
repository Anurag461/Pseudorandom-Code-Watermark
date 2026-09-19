# Closed comparison experiment ledger

**Status: complete and stopped, 19 September 2026.**
The user closed the comparison campaign. No new generation, validation run,
parameter sweep, model replay, or threshold change is pending.

The [consolidated report](../reports/comparisons/REPORT.md) is the final account
of what was run, what was learned, and what the evidence does not establish.
The [paper assets](../reports/comparisons/README.md) include reproducible figures
and tables. The [results index](README.md) links each immutable run report.

## Completed

1. Audited original experiment settings, papers/implementations, fixed keys and
   response randomness. Kept two responses per prompt for the paired study.
2. Validated completion-only PRC/TextSeal detection and kept the historical
   500-prompt comparison separate from the 50-prompt diversity cohort.
3. Validated and analyzed the five-setting native-policy Stage A pilot.
4. Ran SynthID repeat fallback OFF, then TextSeal/Gumbel fallback ON, with full
   paired trajectory diagnostics and no divergence before the first repeat.
5. Consolidated saved-output Self-BLEU, repetition, direct paired intervals and
   corrected null counts under native and fallback-on policies.
6. Generated SynthID depths 2 and 30 and compared with saved depth 10, PRC and
   ordinary outputs; then scored saved 64/128/256-token prefixes.
7. Completed the matched 8B top-100 batch: 500 responses across five settings.
8. Completed the matched 0.6B full-vocabulary batch: 600 responses across six
   settings, with repeat fallback ON for all contextual baselines.
9. Completed the final precision-faithful 8B T=.7 batch: exactly 400 responses,
   with the setup committed/pushed before full dispatch and validation passing.
10. Created the offline final comparison report and paper assets; no new
    generation or inference was used for consolidation.

The paired studies use 50 prompt clusters, two response slots per setting and
1,024-token generation. Reused controls, duplicate deterministic responses and
validation outputs are not counted as independent evaluation data.

## Not completed, and not pending

- Full-length SynthID depth 20: cancelled; only earlier short validation exists.
- Full TextSeal alpha sweep: alpha=0 pairs and alpha=.5 short checks were
  validation-only; the scored main setting is alpha=.1.
- PRC eta sweep, additional temperatures/models/prompts, Bayesian SynthID,
  attacks, multiple independent watermark keys, held-out confirmation and
  empirical null calibration: not run in this campaign.
- The historical common-method proxy comparison was not fully repaired to the
  new completion-only protocol; its prompted results are archival, not current
  paper-ready evidence.

## Decisions to preserve in the paper

Show shallow SynthID as well as deeper settings. State native versus modified
repeat handling. Separate within-response repetition from between-response
Self-BLEU. Show detection beside diversity and raw null counts beside nominal
thresholds. Document the FP32 probability paths in top-100/0.6B separately from
the original BF16 bucket/update paths retained by the temperature study.
Do not claim matched empirical FPR, equivalence from nonsignificance, a complete
parameter frontier, or general superiority.

The campaign's final planning charge is **$12.51222/$200**, with all recorded
allowances; no spending target or remaining-experiment authorization follows.

## Historical plans and reproduction

The chronological 917-line plan is preserved in Git at `69c7ea2:self_bleu/plan.md`.
Its proposed follow-ups are superseded by the closed ledger above. Historical
manifests, runbooks, source snapshots, raw completions and detector artifacts
remain unchanged. The [source map](README.md#source-map) describes the retained
modules; moving or merging frozen numerical files would break their source hashes.
