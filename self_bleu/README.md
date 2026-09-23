# Completed detectability/diversity comparisons

**Campaign closed. No additional runs are planned or queued.**

Start with the [consolidated comparison report](../reports/comparisons/REPORT.md)
for highlighted takeaways, all completed comparisons, corrected null counts,
limitations, and the distinction between the 500-prompt historical cohort and
the 50-prompt paired studies. The [paper asset catalogue](../reports/comparisons/README.md)
contains vector figures, LaTeX/CSV tables and an offline reproduction command.

The results establish lower PRC Self-BLEU than the evaluated TextSeal/Gumbel
configurations, but no established advantage over SynthID depth 2 at T=1.
Matching repeat fallback substantially reduces within-response repetition gaps.
The T=.7 primary Self-BLEU contrast favors PRC, but PRC detection falls to 3/100.
The full report retains favorable, unfavorable and inconclusive outcomes.

## Results index

| Completed study | Cohort and purpose | Authoritative report |
|---|---|---|
| Two-response validation | Fixed keys, separate response RNGs, cache checks, actual model parity | [Validation](../outputs/self_bleu_validation/step3-v4/REPORT.md) |
| Stage A | 50 prompts x two responses x five native settings | [Pilot](../outputs/self_bleu_pilot/stage_a_v2/REPORT.md) |
| Repeat policies | SynthID OFF; TextSeal/Gumbel ON, 100 new responses each | [SynthID](../outputs/self_bleu_repeat/setup_v4/REPORT.md), [follow-ups](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md) |
| Saved-output synthesis | Paired Self-BLEU/repetition and corrected null counts | [Paired comparison](../outputs/self_bleu_repeat/paired_comparison/REPORT.md), [repetition](../outputs/self_bleu_repeat/matched_repetition/REPORT.md) |
| SynthID depths 2/30 | 200 new responses, native fallback ON | [Depth comparison](../outputs/self_bleu_depth/depth2_30_v1/REPORT.md) |
| Short-prefix scoring | Saved depths 2/10/30 at 64/128/256 tokens | [Short prefixes](../outputs/self_bleu_depth/short_prefixes/REPORT.md) |
| 8B top-100, T=1 | 500 matched responses, ordinary/PRC/SynthID 2/10/30 | [Top-100](../outputs/self_bleu_topk/matched_v2/REPORT.md) |
| 0.6B full vocabulary, T=1 | 600 matched responses; all contextual fallbacks ON | [0.6B](../outputs/self_bleu_full_vocab/qwen3_0p6b_v1/REPORT.md) |
| 8B full vocabulary, T=.7 | 400 matched responses; original precision paths | [Temperature](../outputs/self_bleu_temperature/t07_v1/REPORT.md) |

Historical reports describe what was known at their publication time; their
old suggested follow-ups are not active instructions. In particular, the paired
comparison corrects Stage A's Gumbel historical-null count at 400 to **2/500**.
Depth 20 was cancelled before full generation. No broad parameter frontier,
Bayesian SynthID evaluation or held-out null calibration was completed.

## Source map

Shared upstream adapters, generator methods and completion-only detectors live
in [`baseline_comparison/`](../baseline_comparison/README.md). This package owns
study orchestration. Completed manifests pin source hashes, so the generation
and scoring modules remain at their existing paths.

| Modules | Responsibility |
|---|---|
| `config.py`, `reference.json`, `generation.py` | Setting identity, fixed-key/response-RNG separation and generation helpers |
| `validation.py`, `validation_modal.py` | Source/cache audit and integrated controls |
| `pilot.py`, `pilot_modal.py` | Completion-only Stage A scoring and paired analysis |
| `repeat.py`, `repeat_modal.py` | Scoped repeat-policy interventions and trajectory checks |
| `depth.py`, `depth_modal.py` | Explicit per-depth SynthID generation/scoring |
| `topk.py`, `topk_modal.py` | Matched truncation-before-watermarking experiment |
| `full_vocab.py`, `full_vocab_modal.py` | Matched 0.6B experiment |
| `temperature.py`, `temperature_modal.py` | Precision-faithful final T=.7 experiment |
| `../reports/comparisons/build.py` | Offline final report, figures and tables; no GPU dispatch |

## Reproduction and archival policy

Use the [offline publication build](../reports/comparisons/README.md) to recreate
paper assets from versioned saved summaries. Run-specific manifests, scores,
archive indices and raw artifacts remain in their original locations. Raw
responses/traces are ignored locally and archived on `prc-completion-only`.
Use the per-study report's **collect** command when restoring artifacts;
restoring data does not require another generation run.

The [closed experiment ledger](plan.md) replaces the chronological planning log.
Frozen implementation runbooks remain available: [repeat policy](repeat_handling_ablation.md),
[top-100](topk.md), [0.6B](full_vocab.md), [temperature](temperature.md).
They document completed requests and do not authorize additional work.

Source history: the original pre-consolidation tree is at `7bde5c6`, the layout
before moving this package is at `7d275d7`, and the full chronological README/plan
before final cleanup is at `69c7ea2`. No immutable result or source snapshot was
removed. The final campaign planning charge is **$12.51222/$200**, including
allowances; this is not a settled account-wide Modal invoice.
