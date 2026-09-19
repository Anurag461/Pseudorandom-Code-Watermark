# Final temperature sensitivity — setup

One bounded 8B follow-up: **T=.7**, full vocabulary, top-p 1, no top-k.
Four arms: ordinary, PRC eta .05, SynthID depth 2, SynthID depth 10.
The same 50 canonical prompts × seeds 12345 and 67890 = **400 evaluation responses**.
Generate exactly 1,024 steps with original EOS/forced-length behavior; primary
evaluation at 1,024, secondary at 400. No other evaluation settings are authorized.

Use the existing Qwen3-8B-Base revision/H100/BF16/static-cache setup and fixed
keys/PRC partition. Native SynthID repeat fallback stays on. Do not use the
later top-k or 0.6B FP32 sampler paths. The unchanged original generator and
PRC sampler are called through a small model-output adapter that divides BF16
logits by .7 once. SynthID's internal temperature remains 1. This preserves:

| Arm | Original and new probability arithmetic |
|---|---|
| Ordinary | BF16 logits → FP32 softmax → original multinomial |
| PRC | BF16 bucket softmax/multiplication/sum; FP32 channel; FP64 bucket uniform; FP32 masked conditional softmax/CDF/token uniform |
| SynthID 2/10 | BF16 native score updates; FP32 final sampling softmax; ordinary repeat fallback |

Only the new BF16 temperature division is added before those paths. Doing the
division after an FP32 cast would change the rounding convention and would not
match native BF16 SynthID temperature handling. No new T=1 GPU responses are
generated. The existing original T=1 outputs provide the separate sensitivity
reference, with their own ordinary control.

The manifest records exact original numerical source hashes, an AST equality
audit of the original ordinary/SynthID generator, actual BF16 partition dtype,
runtime, model and key fingerprints, seeds, and all saved T=1 reference hashes.
The original numerical files and generator are unchanged. Metadata/source
layout changes are documented, as is the original depth-2 follow-up provenance.

## Validation gate

Local checks compare the adapter with unchanged original samplers, exact native
SynthID single-temperature handling, repeat fallback, PRC replay, key separation
and paired bootstrap covariance. The H100 stage runs **three 64-token batches
of 50 prompts for each arm**, using seeds 12345, 12345, 67890. These **600 short
validation responses** are excluded from the 400-response evaluation cohort.

Check same-seed reproduction, fresh-seed changes, original PRC latent streams,
native SynthID batch/single-row score parity, forced-repeat equality to the
temperature-matched ordinary distribution, actual model precision, prompted
generation/replay agreement, and independent raw-completion replay. PRC replay
uses no original prompt/BOS/template/traces; its first coordinate abstains.

**If validation fails, stop and report the specific failure; do not launch the
full batch or automatically retry validation.** After validation passes and the
setup is committed and pushed, run exactly eight full batches (four arms × two
seeds). Check each full trajectory against its saved 64-token smoke prefix.
Persist every batch, then score all 100 PRC and 100 new ordinary responses with
the original detector and T=.7 completion-only BF16 bucket replay. Score both
SynthID configurations locally with exact keys/native masks. Thresholds remain
unchanged; no calibration or threshold tuning.

## Analysis, budget and stop

Primary: PRC-minus-depth-2 decoded-text Self-BLEU at 1,024 tokens, negative
favoring PRC. Preserve SacreBLEU 2.4.3, 13a, exp smoothing, effective order and
the symmetric two-response definition. Use 2,000 paired resamples of 50 prompts,
seed 20260918, retaining both responses and all arms together. Include absolute
and paired Self-BLEU/TPR, contrasts to ordinary and depth 10, repeated token
four-gram fraction, distinct-3, null counts, fallback counts and replay anomalies.
Show original T=1 alongside T=.7. Recompute the T=1 point metrics from saved
outputs and check against the existing report; use common resamples for both
temperature tables. Treat this as exploratory; nonsignificance is not equivalence,
pilot nulls do not establish matched FPR, and no pilot establishes general superiority.

Cost is estimated from measured 8B ordinary/depth-2/depth-10 generation, a
conservative PRC timing proxy from the top-100 run, and the original 200-response
PRC replay. Add 150 seconds for loading/validation and a 25% margin. The manifest
records those measurements and the numerical estimate before dispatch. Use one
H100, four CPUs and 64 GiB, no retries, with 600-second validation and 1,800-second
evaluation/replay limits. Starting cumulative planning charge: $11.00279/$200.
Record timed worker cost and a separate $0.50 overhead allowance, not a settled bill.

All configs/outcomes belong to `outputs/self_bleu_temperature/t07_v1/` with one
versioned `REPORT.md` and frozen manifest. Raw artifacts use the existing Modal
volume at `self_bleu_temperature/<manifest-id>/`. Only `temperature_modal.py`
dispatches GPUs; `temperature.py` prepares, collects and analyzes locally.

User permits validation before pushing; **commit/push before full generation**.
Run explicit stages `validate`, then `batch` only after a passing report and push.
After this batch and analysis, STOP regardless of favorable, unfavorable or
inconclusive results. No extra prompts, temperatures, models, depths, eta or lengths.
