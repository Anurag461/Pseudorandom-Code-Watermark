# Final 8B temperature sensitivity: T=.7

Complete: exactly 400 evaluation responses (four arms × 50 prompts × two seeds), each 1,024 tokens. Temperature .7, full vocabulary, top-p 1, no top-k, original forced-length/EOS policy. No further experiments are queued.

**Predeclared primary:** PRC minus SynthID depth 2 decoded-text Self-BLEU at 1,024 tokens: **-0.01098 [-0.01976, -0.00227]**. Negative values favor PRC. This is exploratory sensitivity analysis on the existing prompt cohort.

## Interpretation

The primary Self-BLEU contrast favors PRC on this cohort, but this is not an improved overall detection–diversity result. At 1,024 tokens, PRC detection fell from **97/100 at T=1 to 3/100 at T=.7**; depth 2 fell from 100/100 to 93/100 and depth 10 remained 100/100. At 400 tokens PRC detects 1/100, versus 87/100 and 100/100 for the two SynthID depths. No thresholds were changed to compensate.

All four arms have substantially greater lexical overlap and within-response repetition at T=.7 than in their saved T=1 outputs. PRC's lower between-response Self-BLEU does not imply less within-response repetition: at 1,024 tokens its repeated-four-gram fraction is .34070 versus .30458 for depth 2, while distinct-3 is .61037 versus .64441. The paired repetition intervals include zero. PRC versus the matched ordinary control also has a Self-BLEU interval crossing zero; this does not establish equivalence.

The sensitivity result therefore combines a favorable primary lexical-diversity contrast with a severe PRC detection loss. The checks below verify the specified arithmetic, keys, seeds and replay protocol; they do not establish the mechanism responsible for that loss. No additional experiments, detector modifications or parameter adjustments were made.

## Frozen configurations

Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; H100, BF16 forward, static KV cache, batch 50, TF32 off, original reduced-precision BF16 reduction setting. Canonical prompts 0–49, exactly 50 stored prompt tokens, unchanged formatting. Sampling seeds 12345 and 67890; fixed watermark keys independent of response seeds. Exactly 1,024 steps, including special/EOS tokens, without early stopping.

Ordinary sampling; PRC eta .05, t=3, row rate 99/100, key seed 12345, unchanged position-addressed document RNG; native-fallback SynthID depths 2 and 10, ngram length 4, two leaves, history size 1024, original zero-context initialization and fresh state per replicate.

PRC key fingerprint: `ed6c81de0bc7cf35cf1c05d3ba0fc0db846e78c31241d680582d0778f7f5e783`. Key/partition artifact SHA256: `58d9636e615465f0818e2a2fe9062d0560b21cfa42a84ebefef986ba0706a8cc`. SynthID keys: depth 2 `[654, 400]`; depth 10 `[654, 400, 836, 123, 340, 443, 597, 160, 57, 29]`. Full prompt, artifact, source and runtime hashes are frozen in the linked manifest.

## Precision audit and temperature placement

- **model:** existing BF16 weights/forward on H100, static KV cache
- **temperature:** divide BF16 logits by Python float .7 once, retaining BF16, before existing samplers
- **ordinary:** scaled BF16 logits -> float32 -> softmax -> existing multinomial
- **prc:** BF16 softmax, BF16 partition multiplication and bucket sum; FP32 channel; FP64 bucket uniform; FP32 masked conditional softmax/CDF and token uniform
- **synthid:** internal temperature remains 1; BF16 score updates; FP32 final sampling softmax; native zero context, state reset and repeat fallback
- **fallback:** softmax(scaled_BF16_logits.float()), exactly ordinary distribution on the same history
- **replay:** original BF16 completion-only bucket path with the same single BF16 temperature scaling; no clipping/repair of bucket masses

Source layout has moved and legacy scoring metadata was corrected; generate_method AST and numerical source files match the original. Depth-2 T=1 responses come from the completed depth follow-up using the same implementation.

The original sampler functions and numerical files are preserved. This does not inherit the later top-k/0.6B FP32 bucket or SynthID score paths. SynthID's internal division remains /1; the shared BF16 logit adapter performs the only nontrivial temperature scaling. Its native /0.7 path and ordinary fallback were checked for exact equality on actual model logits.

## Results beside saved T=1

Self-BLEU is on the 0–1 scale (lower means less overlap). T=1 values use saved original full-vocabulary outputs; they are not T=.7 controls. Each temperature uses its own ordinary sampling responses.

### 1024 tokens — primary length

| Setting | T | Self-BLEU [95% CI] | Detection / TPR 95% CI | Repeated 4-gram | Distinct-3 |
|---|---:|---|---:|---:|---:|
| null | 1 | +0.01928 [+0.01649, +0.02229] | — | 0.02925 | 0.95145 |
| null | 0.7 | +0.05182 [+0.04317, +0.06093] | — | 0.29296 | 0.65601 |
| prc | 1 | +0.01890 [+0.01581, +0.02227] | 97/100 [93.0%, 100.0%] | 0.02383 | 0.95614 |
| prc | 0.7 | +0.04789 [+0.04002, +0.05610] | 3/100 [0.0%, 7.0%] | 0.34070 | 0.61037 |
| synthid_depth2 | 1 | +0.01987 [+0.01592, +0.02418] | 100/100 [100.0%, 100.0%] | 0.03276 | 0.94619 |
| synthid_depth2 | 0.7 | +0.05886 [+0.04987, +0.06748] | 93/100 [88.0%, 97.0%] | 0.30458 | 0.64441 |
| synthid_depth10 | 1 | +0.02206 [+0.01909, +0.02518] | 100/100 [100.0%, 100.0%] | 0.02423 | 0.95538 |
| synthid_depth10 | 0.7 | +0.06070 [+0.05094, +0.07053] | 100/100 [100.0%, 100.0%] | 0.31355 | 0.63735 |

| T | Paired contrast | Self-BLEU difference [95% CI] | TPR difference [95% CI] |
|---:|---|---|---|
| 0.7 | prc − synthid_depth2 | -0.01098 [-0.01976, -0.00227] | -0.90000 [-0.95000, -0.84000] |
| 0.7 | prc − synthid_depth10 | -0.01281 [-0.02233, -0.00360] | -0.97000 [-1.00000, -0.93000] |
| 0.7 | prc − null | -0.00393 [-0.01346, +0.00524] | — |
| 0.7 | synthid_depth2 − null | +0.00704 [-0.00282, +0.01653] | — |
| 0.7 | synthid_depth10 − null | +0.00887 [-0.00139, +0.01980] | — |
| 1 | prc − synthid_depth2 | -0.00097 [-0.00459, +0.00265] | -0.03000 [-0.07000, +0.00000] |
| 1 | prc − synthid_depth10 | -0.00316 [-0.00731, +0.00102] | -0.03000 [-0.07000, +0.00000] |
| 1 | prc − null | -0.00038 [-0.00369, +0.00297] | — |
| 1 | synthid_depth2 − null | +0.00059 [-0.00334, +0.00487] | — |
| 1 | synthid_depth10 − null | +0.00278 [-0.00055, +0.00622] | — |

Pilot-null counts: T=0.7 prc: 0/100; T=0.7 synthid_depth2: 0/100; T=0.7 synthid_depth10: 0/100; T=1 prc: 0/100; T=1 synthid_depth2: 0/100; T=1 synthid_depth10: 0/100.

### 400 tokens — secondary

| Setting | T | Self-BLEU [95% CI] | Detection / TPR 95% CI | Repeated 4-gram | Distinct-3 |
|---|---:|---|---:|---:|---:|
| null | 1 | +0.01871 [+0.01550, +0.02249] | — | 0.01290 | 0.97344 |
| null | 0.7 | +0.05277 [+0.04407, +0.06187] | — | 0.09788 | 0.85887 |
| prc | 1 | +0.01901 [+0.01640, +0.02182] | 55/100 [44.0%, 65.0%] | 0.01086 | 0.97475 |
| prc | 0.7 | +0.04607 [+0.03913, +0.05288] | 1/100 [0.0%, 3.0%] | 0.11322 | 0.84628 |
| synthid_depth2 | 1 | +0.01879 [+0.01599, +0.02206] | 100/100 [100.0%, 100.0%] | 0.01544 | 0.97133 |
| synthid_depth2 | 0.7 | +0.06066 [+0.05190, +0.07006] | 87/100 [80.0%, 93.0%] | 0.10632 | 0.85166 |
| synthid_depth10 | 1 | +0.02187 [+0.01862, +0.02547] | 100/100 [100.0%, 100.0%] | 0.01008 | 0.97653 |
| synthid_depth10 | 0.7 | +0.06368 [+0.05373, +0.07419] | 100/100 [100.0%, 100.0%] | 0.10436 | 0.85312 |

| T | Paired contrast | Self-BLEU difference [95% CI] | TPR difference [95% CI] |
|---:|---|---|---|
| 0.7 | prc − synthid_depth2 | -0.01460 [-0.02410, -0.00532] | -0.86000 [-0.92025, -0.79000] |
| 0.7 | prc − synthid_depth10 | -0.01761 [-0.02802, -0.00829] | -0.99000 [-1.00000, -0.97000] |
| 0.7 | prc − null | -0.00670 [-0.01511, +0.00109] | — |
| 0.7 | synthid_depth2 − null | +0.00790 [-0.00291, +0.01873] | — |
| 0.7 | synthid_depth10 − null | +0.01091 [+0.00242, +0.01968] | — |
| 1 | prc − synthid_depth2 | +0.00022 [-0.00345, +0.00399] | -0.45000 [-0.56000, -0.35000] |
| 1 | prc − synthid_depth10 | -0.00286 [-0.00664, +0.00091] | -0.45000 [-0.56000, -0.35000] |
| 1 | prc − null | +0.00029 [-0.00376, +0.00420] | — |
| 1 | synthid_depth2 − null | +0.00007 [-0.00369, +0.00393] | — |
| 1 | synthid_depth10 − null | +0.00315 [-0.00084, +0.00694] | — |

Pilot-null counts: T=0.7 prc: 0/100; T=0.7 synthid_depth2: 0/100; T=0.7 synthid_depth10: 0/100; T=1 prc: 0/100; T=1 synthid_depth2: 0/100; T=1 synthid_depth10: 1/100.

## Validation, anomalies and reproducibility

Validation passed: 600 short responses, 64 tokens each, excluded from the evaluation cohort. Every arm used batch 50, both seeds and a same-seed rerun; all same-seed checks, fresh randomness checks, forced-repeat checks, original-codeword checks and generation/replay alignment passed. All 400 full trajectories reproduce their validated 64-token prefixes.

Observed PRC draw bucket mismatches: 0. No sampler repair, token deletion, coordinate shift or threshold change was performed.

- prc, 1024 tokens: 0 zero-probability observations; 8 contradictory saved bucket endpoints / 102300 replay positions.
- prc, 400 tokens: 0 zero-probability observations; 3 contradictory saved bucket endpoints / 39900 replay positions.
- null, 1024 tokens: 0 zero-probability observations; 8 contradictory saved bucket endpoints / 102300 replay positions.
- null, 400 tokens: 0 zero-probability observations; 4 contradictory saved bucket endpoints / 39900 replay positions.

All contradictory cases have a saved BF16-aggregated p1 of 1 with an observed bucket-0 token. Every observed token nevertheless has positive replay probability. This distinguishes rounding of the saved bucket scalar from an actually impossible token: the event counts do not show that the entire bucket has zero underlying probability. All cases occur after position 64. The unchanged detector clips the endpoint scalar and returns a full-magnitude score on these contradictions; that score is the existing convention, not a posterior justified by the endpoint scalar. Prefix-specific early/later diagnostics, fallback trajectories, absolute TPR intervals and all paired repetition intervals are included in [summary.json](summary.json).

All means/contrasts use 2,000 paired resamples of the 50 prompts, retaining both seeds and all arms. [Manifest and source audit](manifest.json) · [Validation](validate_report.json) · [Batch report](batch_report.json) · [Prompt metrics](prompt_metrics.json). Raw completions and GPU replay traces are stored on `prc-completion-only/self_bleu_temperature/ab3b8bc1adac0e68c341e5278ca81f9297798d69a0b0501386868cd4ad8b5988`; local token-only scores are reproducible from those artifacts.


### Native fallback counts

| T | Depth | Length | Responses with repeat | Total fallbacks |
|---:|---:|---:|---:|---:|
| 1 | 10 | 1024 | 100/100 | 4553 |
| 0.7 | 10 | 1024 | 100/100 | 37002 |
| 1 | 2 | 1024 | 99/100 | 5492 |
| 0.7 | 2 | 1024 | 100/100 | 36269 |
| 1 | 10 | 400 | 98/100 | 934 |
| 0.7 | 10 | 400 | 100/100 | 5814 |
| 1 | 2 | 400 | 91/100 | 1138 |
| 0.7 | 2 | 400 | 100/100 | 5878 |

The native fallback and detector context mask are unchanged. First-repeat positions and counts for each response are retained in summary.json.

## Cost and stop

Before launch: expected new worker cost $1.568, based on the recorded prior 8B timings with 25% margin. Measured worker time × frozen resource rate: $1.00943. Cumulative planning charge including $0.50 new overhead allowance: $12.51222 / $200. These estimates are not a settled Modal invoice.

- Exploratory sensitivity analysis on the existing 50-prompt cohort and fixed keys; no claim of general superiority.
- Nominal thresholds unchanged; 100 paired pilot nulls per temperature do not establish matched empirical FPR or permit calibration.
- Intervals containing zero do not establish equivalence. Nonprimary intervals are exploratory and unadjusted for multiplicity.
- Self-BLEU is decoded-text lexical overlap, not semantic quality. Repetition uses raw token IDs.
- At boundary outcomes, bootstrap intervals do not establish zero population FPR or perfect detection.
- T=1 results are saved original outputs, not matched controls for T=.7; each temperature has its own ordinary controls. Existing T=1 point metrics were reproduced; intervals use this report's common bootstrap draws.
- Original BF16 bucket/score arithmetic is retained. BF16 temperature division happens before existing method-specific casts; this intentionally differs from scaling FP32-cast logits.

STOP: no extra configurations, prompts, lengths, calibration or eta changes are authorized by this run.

## Run provenance

Setup commit `24f28c4` was pushed to `comparison-with-redetect` before full dispatch. [Validation worker](https://modal.com/apps/new-prc-watermark/main/ap-cdwukeZ9AKBqRF6n1AURDa) passed; [evaluation/replay worker](https://modal.com/apps/new-prc-watermark/main/ap-6iDaRTRbG0rFJxLYPeD7Yn) passed. Validation took 112.115 seconds; evaluation and replay took 669.491 seconds. The local client timed out waiting for final app logs only after the complete passing report was returned; all 24 validation/evaluation artifact hashes were subsequently verified.

All 29 local checks passed before dispatch. The 400 saved T=1 prompt-level Self-BLEU/detection entries were reproduced. An independent calculation from the new completion pairs exactly reproduces the primary mean and paired bootstrap interval; see [independent_primary_check.json](independent_primary_check.json). The final analysis contains 800 prompt records and 2,400 detector score records.
