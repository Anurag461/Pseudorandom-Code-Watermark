# Matched top-100 comparison

Exactly 500 new responses are complete: 100 per setting. No further generation is queued. The reporting correction was committed and pushed as `90057f0` before preflight and generation. The primary comparison was fixed in advance: PRC minus SynthID depth 2 Self-BLEU at 1,024 tokens.

The primary result **does not establish a PRC diversity advantage over SynthID
depth 2**: PRC-minus-depth-2 Self-BLEU is +0.00095 [−0.00503, +0.00734]. PRC
detects 85/100 versus depth 2's 100/100; the paired TPR difference is −15
percentage points [−22, −8]. Thus this setting gives no demonstrated primary
diversity gain for PRC and lower detection at the predeclared nominal thresholds.
This is not a comparison at empirically matched false-positive rates.

PRC has lower Self-BLEU than depth 30 (difference −0.01376
[−0.02065, −0.00717]), with the same detection tradeoff. The depth-10 comparison
is uncertain. Relative to matched ordinary sampling, PRC and depth 2 have
uncertain Self-BLEU differences, while depths 10 and 30 have higher Self-BLEU.
Every paired PRC-minus-SynthID repetition/distinct-3 interval includes zero at
both lengths. Within-response repetition and between-response Self-BLEU remain
different measurements.

**No contradictory bucket endpoints occurred** in either the 102,300 PRC or
102,300 ordinary replay positions. Token-support mismatches did occur: 0.719%
for PRC and 0.770% for ordinary at 1,024 tokens. Their rates are much higher in
positions 2–64 (6.000% and 6.651%) than in positions 65–1,024 (0.373% and
0.384%). There were zero generation support violations. The clipped-endpoint
convention was therefore not triggered by contradictory endpoints in this
cohort; these diagnostics do not establish the cause of PRC's missed detections.

At the secondary 400-token length, PRC detects 33/100 and every SynthID depth
detects 100/100. Depth 10 has one pilot false positive at 400 tokens; every
other method/length null count is zero. The 100 clustered pilot nulls are too
small to establish calibration at .001. No further experiments were launched.

## Primary length: 1,024 tokens

Self-BLEU uses a 0–1 scale; lower means less lexical overlap between the two responses to a prompt. Repeated-four-gram fraction is lower-is-better; distinct-3 is higher-is-better. Brackets give 95% paired prompt-bootstrap percentile intervals. Detection uses the original nominal .001 thresholds, not empirically matched false-positive rates.

| Setting | Self-BLEU | Detected; TPR interval | Repeated 4-grams (%) | Distinct-3 (%) |
| --- | --- | --- | --- | --- |
| Ordinary | 0.03039 [0.02669, 0.03427] | — | 3.49 [2.89, 4.11] | 93.33 [92.56, 94.07] |
| PRC η=.05 | 0.03329 [0.02872, 0.03827] | 85/100; 85.0 [78.0, 92.0]% | 3.61 [2.86, 4.46] | 93.40 [92.42, 94.32] |
| SynthID depth 2 | 0.03234 [0.02827, 0.03672] | 100/100; 100.0 [100.0, 100.0]% | 3.49 [2.83, 4.21] | 93.50 [92.58, 94.33] |
| SynthID depth 10 | 0.03675 [0.03187, 0.04184] | 100/100; 100.0 [100.0, 100.0]% | 3.79 [2.91, 4.75] | 93.33 [92.15, 94.41] |
| SynthID depth 30 | 0.04704 [0.04161, 0.05310] | 100/100; 100.0 [100.0, 100.0]% | 3.59 [2.78, 4.44] | 93.17 [92.20, 94.11] |

### Direct PRC-minus-SynthID contrasts

The depth-2 Self-BLEU contrast is primary. Other contrasts and metrics are exploratory; no multiplicity adjustment is applied. Differences are bootstrapped directly within prompts, not inferred from overlap of separate intervals.

| Direct paired difference | Δ Self-BLEU | Δ TPR (pp) | Δ repeated 4-grams (pp) | Δ distinct-3 (pp) |
| --- | --- | --- | --- | --- |
| PRC η=.05 − SynthID depth 2 | +0.00095 [-0.00503, +0.00734] | -15.0 [-22.0, -8.0] | +0.12 [-0.74, +0.93] | -0.10 [-1.02, +0.84] |
| PRC η=.05 − SynthID depth 10 | -0.00346 [-0.00957, +0.00274] | -15.0 [-22.0, -8.0] | -0.18 [-1.27, +0.88] | +0.07 [-1.11, +1.27] |
| PRC η=.05 − SynthID depth 30 | -0.01376 [-0.02065, -0.00717] | -15.0 [-22.0, -8.0] | +0.02 [-0.90, +0.95] | +0.24 [-0.87, +1.28] |

### Differences from the matched ordinary control

Positive Δ Self-BLEU or Δ repeated-four-gram fraction means less diversity; positive Δ distinct-3 means more diversity.

| Direct paired difference | Δ Self-BLEU | Δ repeated 4-grams (pp) | Δ distinct-3 (pp) |
| --- | --- | --- | --- |
| PRC η=.05 − Ordinary | +0.00290 [-0.00156, +0.00755] | +0.12 [-0.72, +0.96] | +0.07 [-0.90, +1.08] |
| SynthID depth 2 − Ordinary | +0.00195 [-0.00341, +0.00717] | +0.00 [-0.73, +0.75] | +0.17 [-0.79, +1.11] |
| SynthID depth 10 − Ordinary | +0.00636 [+0.00061, +0.01191] | +0.30 [-0.71, +1.43] | +0.00 [-1.25, +1.18] |
| SynthID depth 30 − Ordinary | +0.01665 [+0.01096, +0.02235] | +0.10 [-0.78, +1.00] | -0.16 [-1.18, +0.85] |

## Secondary length: 400 tokens

These are prefixes of the same responses, not additional generations.

| Setting | Self-BLEU | Detected; TPR interval | Repeated 4-grams (%) | Distinct-3 (%) |
| --- | --- | --- | --- | --- |
| Ordinary | 0.02389 [0.02063, 0.02738] | — | 2.06 [1.48, 2.69] | 95.96 [95.16, 96.69] |
| PRC η=.05 | 0.02835 [0.02371, 0.03406] | 33/100; 33.0 [24.0, 42.0]% | 1.60 [1.27, 2.00] | 96.54 [96.01, 97.02] |
| SynthID depth 2 | 0.02711 [0.02338, 0.03119] | 100/100; 100.0 [100.0, 100.0]% | 1.70 [1.24, 2.21] | 96.43 [95.73, 97.09] |
| SynthID depth 10 | 0.03104 [0.02614, 0.03620] | 100/100; 100.0 [100.0, 100.0]% | 1.39 [1.04, 1.79] | 96.81 [96.29, 97.30] |
| SynthID depth 30 | 0.04262 [0.03641, 0.04900] | 100/100; 100.0 [100.0, 100.0]% | 1.30 [0.93, 1.71] | 96.71 [96.12, 97.29] |

| Direct paired difference | Δ Self-BLEU | Δ TPR (pp) | Δ repeated 4-grams (pp) | Δ distinct-3 (pp) |
| --- | --- | --- | --- | --- |
| PRC η=.05 − SynthID depth 2 | +0.00124 [-0.00398, +0.00665] | -67.0 [-76.0, -58.0] | -0.10 [-0.66, +0.41] | +0.11 [-0.56, +0.82] |
| PRC η=.05 − SynthID depth 10 | -0.00269 [-0.00914, +0.00386] | -67.0 [-76.0, -58.0] | +0.21 [-0.25, +0.69] | -0.27 [-0.86, +0.31] |
| PRC η=.05 − SynthID depth 30 | -0.01426 [-0.02081, -0.00715] | -67.0 [-76.0, -58.0] | +0.29 [-0.24, +0.79] | -0.17 [-0.87, +0.54] |

| Direct paired difference | Δ Self-BLEU | Δ repeated 4-grams (pp) | Δ distinct-3 (pp) |
| --- | --- | --- | --- |
| PRC η=.05 − Ordinary | +0.00446 [-0.00068, +0.01021] | -0.46 [-1.16, +0.20] | +0.58 [-0.23, +1.43] |
| SynthID depth 2 − Ordinary | +0.00322 [-0.00138, +0.00799] | -0.36 [-1.13, +0.38] | +0.47 [-0.45, +1.43] |
| SynthID depth 10 − Ordinary | +0.00715 [+0.00218, +0.01203] | -0.66 [-1.27, -0.08] | +0.85 [+0.18, +1.56] |
| SynthID depth 30 − Ordinary | +0.01873 [+0.01206, +0.02552] | -0.75 [-1.41, -0.12] | +0.75 [-0.01, +1.54] |

## Matched pilot nulls

Each detector is applied to the same 100 newly generated top-100 ordinary responses (50 prompts × two seeds). Historical full-vocabulary nulls are incompatible and are not pooled or reused. Counts below are false positives.

| Detector | 400 tokens | 1,024 tokens |
| --- | --- | --- |
| PRC η=.05 | 0/100 | 0/100 |
| SynthID depth 2 | 0/100 | 0/100 |
| SynthID depth 10 | 1/100 | 0/100 |
| SynthID depth 30 | 0/100 | 0/100 |

The sample cannot establish calibration at .001. All-zero or all-success bootstrap intervals are degenerate and do not imply zero population false-positive probability or perfect detection.

## Prompt-free replay diagnostics

These diagnostics are for PRC detection of PRC and ordinary responses. Generation had access to a prompt; completion-only replay does not. A token outside replay’s top-100 set is therefore not automatically a generation violation. The observed token can be absent while other tokens in its bucket retain positive mass.

A **bucket-endpoint contradiction** means the saved FP32 p1 is exactly zero with observed bucket 1, or exactly one with observed bucket 0. This uses the recorded scalar, including possible endpoint rounding. The unchanged detector clips p1 and can assign a magnitude-one soft score in these cases. That is an existing scoring convention, not a posterior justified for an observation assigned zero probability. No tokens were dropped, no coordinates shifted, and no detector or threshold changed.

Completion positions are one-based. Position 1 abstains. The 400-token diagnostic uses the first 399 entries; the 1,024-token diagnostic uses the first 1,023. Early means positions 2–64; later means 65–n. Rates divide event counts by evaluated positions. The two event columns can overlap.

| Response source | Prefix | Positions | Evaluated token positions | Outside top-100: count; % [CI]; affected responses | Endpoint contradiction: count; % [CI]; affected responses |
| --- | --- | --- | --- | --- | --- |
| PRC η=.05 | 1024 | all (2–1024) | 102300 | 736; 0.719 [0.642, 0.801]; 100/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| PRC η=.05 | 1024 | early (2–64) | 6300 | 378; 6.000 [5.270, 6.714]; 97/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| PRC η=.05 | 1024 | later (65–1024) | 96000 | 358; 0.373 [0.323, 0.424]; 97/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| PRC η=.05 | 400 | all (2–400) | 39900 | 617; 1.546 [1.376, 1.724]; 100/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| PRC η=.05 | 400 | early (2–64) | 6300 | 378; 6.000 [5.270, 6.714]; 97/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| PRC η=.05 | 400 | later (65–400) | 33600 | 239; 0.711 [0.598, 0.836]; 89/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 1024 | all (2–1024) | 102300 | 788; 0.770 [0.691, 0.848]; 100/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 1024 | early (2–64) | 6300 | 419; 6.651 [5.984, 7.286]; 99/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 1024 | later (65–1024) | 96000 | 369; 0.384 [0.331, 0.439]; 94/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 400 | all (2–400) | 39900 | 635; 1.591 [1.419, 1.764]; 100/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 400 | early (2–64) | 6300 | 419; 6.651 [5.984, 7.286]; 99/100 | 0; 0.000 [0.000, 0.000]; 0/100 |
| Ordinary | 400 | later (65–400) | 33600 | 216; 0.643 [0.530, 0.765]; 80/100 | 0; 0.000 [0.000, 0.000]; 0/100 |

| Response source | Prefix | p1=0, bucket 1 | p1=1, bucket 0 | Outside and contradictory | Outside without endpoint contradiction |
| --- | --- | --- | --- | --- | --- |
| PRC η=.05 | 1024 | 0 | 0 | 0 | 736 |
| PRC η=.05 | 400 | 0 | 0 | 0 | 617 |
| Ordinary | 1024 | 0 | 0 | 0 | 788 |
| Ordinary | 400 | 0 | 0 | 0 | 635 |

Per-response/prefix diagnostic records and the complete saved boolean vectors permit independent checks; main-result summaries are in `summary.json`. A lack of endpoint contradictions would not establish that prompt-free replay equals the generation distribution.

## Common-history distribution measurements

Preselected before the new responses: prompts 0, 7, 19, 31, 49 at generation positions 0, 32, 128, 400, 1023 (zero-based), using saved ordinary seed-12345 histories. All 50 histories were replayed in the same model batch; measurements use the 25 selected histories. Both decoders use FP32 probability arithmetic. “Full vocabulary” is a controlled reference, not an exact recreation of the historical BF16 SynthID probability path. Native repeat fallback remains on.

Collision probability is Σ p(token)²; maximum probability is max p(token). Retained mass is probability on the ordinary top-100 support after watermarking. Values below are means across the 25 histories; the linked records contain individual values, history hashes and fallback flags. These fixed-history measurements are descriptive and do not by themselves explain complete-response diversity.

| Decoder | Method | Collision probability | Maximum probability | Top-100 retained mass | Base mass before truncation |
| --- | --- | --- | --- | --- | --- |
| full_vocab_fp32_reference | ordinary | 0.493972 | 0.608896 | 0.965926 | 0.965926 |
| full_vocab_fp32_reference | synthid_depth2 | 0.569810 | 0.683616 | 0.969879 | 0.965926 |
| full_vocab_fp32_reference | synthid_depth10 | 0.717703 | 0.799167 | 0.970507 | 0.965926 |
| full_vocab_fp32_reference | synthid_depth30 | 0.724606 | 0.790109 | 0.950562 | 0.965926 |
| top100_fp32 | ordinary | 0.507671 | 0.621708 | 1.000000 | 0.965926 |
| top100_fp32 | synthid_depth2 | 0.586296 | 0.697667 | 1.000000 | 0.965926 |
| top100_fp32 | synthid_depth10 | 0.739752 | 0.816823 | 1.000000 | 0.965926 |
| top100_fp32 | synthid_depth30 | 0.781309 | 0.838859 | 1.000000 | 0.965926 |

All 200 individual measurements: [common_history_metrics.json](common_history_metrics.json). The v2 artifact is byte-identical to v1, consistent with the reporting-only change.

## Native SynthID fallback

| Setting | Prefix | Responses encountering fallback | Fallback events | Mean events per response |
| --- | --- | --- | --- | --- |
| SynthID depth 2 | 1024 | 100/100 | 6629 | 66.29 |
| SynthID depth 2 | 400 | 100/100 | 1418 | 14.18 |
| SynthID depth 10 | 1024 | 100/100 | 6805 | 68.05 |
| SynthID depth 10 | 400 | 99/100 | 1262 | 12.62 |
| SynthID depth 30 | 1024 | 100/100 | 6968 | 69.68 |
| SynthID depth 30 | 400 | 99/100 | 1300 | 13.00 |

## Settings and verification

- Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; pinned checkpoint/tokenizer and original 50 prompts, indices 0–49, 50 tokens each.
- H100 80 GB; BF16 model weights/forward, FP32 logits-to-probability arithmetic in all five arms and PRC replay; Torch 2.4.0, CUDA 12.1; TF32 disabled, BF16 reduced-precision reduction enabled; static KV cache; batch size 50.
- Temperature 1, top-p 1, exactly top-k 100 applied before watermarking. Boundary ties choose lower token IDs; excluded logits are −1e12 with zero FP32 mass. No EOS stopping; all responses contain exactly 1,024 new token IDs.
- Sampling seeds 12345 and 67890. PRC η=.05, check weight 3, row rate 99/100, key seed 12345, original partition and position-addressed randomness. Both original latent bit streams are verified. No key changes across sampling seeds.
- SynthID depths 2/10/30 use the original fixed nested key bank, ngram length 4, two leaves, context history size 1024, native repeat fallback on, skip-first-ngram off, internal top-k off. Exact key lists/fingerprints are in the [manifest](manifest.json).
- Completion-only detection: PRC fresh raw-completion replay with first-coordinate abstention and the unchanged MAP/Hoeffding test; SynthID explicit per-depth keys, official repeated-context mask, and existing layer-weighted normal test (linearly spaced weights 10 to 1, normalized). Nominal FPR .001, no empirical calibration.
- Symmetric sentence Self-BLEU between the two responses to each prompt, SacreBLEU 2.4.3, 13a tokenizer, exponential smoothing, effective order, case-sensitive, divided by 100. Decode with the pinned tokenizer and skip special tokens for BLEU only. Repetition uses raw token IDs: 1−unique4/(T−3) and unique3/(T−2).
- Bootstrap: 2,000 joint resamples of 50 prompt clusters, both seeds retained, seed 20260918; marginal 95% percentile intervals. Draw hash `5ab115a4b5c632f81fd04e9cf02b5d4f67cf68dd60cf622759f99a04fb767c77`.
- 34 local tests passed. CUDA preflight passed stable top-100, empty/full PRC bucket cases, fixed codewords, native SynthID fallback/support, raw-completion input checks, independent replay oracle and prefix/order invariance.
- Generation checked all 512,000 token choices: zero support violations. Impossible inverse-CDF boundary repairs: 0.
- Analysis verified 500 full responses, 500 prompt/prefix records and 1600 detector records. Historical response/replay caches were not reused.
- An independent recount verified all 400 PRC/null prefix diagnostic records
  and all 12 cohort/window summaries directly from the saved vectors. An
  independent calculation reproduced the primary paired bootstrap interval;
  see [independent checks](independent_checks.json).

## Cost and reproducibility

| Item | Estimated USD |
| --- | --- |
| Prior studies | 7.71132 |
| v1 validation only | 0.15581 |
| v2 preflight | 0.11133 |
| 500-response generation plus PRC replay | 1.29572 |
| New overhead allowance | 0.50000 |
| Cumulative planning charge | 9.77418 |

Estimates use the frozen resource rate $0.00129148/s and include the reporting revision’s earlier preflight plus a $0.50 overhead allowance. These are not settled invoices. The study ceiling remains $200. No retry or additional generation was launched.

Source commit before run: `90057f0`. Manifest ID: `85a49cb02e1eb7fa919259e5e3b6625e493aa6e3dfb9f00a1a474ca923323cf0`. Raw generation/replay artifacts are on the existing `prc-completion-only` Modal volume under `self_bleu_topk/85a49cb02e1eb7fa919259e5e3b6625e493aa6e3dfb9f00a1a474ca923323cf0/`. The [validation report](validate_report.json) and [batch report](batch_report.json) give file hashes, runtime and elapsed time.

[Execution record and Modal run links](execution.json) confirm 50 common
prompts, two seeds, five settings, and 100 responses per setting.

Reproduce with the pinned environment and source bytes:

```sh
MODAL_PROFILE=new-prc-watermark python -m self_bleu.topk collect --stage validate --download
MODAL_PROFILE=new-prc-watermark python -m self_bleu.topk collect --stage batch --download
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m self_bleu.topk analyze
```

[Summary and all intervals](summary.json) · [Prompt-level metrics](prompt_metrics.json) · [Common-history records](common_history_metrics.json) · [Frozen runbook](../../../self_bleu/topk.md)

This is one model, one fixed key bank/partition and 50 prompts. Self-BLEU measures lexical overlap, not semantic quality. No temperature, depth, eta or prompt sweep follows this batch.
