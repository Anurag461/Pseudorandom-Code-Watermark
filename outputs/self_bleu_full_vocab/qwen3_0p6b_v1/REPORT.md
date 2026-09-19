# Qwen3-0.6B full-vocabulary comparison

600 new responses: six settings × the same 50 prompts × two seeds. Temperature 1, top-p 1, no top-k; 1,024 tokens each. Fixed keys. Repeat fallback ON for TextSeal α=.1, Gumbel-max, and SynthID depths 2/10. PRC η=.05 is unchanged.

BF16 model execution; FP32 probabilities in generation and PRC replay. TextSeal detection retains upstream BF16 Hugging Face eager entropy, with fresh direct replay at each prefix. All detectors use raw completion tokens only and nominal p < .001.

Intervals are 95% percentile intervals from 2,000 paired prompt-cluster bootstrap resamples. Both seeds remain within each prompt. Self-BLEU is on a 0–1 scale; lower indicates more lexical diversity.

The primary PRC-minus-depth-2 Self-BLEU difference is **+0.00051 [−0.00141, +0.00248]**. This cohort does not establish a PRC diversity advantage over SynthID depth 2; the depth-10 difference is also uncertain at 1,024 tokens. PRC, both SynthID depths, and ordinary sampling have overlapping paired Self-BLEU results. PRC has lower Self-BLEU than TextSeal and Gumbel with fallback enabled.

PRC detects 99/100 at 1,024 tokens (prompt-bootstrap TPR interval 97–100%), versus 100/100 for every baseline. At 400 tokens it detects 79/100 (70–88%), versus 100/100 for every baseline; the paired deficit is 21 percentage points [12, 30]. PRC-minus-baseline repetition intervals at 1,024 tokens all include zero. Self-BLEU differences therefore should not be presented as a demonstrated within-response repetition advantage.

All five detectors have 0/100 matched pilot-null detections at both lengths. There are zero observed-token zero-probability cases and zero contradictory PRC bucket endpoints in both cohorts at both prefixes, including early and later windows. There were zero generation support violations across 614,400 tokens.

[Full metrics and intervals](summary.json) · [Prompt-level paired metrics](prompt_metrics.json) · [Independent verification](analysis_verification.json) · [Execution provenance](execution.json) · [Frozen setup](manifest.json).

## 1024 tokens

| Setting | Detection | Self-BLEU [95% CI] | Repeated 4-gram fraction | Distinct-3 |
|---|---:|---|---:|---:|
| null | — | 0.01415 [0.01242, 0.01601] | 0.02037 | 0.96656 |
| prc | 99/100 | 0.01353 [0.01210, 0.01503] | 0.02626 | 0.95873 |
| synthid_depth2 | 100/100 | 0.01302 [0.01150, 0.01476] | 0.01706 | 0.97034 |
| synthid_depth10 | 100/100 | 0.01442 [0.01240, 0.01664] | 0.02057 | 0.96505 |
| textseal | 100/100 | 0.03802 [0.03395, 0.04221] | 0.02313 | 0.95510 |
| gumbel | 100/100 | 0.21869 [0.19130, 0.24809] | 0.02821 | 0.94770 |

| Paired contrast (left minus right) | Self-BLEU difference [95% CI] | TPR difference [95% CI] |
|---|---|---|
| prc − synthid_depth2 | 0.00051 [-0.00141, 0.00248] | -0.01000 [-0.03000, 0.00000] |
| prc − synthid_depth10 | -0.00089 [-0.00302, 0.00111] | -0.01000 [-0.03000, 0.00000] |
| prc − textseal | -0.02449 [-0.02863, -0.02039] | -0.01000 [-0.03000, 0.00000] |
| prc − gumbel | -0.20516 [-0.23436, -0.17785] | -0.01000 [-0.03000, 0.00000] |
| prc − null | -0.00062 [-0.00290, 0.00172] | — |
| synthid_depth2 − null | -0.00113 [-0.00354, 0.00147] | — |
| synthid_depth10 − null | 0.00027 [-0.00262, 0.00331] | — |
| textseal − null | 0.02387 [0.01927, 0.02850] | — |
| gumbel − null | 0.20454 [0.17749, 0.23353] | — |

Pilot-null detections: prc: 0/100, synthid_depth2: 0/100, synthid_depth10: 0/100, textseal: 0/100, gumbel: 0/100.

## 400 tokens

| Setting | Detection | Self-BLEU [95% CI] | Repeated 4-gram fraction | Distinct-3 |
|---|---:|---|---:|---:|
| null | — | 0.01509 [0.01305, 0.01749] | 0.00940 | 0.97977 |
| prc | 79/100 | 0.01336 [0.01186, 0.01498] | 0.01433 | 0.97354 |
| synthid_depth2 | 100/100 | 0.01456 [0.01246, 0.01692] | 0.01013 | 0.97982 |
| synthid_depth10 | 100/100 | 0.01665 [0.01388, 0.01994] | 0.01919 | 0.96862 |
| textseal | 100/100 | 0.04689 [0.03963, 0.05446] | 0.01786 | 0.96726 |
| gumbel | 100/100 | 0.43261 [0.37023, 0.49801] | 0.01242 | 0.97176 |

| Paired contrast (left minus right) | Self-BLEU difference [95% CI] | TPR difference [95% CI] |
|---|---|---|
| prc − synthid_depth2 | -0.00120 [-0.00395, 0.00117] | -0.21000 [-0.30000, -0.12000] |
| prc − synthid_depth10 | -0.00329 [-0.00706, -0.00014] | -0.21000 [-0.30000, -0.12000] |
| prc − textseal | -0.03353 [-0.04112, -0.02588] | -0.21000 [-0.30000, -0.12000] |
| prc − gumbel | -0.41925 [-0.48470, -0.35643] | -0.21000 [-0.30000, -0.12000] |
| prc − null | -0.00174 [-0.00419, 0.00070] | — |
| synthid_depth2 − null | -0.00053 [-0.00335, 0.00251] | — |
| synthid_depth10 − null | 0.00155 [-0.00177, 0.00538] | — |
| textseal − null | 0.03179 [0.02419, 0.03981] | — |
| gumbel − null | 0.41751 [0.35505, 0.48302] | — |

Pilot-null detections: prc: 0/100, synthid_depth2: 0/100, synthid_depth10: 0/100, textseal: 0/100, gumbel: 0/100.

## Repeat fallback

| Setting | Length | Responses encountering repeats | Total repeats/fallbacks |
|---|---:|---:|---:|
| synthid_depth2 | 1024 | 100/100 | 3031 |
| synthid_depth2 | 400 | 94/100 | 799 |
| synthid_depth10 | 1024 | 100/100 | 3569 |
| synthid_depth10 | 400 | 91/100 | 1243 |
| textseal | 1024 | 100/100 | 4594 |
| textseal | 400 | 98/100 | 1309 |
| gumbel | 1024 | 100/100 | 5352 |
| gumbel | 400 | 98/100 | 1127 |

Per-response repeat counts, first-repeat positions, absolute detection intervals, paired repetition intervals, and prefix-specific early/later PRC/null replay diagnostics are in summary.json. All completions and GPU replay traces are hash-verified in the existing Modal volume under `self_bleu_full_vocab/<manifest-id>`. The local raw score records (including SynthID/Gumbel token evidence) are reproduced from those artifacts by `python -m self_bleu.full_vocab analyze`.

## Verification and cost

Verification: {'passed': True, 'new_full_responses': 600, 'repeat_trajectories_verified': 400, 'support_checks': 614400, 'cdf_boundary_repairs': 0, 'score_records': 2000, 'prompt_records': 600}. Worker resource estimate: $0.72862; cumulative planning charge including allowance: $11.00279 of $200. This is not settled billing.

- Fixed keys, one model and 50 prompts; primary contrast predeclared; other intervals are exploratory without multiplicity correction.
- Pilot nulls are 100 responses clustered in 50 prompts; nominal p<.001 is not a matched empirical FPR.
- Boundary bootstrap intervals do not establish zero population false positives or perfect detection.
- Self-BLEU measures lexical overlap, not semantic quality. Repetition metrics use raw token IDs.
- Replay diagnostics count zero token probabilities and contradictory saved FP32 bucket endpoints. The unchanged detector clips endpoints; it does not drop tokens or shift coordinates.
- All contextual methods have repeat fallback enabled. SynthID initializes zero context; TextSeal/Gumbel start from the last three prompt tokens, preserving established adapters.
- Generation and PRC replay use FP32 probability calculations; TextSeal detection preserves upstream BF16 HF eager entropy at each actual prefix length. Historical 8B full-vocabulary probability paths differ and are not exactly matched controls.

The requested batch is complete. No further generation is queued.
