# SynthID depths 2 and 30: native fallback, completion-only detection

Completed **200 new 1,024-token responses**: depths 2 and 30 × 50 prompts × two seeds. Saved depth-10 SynthID, PRC and ordinary-sampling pairs are reused. No new baseline or null generation was performed.

**Depth 2 preserves the observed 100/100 detection at both lengths, with no established Self-BLEU difference from PRC, depth 10 or ordinary sampling. Depth 30 also detects 100/100, but has higher Self-BLEU than those three references at both lengths, with paired intervals excluding zero.** PRC's lower Self-BLEU than depth 30 therefore does not establish a diversity advantage over the evaluated SynthID settings as a whole. At 400 tokens, its observed detection is substantially lower than every SynthID depth; at 1,024, the three-point TPR difference has a paired interval reaching zero.

## Absolute detection and Self-BLEU

Each entry contains the mean and its 95% prompt-bootstrap interval. Detection is shown as a percentage; Self-BLEU is on the 0–1 scale, with lower values indicating more diversity between the two responses. Ordinary sampling has no watermarked TPR; its false-positive counts are below.

| Tokens | Setting | Detected | TPR % [95% interval] | Self-BLEU [95% interval] |
|---:|---|---:|---:|---:|
| 400 | PRC η=.05 | 55/100 | 55.0 [44.0, 65.0] | 0.01901 [0.01640, 0.02182] |
| 400 | Ordinary sampling | — | — | 0.01871 [0.01550, 0.02249] |
| 400 | SynthID depth 2 | 100/100 | 100.0 [100.0, 100.0] | 0.01879 [0.01599, 0.02206] |
| 400 | SynthID depth 10 | 100/100 | 100.0 [100.0, 100.0] | 0.02187 [0.01862, 0.02547] |
| 400 | SynthID depth 30 | 100/100 | 100.0 [100.0, 100.0] | 0.03304 [0.02688, 0.04008] |
| 1,024 | PRC η=.05 | 97/100 | 97.0 [93.0, 100.0] | 0.01890 [0.01581, 0.02227] |
| 1,024 | Ordinary sampling | — | — | 0.01928 [0.01649, 0.02229] |
| 1,024 | SynthID depth 2 | 100/100 | 100.0 [100.0, 100.0] | 0.01987 [0.01592, 0.02418] |
| 1,024 | SynthID depth 10 | 100/100 | 100.0 [100.0, 100.0] | 0.02206 [0.01909, 0.02518] |
| 1,024 | SynthID depth 30 | 100/100 | 100.0 [100.0, 100.0] | 0.02890 [0.02412, 0.03392] |

## Within-response repetition at 1,024 tokens

The same 100 saved SynthID depth-30 responses have **2.30% repeated token 4-grams** and **Distinct-3 = 0.9589**. These are arithmetic means across two responses for each of 50 prompts, with repeat fallback on. Repeated 4-grams counts occurrences after the first, divided by all contiguous four-token sequences; Distinct-3 divides the number of distinct contiguous three-token sequences by all such sequences.

These two point estimates were added on 25 September 2026. No new responses, detector scores, bootstrap intervals, or paired contrasts were computed. [Full-precision results, response-level counts, and source hashes](repetition_1024.json) · [Calculation script](calculate_repetition_1024.py).

## Direct paired PRC-minus-SynthID contrasts

Negative Self-BLEU differences favor PRC. Positive TPR differences favor PRC; TPR differences below are percentage points. Intervals directly resample the per-prompt differences.

| Tokens | Contrast | Self-BLEU difference [95% interval] | TPR difference, pp [95% interval] |
|---:|---|---:|---:|
| 400 | PRC minus depth 2 | +0.00022 [-0.00345, +0.00399] | -45.0 [-56.0, -35.0] |
| 400 | PRC minus depth 10 | -0.00286 [-0.00664, +0.00091] | -45.0 [-56.0, -35.0] |
| 400 | PRC minus depth 30 | -0.01403 [-0.02071, -0.00831] | -45.0 [-56.0, -35.0] |
| 1,024 | PRC minus depth 2 | -0.00097 [-0.00459, +0.00265] | -3.0 [-7.0, +0.0] |
| 1,024 | PRC minus depth 10 | -0.00316 [-0.00731, +0.00102] | -3.0 [-7.0, +0.0] |
| 1,024 | PRC minus depth 30 | -0.01000 [-0.01546, -0.00506] | -3.0 [-7.0, +0.0] |

## Direct paired depth contrasts

| Tokens | Contrast | Self-BLEU difference [95% interval] | TPR difference, pp [95% interval] |
|---:|---|---:|---:|
| 400 | Depth 2 minus depth 10 | -0.00308 [-0.00683, +0.00036] | +0.0 [+0.0, +0.0] |
| 400 | Depth 30 minus depth 10 | +0.01117 [+0.00535, +0.01796] | +0.0 [+0.0, +0.0] |
| 400 | Depth 30 minus depth 2 | +0.01425 [+0.00776, +0.02175] | +0.0 [+0.0, +0.0] |
| 1,024 | Depth 2 minus depth 10 | -0.00219 [-0.00589, +0.00148] | +0.0 [+0.0, +0.0] |
| 1,024 | Depth 30 minus depth 10 | +0.00684 [+0.00209, +0.01190] | +0.0 [+0.0, +0.0] |
| 1,024 | Depth 30 minus depth 2 | +0.00903 [+0.00347, +0.01472] | +0.0 [+0.0, +0.0] |

## Self-BLEU relative to ordinary sampling

Positive differences mean higher Self-BLEU than ordinary sampling.

| Setting minus ordinary sampling | 400 tokens: difference [95% interval] | 1,024 tokens: difference [95% interval] |
|---|---:|---:|
| PRC η=.05 | +0.00029 [-0.00376, +0.00420] | -0.00038 [-0.00369, +0.00297] |
| SynthID depth 2 | +0.00007 [-0.00369, +0.00393] | +0.00059 [-0.00334, +0.00487] |
| SynthID depth 10 | +0.00315 [-0.00084, +0.00694] | +0.00278 [-0.00055, +0.00622] |
| SynthID depth 30 | +0.01432 [+0.00813, +0.02159] | +0.00962 [+0.00529, +0.01458] |

## Saved unwatermarked outputs: false-positive counts

Nulls were rescored separately with each depth’s exact keys and depth-aware weights. PRC counts are reused from the completion-only comparison. The pilot nulls contain 100 responses from 50 prompts; the 500 historical nulls overlap those prompts and are kept separate.

| Detector | Pilot 400 | Pilot 1,024 | Historical 400 | Historical 1,024 |
|---|---:|---:|---:|
| SynthID depth 2 | 0/100 | 0/100 | 0/500 | 0/500 |
| SynthID depth 10 | 1/100 | 0/100 | 0/500 | 0/500 |
| SynthID depth 30 | 0/100 | 0/100 | 0/500 | 0/500 |
| PRC | 0/100 | 0/100 | 0/500 | 0/500 |

These are nominal-threshold comparisons, not matched empirical FPR. Zero observed false positives do not certify a 0.1% population FPR. All-success bootstrap intervals likewise do not establish perfect population detection.

## Frozen settings and inference

- Qwen3-8B-Base, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, pinned H100 BF16 runtime, original prompts 0–49, batch size 50, seeds 12345/67890, temperature/top-p 1, forced length 1,024.
- Native SynthID repeat fallback remains on. Context initialization, history size, sampling stream and generation implementation are unchanged.
- Depths use prefixes of the predeclared 30-key bank: the original first ten keys and the previously frozen extension. Keys stay fixed across prompts and seeds. Generation and detection explicitly receive the same per-depth key list.
- Completion-only detection receives only generated token IDs, with the existing official context-repetition mask and the existing depth-aware weighted-normal test at nominal p < .001. No prompts, generation entropies or other generation diagnostics enter detection.
- This is the existing frequentist SynthID variant. It does not evaluate the original paper’s Bayesian detector.
- Self-BLEU is symmetric sentence BLEU for each prompt’s two responses, using SacreBLEU 2.4.3, 13a tokenization, exponential smoothing and effective order, divided by 100. The pinned model tokenizer decodes completion prefixes with special tokens skipped.
- Resample 50 paired prompt clusters jointly, preserving both response slots and every setting: 2,000 draws, seed 20260918; percentile endpoints 2.5%/97.5%. Differences are computed per prompt before resampling. Intervals are exploratory, marginal and unadjusted for multiplicity.
- One fixed-key, single-model, 50-prompt cohort does not establish a broad parameter frontier or generalize across keys/models.

## Verification and reproducibility

- All four requested batches passed; 200 new full-length responses are saved. No automatic retries.
- Both 50-response depth-10/64-token controls reproduced saved prefixes exactly. Both first-seed, 50-response depth-2/30 batches reproduced the earlier 128-token parameter-smoke prefixes exactly.
- Forced-repeat probes passed for each requested depth on CPU and H100. Every generation step also passed the existing official batch/single-row processor parity check.
- CPU analysis reproduced 1,400 saved depth-10 decisions/p-values and all 100 depth-10 prompt-level Self-BLEU values.
- Direct-prefix versus full-trajectory extraction checks passed at both lengths on the first record of every scoring batch. Evidence depth and scorer weight length were checked explicitly.
- Saved 500 prompt-level metric records and 4,200 completion-only score records, including eligible positions, evidence hashes, exact keys and per-depth evidence sums.
- Independent count-weight bootstrap calculations reproduced all 32 contrast intervals and 18 absolute intervals. An independent calculation from the saved per-layer evidence sums reproduced all 4,200 weighted-normal scores/decisions; all 12 null-count cells were recounted.
- Bootstrap draw hash: `5ab115a4b5c632f81fd04e9cf02b5d4f67cf68dd60cf622759f99a04fb767c77`.

[Results](summary.json) · [Prompt-level metrics](prompt_metrics.json) · [Frozen manifest](manifest.json) · [Generation report](generation_report.json) · [Independent verification](verification.json) · [Execution and remote location](execution.json).

Run: [Modal execution](https://modal.com/apps/new-prc-watermark/main/ap-SZjiparhtpncbU7r7jZz1r). Remote raw batches and controls are on `prc-completion-only/self_bleu_depth/c1edcaca75fed307c01a40c85d51399678159b52aa6857099ac83bb233ddffd4`. Existing baseline artifacts are restored from the Stage A archive. Collect without dispatching another GPU job:

```sh
MODAL_PROFILE=new-prc-watermark python -m self_bleu.depth collect --setup outputs/self_bleu_depth/depth2_30_v1 --download
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 PYTHONPATH=. python -m self_bleu.depth analyze --setup outputs/self_bleu_depth/depth2_30_v1
```

The worker ran for **391.39 seconds**, estimated at **$0.50548** using the same frozen resource rate as prior stages. Including the new $0.50 allowance, the cumulative planning charge is **$7.71132**, within the initial $10 allocation and the overall $200 ceiling. These are planning estimates, not a settled Modal bill. Detection and Self-BLEU analysis ran locally; no new detector GPU replay was needed.
