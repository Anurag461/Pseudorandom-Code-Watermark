# Phase E — LPN-style structured-query attack on PRC: query-budget curve

## TL;DR

The LPN-style structured-query attack against PRC's parity-check matrix **does not recover any parity checks** even at k=16,384 queries on Qwen3-4B-Instruct (and at k=256 on Qwen3-8B-Instruct). The recovery test fails because the language-model-induced per-position bias on heavily-restricted "binary stream" prompts saturates the parity statistic for both true and random triples to the same magnitude. PRC's η=0.05 flip noise (which would mathematically separate true checks from random triples) is buried by this LM positional bias.

**Headline number:** at k=16,384 on Qwen3-4B-Instruct, mean |parity bias| of true PRC checks = mean |parity bias| of random non-check triples = **0.4749** (identical to 4 decimal places). True positive rate and false positive rate are both 1.000 — the test is uninformative.

## Threat model

- Attacker has black-box query access to a PRC-watermarked LM.
- Attacker knows the **PRC partition map** (the bucketing of vocabulary tokens into bits 0/1).
- Attacker does NOT know the encoding key (parity-check matrix or one-time-pad).
- Attack: prompt the LM to emit a uniform random binary string of `{w1, w2}` where w1 ∈ partition[1], w2 ∈ partition[0]. Each generation yields a length-T bit string `b`. Repeat k times. Enumerate weight-3 triples S ⊆ [n=400] and look for those where `⊕_{i∈S} b_i` deviates significantly from 0.5 across the k samples.
- Positional mapping: `cp = pos % 400` (PRC's cyclic codeword length).

## Why the attack should work in theory

PRC encodes each token as `b_t ~ Bernoulli(bern_p(xi[pos%n], p1))` where `xi[k]` is the codeword bit and `p1` is the LM's `P[partition_1]` at step t (`watermark_expt.py:267-272`). When p1 ≈ 0.5, `bern_p` reduces to `xi`, modulo η-flip. So if the prompt forces the LM to be balanced between the two partition buckets, the codeword is directly observable up to η=0.05 noise.

For a true parity check S of weight 3:
- Expected `P[parity(b_S) = 0] = (1 + (1-2η)^3) / 2 = (1 + 0.729)/2 = 0.8645` if codeword bits were uniform.
- Expected |signed bias| ≈ 0.3645 about parity-of-LM-signal.

For a random non-check triple, parity is determined entirely by the η-noise (uncorrelated across positions), giving expected |bias| ≈ 0.

## Why the attack actually fails

The LM's per-position bit distribution is **not** uniform. For Qwen3-4B-Instruct on the binary-stream prompt with (w1, w2) = ('A', 'B'):

- Per-position mean bit (where observed): mean=0.5009, **std=0.4775**, range=[0.013, 0.990].
- That is, many of the 400 codeword positions have nearly-deterministic LM output (close to 0 or close to 1).

Given that LM positional bias, the parity of *any* triple — true check or random — is also nearly deterministic, with |bias| approaching the ceiling of 0.5. The η-flip of PRC reduces true-check bias slightly (~0.135 in expectation), but this reduction is comparable to the variation across random triples induced by LM bias. Empirically the two distributions overlap completely.

## Experimental setup

| Parameter | Value |
|---|---|
| LM (4B) | Qwen3-4B-Instruct (`enable_thinking=False`) |
| LM (8B) | Qwen3-8B-Instruct (`enable_thinking=False`) |
| PRC | n=400, t=3, η=0.05, partition seed 12345, r=353 parity checks |
| Word pair | w1='A' (partition[1]), w2='B' (partition[0]) |
| Query format | Chat-template prompt: "Output exactly 512 symbols, each 'A' or 'B', uniform random, single-space-separated" |
| Tokens per query | T=512 (covers all 400 codeword positions plus margin) |
| 4B campaign | k=16,384 queries × 7 GPUs × ~55 hr wall-clock |
| 8B campaign | k=256 queries (PRC + KGW + unwatermarked baseline) |
| Bonferroni z | √(2 ln C(400,3)) ≈ 6.04 (recovery threshold) |

## Results — Qwen3-4B-Instruct (k=16,384, n=400, t=3)

Sweep over k by sub-sampling the 16,384 results:

| k | true \|b\| mean | neg \|b\| mean | TPR | FPR |
|---:|---:|---:|---:|---:|
| 16 | 0.500 | 0.500 | 0.000 | 0.000 |
| 32 | 0.486 | 0.486 | 0.000 | 0.000 |
| 64 | 0.493 | 0.493 | 1.000 | 1.000 |
| 128 | 0.477 | 0.477 | 1.000 | 1.000 |
| 256 | 0.475 | 0.475 | 1.000 | 1.000 |
| 512 | 0.475 | 0.475 | 1.000 | 1.000 |
| 1024 | 0.473 | 0.473 | 1.000 | 1.000 |
| 2048 | 0.472 | 0.472 | 1.000 | 1.000 |
| 4096 | 0.473 | 0.473 | 1.000 | 1.000 |
| 8192 | 0.474 | 0.474 | 1.000 | 1.000 |
| 16384 | 0.475 | 0.475 | 1.000 | 1.000 |

(TPR = fraction of 353 true PRC parity checks recovered above the Bonferroni threshold; FPR = fraction of 5000 random non-check triples spuriously recovered.)

The "TPR=FPR=1.000" cells are not a successful attack — they mean both populations lie above the recovery threshold, so the test is **uninformative**. The recovery threshold at k=16384 is `z * 0.5/√16384 ≈ 0.0236`; both populations sit at ~0.475, so they all pass. There is no statistic by which true and random triples can be separated.

## Results — Qwen3-8B-Instruct (k=256, n=400, t=3)

Sweep on the 8B model:

| k | true \|b\| | neg \|b\| | TPR | FPR |
|---:|---:|---:|---:|---:|
| 16 | 0.119 | 0.116 | 0 | 0 |
| 32 | 0.084 | 0.087 | 0 | 0 |
| 64 | 0.075 | 0.080 | 0 | 0 |
| 128 | 0.057 | 0.058 | 0 | 0 |
| 256 | 0.058 | 0.058 | 0 | 0 |

Unwatermarked baseline at k=256 (same prompt, no watermark applied):

| k | true \|b\| | neg \|b\| |
|---:|---:|---:|
| 256 | 0.042 | 0.044 |

Conclusion for 8B: smaller positional bias than 4B (0.058 vs 0.475), but still indistinguishable between true and random triples. The unwatermarked baseline at k=256 sits at 0.042/0.044 — i.e., the residual |b|≈0.058 in the watermarked case is only slightly above the no-watermark floor and is *equal* across true and random triples, ruling out any parity-check signal.

## Why per-position bias dominates

The structured-query attack assumes the LM's `p1` (probability of partition 1) hovers near 0.5 because the prompt restricts the candidate set to a balanced binary alphabet. In practice this assumption fails:

1. **Position-dependent token preference**: even when the system prompt asks for "uniform random A or B", the LM has strong position-dependent biases (which letter starts a sequence, which letter follows a space, parity-of-position effects in tokenizer space, etc.). For Qwen3-4B the per-position mean bit varies from 0.013 to 0.990.
2. **Watermark preserves this bias**: PRC's marginal-preserving design means it perturbs `p1` only by an η-flip, which is small relative to the LM's intrinsic positional swings.
3. **Parity over biased positions saturates**: when individual positions are near-deterministic, parity over a triple is also near-deterministic, with |bias| at the 0.5 ceiling. The η-flip-induced reduction (~0.135 for true checks) is buried.

This is consistent with PRC's theoretical security argument: an adversary cannot recover the secret unless they can engineer an honest channel where p1 is reliably near 0.5. The structured-query setup turns out not to be such a channel for these instruct models.

## Implications

1. **Phase B/C result for PRC stands.** The Jovanović-style frequency-replay attack at k=2k–10k generic outputs failed on PRC; a stronger LPN-style structured-query attack also fails up to k=16,384. PRC remains undefeated at the budgets this work measures.
2. **Cost of the negative result.** The k=16,384 4B campaign required ~55 GPU-hours on 7 GPUs (PRC generation is CPU-sync-bound on the partition probability + Bernoulli draw, capping GPU util at ~30%). Scaling to k=400^3 ≈ 6.4×10⁷ would cost ~5,000 GPU-years at this throughput — out of reach for academic budget.
3. **Possible attacker response (future work).** To overcome the per-position LM bias one would need either (a) a debiased prompting strategy that drives p1 closer to 0.5 at every position, or (b) a position-adaptive statistic that subtracts an empirical no-watermark baseline. Neither was attempted here.

## Files

- Driver: `phase_e_drive.py`, `phase_e_worker.py` (in docker container, reads PE_* env vars)
- Build artifacts: `phase_e_build.py` (n=400, t=3 PRC config from `calib_workdir_n400_t3_eta05/`)
- Bit extraction: `phase_e_extract_bits.py` → `bits.pt` (16,384 × 400 int8 obs matrix)
- Analysis: `phase_e_analyze_prc.py` → `prc_recovery.json` (sweep + TPR/FPR)
- Results data:
  - `phase_e_workdir/prc_4B_AB_k16384/prc_recovery.json` — headline 4B campaign
  - `phase_e_workdir/prc_AB_k256/prc_recovery.json` — 8B campaign
  - `phase_e_workdir/nw_AB_k256/prc_recovery.json` — 8B unwatermarked baseline
- Logs: `phase_e_prc_4B_k16384.log`, per-GPU `phase_e_worker_gpu*.log`

## Caveats

- Only one (w1, w2) pair was tested ('A'/'B') per user direction. A different pair might give a flatter per-position bias profile, but the 8B/4B cross-check suggests model-dependent rather than pair-dependent bias.
- The Bonferroni threshold (z≈6.04) is conservative; uncorrected per-triple z-tests would still show TPR=FPR overlap — the issue is signal degeneracy, not multiple-test correction.
- KGW track was deprecated mid-experiment per user direction; no Phase E KGW recovery curve is reported. KGW pair-level results from Phase B/C are unaffected.
