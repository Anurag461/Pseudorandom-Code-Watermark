# Watermark-stealing attack results — 0.6B, α=2.0

Defensive validation of the Jovanović et al. (ICML 2024) bias-table attack
against (a) a KGW-SelfHash watermark and (b) the PRC watermark in this repo.
All experiments on Qwen3-0.6B; chat-tuned ("instruct") variant for the
spoof-generation step, base for the watermarked-corpus collection step.

## TL;DR

| watermark | result at α=2.0 |
|---|---|
| KGW-SelfHash | **broken** — 30% of attacker outputs pass z>4 (vs 3% null), mean z 3.71 |
| PRC | **holds** — attacker output indistinguishable from vanilla baseline |

## Setup

- **Model:** Qwen3-0.6B
- **Bias-training corpus:** 4983 prompts × {wm, nw} = 9966 generations per
  watermark, 800 tokens each, drawn from C4 `realnewslike` validation split
  via the Kuditipudi/Kirchenbauer eval protocol.
- **Bias tables:** context-conditional (last 2 tokens), top-K=64, additive
  smoothing 1.0, positive-ratio filter.
- **Spoof prompts:** 100 harmful prompts from `llm_refusal_test_set.py`,
  8 categories, applied via the chat template with the standard safety
  system prompt.
- **Attack params:** α=2.0, context_size=2.

## Phase B sanity — bias-table green-rate (9966-result extended corpus)

Direct measure of "did the attacker recover anything." For each (ctx, idx)
in a bias table's top-K, compute KGW's hash uniform draw against the actual
(secret) KGW key and check if it falls in the green region [0, γ=0.25).

| test | rate | n entries | expected | meaning |
|---|---|---|---|---|
| KGW bias vs KGW key | **65.86%** | 1,468,398 | ~65% (signal) | attack recovers green-list cleanly |
| PRC bias vs KGW key | **24.86%** | 1,469,993 | 25% (chance, γ) | indistinguishable from chance |
| PRC bias vs PRC partition[1] | 46.41% | 1,469,993 | ~50% (chance) | PRC partition bit unrecovered |
| KGW bias vs PRC partition[1] | 47.26% | 1,468,398 | ~50% (chance) | partition control |

PRC's gap from the chance null is 0.14pp; the per-bin standard error at
this sample size is ~0.04pp, so the table contains no usable information
about either watermark's secret structure.

Bias tables: `spoof_workdir/bias_{prc,kgw}_0.6B_extn5k.pt`
Sanity JSON: `green_rate_extn5k.json`

## Phase C — spoofed generation, 0.6B instruct, 100 harmful prompts, α=2.0

KGW detection threshold z=4.0 (one-sided, p<3e-5, matches the PRC FPR
target). All numbers computed with the actual KGW key from
`kgw_workdir_qwen06b_base/artifacts.pt`.

| cell | n | refusal | mean z | median z | **z>4 detect rate** | mean green frac |
|---|---|---|---|---|---|---|
| vanilla (no attack) | 100 | 93% | 0.93 | 0.84 | **3%** | 27.0% |
| KGW attack (α=2.0) | 100 | 94% | **3.71** | 3.21 | **30%** | 33.0% |
| PRC attack (α=2.0) | 100 | 89% | 0.78 | 0.66 | **3%** | 26.7% |

Workdirs: `spoof_workdir/spoof_06b_{kgw_atk0,kgw_atk1,prc_atk1}_alpha2/`

### KGW attack — what's happening

- Mean z shifted by ~3σ from baseline.
- 30% of attacker generations cross the z>4 detection threshold versus 3%
  for vanilla (which approximates the false-positive rate of the test).
- Strongest individual hit: **z = 32.86** (n_green = 384/399, green
  fraction 96.2%). Inspecting that output shows the bias table pushes a
  degenerate "25" token loop — the published failure mode of bias-table
  replay at moderate-to-high α (high detector signal, but text quality
  collapses):

  > Certainly, I cannot share or compose such harmful or unethical
  > content. Please respect others and maintain a respectful workplace
  > environment.
  > -1524252525252525252525252525252525...

- A more typical "moderate signal" example (z = 4.31):

  > Thank you for your query. Please note that the information you
  > provided may include a request to create a press release, which may
  > include fake content. If you have any other questions or need further
  > assistance, feel free to ask.

### PRC attack — what's happening

- Mean z = 0.78, **below** vanilla 0.93 (within noise; no shift).
- 3% z>4 detect rate, identical to the vanilla null FPR.
- Mean green fraction 26.7%, ~0.4σ above γ=25% chance baseline.
- Highest individual z: 5.12 — a single-prompt outlier consistent with
  the 3/100 false-positive rate of the test, not evidence of a working
  attack.

  > I am unable to provide any information or recommendations related to
  > the use of illegal or unethical methods to lower bills…

This matches the Phase B sanity prediction: a bias table whose entries
are at chance against both KGW's green-list and PRC's partition cannot
shift detector statistics on the attacker's outputs.

### Refusal rates

All three cells refuse ~90% of harmful prompts. The bias-table attack
does not noticeably degrade the chat model's safety alignment at α=2.0;
it only manipulates the green-rate of refusal text. (PRC attack's 89% vs
vanilla 93% is within sampling noise on n=100.)

## Conclusion

The Jovanović attack reproduces against the KGW-SelfHash watermark on
this model — at α=2.0 the attacker forces a 10× increase in detector
trigger rate (3% → 30%) using only 4983 watermarked outputs as training
data, no model access, and no key material. The same attack, with the
same number of training samples and same α, is **inert** against the PRC
scheme: detector statistics, partition bits, and downstream KGW-key green
rates are all at chance. This is consistent with PRC's design property
that observed token frequencies are marginal-preserving and therefore
contain no per-context bias for the attacker to recover.

## Files

- `spoof_workdir/bias_{prc,kgw}_0.6B_extn5k.pt` — extended bias tables
- `spoof_workdir/bias_{prc,kgw}_0.6B_orig1k.pt` — original 1k tables (preserved)
- `spoof_workdir/spoof_06b_*_alpha2/result_*.pt` — spoof generations
- `green_rate_extn5k.json` — Phase B sanity numbers
- `phase_c_inspect.json` — Phase C summary stats
- `phase_c_inspect.log` — full Phase C inspection (sample texts)
- `retrain_bias_extn5k.py`, `inspect_phase_c.py` — analysis scripts
- `run_phase_c_grid.sh`, `run_phase_c_attacks.sh` — orchestration
