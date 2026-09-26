# Watermark stealing, black-box detection and substitution robustness: PRC vs KGW-2.0, EXP and SynthID-Text

Branch `kth-attacks`. All numbers below come from the committed files in `outputs/attacks/`; confidence
intervals are 95% Wilson score intervals. Model: Qwen3-0.6B (Base for generation and stealing, the chat
model with thinking disabled for black-box detection). Temperature 1, full vocabulary.

## Summary

- **Watermark stealing / spoofing (Jovanović et al., ICML 2024).** With the official scoring rule and one
  deployed key per scheme, an attacker spoofs KGW-2.0 (100% of quality-filtered texts) and EXP without key
  offset (98%), and partially spoofs SynthID-Text (44% at 30k queries, still rising). PRC is never spoofed
  above its false-positive rate in any of 35 attack cells (17,500 texts): 15 empirical-threshold hits vs
  17.5 expected by chance, and 1 at PRC's proven threshold. The stolen tables recover the secret for KGW
  and EXP (74.7% green vs 25% chance; key values 0.84 vs 0.50) and nothing for PRC (partition-1 share
  0.48–0.51 vs 0.50).
- **Black-box detection (Gloaguen et al., ICLR 2025), one repetition.** The Red-Green test detects KGW-2.0
  and SynthID-Text (p ≈ 0), the Fixed-Sampling test detects EXP (252/1000 distinct outputs,
  p = 2e-124), and neither test detects online PRC (Red-Green p = 0.22 after Bonferroni; 1000/1000
  distinct outputs, same as the unwatermarked model). A PRC deployment that ties the sampling seed to the
  prompt is trivially detectable (1/100 distinct outputs), so every request needs a fresh seed.
- **Substitution robustness (Kuditipudi et al. attack).** Robustness depends strongly on length. At
  **400 tokens** PRC is far less robust than all three baselines: 64.6% detection at 5% substitutions and
  2.6% at 30%, while EXP and KGW-2.0 stay at 100% and SynthID-Text at ≥89%. At **4096 tokens** (fixed
  PRC n = T = 4096) PRC keeps **98.2% [96.6%, 99.1%] detection at 20% substitutions** and **94.6%
  [92.3%, 96.3%] at 30%** (MAP; 0/500 false positives), close to the baselines' 99–100% at 20% (200 texts
  each; the baselines were not run at 30%).

The paper can therefore claim resistance to stealing-based spoofing and to black-box detection. Robustness to
random token substitution holds for long texts (4096 tokens) but not for short ones (400 tokens).

## 1. Common setup

- Prompts: 50-token prefixes of C4 RealNewsLike (validation split for evaluation, the first 500 of
  `prompts.jsonl`; train split, disjoint documents, for attacker queries).
- PRC: fixed construction, n = T = 400, t = 3, η = 0.05 (stealing and substitution); online construction,
  η = 0.05 (black-box detection). Detection uses completion-only BF16 replay and the MAP-weighted
  Hoeffding detector at FPR 1e-3.
- KGW-2.0: γ = 0.25, δ = 2.0, previous-token hash (Kuditipudi et al.'s code, commit `80d4ec8`).
- EXP: Kuditipudi et al.'s code, key length 256. **Stealing and substitution use no random key offset**
  (the setting of their experiments); black-box detection uses the random offset they recommend.
- SynthID-Text: Hugging Face `transformers` implementation, MarkLLM/DITTO configuration (n-gram 5, 30 keys).

## 2. Watermark stealing and spoofing

Code: `stealing.py`. Attacker: Qwen3-0.6B-Base without the watermark, N watermarked query responses per
scheme plus N of its own unwatermarked responses (400 tokens each). Scoring follows the official
`SpoofedProcessor.get_boosts` (eth-sri/watermark-stealing `b8d207d`): clip c = 2, min count 2, logits
+= α·boost. Five context variants: previous 1–4 tokens (ctx1–ctx4) and position mod the scheme's period
(pos, an adaptive variant for position-keyed schemes). α ∈ {1, 2, 4.5, 8}. 500 spoofed texts per cell,
answering the 500 evaluation prompts.

**Detection threshold**: for every scheme, the 1e-3 quantile of its statistic on the same 5,000
unwatermarked calibration texts (PRC's statistic is its Hoeffding p-value bound). **Quality filter**:
perplexity under Qwen3-4B-Base at most the 95th percentile of the calibration texts (136). The paper's
GPT-4 judge was not run; very low perplexity (repetitive text) passes this filter.

### E2: how much of the secret the stolen table recovers (N = 10,000)

- kgw2 / ctx1: fraction of boosted pairs that are green = 0.747 (chance 0.25)
- exp / pos: mean key value xi[position, token] of boosted pairs = 0.842 (chance 0.5)
- prc / ctx1: fraction of boosted tokens in partition 1 (codeword is fresh per text) = 0.480 (chance 0.5)
- prc / pos: fraction of boosted tokens in partition 1 (codeword is fresh per text) = 0.508 (chance 0.5)

### E3: spoof rate at N = 10,000, best α per attack variant

| Scheme | Genuine TPR | ctx1 | ctx2 | ctx3 | ctx4 | pos |
|---|---|---|---|---|---|---|
| kgw2 | 100.0% | 500/500 (100.0%) [99.2%, 100.0%], α=4.5 | 499/500 (99.8%) [98.9%, 100.0%], α=8 | 335/500 (67.0%) [62.8%, 71.0%], α=8 | 14/500 (2.8%) [1.7%, 4.6%], α=8 | 12/500 (2.4%) [1.4%, 4.1%], α=8 |
| exp | 100.0% | 11/500 (2.2%) [1.2%, 3.9%], α=8 | 7/500 (1.4%) [0.7%, 2.9%], α=8 | 7/500 (1.4%) [0.7%, 2.9%], α=8 | 3/500 (0.6%) [0.2%, 1.7%], α=8 | 490/500 (98.0%) [96.4%, 98.9%], α=2 |
| synthid | 100.0% | 17/500 (3.4%) [2.1%, 5.4%], α=8 | 8/500 (1.6%) [0.8%, 3.1%], α=4.5 | 85/500 (17.0%) [14.0%, 20.5%], α=8 | 46/500 (9.2%) [7.0%, 12.1%], α=8 | 1/500 (0.2%) [0.0%, 1.1%], α=4.5 |
| prc | 91.0% | 1/500 (0.2%) [0.0%, 1.1%], α=1 | 2/500 (0.4%) [0.1%, 1.4%], α=8 | 1/500 (0.2%) [0.0%, 1.1%], α=1 | 1/500 (0.2%) [0.0%, 1.1%], α=4.5 | 1/500 (0.2%) [0.0%, 1.1%], α=4.5 |

### E4: spoof rate vs number of attacker queries

| Scheme / attack (α) | N=1,000 | N=3,000 | N=10,000 | N=30,000 |
|---|---|---|---|---|
| kgw2 / ctx1 (4.5) | 500/500 (100.0%) [99.2%, 100.0%] | 499/500 (99.8%) [98.9%, 100.0%] | 500/500 (100.0%) [99.2%, 100.0%] | 500/500 (100.0%) [99.2%, 100.0%] |
| exp / pos (2) | 473/500 (94.6%) [92.3%, 96.3%] | 492/500 (98.4%) [96.9%, 99.2%] | 490/500 (98.0%) [96.4%, 98.9%] | 492/500 (98.4%) [96.9%, 99.2%] |
| synthid / ctx3 (8) | 4/500 (0.8%) [0.3%, 2.0%] | 13/500 (2.6%) [1.5%, 4.4%] | 85/500 (17.0%) [14.0%, 20.5%] | 221/500 (44.2%) [39.9%, 48.6%] |
| synthid / ctx4 (8) | 1/500 (0.2%) [0.0%, 1.1%] | 12/500 (2.4%) [1.4%, 4.1%] | 46/500 (9.2%) [7.0%, 12.1%] | 123/500 (24.6%) [21.0%, 28.6%] |
| prc / ctx1 (8) | 0/500 (0.0%) [0.0%, 0.8%] | 0/500 (0.0%) [0.0%, 0.8%] | 0/500 (0.0%) [0.0%, 0.8%] | 0/500 (0.0%) [0.0%, 0.8%] |
| prc / ctx2 (8) | 0/500 (0.0%) [0.0%, 0.8%] | 0/500 (0.0%) [0.0%, 0.8%] | 2/500 (0.4%) [0.1%, 1.4%] | 0/500 (0.0%) [0.0%, 0.8%] |
| prc / ctx3 (8) | 0/500 (0.0%) [0.0%, 0.8%] | 4/500 (0.8%) [0.3%, 2.0%] | 0/500 (0.0%) [0.0%, 0.8%] | 1/500 (0.2%) [0.0%, 1.1%] |
| prc / ctx4 (8) | 0/500 (0.0%) [0.0%, 0.8%] | 2/500 (0.4%) [0.1%, 1.4%] | 0/500 (0.0%) [0.0%, 0.8%] | 0/500 (0.0%) [0.0%, 0.8%] |
| prc / pos (4.5) | 0/500 (0.0%) [0.0%, 0.8%] | 1/500 (0.2%) [0.0%, 1.1%] | 1/500 (0.2%) [0.0%, 1.1%] | 0/500 (0.0%) [0.0%, 0.8%] |

PRC aggregate over all 35 spoofing cells (17,500 texts): 15 detected at the empirical 1e-3 threshold (expected 17.5 false positives), 1 at the proven Hoeffding threshold. The PRC cells with 2–4 hits are within that noise (P(≥ 15 hits | chance) = 0.76).

### Caveats

- EXP is spoofable only because it is deployed without a random key offset; with the offset it would
  likely resist the position attack.
- SynthID-Text is still rising at 30k queries (the original paper used 30k); larger N would likely go higher.
- One key per scheme, one model size (0.6B), one prompt distribution. No LLM-judge quality check.

## 3. Black-box watermark detection (Gloaguen et al.)

Code: `bbd.py`, reimplemented from the paper and its released code (eth-sri/watermark-detection `d4ca66f`,
no licence, so only prompts are reused). Victim: Qwen3-0.6B chat, thinking disabled, T = 1.

- **Red-Green test**: 10 prefixes × 9 digits, context width H = 4 and 5 (SynthID requires H = 4 exactly),
  100 parsed answers per cell, word list apples/bananas/oranges/pears (the most balanced of the paper's
  lists in a pilot; top share 0.46), the paper's released statistic and 10,000 permutations; p-values
  Bonferroni-corrected over H. Parse rates were 77–86%.
- **Fixed-Sampling test**: "This is the story of", 1,000 responses of 100 tokens, rarefaction curve +
  Mann–Whitney U.
- The Cache-Augmented test was not run: none of the schemes has a cache.
- Online PRC draws a fresh secret document seed per response.

| Scheme | Red-Green p (Bonferroni, H = 4, 5) | Fixed-Sampling: distinct / 1000 | Fixed-Sampling p |
|---|---|---|---|
| Unwatermarked | 1.0 | 1000 | 0.94 |
| **PRC (online, η = 0.05)** | **0.22** | **1000** | **0.94** |
| KGW-2.0 | 0.0 | 1000 | 0.94 |
| SynthID-Text | 0.0 (detected at H = 4 only) | 1000 | 0.94 |
| EXP (random offset) | 1.0 | 252 | 2.2e-124 |

**Seed handling matters.** In the pilot, online PRC with the seed tied to the prompt gave **1 distinct
output in 100** (with a fresh seed: 100/100). A deployment must draw a fresh secret seed per request.

Caveat: this is **one repetition** per scheme (the paper reports medians over many). The PRC Red-Green
p-value at H = 5 was 0.11 (0.22 after correction), not significant, but repetitions would make the
null result firmer.

## 4. Substitution robustness (Kuditipudi et al. attack)

Code: `attacks.py`, `kth_baselines.py`, and `modal_run.py::redetect --attack`. Uniform random token
substitution at the same positions and with the same replacement tokens for every scheme; detection at
FPR 1e-3 (PRC: proven Hoeffding bound; baselines: the paper's empirical p-values against 4,483 human C4
continuations). 400-token texts, **500 watermarked + 500 unwatermarked texts per scheme and rate**.

Detection rate [95% CI]:

| Substitution rate | PRC (MAP) | EXP | KGW-2.0 | SynthID-Text |
|---|---|---|---|---|
| 0 | 86.0% [82.7%, 88.8%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] |
| 0.05 | 64.6% [60.3%, 68.7%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] |
| 0.1 | 41.6% [37.4%, 46.0%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 99.8% [98.9%, 100.0%] |
| 0.15 | 23.6% [20.1%, 27.5%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 99.8% [98.9%, 100.0%] |
| 0.2 | 13.2% [10.5%, 16.4%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 99.2% [98.0%, 99.7%] |
| 0.25 | 6.0% [4.2%, 8.4%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 98.2% [96.6%, 99.1%] |
| 0.3 | 2.6% [1.5%, 4.4%] | 100.0% [99.2%, 100.0%] | 100.0% [99.2%, 100.0%] | 89.2% [86.2%, 91.6%] |

False positives at 1e-3 (500 unwatermarked texts per cell): PRC 0/500; SynthID 3/500, 1/500, 1/500, 3/500, 6/500, 4/500, 2/500


SynthID's false-positive rate on unwatermarked model text exceeds the nominal 0.1% because its null
reference is human text.

### 4096-token texts

Code: `kth_long.py`. Fixed PRC n = T = 4096, η = 0.05 on **all 500 prompts** (500 watermarked + 500
unwatermarked texts), redetected with the same `modal_run.py::redetect --attack` pipeline (completion-only
BF16 replay, 0.6B detector) as `main`. Each baseline uses **the first 200 prompts only** (200 watermarked +
200 unwatermarked texts). Baselines use analytic p-values (human text is too short for a 4096-token null):
EXP a Gamma tail with Bonferroni over its 256 key shifts, KGW-2.0 a one-sided z-test, SynthID-Text its
mean-score z-test. Same substitution procedure as above. Detection at FPR 1e-3, [95% Wilson CI]:

| Substitution rate | PRC MAP (n = 500) | PRC entropy (n = 500) | EXP (n = 200) | KGW-2.0 (n = 200) | SynthID-Text (n = 200) |
|---|---|---|---|---|---|
| 0 | 100.0% [99.2%, 100.0%] | 99.6% [98.6%, 99.9%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] |
| 0.05 | 99.6% [98.6%, 99.9%] | 99.4% [98.3%, 99.8%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] |
| 0.1 | 98.8% [97.4%, 99.4%] | 98.8% [97.4%, 99.4%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] | 99.5% [97.2%, 99.9%] |
| 0.15 | 98.6% [97.1%, 99.3%] | 98.8% [97.4%, 99.4%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] | 99.5% [97.2%, 99.9%] |
| 0.2 | 98.2% [96.6%, 99.1%] | 97.2% [95.4%, 98.3%] | 100.0% [98.1%, 100.0%] | 100.0% [98.1%, 100.0%] | 99.0% [96.4%, 99.7%] |
| 0.3 | 94.6% [92.3%, 96.3%] | 94.0% [91.6%, 95.8%] | not run | not run | not run |

Rate 0.3 was run for PRC only, on the same frozen texts and pipeline. False positives: PRC 0/500 at every
rate (both detectors); EXP 0/200; KGW-2.0 2/200, 2/200, 1/200, 1/200,
1/200; SynthID-Text 0/200, 0/200, 1/200, 1/200, 0/200 (rates 0 to 0.2). Results:
`outputs/attacks/kth_long_results.csv`.

## 5. Reproducibility and cost

| Result file | Produced by |
|---|---|
| `outputs/attacks/stealing_spoof_results.csv` | `modal run stealing.py::summarize_attack` |
| `outputs/attacks/stealing_signal_recovery.jsonl` | `modal run stealing.py::recovery` |
| `outputs/attacks/bbd_results.csv`, `bbd_pilot.json`, `bbd_wordlists.json` | `bbd.py` |
| `outputs/attacks/kth_baseline_results.csv`, `kth_attack_results.csv` | `kth_baselines.py::summarize`, `modal_run.py::redetect --attack` |
| `outputs/attacks/kth_long_results.csv` | `modal run kth_long.py::summarize` (profile `anurag461`) |

Raw generations, scores and traces are in the Modal volume `prc-attacks` (workspaces `new-prc-watermark`
for stealing and the 400-token study, `anurag461` for black-box detection and the 4096-token study).
Total Modal spend for these experiments is about $166 of a $200 budget (provider billing: $49.18 on
Sep 23 and $61.24 on Sep 24 in `new-prc-watermark`; $15.10 on Sep 24 and $40.27 on Sep 25 in `anurag461`,
of which the 4096-token study was about $41).
