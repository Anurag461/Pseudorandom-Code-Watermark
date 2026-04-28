# Outlier prompt 23 — naive vs entropy fold

Worst-case watermarked prompt under naive folding from the n=128 run. Demonstrates how entropy weighting recovers signal that naive fold loses to deterministic tokens.

## Setup

- Prompt: "Produce an alphabetized encyclopedia of every named celestial body in our solar system: all eight planets, all five officially recognized dwarf planets, all major moons (at least the top 25 by size), …"
- n = 128, parity checks r = 91, weight t = 3, noise_rate = 0.05
- Tokens generated: 1024 (8 observations per slot)

## Statistic decomposition

A parity check is labeled "corrupted" here if at least one of its t=3 slots has a mean per-token entropy weight below 0.05 (i.e. that slot's 8 observations are mostly deterministic argmax tokens carrying no codeword signal).

| fold    | total   | corrupted checks (13/91) | clean checks (78/91) |
|---------|---------|--------------------------|----------------------|
| naive   | −0.239  | +0.057                   | −0.296               |
| entropy | +4.599  | +0.554                   | +4.044               |

- 5 of 128 slots are low-entropy (mean weight < 0.05).
- 13 of 91 parity checks touch at least one low-entropy slot.

## What entropy weighting actually does here

Two effects, both visible in the decomposition above:

1. **Suppress corrupted checks.** A check that touches a low-entropy slot has its product Π ≈ random ±|signal|, contributing noise. Entropy weighting drives that slot's posterior toward 0, collapsing Π → 0 and silencing the check. Naive: the 13 corrupted checks net out to +0.06 (random cancellation). Entropy: +0.55 (silenced ≈ zero, with a small residual sign-aligned bias).

2. **Clean up surviving checks at the observation level (the bigger effect).** A check labeled "clean" still has every slot averaging 8 observations, and *individual* observations within a slot can be deterministic — those drag the slot's naive posterior toward random ±1, weakening the product Π even when no whole slot is bad. Entropy weighting downweights each individual deterministic observation, so each slot's posterior is dominated by the high-entropy observations whose signs actually correlate with the codeword. This is where the +4.34-point swing on clean checks (−0.30 → +4.04) comes from.

Net: total stat goes from −0.24 (naive, sub-null) to +4.60 (entropy, ~1.1σ above null). The check-level "drop the corrupted ones" framing is only the smaller of the two effects — most of the recovery is at the observation level inside every slot.

## Demo data

`outlier_demo.json` carries the full per-token, per-slot, and per-parity-check arrays for this prompt:

- `per_token` (length 1024): `i`, `slot`, `token_id`, `token_str`, `p1`, `weight = H_2(p1)/log2`, `bit`, `low_entropy` (weight < 0.05).
- `per_slot` (length 128): `n_obs`, `mean_weight`, `naive_posterior`, `entropy_posterior`, `low_entropy` (slot-level).
- `per_check` (length 91): `slot_indices` (length 3), `Pi_naive`, `Pi_entropy`, `contrib_naive`, `contrib_entropy`, `corrupted`.

Visualization ideas for the interactive demo:

- **Token strip.** Render the generated text token-by-token, colored by `weight`. Tokens with `low_entropy=true` glow red — these are the steps the watermark cannot inject any signal into. Hover shows `p1`, `weight`, `bit`, and which slot the token landed in.
- **Slot grid (n=128).** 16×8 heatmap. Three layers togglable: mean weight, naive posterior, entropy posterior. Low-entropy slots are immediately visible as dark spots in the weight layer and as desaturated cells in the entropy posterior.
- **Parity-check panel.** Sortable list of all 91 checks. Each row shows the 3 slot indices (with low-entropy slots marked), `Pi_naive` vs `Pi_entropy`, and `contrib_naive` vs `contrib_entropy`. A "drop corrupted" toggle re-aggregates the statistic without the 13 corrupted checks; a slider can drop additional checks ranked by `|contrib_naive - contrib_entropy|` to show how the naive stat climbs as the user manually filters out the noisiest checks.
- **Live statistic readout.** As the user toggles checks on/off, recompute `sum(contrib)` for both folds and show the running stat alongside the FPR=1e-9 threshold (24.0). The pedagogical "aha" is that under naive fold you'd have to hand-prune dozens of checks to recover signal, whereas entropy weighting does it automatically and at the right granularity (per observation, not per check).
