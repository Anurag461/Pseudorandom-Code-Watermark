# Controlled Qwen3-8B baseline smoke report

Date: 2026-08-22 (runs crossed 2026-08-23 UTC)

## Outcome

The authorized five-prompt comparison passed. Online causal PRC, TextSeal,
SynthID-Text, and Gumbel-Max produced complete, finite scores for canonical
prompt rows `0..4` at exact continuation prefixes
`T={128,256,400,512,768,1024}`. TextSeal, SynthID, and Gumbel generated exactly
1,024 tokens per primary prompt. Online PRC and the shared nulls were read only
from validated caches; their generation-attempt counters remained zero.

The 500-prompt generation has not been launched. Fixed PRC was omitted from
this smoke with explicit approval because no compatible cached 8B,
`eta=0.05`, `T=1024` fixed artifact exists and regeneration was forbidden.

## Frozen implementation and provenance

- Model/tokenizer: `Qwen/Qwen3-8B-Base`, revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; tokenizer SHA-256
  `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
- TextSeal: `https://github.com/facebookresearch/textseal` at
  `c60d0d1da2e59f09a698438e218a07ee779b4616`, Apache-2.0.
- SynthID-Text: `https://github.com/google-deepmind/synthid-text` at
  `addb4a158143c7c6851a1308f78b89fceed59683`, Apache-2.0.
- Uploaded paper source SHA-256:
  `dc703a7a61ef219b55b70ae12f24839f56560bd38da985af801fcb1e270095ed`.
- Python 3.11 Modal image definition SHA-256:
  `36ef066d906bf9df05801790b9327b8f2a8854d516add87b3f36d47afcd40217`.
- Main generation image: `im-lqcA41yGVtqwIMu2GARgZZ`; final CPU scoring
  image: `im-4YmW1ZVhd2wu954UvbtcGQ`.
- Canonical prompt corpus SHA-256:
  `0bf0560438d9d4b7a85ebf8b7349d6d028aa02b11a381982742ee55b4430530c`.
- Raw generation artifact:
  `/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt`,
  semantic fingerprint
  `58fa13633cef1e592038cf110f7d3d05de00972145a9a09c983db7e5a87be221`.

The complete dependency resolution, PRC file hashes, model-shard sizes,
official source metadata, paper configuration, metric definitions, and
paper-versus-code discrepancies are in
`baseline_comparison/pinned_sources_manifest.json`.

## Detection results

Observed five-prompt TPR by exact prefix (these are smoke values, not final
estimates):

| Method | 128 | 256 | 400 | 512 | 768 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Online PRC | 0.00 | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 |
| TextSeal | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| SynthID-Text | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Gumbel-Max | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Every observed null FPR was `0/5` at every prefix. Five nulls provide no useful
calibration guarantee. Even the planned 500 nulls cannot tightly validate a
nominal 0.1% FPR; final results must report false-positive counts separately
and retain these non-equivalent labels:

- PRC: Hoeffding p-value upper bound.
- TextSeal: moment-matched Gamma approximation.
- SynthID-Text: normal approximation.
- Gumbel-Max: exact Gamma test.

No comparison should treat these p-values as identically calibrated, and PRC
`eta` must not be interpreted as TextSeal `alpha`.

## Reference, prefix, and seed validation

- All official CPU reference checks passed. Common TextSeal and the pinned
  detector agreed numerically; the maximum smoke absolute p-value difference
  was `3.340031283771694e-7`, arising from the released scalar-float32 versus
  common float64 reduction path. All nominal `p<1e-3` decisions were identical.
- SynthID generation-time score updates matched a second independent Google
  processor at every smoke token: indices equal and maximum absolute score
  difference `0.0`. Synthetic and smoke g-values matched the pinned Google
  reference exactly.
- Exact-prefix evidence sliced from the full continuation matched direct
  causal-prefix evidence with maximum absolute difference `0.0`.
- `(context_3, token)` deduplication matched TextSeal v2, including its released
  `k+1` start convention. Counts are preserved for every row.
- Equal-shape fixed-seed replay passed for TextSeal and SynthID. Both changed
  under the second seed at batch size 1 (prompt-0 token agreement `0.0205` and
  `0.0127`, respectively).
- Gumbel-Max was deterministic across seeds when prompt and batch shape were
  held fixed: both batch-1 runs had token hash
  `7325894598cca4da9db2d0818fd35103fb43d7d733607e65a958819817c6632f`.
  The batch-5 prompt-0 hash differed, establishing batch-shape sensitivity.
  A follow-up diagnostic found the released power form and exact log-space
  equivalent token-identical at both batch shapes, locating the sensitivity in
  batch-dependent logits/numerical execution rather than the argmax formula.
  This is not an RNG failure.

## Quality sanity check

All quality fields were finite and lengths were exact. Median prompt-level
metrics at `T=1024` were:

| Method | Base NLL | Perplexity | Repeated 4-gram fraction | distinct-2 | distinct-3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Online PRC | 3.196 | 24.43 | 0.0059 | 0.904 | 0.981 |
| TextSeal | 0.961 | 2.61 | 0.6405 | 0.323 | 0.350 |
| SynthID-Text | 3.168 | 23.76 | 0.0127 | 0.896 | 0.979 |
| Gumbel-Max | 0.794 | 2.21 | 0.5357 | 0.382 | 0.437 |

The numbers are internally valid, but TextSeal and Gumbel-Max show severe
repetition/diversity collapse on several prompts under the frozen Qwen3-8B,
temperature-1/top-p-1, 1,024-token setup. That is scientifically material and
must accompany their very strong detection; it is not a reason to change the
frozen settings silently. The paper itself notes deterministic Gumbel loops,
but its main comparison used Qwen3.5-27B, temperature 0.8, top-p 0.9, and 400
tokens, so its quality result is not directly transferable.

## Runtime, memory, cost, and full-run projection

Primary batch-5 generation timings were:

| Method | Five prompts | Seconds per prompt | Tokens/s (batch aggregate) |
| --- | ---: | ---: | ---: |
| TextSeal | 37.006 s | 7.401 s | 138.36 |
| SynthID-Text | 40.504 s | 8.101 s | 126.41 |
| Gumbel-Max | 35.554 s | 7.111 s | 144.01 |

The six primary/secondary generation loops totaled `219.834` seconds. The
main generation worker consumed `0.31757434` dollars of H100 billing, equivalent
to `280.350` billed H100 seconds at 4.078 dollars/hour. Measured peak CUDA
memory was `17,668,689,920` bytes allocated and `17,702,060,032` bytes reserved
on `NVIDIA H100 80GB HBM3`.

Exact final smoke/integration spend is `1.39371804` dollars, itemized
into GPU, CPU, and memory in
`outputs/controlled_baseline_smoke_cost_ledger.csv`. The main completed
generation-plus-first-CPU-resume path cost `0.37493568` dollars; the campaign
total also includes fail-closed integration diagnostics and the controlled
seed checks. It remained far below the 10-dollar hard cap.

Scaling the measured primary batch-5 loops to 500 prompts requires about
`11,306.4` aggregate GPU-seconds (`3.141` GPU-hours). Adding ten measured model
loads projects about `13.09` dollars of H100 cost. Allowing provider CPU,
memory, cache validation, CPU scoring, and operational variance gives a
conservative all-resource estimate of **14--18 dollars**. Ten 50-prompt shards
project **20--35 minutes** wall time, followed by CPU scoring/finalization.
This is now retained as the batch-5 upper-bound projection. The current plan
validates batch 50 on one standalone 50-prompt shard before replacing the cost
and wall-time estimate.

## Readiness decision

The full run is engineering-ready but remains approval-gated. No correctness
blocker remains: official references, schema, exact prefixes, caches, lengths,
fixed seeds, quality fields, and provenance passed. Approval should explicitly
acknowledge two scientific observations: severe TextSeal/Gumbel repetition on
this frozen setup and Gumbel's released-path batch-shape sensitivity. The
exact guarded commands are in `controlled_baseline_full_run_runbook.md`; do
not launch them without a new explicit approval.
