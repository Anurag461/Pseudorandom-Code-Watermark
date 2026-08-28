# Controlled Qwen3-8B baseline comparison

Status: **complete and validated**, with a material quality-diversity tradeoff.
The experiment covers all 500 canonical prompts. It reused the validated first
50-prompt shard and generated only prompts 50–499 after explicit approval. No
27B, 50x5 diversity, or proxy-model experiment was launched.

## Frozen setting

Qwen3-8B-Base revision
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; exactly 1,024 continuation
tokens; temperature 1.0; top-p 1.0; reasoning off; batch size 50. Methods were
online PRC (`t=3`, `eta=0.05`), TextSeal (`alpha=0.1`), SynthID-Text (depth 10,
frequentist detector), and the pinned TextSeal Gumbel-Max comparison. Baseline
detectors used context length 3 and exact `(context, token)` deduplication.

## Detection

| Method | T=128 | T=256 | T=400 | T=512 | T=768 | T=1024 |
|---|---:|---:|---:|---:|---:|---:|
| Online PRC | 13.6% | 41.0% | 67.8% | 80.0% | 93.8% | 96.0% |
| TextSeal | 100% | 100% | 100% | 100% | 100% | 100% |
| SynthID-Text | 100% | 100% | 99.8% | 99.8% | 100% | 100% |
| Gumbel-Max | 100% | 100% | 100% | 100% | 100% | 100% |

At the nominal `p < 1e-3` rule, online PRC had zero false positives at every
prefix. TextSeal had one at T=128 and one at T=768; SynthID had one at T=128;
Gumbel-Max had one at T=128, one at T=256, and two at T=400. All other cells
had zero. These are observed counts, not proof of calibration: 500 nulls cannot
tightly validate a nominal 0.1% FPR. The calibration labels remain distinct:
PRC Hoeffding upper bound, TextSeal moment-matched Gamma approximation, SynthID
normal approximation, and Gumbel-Max exact Gamma test.

## Quality and diversity

Median full-continuation metrics were:

| Output | Base NLL | Perplexity | Repetition | Distinct-2 | Distinct-3 |
|---|---:|---:|---:|---:|---:|
| Null | 3.004 | 20.159 | 0.014 | 0.888 | 0.967 |
| Online PRC | 2.999 | 20.057 | 0.014 | 0.887 | 0.968 |
| SynthID-Text | 3.060 | 21.326 | 0.012 | 0.893 | 0.972 |
| TextSeal | 1.590 | 4.905 | 0.262 | 0.640 | 0.721 |
| Gumbel-Max | 0.890 | 2.436 | 0.633 | 0.334 | 0.361 |

TextSeal had repetition above 0.1 on 367/500 prompts and distinct-3 below 0.8
on 304/500. Gumbel-Max had those outcomes on 469/500 and 439/500 prompts.
Online PRC counts were 19/500 and 8/500; SynthID counts were 25/500 and 8/500.

The collapse is length-dependent. TextSeal median distinct-3 declines from
0.992 at T=128 to 0.937 at T=400, 0.900 at T=512, and 0.721 at T=1,024.
Gumbel declines from 0.992 to 0.830, 0.691, and 0.361. The unusually low
TextSeal/Gumbel NLL shows that selected tokens are individually high
probability under the base model; it does not negate sequence-level looping.

Consequently, TextSeal and Gumbel's stronger detection cannot be interpreted
as a quality-matched win over PRC. The scientifically supported result is a
joint detectability-quality-diversity tradeoff under the frozen setting.

## Validation and provenance

All 1,500 newly generated method outputs are exactly 1,024 tokens, and all
stored entropies/log-probabilities are finite. All 24,000 prompt-prefix rows
passed schema and uniqueness checks. All 18,000 baseline exact-prefix evidence
comparisons had zero delta. TextSeal official/common decisions agreed; the
maximum p-value difference was `3.706e-7`, within the predeclared tolerance.
SynthID's generation/reference check was exact. PRC and null generation
attempts were zero.

Shard 0 has an older whole-tree integration fingerprint because it predates
the resume/finalizer orchestration. Its raw and scored SHA-256 hashes were
validated before reuse. Generation/scoring algorithms, model code, dependency
image definition, settings, batch shape, hardware class, seed, prompts, and
keys did not change; this discrepancy is recorded rather than hidden.

## Runtime and cost

All workers used NVIDIA H100 80GB HBM3 GPUs. Mean 50-prompt method times were
44.68 seconds for TextSeal, 68.70 for SynthID, and 43.97 for Gumbel. Mean total
shard time was 165.87 seconds; the slowest shard took 169.09 seconds. Peak CUDA
reserved memory was 27,839,692,800 bytes (25.93 GiB).

Exact generation-and-scoring spend for all 500 prompts was **$2.65877380**:
$0.27034319 for the reused validation generation, $0.00604867 for its scoring,
$2.31950392 for the remaining generation, and $0.06287802 for remaining
scoring. This is below the $5 approved cap. Including earlier integration,
smoke, and diagnostics, cumulative controlled-baseline spend is **$4.22362341**.

The complete 24,000-row JSONL (74,941,242 bytes) and 4,000-row quality CSV are
under
`outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/`.
Raw and scored shards remain on the `prc-data` Modal volume under the same run
ID. The large prompt-level JSONL is intentionally not committed; its SHA-256 is
`7b02c3b6393b4b988a40b3b22456876a89571158898cf93c9c55f4c5cd66d0b1`.
