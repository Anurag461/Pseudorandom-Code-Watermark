# Controlled 8B baseline handoff

Status: **500-prompt controlled comparison complete and validated** on branch
`controlled-baseline-comparison`. The approved run did not launch the 50x5
diversity experiment, 27B replication, or proxy-model experiments.

## Frozen comparison

- Run ID: `qwen3-8b-batch50-validation-20260823-v1`.
- Qwen3-8B-Base revision:
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Canonical prompts `0..499`; exactly 1,024 continuation tokens; temperature
  1.0; top-p 1.0; reasoning off; production batch size 50.
- Online PRC `t=3, eta=0.05` reused cache only; TextSeal `alpha=0.1`;
  SynthID-Text depth 10 with the frequentist detector; pinned TextSeal
  Gumbel-Max comparison path.
- Context length 3, exact TextSeal-v2 `(context, token)` deduplication, prefixes
  `128,256,400,512,768,1024`, and nominal `p < 1e-3` decisions.

## Results and artifacts

The compact scientific report is `controlled_baseline_full_report.md`. The
authoritative result directory is:

`outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/`

It contains the 24-row detection summary, 4,000-row prompt-quality CSV,
quality summaries, exact cost ledger, runtime record, audit, validation, and
artifact/provenance manifests. The complete 24,000-row prompt-level JSONL is
also present locally but intentionally not committed because it is 74.9 MB;
its SHA-256 is
`7b02c3b6393b4b988a40b3b22456876a89571158898cf93c9c55f4c5cd66d0b1`.
Raw and scored shards remain on the Modal `prc-data` volume under the run ID.

All 1,500 generated continuations are exactly 1,024 tokens. All 24,000 rows
pass schema/coverage validation, all 18,000 baseline exact-prefix evidence
checks have zero delta, TextSeal official/common decisions agree, SynthID's
official generation/reference check is exact, and PRC/null generation attempts
are zero.

At prefixes `128,256,400,512,768,1024`, TPR is:

| Method | 128 | 256 | 400 | 512 | 768 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| Online PRC | .136 | .410 | .678 | .800 | .938 | .960 |
| TextSeal | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SynthID-Text | 1.000 | 1.000 | .998 | .998 | 1.000 | 1.000 |
| Gumbel-Max | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Observed false positives among 500 shared nulls are recorded per prompt and
prefix in the audit. Five hundred nulls cannot tightly validate nominal 0.1%
FPR, and the four analytic/frequentist p-value calibrations are not equivalent.

At 1,024 tokens, median distinct-3/repetition is 0.968/0.014 for PRC,
0.972/0.012 for SynthID, 0.721/0.262 for TextSeal, and 0.361/0.633 for Gumbel,
versus 0.967/0.014 for null. Treat this as a joint
detection-quality-diversity result, not a quality-matched detector ranking.

## Runtime and billing

All ten workers used H100 80GB GPUs. Mean 50-prompt shard time was 165.873
seconds, and peak CUDA reserved memory was 27,839,692,800 bytes (25.93 GiB).
Exact full comparison generation-and-scoring spend was **$2.65877380**, below
the approved $5 cap. Cumulative controlled-baseline integration, smoke,
diagnostic, and full-run spend is **$4.22362341**.

Generation apps:

- shard 0: `https://modal.com/apps/new-prc-watermark/main/ap-INjm5299E4tI0jNgiqOx4U`
- shards 1--9: `https://modal.com/apps/new-prc-watermark/main/ap-5bAllM2qADIA80FtqwdTpX`

Scoring apps:

- shard 0: `https://modal.com/apps/new-prc-watermark/main/ap-TiyaTghkwGCV6xEcLkk8SU`
- shards 1--9: `https://modal.com/apps/new-prc-watermark/main/ap-e3ENaI1x4V0MVHvbBTPVqi`

## Reproduction and next approval gate

The exact executed resume, scoring, download, and finalization commands are in
`controlled_baseline_full_run_runbook.md`. Do not rerun production against this
run ID: workers fail closed rather than overwrite saved shards.

No further command is required for the controlled 500-prompt result. A new,
explicit approval is required before launching the 50-prompt x five-seed
diversity experiment, any 27B replication, or any proxy-model comparison.
