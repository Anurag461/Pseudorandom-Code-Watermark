# Saved-token numerical diagnostic — September 23, 2026

**Both predeclared controls passed.** On the saved failing null completion,
static and concatenating caches produced bitwise-identical logits at all four
tested prefixes, in both BF16 and FP32. FP32 cached/uncached differences were
well inside the declared TV≤1e-4 and max-logit-error≤.002 limits.

| Prefix length | Static/concat exact, both precisions | BF16 cached/uncached TV | FP32 cached/uncached TV |
| --- | --- | --- | --- |
| 1 | Yes | 0 | 0 |
| 4 | Yes | 0.02201158 (2.2012%) | 0.0000009359 (0.0000936%) |
| 8 | Yes | 0.04039494 (4.0395%) | 0.0000114719 (0.0011472%) |
| 16 | Yes | 0.02335232 (2.3352%) | 0.0000105681 (0.0010568%) |

The largest FP32 logit difference was 0.00011635. The original BF16 discrepancy
at length 4 reproduced exactly; longer prefixes show that it was not merely a
one-off near the 2% cutoff. Cross-precision comparisons are retained in
`numerical_diagnostic.json`, including 4.48% BF16/FP32 TV at the single-token prefix.

These observations support **precision-related differences between incremental
and full-prefix arithmetic**, rather than a static-cache implementation defect,
on the tested completion. They do not identify an individual kernel or establish
a bound across prompts, long contexts or batch sizes. PyTorch documents that
mathematically equivalent batched/sliced computations need not be bitwise
identical; that general explanation is consistent with this result, rather than
proof of its detailed cause. See the official
[numerical accuracy notes](https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html#batched-computations-or-slice-computations).
The experiment itself ran pinned PyTorch 2.4.0; the linked current documentation
is background, not a claim that it ran under PyTorch 2.14.

## Scope and cost

The user approved this one diagnostic with “go ahead” after the $0.25–$0.75 quote.
It ran from pushed commit `eaaccdb3a37d151fbe864dde8d6af4dd939d88ed` in app
`ap-2eJCrKhO7F3RXNlvGpKq7S`, now stopped with zero tasks.

One H100, four CPU cores, 64 GiB host RAM, batch 1; prefix lengths 1/4/8/16 from
`null_g00_p00_t1.0`. The checkpoint and saved trace were verified. It performed
122 teacher-forced token positions across BF16 and FP32, with TF32 disabled for
the FP32 reference. **No new text was generated.** The production model precision
and detector definitions were not changed.

GPU function time: **102.73 seconds**, or **0.02854 GPU-hours**. Runtime-based
resource estimate: **$0.1327**, excluding unmeasured startup/build/teardown/storage.
Provider billing for this diagnostic has not posted. The earlier sanity attempt
has now posted at **$0.14605021** including its reported H100/CPU/memory charges.
Known first-attempt cost plus this runtime estimate is approximately **$0.279**;
this is a mixture of billed and estimated costs, not a final combined bill.
See `billing_reconciliation.json` for the actual returned provider rows.

## What remains

The original 2% BF16 smoke check is **still recorded as failed**, with its
tolerance unchanged. The remaining T=1.8 smoke and the full experiment have
not run. This diagnostic does not automatically authorize or unblock production.

Recommendation for review: keep the requested BF16 generation/replay stack,
use direct static/concat equality and an FP32 reference as the cache-correctness
controls, and retain BF16 cached/uncached drift as an explicit numerical diagnostic.
This would be a documented change to validation policy, not a claim that the old
2% check passed or that numerical differences cannot affect posterior scores.
Complete the remaining short smoke before proposing production, with a new
workload/cost quote and approval. No additional paid run is prepared as approved.

Raw result and provenance: `numerical_diagnostic.json`, `run_ledger.json`,
`approval.json`, `quote.json`. Compact metrics: `prefix_comparison.csv`, `report.json`.
