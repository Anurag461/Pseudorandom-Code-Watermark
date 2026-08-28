# Controlled 8B baseline full-run runbook

Status: **executed successfully on 2026-08-23**. This is the standalone record
of the exact frozen experiment, commands, validation gates, and recovery paths.
It does not authorize the separate diversity, 27B, or proxy-model experiments.

## Frozen experiment

- Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Canonical prompt rows `0..499`, unchanged ordering and IDs.
- Exactly 1,024 continuation tokens, temperature 1.0, top-p 1.0, reasoning off.
- Online PRC `t=3, eta=0.05` from cache only; TextSeal `alpha=0.1`; Google
  SynthID-Text depth 10 with the frequentist normal detector; TextSeal Gumbel
  comparison path.
- Context length 3 and TextSeal-v2 unique `(context, token)` deduplication.
- Prefixes `128,256,400,512,768,1024`; nominal decision `p < 1e-3`.
- Ten shards of 50 prompts, fixed batch size 50, `H100` requested. Each worker
  loaded Qwen once and made one generation call per new method.
- Run ID `qwen3-8b-batch50-validation-20260823-v1`.

## Source and configuration pins

The authoritative dependency/license/source record is
`baseline_comparison/pinned_sources_manifest.json`. TextSeal is pinned to
`c60d0d1da2e59f09a698438e218a07ee779b4616`; Google's SynthID-Text reference is
pinned to `addb4a158143c7c6851a1308f78b89fceed59683`; the isolated image
definition SHA-256 is
`36ef066d906bf9df05801790b9327b8f2a8854d516add87b3f36d47afcd40217`.

## Executed commands

The batch-50 validation generated prompt rows `0..49`:

```bash
modal run baseline_comparison/modal_app.py::app.batch50-validation \
  --approval-token APPROVE_50_PROMPT_BATCH50_VALIDATION \
  --run-id qwen3-8b-batch50-validation-20260823-v1
```

It was scored independently on Modal CPU. After the user explicitly approved
the remaining comparison, fail-closed resume code verified the raw and scored
shard-0 hashes and generated only prompt rows `50..499`:

```bash
modal run baseline_comparison/modal_app.py::app.remaining-run \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-batch50-validation-20260823-v1
```

Then the saved new shards were scored, again with no model load or generation:

```bash
modal run baseline_comparison/modal_app.py::app.remaining-score \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-batch50-validation-20260823-v1
```

The two production app URLs were:

- generation: `https://modal.com/apps/new-prc-watermark/main/ap-5bAllM2qADIA80FtqwdTpX`
- scoring: `https://modal.com/apps/new-prc-watermark/main/ap-e3ENaI1x4V0MVHvbBTPVqi`

The saved artifacts were downloaded for local streaming finalization:

```bash
modal volume get prc-data \
  controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/scored \
  /private/tmp/qwen3-8b-controlled-full-scored

modal volume get prc-data \
  controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/generated \
  /private/tmp/qwen3-8b-controlled-full-generated
```

```bash
PYTHONPATH=. python -m baseline_comparison.full_run finalize \
  --scored-dir /private/tmp/qwen3-8b-controlled-full-scored/scored \
  --output-dir outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1
```

The finalizer required exactly ten shards, 24,000 unique schema-valid rows,
and complete prompt `0..499` coverage for every method/sample type.

## Validation gates and outcome

- All 1,500 generated outputs contain exactly 1,024 continuation tokens.
- All generated entropy/log-probability and scored statistic/p-value/quality
  fields are finite.
- All 18,000 baseline exact-prefix evidence comparisons have zero delta.
- TextSeal official/common decisions all agree; maximum p-value difference is
  `3.7056737051122113e-7`, within the declared tolerance.
- SynthID generation/reference indices are exact, with zero score difference.
- Online PRC and null generation attempts are both zero.
- Shard 0's older whole-tree fingerprint is preserved as a disclosed
  orchestration-only difference; its exact raw/scored hashes were checked
  before reuse.

The compact report is `controlled_baseline_full_report.md`; the audit,
validation, detection summary, quality tables, runtime, billing, and manifests
are under the output directory named above. The complete 74.9 MB prompt-level
JSONL is intentionally excluded from git but remains locally and is
fingerprinted in the manifest.

## Billing reconciliation

The final provider billing was fetched with:

```bash
modal billing report --for today --resolution h --show-resources --json
```

All 500 prompts, including reused shard 0, cost **$2.65877380** for generation
and scoring. The new nine-shard generation was $2.31950392 and its CPU scoring
was $0.06287802. Mean 50-prompt shard time was 165.873 seconds; peak CUDA
reserved memory was 27,839,692,800 bytes (25.93 GiB). Exact per-resource rows
and Modal app URLs are in `controlled_baseline_full_cost_ledger.csv`.

## Interpretation constraints

- Report the four calibration labels separately: PRC Hoeffding upper bound,
  TextSeal moment-matched Gamma approximation, SynthID normal approximation,
  and Gumbel-Max exact Gamma test.
- Five hundred nulls cannot tightly validate a nominal 0.1% FPR.
- Interpret detection jointly with quality and diversity. TextSeal and Gumbel
  show material length-growing repetition under the frozen setting.
- Gumbel is seed-deterministic at fixed batch shape. Preserve batch size,
  prompt grouping/order, hardware provenance, key, and model revision.
- Retain the project-Qwen/Hugging-Face numerical portability discrepancy from
  `controlled_baseline_diagnostic_report.md`; it does not invalidate the
  internally controlled fixed-batch result.

## Recovery and no-overwrite behavior

Do not invoke generation again for this run ID. Existing shards are immutable,
and the workers fail closed instead of overwriting them. If compact local
outputs are lost, redownload `scored/` and rerun only the local finalizer. If a
scored shard is lost remotely, verify all raw shard hashes against
`controlled_baseline_full_provenance_manifest.json` before invoking a scoped
CPU-only rescore.

The 50-prompt x five-seed diversity experiment requires separate approval. Its
analysis utilities exist, but this run did not invoke any diversity-generation
entrypoint.
