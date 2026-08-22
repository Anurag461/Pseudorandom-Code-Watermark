# Fixed versus online cache-only runbook

This runbook is approval-gated. The commands below perform five exact-prefix
audits from existing Modal caches. They must not generate text, start a model
worker, allocate a GPU, or download a model. Run them sequentially so the
dedicated summary CSV has a single writer.

## Frozen design

- Seven matched cells and their immutable source paths are declared in
  `fixed_vs_online_analysis_config.json`.
- Analysis seed: `20260822`; experiment seed: `12345`.
- 10,000 paired prompt bootstrap resamples per comparison.
- Exact two-sided McNemar tests, with Holm correction across all 21 tests.
- Smallest meaningful difference: 5 percentage points.
- Five new 8B audits only; the two 0.6B full audits are already local.
- Expected total provider cost: below $0.10. Hard stop: $0.25.

## Audit commands

Run one command at a time from the repository root. The `--cache-only` guard
checks that all 500 watermarked and 500 null records exist before the code can
construct a model worker. `--gpu` is intentionally omitted; the only remote
functions that may execute are CPU artifact/planning/detection functions.

```sh
modal run modal_online_run.py::main --num-prompts 500 --n 416 --t 3 --eta 0.05 --fpr 0.001 --experiment-seed 12345 --generation-model-size 8B --kv-cache-implementation static --null-kv-cache-implementation static --detection-shard-size 50 --detection-max-containers 10 --cache-only --csv-out outputs/fixed_vs_online_online_audits_summary.csv
```

```sh
modal run modal_online_run.py::main --num-prompts 500 --n 749 --t 3 --eta 0.05 --fpr 0.001 --experiment-seed 12345 --generation-model-size 8B --kv-cache-implementation static --null-kv-cache-implementation static --detection-shard-size 50 --detection-max-containers 10 --cache-only --csv-out outputs/fixed_vs_online_online_audits_summary.csv
```

```sh
modal run modal_online_run.py::main --num-prompts 500 --n 768 --t 3 --eta 0.10 --fpr 0.001 --experiment-seed 12345 --generation-model-size 8B --kv-cache-implementation static --null-kv-cache-implementation static --detection-shard-size 50 --detection-max-containers 10 --cache-only --csv-out outputs/fixed_vs_online_online_audits_summary.csv
```

```sh
modal run modal_online_run.py::main --num-prompts 500 --n 1382 --t 3 --eta 0.10 --fpr 0.001 --experiment-seed 12345 --generation-model-size 8B --kv-cache-implementation static --null-kv-cache-implementation static --detection-shard-size 50 --detection-max-containers 10 --cache-only --csv-out outputs/fixed_vs_online_online_audits_summary.csv
```

```sh
modal run modal_online_run.py::main --num-prompts 500 --n 1625 --t 3 --eta 0.10 --fpr 0.001 --experiment-seed 12345 --generation-model-size 8B --kv-cache-implementation static --null-kv-cache-implementation static --detection-shard-size 50 --detection-max-containers 10 --cache-only --csv-out outputs/fixed_vs_online_online_audits_summary.csv
```

## Fail-closed checks after every audit

The log must contain all of the following before accepting the result:

1. `wm_missing=0` and `null_missing=0` in the generation plan.
2. `cache-only guard passed; GPU generation is disabled`.
3. `all generation records cached`.
4. A local JSON path matching the path frozen in the analysis config.
5. 500 watermarked and 500 null decisions for MAP, entropy, and naive.

Abort the sequence immediately if any cache is missing. Do not rerun without
`--cache-only`, and do not launch a generation command to repair a cache.

After each accepted audit, record its Modal run URL and provider cost in
`outputs/fixed_vs_online_cache_only_cost_ledger.csv`; change that row's status
to `complete`. Keep generated-token and GPU fields at zero only when the run
log confirms the cache-only guard. Reconcile cumulative provider cost before
starting the next audit. Do not start another run once the cumulative cost
reaches $0.25.

## Paired analysis

Do not run this while the user's other heavy local process is active. Once all
five audits and ledger rows are complete, run:

```sh
python fixed_vs_online_analysis.py
```

The analysis uses only Python's standard library and the local JSON/CSV files.
It performs no network access, generation, model loading, or GPU work. It
refuses to write results unless all source, prompt-coverage, detector-field,
configuration, cache-only, and cost checks pass. Its atomic outputs are:

- `outputs/fixed_vs_online_prompt_level.jsonl` (3,500 prompt-paired rows);
- `outputs/fixed_vs_online_paired_summary.csv` (21 comparisons);
- `fixed_vs_online_analysis.md`;
- `outputs/fixed_vs_online_validation.json`.

The validation record includes SHA-256 fingerprints for every source and
generated artifact, prompt coverage, cache provenance, analysis seed, family
size, and reconciled total cost.
