# Controlled 8B baseline full-run runbook

This runbook is standalone. It describes the later 500-prompt comparison; it
does not authorize it. The code fails closed unless the exact approval token is
provided.

## Frozen experiment

- Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Canonical prompt rows `0..499`, unchanged ordering and IDs.
- Exactly 1,024 continuation tokens, temperature 1.0, top-p 1.0, reasoning off.
- Online PRC `t=3, eta=0.05` from cache only; TextSeal `alpha=0.1`; Google
  SynthID-Text depth 10 with the frequentist normal detector; TextSeal Gumbel
  comparison path.
- Context length 3 and TextSeal-v2 unique `(context, token)` deduplication.
- Prefixes `128,256,400,512,768,1024`; nominal decision `p<1e-3`.
- Ten generation shards of 50 prompts, batch size 5, at most ten `H100`
  workers. Generated shards are committed before CPU scoring.

## Approval gate

Before generation, report the smoke result and measured estimate (14--18
dollars; 20--35 minutes with ten GPUs) and obtain explicit user approval for
the 500-prompt run. Only then may the literal token below be passed:

`APPROVE_500_PROMPT_CONTROLLED_BASELINE`

Without it, `app.full-run` raises `PermissionError` before preflight or GPU
launch. Do not weaken or bypass this guard.

## Exact commands after approval

Run from the repository root. Choose a new immutable run ID; the workers refuse
to overwrite an existing shard.

```bash
modal run baseline_comparison/modal_app.py::app.full-run \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-controlled-20260823
```

This command runs CPU reference checks and validates all 500 online-PRC/null
cache pairs before launching any GPU. It then maps ten 50-prompt shards across
up to ten H100s, loads Qwen once per shard, generates each of the three new
methods in batches of five, and commits raw shards under:

`/data/controlled_baseline_full/qwen3-8b-controlled-20260823/generated/`

After all ten generation shards pass, score them on Modal CPU:

```bash
modal run baseline_comparison/modal_app.py::app.full-score \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-controlled-20260823
```

This produces ten shared-schema JSONL shards (2,400 rows each; 24,000 total)
and validation records without loading Qwen or regenerating PRC/null text.

Download only the compact scored artifacts to a new local directory:

```bash
modal volume get prc-data \
  controlled_baseline_full/qwen3-8b-controlled-20260823/scored \
  /private/tmp/qwen3-8b-controlled-20260823-scored
```

Stream-validate, merge, and summarize them locally:

```bash
python -m baseline_comparison.full_run finalize \
  --scored-dir /private/tmp/qwen3-8b-controlled-20260823-scored \
  --output-dir outputs/controlled_baseline_full/qwen3-8b-controlled-20260823
```

The finalizer requires exactly ten shards, 24,000 unique schema-valid rows,
and complete `0..499` coverage for every method/sample type. It writes the
prompt-level JSONL, 24-row prefix summary, quality CSV, validation JSON, and
artifact fingerprint manifest.

## Billing reconciliation

Fetch provider billing after the final hourly interval is complete:

```bash
modal billing report --for today --resolution h --show-resources --json
```

Record CPU, memory, and H100 charges for every full-run app URL. Compare actual
spend with the 14--18 dollar projection and stop if the approved cap would be
exceeded. Preserve per-shard function seconds, model-load time, tokens/s, peak
CUDA allocation/reservation, actual GPU name, image ID, task ID, and app URL.

## Required review before interpreting results

- Confirm every generated sequence is exactly 1,024 tokens and every p-value,
  threshold, statistic, NLL, perplexity, repetition, distinct-2, and distinct-3
  value is finite.
- Confirm online PRC/null generation attempts remain zero and all source
  fingerprints match the smoke manifest.
- Confirm exact-prefix deltas are zero and TextSeal official/common decisions
  agree. Do not alter tolerances or frozen settings to hide a mismatch.
- Report observed false-positive counts over 500 separately. State that 500
  nulls cannot tightly validate a nominal 0.1% FPR.
- Keep the four calibration labels explicit and do not compare raw p-values as
  though calibration were identical.
- Interpret detection jointly with quality and diversity. In particular,
  inspect TextSeal/Gumbel repetition and distinct-n distributions, not only
  median NLL/perplexity.
- Gumbel is seed-deterministic at fixed batch shape, but the released power-form
  path showed batch-shape sensitivity. Keep batch size 5 fixed for production.

The 50-prompt x five-seed diversity experiment remains separately unauthorized.
The utilities `pairwise_token_agreement` and `self_bleu_token_ids` support its
planned analysis, but no diversity-generation entrypoint is invoked here.
