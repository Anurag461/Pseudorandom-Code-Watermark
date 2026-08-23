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
- Ten generation shards of 50 prompts, batch size 50, at most ten `H100`
  workers. Each worker makes one generation call per method after one model
  load. Generated shards are committed before CPU scoring.

## Batch-50 validation gate (completed)

The standalone shard on prompt rows `0..49` completed successfully through
TextSeal, SynthID, and Gumbel at the frozen 1,024-token settings:

```bash
modal run baseline_comparison/modal_app.py::app.batch50-validation \
  --approval-token APPROVE_50_PROMPT_BATCH50_VALIDATION \
  --run-id qwen3-8b-batch50-validation-20260823-v1
```

It produced 50 exact-length outputs per method in 166.386 seconds and used
27,839,692,800 bytes (25.93 GiB) peak CUDA reserved. The exact all-resource
bill was $0.27034319. The raw artifact, fingerprints, per-method timings, and
billing split are recorded in `controlled_baseline_batch50_validation_report.md`
and `outputs/controlled_baseline_batch50_validation.json`.

## Approval gate

The batch-50 gate passed. The measured projection is $2.70 for generation and
$3--4 end to end including CPU scoring; use a $5 hard cap. Expected end-to-end
wall time with ten available H100s is 8--15 minutes, excluding unusual queue
delay. Explicit user approval for the 500-prompt run is still required. The
earlier $14--18, 20--35-minute estimate is retained only as the obsolete
batch-5 upper bound. Only after approval may the literal token below be passed:

`APPROVE_500_PROMPT_CONTROLLED_BASELINE`

Without it, `app.full-run` raises `PermissionError` before preflight or GPU
launch. Do not weaken or bypass this guard.

## Exact commands after approval

Run from the repository root. Choose a new immutable run ID; the workers refuse
to overwrite an existing shard.

```bash
modal run baseline_comparison/modal_app.py::app.full-run \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-controlled-batch50-20260823
```

This command runs CPU reference checks and validates all 500 online-PRC/null
cache pairs before launching any GPU. It then maps ten 50-prompt shards across
up to ten H100s, loads Qwen once per shard, generates each of the three new
methods in one batch of 50, and commits raw shards under:

`/data/controlled_baseline_full/qwen3-8b-controlled-batch50-20260823/generated/`

After all ten generation shards pass, score them on Modal CPU:

```bash
modal run baseline_comparison/modal_app.py::app.full-score \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-controlled-batch50-20260823
```

This produces ten shared-schema JSONL shards (2,400 rows each; 24,000 total)
and validation records without loading Qwen or regenerating PRC/null text.

Download only the compact scored artifacts to a new local directory:

```bash
modal volume get prc-data \
  controlled_baseline_full/qwen3-8b-controlled-batch50-20260823/scored \
  /private/tmp/qwen3-8b-controlled-batch50-20260823-scored
```

Stream-validate, merge, and summarize them locally:

```bash
python -m baseline_comparison.full_run finalize \
  --scored-dir /private/tmp/qwen3-8b-controlled-batch50-20260823-scored \
  --output-dir outputs/controlled_baseline_full/qwen3-8b-controlled-batch50-20260823
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
spend with the post-validation batch-50 projection and stop if the approved cap
would be exceeded. Preserve per-shard function seconds, model-load time, tokens/s, peak
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
- Gumbel is seed-deterministic at fixed batch shape. The released power form
  and diagnostic log-space form were token-identical at batch sizes 1 and 5;
  the remaining batch-shape sensitivity comes from model logits/numerical
  execution. Keep batch size 50, prompt grouping/order, hardware, key, and
  model revision fixed for production.
- Retain the strict project-Qwen/Hugging-Face parity discrepancy in provenance:
  maximum batch-5 JSD was `8.726e-4` against a predeclared `1e-4` criterion,
  though all top-1 tokens agreed and native Hugging Face itself had
  `1.622e-3` maximum batch-shape JSD. Do not use this check to substitute the
  Hugging Face implementation for the frozen project-Qwen path.

The 50-prompt x five-seed diversity experiment remains separately unauthorized.
The utilities `pairwise_token_agreement` and `self_bleu_token_ids` support its
planned analysis, but no diversity-generation entrypoint is invoked here.
