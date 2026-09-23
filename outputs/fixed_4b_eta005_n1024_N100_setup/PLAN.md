# Fixed PRC: 4B generation, paired 4B and 0.6B detection

Prepared September 20, 2026, on branch `redetection` (`998f9cc`).
**All six approved stages are complete, including cloud CPU scoring.**
4B detector: MAP 98/100, entropy 93/100. 0.6B detector: MAP 93/100, entropy
90/100. Both used the same 100 saved completions; no nulls were generated.
Actual provider-reported spending is $0.40304877, with approximately $6.10
remaining. See `RESULTS.md` and `billing_final.json` for the completed results.
Generation produced 100 saved 1024-token watermarked completions exactly once.
See `GENERATION_RESULT.md`, `cache_index.json`, and `progress.json`.
Preparation took 105.68 seconds and cost $0.00988003 according to the provider
report. Both pinned checkpoints, the original artifact/partition, and all 100
prompts passed verification. See `PREPARATION_RESULT.md` and `progress.json`.
The current balance estimate includes intervening usage of approximately
$0.01947 from a separate diagnostic app. The original
proposal and estimates below are retained for reference; `setup.json` is frozen.
The user initially confirmed approximately **$6.52 remaining**, and selected **watermarked
completions only, with no empirical FPR evaluation**.

The staged adapter reuses the existing generation, replay and scoring routines.
Shared changes add revision pinning, 4B checkpoint validation and explicit
watermarked-only CSV handling. Sampling and PRC formulas are unchanged. Existing
working-tree changes were preserved, including the NumPy loading-order
correction in `modal_run.py`. Primary generation outputs were committed in
`a9f3e42`; nothing was pushed or uploaded as an additional archive. Exact
execution sources are retained locally with hashes.

## 1. Exact experiments and reuse

- Generate **100 watermarked completions once** using Qwen3-4B-Base, for prompt
  indices 0–99 from the existing `prompts.jsonl` cache. Each prompt has exactly
  50 tokens. Use the saved token IDs directly, with no chat template or added
  special token. The selected prompts are frozen in `prompts_N100.jsonl`.
- Fixed PRC: **eta=.05, n=T=1024, t=3, r=1014=round(.99n), one block**, forced
  length, no EOS early stopping. Thus 102,400 generated completion tokens.
- Detect exactly those same 100 saved token sequences with **4B and 0.6B**.
  Each detector recovers 100 × 1023 completion-only partition probabilities;
  MAP and entropy scoring share that detector's trace. Total: 204,600 recovered
  probabilities and 400 candidate/weight decisions.
- **Null generations: 0; null-generation cost: $0; null replay cost: $0.**
  Empirical FPR is not evaluated. Target FPR=.001 remains the analytical
  detector setting. CSV FPR cells must say `skipped`, never `0/0` or `0%`.
- Reuse the existing archived fixed eta=.05, n=T=1024 key and partition. Its
  original seed is **12345**. Do not build a new key or regenerate the partition.
  Original artifact SHA-256:
  `5915dc825d94b0d248ec99df167f7f99a7f9c4347bce2ea217934f49b0b444cd`.
  Archive location: `prc-research-archive/objects/sha256/59/5915dc825d94b0d248ec99df167f7f99a7f9c4347bce2ea217934f49b0b444cd`.
  The artifact was downloaded, checksum-verified and its scalar metadata read
  without unpickling or running scientific code. The approved CPU preparation
  must verify its deserialized key dimensions, eta/t, prompts and partition.
- Original partition: two equal 75,968-token buckets over 151,936 model output
  rows. Partition tensor fingerprint:
  `503d1cf93958f0d765606ed3e25aa87a1a777cd08d8f44436b0c6ae9716aa184`.
- Reuse the existing 0.6B checkpoint in `prc-hf-cache`; recheck its pinned hashes
  on the approved CPU stage. The active cache has no 4B checkpoint; download
  its approximately 8.05 GB of weights on CPU, once, during that stage.
- No 4B-generated completion or null cache was found in the current `prc-data`
  root/model-specific null directories or archived provenance catalog.
  Historical `qwen3_4b_base` trace/shard labels describe 4B detection of
  other-model generations and are not reusable as this generation cohort.
  Existing completion-only traces also cannot stand in for the new completions.
- A subsequent read-only check of `watermark-prc/main` found the matching
  n=T=1024 directory with 500 saved completions, but sampled records 0, 99 and
  499 exactly match the existing **0.6B-generation** manifest. Its 4B-labeled
  files are detector traces. No verified 4B-generation cache or cached 4B
  weights were found there. See `WATERMARK_PRC_CACHE_CHECK.md` and its JSON
  evidence; no paid computation was launched for this check.

## 2. Checkpoints, token compatibility and detector settings

| Role | Exact checkpoint | Weight and tokenizer revision |
|---|---|---|
| Generation and same-model detection | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Cross-model detection | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

The official [4B Base checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Base/tree/906bfd4b4dc7f14ee4320094d8b41684abff8539)
exists. Its configuration agrees with `qwen.py::return_qwen_config("4B")`:
36 layers, hidden size 2560, intermediate size 9728, 32 query heads, 8 KV heads,
head dimension 128, RoPE theta 1,000,000, vocabulary 151,936, BF16 and tied
embeddings. The custom loader handles tied embeddings and sharded weights.
The local RoPE allocation limit is 40,960 versus the official 32,768 context
limit; the proposed 50+1024 tokens are within both. Metadata agreement is not
a claim that a new 4B execution has already been tested.

The two revision-pinned `tokenizer.json` files are **byte-identical**:
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
Token-to-ID maps, merges and added tokens were also compared directly and match.
Both models have 151,936 output rows. Therefore direct 4B→0.6B token-ID replay
is compatible; no decode/re-tokenize step is needed. Full checkpoint shard,
index and tokenizer hashes are in `checkpoint_verification.json`.

Use **`completion_only_raw_abstain_v1`**, as in the recent fixed experiments:

- BF16 model inference; TF32 disabled; static KV cache for replay; one token
  per replay step. Raw completion token 1 begins a fresh cache at position 0.
- No original prompt, BOS/EOT proxy or chat wrapper is supplied to either
  detector. Coordinate 1 has score zero. Recover probabilities for coordinates
  2–1024, stored as float32; score with the existing float64 CPU routines.
- Existing MAP/posterior weighting and entropy weighting, with unchanged PRC
  indices and `detect_hoeffding` threshold. Fixed-construction policy is
  `block_or_bonferroni`; one block means its FPR allocation is exactly .001.
- Generation-time `p_trace` is retained as provenance only and **never used as
  the completion-only trace**, including for 4B→4B detection.

Generation preserves `generate_batch_and_collect`: temperature 1, no top-k or
top-p truncation, the existing Bernoulli PRC bucket channel, then multinomial
sampling within that bucket using float32 masked softmax. Keep the existing
concat KV cache for generation rather than silently switching its numerics.
This is the implemented sampler; the older single-token argmax description
does not describe this batched fixed-generation path.

**Seed limitation:** 12345 is verified for the archived key/partition. The
legacy fixed worker does not explicitly seed every generation RNG;
`Encode` calls `GF.Random` without a supplied generator. Do not claim that
the completions can be regenerated from 12345 alone. Preserve the realized
codewords, original token outputs, available RNG state and batch order.
Any change to explicit generation seeding needs to be reviewed as a separate
implementation choice; no RNG or PRC construction change remains in the code.

## 3. Resources, stages and cost

One worker at a time. No benchmark, automatic retry, smoke run, full reference
replay, reversed-batch replay, or extra validation pass is included.
Routine hashes, dimensions and coverage checks operate on the saved inputs
and primary outputs as part of the listed stages.

| Paid stage | Work and resources | Expected time | Expected cost | Conservative allowance |
|---|---|---:|---:|---:|
| CPU preparation | Cached artifact/prompt verification; pinned checkpoint download/hash. 4 physical cores, 16 GiB RAM | 2–5 min | $0.02–0.04 | $0.10 |
| 4B generation | 1 H100 80GB, 4 physical cores, 64 GiB host RAM; one batch of 100 × 1024 | 3–6 min | $0.24–0.47 | $0.90 |
| CPU manifest freeze | Convert saved batch to existing per-candidate records; freeze two manifests referencing identical completions/key/partition. 4 cores, 16 GiB | 0.5–2 min | $0.003–0.011 | $0.05 |
| 4B primary replay | 1 H100 80GB, 4 cores, 64 GiB; one batch of 100 | 2–4 min | $0.16–0.31 | $0.90 |
| 0.6B primary replay | 1 A100 80GB, 4 cores, 16 GiB; one batch of 100 | 1–3 min | $0.05–0.15 | $0.40 |
| CPU scoring | Existing `_score_redetection`, both detectors, MAP + entropy. 4 cores, 8 GiB | 1–3 min | $0.005–0.013 | $0.05 |
| **Total** | **No nulls** | **About 10–25 min compute; 15–30 min elapsed allowing startup/transfers** | **About $0.50–1.00 before startup uncertainty** | **$2.40 + $0.10 reserve = $2.50** |

The conservative proposal would leave approximately **$4.02 of $6.52**.
This is an estimate, not authorization or a provider-enforced dollar cap.
The implemented work timeouts are 10 minutes for preparation and each H100 stage,
6 minutes for A100 replay, and 3 minutes each for manifest freeze and scoring.
Each stage has a 30-second startup timeout and a 30-second outer-function
margin. Four physical CPU cores are both reserved and limited; one worker is
allowed, with retries disabled. A bounded child process turns native crashes
into ordinary failures. Any timeout/failure stops without a retry and retains
saved outputs.

Rates checked against [Modal pricing](https://modal.com/pricing): H100
$0.001097/s, A100 80GB $0.000694/s, physical CPU core $0.0000131/s,
memory $0.00000222/GiB/s. Estimates include the listed CPU and RAM reservations
and do not assume credits or discounts. Normal cached output storage is small;
the 4B weight cache adds about 7.5 GiB, within Modal's published 1 TiB monthly
free volume allowance if available (otherwise about $0.68/month).

Evidence, without new benchmarking:

- Previous 8B H100 replay at T1280/batch125: seven primary-only batches took
  111.1–113.4 seconds each and 42.73 GB peak allocation. The representative
  batch with the extra full reference took 223.9 seconds. The proposed model,
  length and batch are smaller, but the 4B estimate is still an extrapolation.
- Previous fixed 0.6B A100 T1024/batch125: 1000 candidates, eight batches and
  one reference check took 503.77 aggregate GPU-method seconds, with 17.01 GB
  peak allocation. Batch100 fits below that proven workload.
- `timing_references.json` retains these existing measurements. No tensor
  deserialization, model execution or experiment scoring occurred locally.

`billing_readonly.json` is a provider usage report fetched during setup:
$31.78713855 reported for September 20 UTC through completed hourly intervals,
before credits. It is **not a remaining-balance report** and excludes the last
partial hour. The prior experiment's saved complete report is $13.48415601.
The user explicitly clarified that the earlier $20 estimate preceded that
charge, so the operative remaining-budget estimate is $6.52. Do not subtract
the entire daily report again. Setup itself launched zero billed workers.

The read-only refresh at 11:54 UTC is in `billing_before_prepare.json` and
reports $34.87970689 for completed hourly intervals. The newly visible
$3.09256834 belongs to the prior 8B experiment's already-accounted-for final
replay/scoring, not new spending by this setup. Its app totals agree with the
prior $13.48415601 report to provider rounding. No `prc-fixed-4b-100` billing
rows are present. The working balance estimate remains approximately $6.52.

## 4. Necessary code adaptations and unresolved checks

Implemented in `fixed_4b_comparison.py`, with narrow shared edits in
`modal_run.py` and `watermark_expt.py`:

1. `_redetect_model_spec` validates pinned 4B shards and index. The adapter uses
   the existing model loader; online model normalization and `RedetectionModel`
   are unchanged.
2. The adapter selects exactly 100 watermarked generations. An explicit
   `null_policy=not_evaluated` permits this cohort; existing callers still
   require both watermarked and null candidates by default.
3. Primary replay uses `validate=False` and commits each primary trace. The
   stock full-reference workflow is unchanged and is not used for this run.
4. Optional revision pinning covers weight downloads and the verified local
   tokenizer. Runtime source hashes are checked before each stage. Native and
   model imports precede NumPy pickle compatibility aliases.
5. Existing cloud CPU scoring is unchanged. CSV rows report `N=100; null N=0`,
   empirical FPR `skipped`, and working-tree source hashes as execution evidence.

Five metadata-only tests passed: checkpoint/index validation, explicit cohort
policy, CSV formatting/idempotence and launch guards. Syntax checks passed.
These checks do not import models, deserialize tensors or score experiments.
No paid smoke test or benchmark was performed.

The approved CPU preparation still needs to deserialize the archived artifact,
check all 100 prompts and key/partition compatibility, and verify/download model
files. Failure must stop before GPU generation. Current 4B inference has not
been benchmarked or separately validated; no such paid pass is included.

### Staged launch runbook

Only after explicit approval for CPU preparation with a $0.10 allowance:

```sh
MODAL_PROFILE=new-prc-watermark /opt/anaconda3/bin/python -m modal run \
  fixed_4b_comparison.py --stage prepare \
  --approval-reference 'Record the actual user approval for CPU preparation here'
```

Later stage names are `generate`, `freeze`, `replay_4b`, `replay_0p6b`, and
`score`. Each requires its own approval and collected prerequisite outputs.
There is no all-stages command. An already-attempted stage refuses to relaunch;
the reviewed setup becomes immutable after the first attempt.

If only a local transfer was interrupted, repeat the storage-only collection:

```sh
MODAL_PROFILE=new-prc-watermark /opt/anaconda3/bin/python fixed_4b_comparison.py collect STAGE_NAME
```

This runbook does not authorize any paid launch.

## 5. Saving, reporting and approval boundary

Cache prompts, original and adapted artifacts, source hashes, checkpoint
revisions, exact completion tokens, PRC codewords, available RNG state,
generation provenance, completion-only inputs, both detector traces, per-record
scores, manifests, logs, peak memory, elapsed times and provider billing IDs.
Use existing `prc-data`, `prc-hf-cache`, and `prc-completion-only` volumes for
normal execution outputs. Retrieve the primary output files locally after each
stage. Keep additional reproducibility archives local.

After scoring, append exactly two completed rows to the existing redetection
CSV: 4B→4B and 4B→0.6B, each with explicit /100 TPR denominators and N=100 in
Notes. Historical TPR fields are `unavailable`, naive fields `skipped`, and
empirical FPR fields `skipped`. Preserve unrelated rows and working-tree edits.
Save and narrowly commit completed generation/primary detection results before
any optional checks; do not broadly stage, reset or push. Report actual provider
cost per approved stage and the total.

**Every listed CPU/GPU stage received explicit user approval for that workload
and allowance before launch. All six stages are now complete.** Any retry,
benchmark, extra validation, null cohort, or new archive upload would require a
separate proposal and approval. The previous 8B eta=.15 experiment is excluded.
