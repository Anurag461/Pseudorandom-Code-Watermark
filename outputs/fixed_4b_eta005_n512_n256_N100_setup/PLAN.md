Prepared on `redetection` at `d6f618937d82fb46813e4d38eb015879fe3f8fca`.
Status: awaiting explicit approval for the twelve named paid stage runs below.
No paid computation for these cohorts has started.

| Cohort | n | T | Blocks | N | eta | t | r | Generator | Detectors |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| fixed_4b_eta005_n512_N100_v1 | 512 | 512 | 1 | 100 | 0.05 | 3 | 507 | 4B | 4B and 0.6B |
| fixed_4b_eta005_n256_N100_v1 | 256 | 256 | 1 | 100 | 0.05 | 3 | 253 | 4B | 4B and 0.6B |

Generate once per cohort and detect exactly those saved token IDs with both models.
This requires 200 new watermarked completions, 76,800 generated tokens, 153,200
recovered completion probabilities and 800 MAP/entropy decisions. Each cohort
uses the same original prompt indices 0–99, with 50 prompt tokens per record.
The fixed keys differ by n, so truncating the completed n=1024 generations would
not produce either requested experiment.

**Null generations: zero. Null generation/replay/scoring cost: $0.** Empirical FPR
is not evaluated, following the user's previous choice. Target FPR remains 0.001
with the existing `block_or_bonferroni` policy; each candidate has one block.

The archived fixed-PRC artifacts confirm seed 12345, eta 0.05, t=3, T=n and
r=round(0.99n). Reuse each length's original key and partition, plus the cached
prompt IDs. The two partition buckets contain 75,968 IDs each; partition SHA-256
is `503d1cf93958f0d765606ed3e25aa87a1a777cd08d8f44436b0c6ae9716aa184`.
The length-specific artifact and semantic key hashes are frozen in
[proposal.json](proposal.json) and the two setup manifests:
[n512](../fixed_4b_eta005_n512_N100_setup/setup.json),
[n256](../fixed_4b_eta005_n256_N100_setup/setup.json).

Read-only Modal inventory found both checkpoints in `prc-hf-cache` and no
compatible n256/n512 4B generation cohort in the inspected caches. Earlier
`watermark-prc` inspection found 0.6B-generated fixed cohorts and 4B detector
caches, which cannot replace 4B generations. Archived source artifacts were
downloaded and hash-verified locally; only non-executing pickle metadata was
inspected. Full key/partition and prompt verification occurs in each approved
CPU preparation stage. Cache evidence is in [cache_inventory.json](cache_inventory.json).

| Role | Checkpoint | Model and tokenizer revision |
|---|---|---|
| Generate and detect | Qwen/Qwen3-4B-Base | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| Detect | Qwen/Qwen3-0.6B-Base | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

These exact checkpoints already completed the n1024 experiment through the
repository's existing Qwen loader and replay routine. Their tokenizer.json files
are byte-identical (SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`),
and both models have 151,936 output rows. Direct 4B→0.6B token-ID replay is
compatible; no text round-trip or retokenization is used. CPU preparation
reverifies cached weights and metadata against the frozen hashes. Missing
checkpoint files stop the stage; downloading replacements is not authorized.

Generation preserves the successful sampler: BF16 model, TF32 disabled,
temperature 1, no top-k/top-p restriction, Bernoulli PRC bucket selection and
float32 masked-softmax multinomial sampling within the selected bucket. Use
concat KV caching, no chat template, and force exactly T tokens without EOS
termination. Seed 12345 describes the archived key/partition; the legacy sampler
does not explicitly seed every RNG. Preserve actual codewords, tokens and
available RNG states rather than claiming seed-only regeneration.

Detection uses `completion_only_raw_abstain_v1`, the same protocol as the
completed n1024 experiment: raw completion tokens without a prompt or special
prefix, first-coordinate abstention, static KV cache, BF16 models, float32
recovered probabilities, and existing float64 cloud CPU MAP plus entropy
scoring. Coordinates 2 through T are recovered from each detector. Generation
`p_trace` is saved as provenance only and is never substituted for detection
traces. There is no independent reference replay, benchmark or optional
validation pass in this proposal.

The following table lists **twelve paid runs**: each row is executed separately
for n512 and n256. Run sequentially with one worker, no retries, and one batch
of 100 for each GPU stage. CPU stages also process only the named 100-record
cohort. All stages reserve and limit CPU to four physical cores.

| Stage (each n) | Hardware / host RAM | n512 time / expected cost | n256 time / expected cost | Allowance per run |
|---|---|---|---|---:|
| prepare | CPU / 16 GiB | 0.25–1 min / $0.002–0.006 | 0.25–1 min / $0.002–0.006 | $0.025 |
| generate | H100 80 GB / 64 GiB | 1.5–3 min / $0.12–0.24 | 1–2.5 min / $0.08–0.20 | $0.40 |
| freeze | CPU / 16 GiB | 0.25–0.75 min / $0.002–0.004 | 0.25–0.75 min / $0.002–0.004 | $0.025 |
| replay_4b | H100 80 GB / 64 GiB | 1.25–2.5 min / $0.10–0.20 | 1–2 min / $0.08–0.16 | $0.40 |
| replay_0p6b | A100 80 GB / 16 GiB | 0.75–1.5 min / $0.04–0.07 | 0.5–1.25 min / $0.025–0.06 | $0.20 |
| score | CPU / 8 GiB | 0.17–0.75 min / $0.001–0.004 | 0.17–0.75 min / $0.001–0.004 | $0.025 |

Preparation copies and verifies the archived artifact and cached checkpoints.
Generation saves the single primary batch before any conversion. Freeze builds
the existing candidate records and two manifests referencing identical tokens.
Each replay saves one primary trace. Scoring calls the usual `_score_redetection`
routine for the two models and both weights, then exports the existing CSV
schema with N=100 explicit, /100 denominators, null N=0 and FPR skipped.

Expected cost is **$0.265–0.524 for n512**, **$0.190–0.434 for n256**, or
**$0.455–0.958 combined**. Allowances total $2.15; an additional $0.10 startup
reserve gives a **$2.25 conservative total**. These are estimates, not a
provider-enforced dollar cap. Child work timeouts are 120 seconds for each CPU
stage, 240 seconds for each H100 stage and 180 seconds for each A100 stage;
the function adds 30 seconds and startup is bounded to 30 seconds.

Estimated combined worker time is 7.3–17.8 minutes, approximately **12–25 minutes
elapsed** with startup, transfers and local commits, excluding approval waits
and provider queues. Estimates use the successful n1024 run's identical
hardware and batch size, retaining model-loading overhead; no benchmark was
run. That experiment cost $0.40304877, with generation/4B replay/0.6B replay
worker times of 130.5/110.8/76.2 seconds. See
[timing_references.json](timing_references.json).

The read-only provider billing check found no additional usage since that
experiment's final accounting. Estimated balance is **$6.09748620** and would
be **$3.84748620** after the conservative $2.25 total. This is derived from the
user's approximate original balance and recorded charges, not a provider
credit-balance guarantee. Evidence: [billing_before_setup.json](billing_before_setup.json).

Only the successful staged runner's length selection/paths/shape checks and
shorter timeouts were adapted, with a missing-checkpoint guard. Sampling,
replay, CPU scoring and corrected native-import-before-NumPy-alias order are
unchanged. Six lightweight metadata tests passed, including cohort isolation
and wrong-manifest launch rejection. Exact runtime sources are hashed in each
setup and snapshotted locally. The completed n1024 setup and its original
execution snapshots remain intact. No expensive local work occurred.

After approval, record the approval separately for each of the twelve named
runs, check billing before launches, stop on failure, and retrieve outputs and
evidence after each stage. Save and narrowly commit completed generation and
primary detection outputs before any optional checks. Do not retry or add a
paid pass without a new estimate and explicit approval. Preserve unrelated
working-tree changes and do not push. Additional source/reproducibility archives
stay local; routine cloud setup/manifests and experiment outputs follow the
existing workflow.

Run a single approved stage with the existing cloud entrypoint, for example:

```sh
MODAL_PROFILE=new-prc-watermark /opt/anaconda3/bin/python -m modal run \
  fixed_4b_comparison.py --n 512 --stage prepare \
  --approval-reference 'REPLACE WITH THE ACTUAL EXPLICIT USER APPROVAL'
```

Use the corresponding n and stage only after it is approved. Each stage refuses
an existing attempt; the setup becomes immutable after preparation. Read-only
collection can be repeated using `python fixed_4b_comparison.py collect STAGE N`.
No command in this plan has been launched for either new cohort.
