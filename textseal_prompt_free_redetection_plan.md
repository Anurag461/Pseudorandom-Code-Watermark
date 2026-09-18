# TextSeal and PRC prompt-free redetection plan

Prepared 2026-09-17. Status: source/cache preflight complete, shared runner
renamed, and completion-only upstream TextSeal adapter implemented. All 38
focused tests pass. No real-checkpoint replay, generation, or paid Modal compute
job launched. Repeat handling is deferred at the user's request.
Working branch: `comparison-with-redetect`, created directly from `redetection`
at `1bd7ce2`. All existing working files were preserved when switching branches.
The branch now includes current `redetection` through `04f34d3`; its new native
8B PRC traces and prefix results are reflected in the reuse plan below.

## Objective and scope

Repair the existing 500-prompt Qwen3-8B comparison using saved completions.
Recompute TextSeal entropy and PRC partition probabilities without the original
prompt, and preserve the authors' released TextSeal algorithm by calling its
implementation directly. Repeat handling and new generations are outside the
current scope; keep the original generation behavior and saved completions.

The initial repair covers the six existing lengths
`T={128,256,400,512,768,1024}`, TextSeal `alpha=0.1`, and online PRC
`eta=0.05, t=3`, with the original 500 watermarked completions per method and
500 shared null completions. Produce separate native-8B and proxy-0.6B detector
panels. Reuse SynthID-Text and Gumbel-Max detection records, whose inputs are
already completion-only. Keep generation-conditioned NLL as a separately
labelled quality metric; it is not a detection input.

This is a repair of the controlled comparison, not the larger Self-BLEU sweep
or a new model/key/temperature study. Further historical PRC sweeps are an
explicit extension, not silently included in the cost below.

## Findings that constrain exact upstream equivalence

Audited TextSeal commit: `c60d0d1da2e59f09a698438e218a07ee779b4616`.
The GitHub API resolved `main` to this same commit on 2026-09-17. File hashes
and findings are recorded in
[`baseline_comparison/textseal_source_audit.json`](baseline_comparison/textseal_source_audit.json).

1. The released `TextSealDetector._compute_entropies` consumes raw token IDs,
   forwards them through the model, and returns the first `T-1` next-token
   entropies. No generation prompt or synthetic prefix is required.
2. `_score_text` uses `entropies[pos-1]`, begins at zero-based position
   `ngram+1`, and deduplicates `(context, target)` under `v2`. Preserve these
   conventions, including the extra skipped initial position.
3. The official result contains weighted and unweighted p-values, their
   minimum as `p_value`, and `detected = p_value < 0.01`. Our historical
   comparison instead used `p_value_weighted < 0.001`. These are distinct
   decision rules even if the underlying weighted statistic agrees.
4. The released `TextSealGenerator.sample_next` samples the key every token;
   neither it nor its inherited generation loop tracks repeated contexts.
   The checked Python/config/docs source tree contains no generation-time
   repeat-handling switch. Detector deduplication is not generation fallback.

Sources: [pinned detector](https://github.com/facebookresearch/textseal/blob/c60d0d1da2e59f09a698438e218a07ee779b4616/textseal/watermarking/detector.py),
[pinned generator](https://github.com/facebookresearch/textseal/blob/c60d0d1da2e59f09a698438e218a07ee779b4616/textseal/watermarking/generator.py),
and [paper Remark 1](https://arxiv.org/html/2605.12456v1#S2.Thmtheorem3).

**Exactness requirement:** use the pinned upstream detector and sampler as the
executed implementation, not a rewritten formula that merely has small
p-value differences. Pin dependencies, model revision, dtype, and execution
shape. A change of model backend, precision, cache, or batching is a separate
numerical change and must be validated and recorded. This plan does not claim
that the existing custom Qwen execution is bitwise identical to upstream HF
execution or that repeat handling already exists in released code.

## 1. Code organization and source inventory

Completed in this task: rename `baseline_comparison/smoke_runner.py` to
`baseline_comparison/comparison_runner.py`; update active worker, diagnostic,
resume, finalizer, test, and documentation references. Keep genuine smoke-test
function names. Preserve archived source snapshots, manifests, and historical
fingerprints; the renamed source receives a new integration fingerprint.
The comparison/proxy tests and new upstream/input-contract tests pass: 38 tests.
Both historical native and proxy TextSeal entropy-scoring paths now fail closed.

The read-only SDK preflight (`baseline_comparison/textseal_preflight.py`) has
verified 1,010 source files and exported exactly 1,500 completion-only rows
(500 TextSeal, 500 online PRC, 500 shared null), all matching the historical
1,024-token hashes. Local artifacts are under `outputs/comparison_redetect/preflight/`:
`preflight.json` records source hashes, PRC artifacts/key, model revisions and
cache inventory; `completion_inputs.jsonl` contains only method, pairing index,
completion IDs, and historical token hash. No prompt or generation trace is
exported. The preflight read 458,397,255 source bytes and launched zero compute jobs.

Model download metadata matches both pinned revisions, including every weight
shard. Weight bytes were inventoried, not rehashed or loaded. The 0.6B cache
contains weights and tokenizer JSON but lacks `config.json` and
`tokenizer_config.json`: stage those small files from the pinned revision before
HF replay. This is recorded as a replay prerequisite, not hidden by the inventory
passing.

Inventoried sources:

- The ten generated TextSeal shards under
  `/data/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/generated/`.
- Online PRC source
  `online_causal_prc_v1/qwen3_8b_base/n1280_T1280_t3_eta0.05_rr99of100_sampler-poscdf-v1_kvcache-static-v1`,
  retaining its original online key, partition, OTP, parity supports, and token IDs.
- The shared null source `/data/_nulls/qwen3_8b_base/T13088`, sliced to the
  required prefix without changing tokenization.
- Generator/detector model revisions: 8B
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; proxy 0.6B
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.

Load models from existing remote caches. Keep prompts in the inventory process
for pairing/provenance only. Pass the replay worker an allowlisted payload of
completion token IDs, detector configuration, and, for PRC, the partition.
Do not pass prompt tokens, generation caches, or saved generation probabilities.

## 2. One raw-completion input contract

Use `completion_only_raw_abstain_v1`, consistent with the completed September
17 PRC reruns: no prompt, BOS, EOT, chat template, padding prefix, or other
prepended token. This supersedes the older EOT-seeded comparison proposal.
Use fresh model state for every independent completion batch and model
positions beginning at zero. Preserve stored token IDs without decode/re-encode.

For zero-based completion position `p>=1`, logits after token `p-1` predict
token `p`. TextSeal receives a `T-1` entropy vector with this alignment. PRC
retains all original coordinates and gives coordinate 1 (zero-based index 0)
soft score zero; its token remains context for coordinate 2. Never fabricate
a probability for the first token or shift the remaining coordinates.

Cache identity includes token hash, model/revision, runtime/dtype, input
protocol, length, execution shape, and upstream source hash. PRC probabilities
also include partition identity. Full-vocabulary entropy can be shared across
uses of the same null text only when all relevant identity fields match.
PRC partition probabilities alone cannot supply TextSeal entropy.

## 3. Exact TextSeal redetection

Implemented reference adapter: `baseline_comparison/textseal_completion.py`.
It verifies the four pinned upstream source hashes before loading the original
modules, rejects prompt/cached-entropy inputs, and returns the original result
dictionary unchanged alongside the comparison decision. Its numerical calls
match upstream exactly on deterministic fixtures and a small randomly initialized
HF Qwen3 model; production checkpoint/GPU parity is still pending. The local
test runtime uses CPU PyTorch 2.5.1, distinct from the planned PyTorch 2.4 runtime.

Construct the real upstream `TextSealDetector` with the existing keys,
`ngram=3`, `mixing_alpha=0.1`, and `scoring_method='v2'`.

The reference path calls upstream `_compute_entropies(completion_ids)` and
`_score_text(completion_ids, entropies, 'v2')` directly. These are precisely the
internal operations of upstream `detect`, with stored IDs replacing its text
tokenization step. Verify `detect(decoded_text)` equality when tokenizer
round-tripping preserves the saved IDs; otherwise retain the IDs and document
the token-level entry point. Do not silently retokenize the experiment.

Prefer the upstream HF model interface for this reference path. An optimized
custom-Qwen/chunked path is optional and cannot replace it unless it passes
the checks below. Keep upstream entropy dtype and arithmetic: do not insert
an undocumented `.float()`, change softmax temperature, truncate the vocabulary,
or substitute binary-partition entropy. Full-sequence logits can consume large
memory, so begin with one completion, then small equal-length batches; benchmark
before increasing batch size. No left padding is necessary for this corpus.

For every prefix, call `_score_text(ids[:T], H[:T-1], 'v2')` so entropy
normalization and deduplication are prefix-local. Reuse a longer causal entropy
trace only after checking it against direct-prefix replay. Do not use the
custom `textseal_gamma_test` output as the authoritative result; retain it only
as a diagnostic reference. Preserve raw upstream outputs, including zero
p-values, rather than silently clipping them in stored results.

Report three clearly named outputs:

- **Upstream result:** unchanged returned dictionary, including its combined
  `p_value` and `detected` at 0.01.
- **Controlled-comparison result:** the upstream `p_value_weighted` at the
  existing nominal 0.001 cutoff, for continuity with historical figures.
- **Combined score at 0.001:** optional supplementary comparison using the
  upstream minimum, explicitly labelled as such. Do not assume taking a
  minimum preserves nominal FPR; report observed null counts separately.

This preserves the authors' code while making our benchmark's threshold and
choice of output visible. Do not substitute the upstream 0.01 boolean into a
table described as nominal 0.001 detection. A new independent calibration/null
study is outside this repair; 500 nulls cannot establish 0.1% FPR tightly.

## 4. PRC: what needs replay and what can be reused

| Result | Action |
|---|---|
| Fixed PRC 0.6B→0.6B, n=400 | Already redetected under raw-completion abstention; reuse after identity checks |
| Online PRC 8B→0.6B, n=640, eta=.05 | Already redetected; reuse as a validated endpoint/reference, not as evidence for every prefix/model |
| Online PRC 0.6B→0.6B, n=3104, eta=.20 | Already redetected; retain the indexed results and traces without rerunning |
| Online PRC 8B→8B, n=1280 and n=1024, eta=.05 | Already redetected on current `redetection`; n=1024 reuses the n=1280 traces without inference |
| Online PRC eta=.05, all six comparison prefixes, native 8B | Reuse matching watermarked traces after token/key/runtime checks; audit shared-null identity and replay any missing coverage |
| Same six prefixes, proxy 0.6B | Inventory existing raw traces, then recover missing coverage through 1,024 |
| Other eta values, lengths, or 14B panels | Audit and flag prompted results; extend only if those panels are being republished |

Authoritative existing raw results and identities:
[`outputs/redetection/README.md`](outputs/redetection/README.md) and
[`outputs/redetection/cache_index.json`](outputs/redetection/cache_index.json).
The new native-8B run used nulls from T=1382, whereas this comparison uses the
T=13088 shared-null source. Its null scores cannot replace the comparison's
null scores without matching token identities. Preserve the frozen comparison
cohort and replay its nulls when the existing traces do not match.
They contain partition probabilities, not the full-vocabulary entropy needed
for TextSeal. A 640-token trace cannot simply be extended from a probability
array without reconstructing its model context; budget a full 1,024-token
pass unless compatible longer traces or replay state actually exist.

Reuse the redetection branch's integrated implementation:
`qwen.completion_only_partition_trace_batch`, the completion-only defaults in
`detectors.py`, and `modal_run.py::redetect` for manifests, validated batching,
trace-cache identities, and scoring. The older `completion_only/` scripts are
reference diagnostics, not a second production implementation. The current
integrated CLI supports both BF16 Qwen3-0.6B-Base and pinned Qwen3-8B-Base,
including sharded-weight validation and the native-8B memory reservation.

The legacy comparison PRC function still passes a T-length prompted `p_trace`;
the redetection branch's current detector requires T-1 completion-only values
and rejects that old call. Wire the comparison to the integrated raw trace
path rather than turning on `completion_only=False`. Use that explicit opt-in
only for a deliberately labelled matched prompted control.

Reuse the project `OnlinePRCKey` and detector scoring code for parity products
and thresholds. Recompute the usual
score-dependent Hoeffding threshold at nominal .001 from the new soft scores;
retain the old literal numeric thresholds only in a separately labelled
sensitivity table. Preserve original online supports at each prefix. Do not
regenerate a fixed n=400 key or treat an online prefix as a new fixed code.

## 5. Repeat handling: deferred

Per the latest user instruction, implement no generation-time repeat handling,
new generation arm, fallback state, block resets, or repetition penalty. Keep
upstream detector deduplication unchanged. Any later repeat-handled generation
experiment is separate scope: the currently pinned released sampler has no
such state machine, so a paper-defined extension cannot be described as
identical to the authors' released implementation.

## 6. Validation before production replay

1. **Source and call parity:** verify module hashes and real upstream method
   calls. Compare every returned detector field on empty/short, constant-entropy,
   repeated-context, and ordinary fixtures. Same-code/same-input CPU scoring
   must agree exactly. Preserve key-A alpha orientation, epsilon handling,
   initial eligibility, deduplication, weight normalization, and Gamma tails.
2. **Prompt exclusion:** instrument model inputs; assert their concatenation is
   exactly the stored raw completion (or its causal replay prefix). Delete or
   replace prompt metadata while holding completion IDs fixed and require
   identical entropy/probability traces and scores. Poison old cached entropy
   and probability fields to prove the new path cannot read them.
3. **Model parity and precision:** validate the actual target-device model
   inputs/logits, dtype, entropy vectors, and final decisions against the direct
   upstream HF reference. An optimized path needs exact agreement at the same
   execution settings to claim bitwise equivalence; tolerance-only agreement
   is reported as such, not relabelled exact. If it fails, use the upstream
   reference path and update timing estimates.
4. **Causality and alignment:** change suffix tokens and check earlier traces;
   compare all six prefixes to direct prefix calls. Confirm PRC coordinate-1
   abstention and TextSeal `H[pos-1]`; check a batch row cannot change another's
   keys, seen-context state, or cache.
5. **Separate numerical/context effects:** on the 50-prompt validation subset,
   run a matched prompted control through the same new model/entropy execution.
   Its completion-aligned entropy slice excludes the prompt from hash scoring.
   Compare it with raw-input replay and the historical cached trace. Attribute
   full historical differences only to the changes actually held fixed; a
   full matched-control run is additional work if needed to isolate precision.
6. **Result integrity:** check all 500 watermarked/null rows and prefix coverage,
   token/key hashes, source immutability, and unchanged SynthID/Gumbel scores.
   Run the existing comparison, proxy, `tests/test_prompt_free.py`, and
   applicable model-cache and online-PRC tests.

## 7. Cost estimate and execution order

Estimates are incremental compute charges, not an exact future bill or a
request to launch work now. Revised allocation without repeat handling:
**$5–15**, retaining a conservative **$25 planning ceiling** for the core repair. The pilot replaces estimates with
measured throughput before full dispatch. No blanket rerun of older sweeps is
included, and no credit balance is assumed.

Historical anchors:

- All three baselines together generated 1,500 × 1,024 tokens and scored the
  comparison for $2.6588. TextSeal alone averaged 44.68 seconds per 50 responses
  on H100. The compute-only lower anchor for another 500 TextSeal responses
  is therefore about 447 H100 seconds, not a promise for repeat handling.
- The completed raw 8B→0.6B PRC run processed 1,000 × 639 next-token positions
  in 494.7 seconds, or 570.7 seconds including model loading and validation.
- The earlier mixed PRC/null/TextSeal proxy replay cost $2.1948, with an
  additional $0.0766 for official TextSeal scoring; its chunked and prompted
  execution is a scaling reference, not an exact forecast for this protocol.

Sources: [baseline runtime](outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/controlled_baseline_full_runtime.json),
[baseline costs](outputs/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/controlled_baseline_full_cost_ledger.csv),
[raw replay timing](outputs/completion_only/n640-raw-69d2e3b80f3483ff/RESULTS.md),
[proxy costs](outputs/proxy_8b_cost_ledger.csv).

Current [Modal rates](https://modal.com/pricing), checked 2026-09-17: H100
$0.001097/s, A10 $0.000306/s, CPU $0.0000131/core/s, memory
$0.00000222/GiB/s. For example, 4 CPU cores and 48 GiB host RAM make an H100
worker approximately $0.001256/s before region/other premiums. Use total billed
worker time, including loading and validation, rather than kernel time alone.

| Work | Planning allowance |
|---|---:|
| Inventory, upstream/reference checks, five-/50-prompt pilots and matched controls | $1–4 |
| Native-8B prompt-free replay of original TextSeal, PRC eta=.05, shared null | $3–8 |
| Proxy-0.6B replay of those same three sets, reusing verified complete traces | $0.5–2 |
| Direct upstream CPU scoring, PRC scoring, paired statistics | $0.2–1 |
| **Rounded planning range including small overhead variation** | **$5–15** |

The original repair has 1,500 distinct length-1,024 sequences per detector
model: 500 TextSeal + 500 PRC + 500 shared null. That is 1,534,500 useful
next-token positions per model, before compatible-cache reuse. The literal
upstream entropy method also forwards the
last token before discarding its final entropy; include that overhead in the
benchmark. Replaying the shared null once serves both methods when execution
identity is identical. Scoring six prefixes does not require six full model
passes once causal-prefix parity has passed. Until then, the new reference
adapter replays each requested prefix independently; six prefixes total 3,088
forwarded tokens per TextSeal/null sequence. The pilot must measure this direct
path before treating a single-pass estimate as achievable.

Order: local rename/tests → read-only inventory → pinned upstream reference and
input-contract validation → costed pilot → core replay and scoring → tables. Reserve measured outstanding worker costs
and bounded retries before dispatch; stop if the projected total exceeds $25.
Exact upstream replay may be slower than the historical custom/chunked path;
do not weaken the parity requirement to fit a speculative cost estimate.

Optional later scope: other PRC eta panels require up to four watermarked sets
plus one shared null set at common lengths. Reproducing the old extended
PRC/null workload at the selected 640/1407/4096/13088 endpoints, plus its
TextSeal comparison, is approximately 16.9 million model-token positions per
detector model before reuse, over ten times the core replay. It requires
a separate measured budget; no new completion-only n90 boundary can be claimed
from just rescoring an old prompted boundary.

## Deliverables and acceptance

Store new inputs, traces, source/runtime manifests, raw upstream result dicts,
prefix tables, paired changes, and cost ledger under a separate protocol/run
namespace. Keep historical results as prompted controls with their original
hashes. Mark plots with generator model, detector model, context protocol,
TextSeal generation variant, score selection, and nominal cutoff.

The repair is complete when native and proxy TextSeal/PRC scores have verified
raw-completion inputs and complete coverage, and SynthID/Gumbel reuse is
verified. Repeat handling is deferred and is not part of acceptance for this repair.
