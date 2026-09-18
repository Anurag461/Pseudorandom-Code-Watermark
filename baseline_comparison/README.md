# Completion-only TextSeal comparison

The shared historical runner is `comparison_runner.py` (formerly
`smoke_runner.py`). It served the full comparison as well as smoke runs.
Its native and proxy TextSeal paths that consumed prompt-conditioned entropy
are retired and raise an error. Generation behavior is unchanged; repeat
handling is deferred.

## Detector

`textseal_completion.TextSealCompletionDetector` calls the original
`TextSealDetector._compute_entropies` and `_score_text` from Meta's pinned
[TextSeal source](https://github.com/facebookresearch/textseal/tree/c60d0d1da2e59f09a698438e218a07ee779b4616).
There is no local replacement for entropy, PRFs, deduplication, weighting, or
Gamma scoring. `textseal_source_audit.json` pins the exact bytes of the four
upstream numerical/configuration files; changed files are rejected before import.
The loader skips upstream package initializers that eagerly import unrelated
post-hoc/evaluation dependencies, and executes the original numerical modules
unchanged under their original module names.

```python
from baseline_comparison.textseal_completion import TextSealCompletionDetector

# model: an eval-mode Hugging Face causal LM on the desired device.
# completion_ids: original saved Python integer IDs, without decode/re-encode.
detector = TextSealCompletionDetector(model)
result = detector.detect(completion_ids[:400])
original_result = result["upstream"]
comparison_decision = result["comparison"]["decision"]
```

The only detection input is the raw completion ID list. The API has no prompt,
entropy, probability, or KV-cache parameter and rejects cache-record mappings.
It prepends no BOS/EOT or other token. Every call forwards the completion anew,
starting at position zero; upstream produces T−1 entropies and indexes H[pos−1].
For ngram=3, upstream's first eligible zero-based position is 4. Repeated
context/target pairs are deduplicated by upstream v2, without generation changes.

`upstream` preserves the entire original returned dictionary, including
`p_value = min(p_value_weighted, p_value_unweighted)` and the authors' 0.01
decision. `comparison` separately selects `p_value_weighted < 0.001`. Missing
weighted evidence yields an abstention/false comparison decision; the upstream
short-input result remains untouched. The settings retain the original comparison's
keys 42/12387, ngram 3, alpha 0.1, and scoring method v2.

Use `detect_prefixes(completion_ids[:1024])` to run upstream entropy and scoring
independently at each of the six actual lengths. `validate_prefixes=True` also
calls upstream's public `detect` separately on the identical raw IDs at each
length and requires exact entropy and result agreement. Per-prefix entropy
vectors are stored separately; production manifests reject longest-trace reuse.

The [GPU diagnostic](../outputs/comparison_redetect/textseal_setup/prefix_diagnostic/REVIEW.md)
identified BF16 matrix-shape dependence: all ten pilot examples differed at
n=128 when slicing n=1024 entropy. There were no comparison decision flips in
that small pilot. The first differing activation on null/0000 was layer 0's
key projection, reproducible with identical input prefixes and weights in an
isolated linear operation. Same-shape repeats and a causal suffix intervention
passed. `detect_prefixes_reusing_longest` remains diagnostic-only; matching on
other lengths in ten examples is not a guarantee for all records. Batching,
chunking, and alternate model backends remain outside the reference path.

## Reproducible setup and checks

Use an isolated Python 3.11 environment for the planned production runtime:

```sh
python -m pip install -r baseline_comparison/requirements-textseal.txt
python -m pip install --no-deps 'git+https://github.com/facebookresearch/textseal.git@c60d0d1da2e59f09a698438e218a07ee779b4616'
python -m pip install pytest==8.3.3
python -m pytest tests/test_textseal_completion.py -q
```

Alternatively, pass `source_root=path_to_pinned_checkout` to the constructor;
set `TEXTSEAL_SOURCE_ROOT` to that checkout for tests. All four audited files
must be present and unchanged. No package downloads occur inside the detector.

Local validation on 2026-09-17 used Python 3.12 / CPU PyTorch 2.5.1 and
Transformers 4.51.3: 52 tests passed across TextSeal completion/preflight/replay,
baseline comparison, and proxy analysis. They check exact dictionary equality
against upstream's public `detect` on ordinary/repeated/short fixtures and a
small randomly initialized HF Qwen3 model; actual forwarded IDs; prefix causality;
cross-response isolation; poisoned cached fields; source tampering; and retired
prompted entry points. Tests do not download or load production weights.
These checks establish source/call parity, not production GPU numerical parity.

## Completed source preflight

`textseal_preflight.py` is a local Modal SDK reader, not a remote Modal app.
It reads cached files, checks historical completion hashes, and exports only
method, pairing index, raw token IDs, and the historical token hash.

```sh
MODAL_PROFILE=new-prc-watermark python -m baseline_comparison.textseal_preflight \
  --cache-dir /tmp/comparison-redetect-cache \
  --output-dir outputs/comparison_redetect/preflight
```

The preflight has validated 1,010 files / 458,397,255 source bytes and all
1,500 length-1,024 completions (500 TextSeal, 500 PRC, 500 shared null).
`outputs/comparison_redetect/preflight/preflight.json` records every source
hash, PRC artifact/key identity, and model download revisions. Its
`completion_inputs.jsonl` is the clean detector-input export. Raw downloaded
source files are held outside the repository and must never be passed to the
new detector as input records.

The existing 0.6B cache lacks `config.json` and `tokenizer_config.json`.
Stage those files from revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
before HF replay. The 8B cache revision is
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Weight-file sizes and download
metadata were verified during preflight. The later replay also verified all
weight bytes against the frozen checkpoint hashes before loading.

The TextSeal prefix discrepancy is resolved by direct per-length replay. The
pilot passed all 60 exact upstream checks, and the full 500 TextSeal + 500 shared
null cohort is complete: TPR is 500/500 at every length; FPR is 0/500 except
n=768 at 1/500. All 1,000 records and 6,000 prefix results passed readback checks.
Six TextSeal rows were added to `baseline_comparisons.csv`, preserving all
original PRC cells. PRC shared-null alignment is complete, as recorded below.
The native-8B pilot/full worker is defined in `textseal_modal.py`; local setup
is prepared by `python -m baseline_comparison.textseal_redetect`. The
[launch review](../outputs/comparison_redetect/textseal_setup/direct_prefix/REVIEW.md) describes
the frozen manifest, ten-record pilot, exact gates, manual dispatch and costs.
The pilot never launches the full stage automatically. The
[redetection plan](../textseal_prompt_free_redetection_plan.md)
specifies PRC reuse, remaining integration, and the provisional $5–15 core-repair
budget with a $25 planning ceiling. The shared native comparison CSV now
contains all four methods: PRC, TextSeal, SynthID and GumbelMax. The proxy
panel remains a separate step; historical full-score orchestration still references retired
paths and is not used by the new replay/collector/publisher.

## Cached PRC prefix comparison

`prc_prefix_comparison.py` rescores the published native-8B online PRC n=1024
cohort to the existing comparison grid: 128, 256, 400, 512, 768, and 1024 tokens.
It calls the integrated completion-only scorer on the original cached traces,
keys and partition, and saves a separate
`outputs/comparison_redetect/baseline_comparisons.csv` with old/new posterior
and entropy-weighted TPRs, new FPRs, and percentage-point changes.

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m baseline_comparison.prc_prefix_comparison \
  --generation-cache /tmp/comparison-redetect-cache \
  --output /tmp/prc_original_cohort_prefixes.csv
```

The generation cache is the local `prc-data` export produced by the source
preflight. The indexed native-8B run must also be available locally at the
location in `outputs/redetection/cache_index.json`, including its prepared
manifest, artifact, inputs and traces. No remote call or inference is made.
Use `--lengths` to request another grid through 1024; the script always includes
1024 as an exact regression check against all saved scores and old decisions.

Each row is a separate one-shot test at nominal FPR 0.001, retaining the
original online prefix supports and coordinate-1 abstention. This is not an
OR decision across lengths. This original-cohort command uses the published
n=1024 null cohort (T1382 source). The shared comparison CSV now uses the T13088
nulls after the alignment below; choose a separate output to reproduce T1382.
The adjacent provenance JSON records input/output hashes and validation results.
Detailed per-candidate scores remain in a separate local archive namespace;
the original redetection reports and summary CSV are unchanged by this script.

## Align PRC with the comparison's shared nulls

`prc_shared_nulls.py` prepares a null-only native-8B replay using the existing
integrated PRC worker and scorer. All 500 T1382 null prefixes differ from the
original comparison's T13088 prefixes; the indexed native-8B cache cannot be
relabelled as the shared cohort. All 500 watermarked traces are reusable, with
exact agreement of all 6,000 per-prefix posterior/entropy score dictionaries.

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m baseline_comparison.prc_shared_nulls \
  --stage prepare --generation-cache /tmp/comparison-redetect-cache
```

Preparation runs locally, verifies the historical token/source identities, and
freezes the clean inputs, original key/partition and checkpoint. Launch requires
the exact plan hash via `--stage run --approved-plan-sha256 ...`; see the
[PRC launch review](../outputs/comparison_redetect/prc_shared_nulls/REVIEW.md).
Only four null batches of 125 completions reach the GPU, with a validated first
batch. The n=1024 probabilities supply all six prefixes without further inference.
`--stage collect` retrieves already computed traces and scores locally; it never
launches a GPU. Publication checks every watermarked score and updates only
the six PRC FPR fields/notes and provenance in `baseline_comparisons.csv`.
The original prefix command refuses to overwrite an already aligned table.

Current status: **completed and validated**. All 500 original T13088 shared nulls
were replayed without prompts; posterior and entropy-weighted FPRs are 0/500 at
all six lengths. Every TPR field and all 6,000 watermarked score dictionaries are
unchanged. See [verification.json](../outputs/comparison_redetect/prc_shared_nulls/verification.json)
and [execution.json](../outputs/comparison_redetect/prc_shared_nulls/execution.json).
The reports are archived with verified member checksums on `prc-completion-only`;
[result_index.json](../outputs/comparison_redetect/prc_shared_nulls/result_index.json)
gives the `reports.tar.gz` retrieval path and raw trace locations.
The frozen preparation plan is retained for provenance; rerunning it against the
aligned table is intentionally rejected. TextSeal native-8B replay is also complete.

## Published TextSeal results

`textseal_results.py` verifies the completed replay without launching compute.
`publish_textseal_comparison.mjs` appends six rows and common primary-test
columns to the shared CSV. Generic `TPR`/`FPR` use PRC posterior and TextSeal
weighted p-values; PRC entropy-aware results remain in their existing columns.
The publication checks preserve all original PRC cells and source provenance.
Full replay took 666.49 seconds, approximately $0.86 in measured resource time
before startup/image/storage overhead. See the [full summary](../outputs/comparison_redetect/textseal_setup/direct_prefix/full_summary.json).

## Reused SynthID and GumbelMax results

The shared CSV contains all 24 method/length rows. `reuse_token_baselines.py`
verified the 12,000 existing SynthID/Gumbel records against original generation
shards and the same T13088 nulls, then aggregated their saved decisions.
`publish_cached_baselines.mjs` appended 12 rows with the existing schema.
Neither detector uses prompt-conditioned entropy, so no redetection or model
replay was performed. All preceding PRC and TextSeal values are unchanged.
See the [reuse audit](../outputs/comparison_redetect/token_baseline_reuse/REVIEW.md).

## Repetition audit

[repetition_audit.json](../outputs/comparison_redetect/repetition_audit.json)
recomputes repetition and distinct-2/3 on the actual completion prefixes at all
six lengths for all four methods and the shared nulls. All 2,500 completions
match the source hashes, including PRC/TextSeal replay inputs; the recalculated
full-length metrics agree with all 24,000 historical detection records.
Historical prompt-level quality fields describe the full 1,024-token response
even when its detection prefix is shorter; the new audit explicitly truncates
tokens before measuring prefix quality.

At 1,024 tokens, mean repeated-token-4-gram fractions are 2.70% for PRC, 33.04%
for TextSeal, 2.58% for SynthID, 57.38% for GumbelMax, and 2.89% for the shared
nulls. Redetection reused generated text, so these metrics are unchanged.
TextSeal generation-time repeat handling remains disabled.

## Self-BLEU study controls (preparation only)

The experiment plan is [detectability_self_bleu_plan.md](../detectability_self_bleu_plan.md).
`self_bleu_reference.json` freezes the completed comparison at commit `4696382`,
including 30 source hashes and nine artifact/provenance records. Call
`self_bleu_config.verify_reference()` locally to verify the historical git blobs
and result records without changing them. The current study source is allowed
to differ; each new generation manifest records its own implementation hashes.

`StudySetting` configures online PRC eta/key seed, TextSeal alpha, or SynthID
depth. Its 30-key SynthID bank preserves the original first ten keys, then
uses the predeclared SHA256 domain in `self_bleu_config.py`. Generation and
evidence extraction must use the same key list. `pilot_settings()` returns the
five Stage A configurations; `pilot_settings("B")` and `pilot_settings("depth30")`
describe the later checks. Sampling seeds are 12345 and 67890, independent of
the fixed PRC key seed 12345.

`self_bleu_generation.generate_response_batch` wraps already-loaded models;
it does not load weights, dispatch Modal jobs, or write caches. For example,
once the existing generation runtime and original artifact have been loaded:

```python
from baseline_comparison.self_bleu_config import StudySetting
from baseline_comparison.self_bleu_generation import generate_response_batch

setting = StudySetting("online_prc", eta=.05, key_seed=12345)
second_response = generate_response_batch(
    we.model, prompts, prompt_indices,
    setting=setting, sampling_seed=67890, response_index=1,
    execution=actual_runtime_identity,  # record actual versions/device/precision
    prc_artifact=original_artifact,
    online_sampler=we.generate_batch_and_collect_online,
)
```

The wrapper verifies the artifact key against the setting and passes an
independent document seed into the existing sampler. It records partition,
key, ordered prompt hashes, execution, sample seed and response IDs under
`self_bleu_v1/<batch-hash>`. Existing `modal_run` artifact and cache behavior is
untouched. Ordinary sampling is supported as `StudySetting("null")`; baseline
methods use the existing `generate_method` with explicit alpha/keys.

For TextSeal detection, instantiate `TextSealCompletionDetector(model,
alpha=setting.alpha)` and call `detect_prefixes` on the raw saved completion
IDs. Preserve direct per-length replay. For SynthID, pass
`keys=setting.synthid_keys` to both `synthid_processor` and
`official_synthid_g_values`. The historical alpha .1 / depth 10 defaults remain.
Eligibility/repeat handling and detector formulas have not changed. Generation
diagnostics are explicitly separated from completion-only detection inputs.

Local checks require the existing numerical test environment, the pinned
TextSeal checkout (or installed package), and the pinned SynthID package:

```sh
python -m pip install --no-deps 'git+https://github.com/google-deepmind/synthid-text.git@addb4a158143c7c6851a1308f78b89fceed59683'
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 TEXTSEAL_SOURCE_ROOT=/path/to/pinned/textseal \
  python -m pytest tests/test_self_bleu_controls.py -q
```

These CPU checks exercise the original PRC sampler with a small deterministic
model, upstream parameter propagation, default generation versus the frozen
function, and identity separation. They are not production GPU validation.
The next step is equal-geometry GPU validation and an audit of reusable first
responses before any 50-prompt pilot dispatch. No paid job was launched by
this preparation.
