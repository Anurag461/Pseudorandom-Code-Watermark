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

Call `detect` separately for every requested prefix. Prefix-trace reuse,
batching, chunking, and alternative model backends require later target-device
validation and are not implemented in this reference adapter.

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
Transformers 4.51.3: 38 tests passed across TextSeal completion/preflight,
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
metadata were verified; weight bytes were neither loaded nor rehashed.

No real-checkpoint replay or remote compute job has run for this setup.
Next are pinned-model/device validation and a measured pilot, then the full
comparison integration. The [redetection plan](../textseal_prompt_free_redetection_plan.md)
specifies PRC reuse, remaining integration, and the provisional $5–15 core-repair
budget with a $25 planning ceiling. Existing full-score orchestration still
calls retired paths and must be migrated before use; this setup does not claim
to have rebuilt the final comparison tables.

## Cached PRC prefix comparison

`prc_prefix_comparison.py` rescores the published native-8B online PRC n=1024
cohort to the existing comparison grid: 128, 256, 400, 512, 768, and 1024 tokens.
It calls the integrated completion-only scorer on the original cached traces,
keys and partition, and saves a separate
`outputs/comparison_redetect/prc_prefix_comparison.csv` with old/new posterior
and entropy-weighted TPRs, new FPRs, and percentage-point changes.

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m baseline_comparison.prc_prefix_comparison \
  --generation-cache /tmp/comparison-redetect-cache
```

The generation cache is the local `prc-data` export produced by the source
preflight. The indexed native-8B run must also be available locally at the
location in `outputs/redetection/cache_index.json`, including its prepared
manifest, artifact, inputs and traces. No remote call or inference is made.
Use `--lengths` to request another grid through 1024; the script always includes
1024 as an exact regression check against all saved scores and old decisions.

Each row is a separate one-shot test at nominal FPR 0.001, retaining the
original online prefix supports and coordinate-1 abstention. This is not an
OR decision across lengths. The null cohort is the published n=1024 cohort
(T1382 source); it differs from the full TextSeal comparison's T13088 nulls.
The adjacent provenance JSON records input/output hashes and validation results.
Detailed per-candidate scores remain in a separate local archive namespace;
the original redetection reports and summary CSV are unchanged by this script.
