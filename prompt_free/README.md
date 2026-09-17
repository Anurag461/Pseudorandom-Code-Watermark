# Prompt-free paper redetection

This is the canonical runner for future redetection. It uses **raw stored
completion token IDs, no prepended token, and score zero at coordinate 1**.
The historical `modal_run.py`, `modal_online_run.py`, and `completion_only`
diagnostic scripts are not this runner and are not imported or deployed by it.

## Protocol

- Detector: frozen Qwen3-0.6B-Base checkpoint, BF16, temperature 1, no filtering.
- Model inputs are exactly completion tokens 1 through T−1, at model positions
  0 through T−2. The next-token logits supply probabilities for original PRC
  coordinates 2 through T. The last completion token is scored; forwarding it
  would only produce an unused prediction for coordinate T+1.
- Coordinate 1 stays in the PRC vector and has score zero. No probability for
  that coordinate is fabricated. Every later score uses its own observed bit
  and response-only partition probability. Posterior mean is primary; entropy
  weighting is also available.
- Original candidate tokens, keys, partitions, OTP and parity supports are
  retained. No text or key generation is performed.
- Thresholds use the original formula with **V computed from the new scores**.
  Old prompted/EOT numerical thresholds are not accepted as production inputs.
- Fixed PRC preserves prefix-column scoring when T<n, and complete-block OR
  with FPR divided across blocks when T≥n. As in the original detector,
  incomplete trailing blocks are ignored. Only the first completion coordinate
  is zeroed; later block starts retain their normal scores.
- Online PRC preserves its one-shot or `alpha_spending_v1` FPR policy. Multiple
  listed lengths produce per-prefix results; they are not silently ORed into
  an anytime decision. Preserve the policy of the experiment being reproduced.
- Zero usable variance produces a negative decision with
  `status="insufficient_evidence"` and JSON `threshold=null`.

## Installation and local checks

Use Python 3.11 in an isolated environment. The pinned runtime matches the
completed BF16 pilots; it is separate from the older root `pyproject.toml`.

```sh
python -m pip install -r prompt_free/requirements.txt 'modal==1.5.1' pytest
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_prompt_free.py tests/test_online_prc.py tests/test_qwen_kv_cache.py -q
python -m prompt_free.manifest prompt_free/manifests/pilots.json
```

The last command validates and prints a plan locally. It does not access Modal,
load model weights or launch inference.

## Explicit execution stages

The included manifest pins the two verified pilot settings, with all 1,000
candidate-file and token hashes per setting:

| Case | Construction | Batch | Cache | Length | Target FPR |
|---|---|---:|---|---:|---:|
| `same_0p6b_n400` | Fixed block | 100 | concat | 400 | .001 |
| `cross_8b_0p6b_n640` | Online | 50 | static | 640 | .001 |

All stages require the execution sources and input manifest to match the
current Git commit. The Modal image includes only the explicitly listed source
files; untracked diagnostics are not deployed. Keep the branch's sources and
manifests committed before execution.

```sh
# CPU: verify all sources and create sanitized inputs. This is the default stage.
MODAL_PROFILE=new-prc-watermark python -m modal run --detach \
  -m prompt_free.modal_redetect --stage preflight

# GPU: one real batch per case, with full-length independent replay validation.
MODAL_PROFILE=new-prc-watermark python -m modal run --detach \
  -m prompt_free.modal_redetect --stage smoke

# Explicit full execution of every candidate in the selected manifest.
MODAL_PROFILE=new-prc-watermark python -m modal run --detach \
  -m prompt_free.modal_redetect --stage full
```

`--manifest path/to/committed.json` selects a different frozen campaign.
`--case same_0p6b_n400` selects one case. The bundled pilot manifest is not an
inventory of all paper experiments. Add the remaining paper cases with their
original artifact references, candidate hashes, target lengths and FPR policy
before launching the full paper campaign. The runner supports additional
lengths and generation models using the same audited 0.6B detector.

The manifest schema is enforced in `manifest.validate`. Each source reference
is `{volume, path, sha256, bytes}`, with `volume` equal to `archive` or `data`.
Each `tokens_sha256` hashes the original int64 completion prefix through the
largest requested length, in little-endian CPU storage order as in the pilots.
The order of `records` defines document batches. Lengths are replayed to their
maximum once and shorter prefixes are scored from the same causal trace.
The artifact supplies the existing fixed decoding key or online key.

## Batching, validation and resume

One warm A10 holds the 0.6B model. Each remote call processes one fixed document
batch, one token per step, with a fresh per-batch KV cache. Final partial batches
have explicit identities. No padding, automatic batch resizing, precision
fallback, or automatic retry is used. Additional sequence lengths require an
appropriate explicit batch size and a successful smoke check before full use.

The first uncached batch of each configuration is checked against independent
token-step replay, captured raw model inputs, prefix alignment, batch reversal,
causality and a GPU memory margin. Any failure stops that batch without saving
an accepted trace. Keys and scoring stay on CPU; GPU payloads contain exactly
`tokens` and `partition`.

Each completed batch is saved in `prc-completion-only` under
`completion_only_raw_abstain_v1/<case>/<run-id>/`. Source caches in `prc-data`
and `prc-research-archive` are read, never modified. A trace contains T−1
probabilities and a full identity covering protocol, checkpoint, code, source
tokens, ordered batch membership, batch size, cache and maximum length. Legacy
generation/EOT traces, changed inputs and incompatible batches are rejected.

Repeating a stage on the same commit and manifest validates and reuses completed
batches. GPU inference and CPU aggregation are separate; aggregation failure
does not discard recovered probabilities. The runner never reports a complete
result if a candidate or trace is missing. Local plans/results are written
under `outputs/prompt_free/`; historical manuscript tables are not overwritten.

## Audit and pilot regression

See [AUDIT.md](AUDIT.md) for the inspected paths and checks. The CPU regression
script compares both scoring methods on every candidate against the frozen
raw-completion pilots, including decisions, statistics, V and thresholds:

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m prompt_free.audit_pilots \
  --artifact-directory /path/to/hash-verified-original-artifacts \
  --results-directory outputs/completion_only
```

The artifact directory contains `same_0p6b_n400.pt` and
`cross_8b_0p6b_n640.pt`, obtained from the original references in `pilots.json`.
The prior pilot result directories must include `inputs.pt`, the trace shards
and `full.json`; the script verifies their frozen hashes. This check does not
run model inference. Raw candidates, keys, trace binaries and model weights
are not committed to Git.
