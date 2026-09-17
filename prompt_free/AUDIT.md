# Prompt-free redetection implementation audit

Audit date: September 17, 2026. Branch based on main revision
`78ade0865f2f1f06327f9498c8cc5b9cf366249c`.

Scope: the dedicated `prompt_free` runner, its committed-source/manifest gate,
input extraction, model input alignment, posterior/entropy scoring, original
fixed/online FPR policies, trace provenance, batching, resume and aggregation.
This does not certify the legacy generation/EOT experiment runners as prompt-free.

## Findings resolved

1. **Pilot-only coupling.** The prior diagnostic runner depended on historical
   EOT and prompted-control results. The new entrypoint has no imports from
   `completion_only`, `watermark_expt`, `modal_run` or `modal_online_run`.
   Production input manifests cannot supply prompts or old probability traces.
2. **First-coordinate alignment.** The GPU loop forwards stored completion
   token j and records its probability for coordinate j+1. It produces exactly
   T−1 probabilities. A zero is placed in score coordinate 1, with no token or
   PRC-index deletion and no synthetic initial probability.
3. **Multiple blocks.** Abstention applies once at the start of the completion.
   Later complete PRC blocks use their original indices and ordinary scores.
   Original block-OR Bonferroni correction and trailing-block behavior remain.
4. **Zero evidence.** The legacy `prc.Detect` can accept a full block with V=0
   because both statistic and threshold are zero. Abstention can create this
   case. The new wrapper returns a negative, insufficient-evidence result.
   Nonzero-evidence pilot decisions are preserved. Online scoring retains its
   existing 1e−15 variance tolerance.
5. **Numerical threshold reuse.** Production scoring accepts no saved numeric
   cutoffs. It applies the original formula using the newly recovered scores
   and the manifest's original FPR policy. Fixed-cutoff pilot sensitivities
   remain diagnostic results only.
6. **Cache confusion.** Cache identity includes raw protocol, model revision
   and hashes, code identity, token hashes, ordered batch membership, actual
   and requested batch size, cache implementation and maximum length. Legacy
   `p_trace` payloads and incompatible/corrupt checkpoints are rejected.
7. **Deployment provenance.** The execution files and selected manifest must
   match Git HEAD. The image explicitly contains only listed source files;
   untracked diagnostic code is excluded. Default remote execution is CPU
   preflight. Full execution requires `--stage full`.
8. **Modal class registration.** Postponed string annotations were incompatible
   with the installed Modal class-parameter serializer. The entrypoint uses a
   concrete `str` annotation and has a no-deployment import test.
9. **Parallel batch execution.** The explicit `--workers` option is bounded to
   1–10 A10 containers. It changes scheduling only: fixed batch membership,
   independent KV caches and distinct atomic shard writes remain unchanged.
   Each stage records the worker limit locally.
10. **Shared full validation.** One safe batch certifies each numerical
    configuration. Workers reuse a hash-verified certificate and retain all
    input, probability and memory checks. Prior-run validation is accepted only
    after its trace, source commit, numerical file hashes, model-loading AST and
    raw replay statements are checked. Configuration mismatches and incomplete
    validation records are rejected. A10 and A10G device labels identify the
    same supported A10 GPU family; other GPU families are rejected.

## Validation

The automated suite covers actual model input IDs, next-coordinate alignment,
fresh caches, causal prefixes, batch ordering, first-coordinate abstention,
short fixed prefixes, multiple blocks and ignored tails, both online FPR
policies, zero evidence, source corruption, legacy caches, exact resume,
missing-candidate rejection, committed-source guards and Modal registration.
It uses a small real Qwen implementation for CPU inference and temporary source
caches for an end-to-end test of preparation, batching and aggregation.

Run:

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_prompt_free.py tests/test_online_prc.py tests/test_qwen_kv_cache.py -q
```

The full cached-pilot regression separately checks **2,000 candidates and 4,000
method/candidate decisions** against the completed raw-completion BF16 runs.
Every decision matches exactly. Largest absolute discrepancies among float64
statistics, V and thresholds are 1.14e−13 (n400) and 5.69e−14 (n640), within the
explicit 1e−12 absolute allowance for NumPy/CPU host rounding. This allowance
does not apply to GPU probability-recovery validation, which requires bitwise
agreement with independent token-step replay.

| Pilot | Posterior-mean TP / 500 | Entropy-weighted TP / 500 | FP / 500, both methods |
|---|---:|---:|---:|
| 0.6B → 0.6B, n400 | 430 | 355 | 0 |
| 8B → 0.6B, n640 | 364 | 300 | 0 |

[audit_results.json](audit_results.json) records source hashes and regression
evidence. [pilot_regression.json](manifests/pilot_regression.json) pins prior
result hashes. Actual candidate texts, keys and trace binaries remain outside
Git in the original and new results volumes.

## Execution boundary

The manifests contain the two pilot settings and the n3104 eta=0.2 case, not every
paper experiment. The common runner supports additional fixed and online
cases, prefixes and multiple blocks. Remaining paper cases must be frozen in
reviewed, committed manifests before their campaign is started.

The production runner establishes GPU validation once per numerical batch
configuration, then shares that evidence across workers. Every batch retains
its memory margin check. A new length/batch configuration is not numerically certified
merely because the two pilots passed. Batches are never silently resized or
converted to another precision on failure. The full paper campaign has not
been launched as part of preparing this branch.
