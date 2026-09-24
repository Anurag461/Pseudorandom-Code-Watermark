# Wang-channel detector ablation on Qwen3-8B-Base

**Full experiment completed September 24, 2026.** See the
[final report](evidence/experiment-20260924/README.md),
[results CSV](evidence/experiment-20260924/results_summary.csv), and
[matched-FPR figure](evidence/experiment-20260924/tpr_matched_fpr_vs_temperature.pdf).

**Main finding: at T=1.2, prompt-free MAP raises detection from 0% to 97.5% on
the same Qwen3-8B-Base outputs under the prescribed thresholds.** After both
detectors are calibrated to target FPR 0.1%, hard reaches 89.38% and MAP reaches
98.13%, a paired gain of 8.75 percentage points (95% CI 2.50–16.88).

At T=1.0, calibrated MAP gains 11.25 points but absolute detection remains only
11.25%. At T=1.4 and above, both calibrated detectors reach 100%. Standard MAP's
pooled observed FPR is **0.0039%, not zero** (4/102400 null/key pairings from
400 distinct held-out null texts). Equal calibration targets do not imply equal
realized FPR. The [main takeaways](evidence/experiment-20260924/README.md#main-takeaways)
and [comparison with Wang's results](evidence/experiment-20260924/README.md#comparison-with-wangs-original-results)
separate threshold effects, scoring gains, and model differences.

All 1600 generation/replay traces are saved. The run took 41.28 minutes and cost
approximately $4.48 from observed runtimes; provider billing is pending.

This is the requested controlled detector comparison on fresh Wang-channel
samples from the cached Base model. It is not a DeepSeek Figure 5 replication.
The [original source check](evidence/sanity-20260923/README.md) passed all 160
individual detector comparisons. Its BF16 prefix check failed; subsequent
[numerical controls](evidence/numerical-20260923/README.md) passed exact cache and
FP32-reference comparisons. The full run used those completed controls, with the
BF16 discrepancy preserved as a limitation. No extra short smoke was run.
The earlier saved-sweep preflight is preserved in `../cryptoanalysis_redetect/`.

All seven approved preparation/GPU/scoring calls finished, with no retries.
The app has stopped. Any further paid work requires a new explicit approval.

## Frozen experiment

- `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`,
  existing BF16 `qwen.py` static-cache backend. Checkpoint hashes are in
  `model_manifest.json`. All model reads are cache-only; no download fallback.
- Raw Base prompt tokenization, no chat or reasoning prefix. Sixteen strings
  exactly extracted from the pinned Wang source (`prompts.json`).
- 10 keys × 16 prompts × 5 temperatures (1.0, 1.2, 1.4, 1.6, 1.8): 800 watermarked
  and 800 ordinary null texts, each forced to exactly 1024 completion tokens.
- Wang fixed PRC: t=3, eta=.1, n=18432, r=17510, default generator width
  `floor(log2(comb(n,t)))`. Token width is derived and validated against the
  model vocabulary (151936 → 18 bits). One noisy codeword per key/prompt is reused
  across all temperatures. Key permutations, OTPs, generator matrices, payloads
  and noise are saved.
- Sampling applies Wang's binary rule to each **conditional** hierarchical branch,
  MSB first. Full-vocabulary probabilities: FP32 log-softmax, FP64 exponentiation,
  normalization and sums. Prefix-sum interval queries have a positive-mass tree
  fallback for cancellation in tiny tails. This avoids replacing conditional
  probabilities with bit marginals. Ordinary null generation walks the same tree
  with unmodified branch probabilities; this is full categorical sampling.
- Seeds are fixed, domain-separated SHA-256 derivations into NumPy PCG64. Uniform
  streams are independent of scheduling. Floating-point model results can vary
  with batch shape; batch identities and software/device metadata are recorded.

## Detector definitions and analysis

**Main comparison:** Wang's published inclusive `H <= r/2-r**.75` versus the
standard posterior inclusive `S >= sqrt(2*V*log(1000))`. Wang uses every recovered
bit. The posterior imports `detectors.py::map_soft_token` unchanged and abstains
on **all 18 coordinates of completion token 0**. For token i≥1, the fresh model
cache has received exactly raw completion IDs `0:i`; no prompt, BOS, EOT or chat
template is prepended. Generated special tokens already in the completion remain.

Posterior row products are corrected by OTP signs. Save S, V and canonical
`Z=S/sqrt(V)` without an epsilon. V=0 rejects/abstains; CSV Z is blank, internal
ranking Z is negative infinity. Main decisions use the S-space rule exactly.

**Secondary matched-FPR ablation:** null source groups 0–4 calibrate, groups 5–9
evaluate, identically at all temperatures. Each of 400 calibration texts is scored
against each of 256 independent calibration keys; evaluation uses 400 held-out
texts and a disjoint set of 256 keys. Neither pool uses WM keys. Both detectors
use the same 102400 calibration pairs pooled across temperatures, allowing at
most 102 detections. Hard uses the most permissive integer cutoff including whole
ties. Posterior uses float64 `nextafter(excluded_boundary, +inf)`, with inclusive
`>=` decisions and whole ties. Threshold JSON and its content hash are written
before WM scoring; an incompatible refreeze is rejected.

All 160 WM texts per temperature use their corresponding true key. TPR intervals
and paired differences use 2000 crossed key-group × prompt bootstrap replicates,
with the same resampling across detectors/temperatures. FPR resampling carries
all 256 keys for each null completion together and resamples seed-group × prompt;
inference is conditional on the frozen evaluation-key pool and frozen thresholds.
ROC/AUC uses held-out nulls only and equal total weight per null text. Cross-key
pairs never count as independent text observations.

`--oracle-prompt-context` is an optional **score-stage diagnostic** using saved
generation probabilities, including token 0. It requires no additional LM pass
and is excluded from both comparison tables and all main figures. It reports
standard-threshold oracle scores only, without an additional calibration exercise.

## Local checks (tiny synthetic data only)

From the isolated worktree:

```bash
cd /private/tmp/prc-cryptoanalysis-redetection
NUMBA_CACHE_DIR=/private/tmp/prc-wang-numba-tests python -m pytest -q \
  wang_prc_detector_ablation/tests cryptoanalysis_redetect/test_preflight.py
python -m compileall -q wang_prc_detector_ablation
python -m wang_prc_detector_ablation.launch quote --stage sanity
```

The tests use no pretrained model and do not score the released dataset. They
compare tiny keys/encoding against the vendored original, test direct-sum versus
fast sampling under identical variates, bit ordering, OTP/permutation conventions,
tail masses, abstention, no-evidence handling, inclusive ties, deterministic stream
separation, corrupted caches, prompt-free input capture with canned logits,
cluster resampling and tied AUC. The source and checkpoint checks have since passed in the first approved attempt;
its GPU prefix check failed. The full experiment subsequently completed using the
separately passed cache/FP32 controls; the generated figures were visually checked.

## Run commands and approvals

The completed experiment used one preparation call, five H100 workers (batch 80),
and one CPU scoring call. Its exact command, source commit, cache location,
measured runtime and cost estimate are in the
[final report](evidence/experiment-20260924/README.md#execution-and-costs).

For any new paid run, first refresh billing and obtain a workload/cost quote:

```bash
export MODAL_PROFILE=new-prc-watermark
modal billing report --start 2026-09-24 --resolution h --show-resources --json
python -m wang_prc_detector_ablation.launch quote --stage experiment

# Only after explicit approval of the quoted workload and current budget:
python -m wang_prc_detector_ablation.launch launch --stage experiment \
  --approval wang_prc_detector_ablation/approvals/new-experiment.json
```

`launch.py` checks approval before importing Modal or starting a build. It requires
committed code pushed to `origin/cryptoanalysis-redetection`, an exact quote hash,
a recent billing review and an unused approval identity. A failed launch consumes
that approval; retries and additional stages require new approval. The quote's
$35 scenario is the historical budget assumption, not a current balance.
Operational billing records are not inputs to the experiment's scientific results.

Approval records are local and ignored by Git. Required fields are `approved`,
`user_approval_text`, unique `run_id`, `stage`, `fingerprint`, `git_commit`,
`profile`, `quote_sha256`, `max_estimated_usd`, `billing_review`, and
`oracle_prompt_context`. The `experiment` package also requires the disclosed
`validation_policy` and complete `included_sequence` from the quote. Populate
these only after approval of that exact workload. `cloud.py` is not a supported
direct-launch entry point.

The initial sanity package is skipped if its concrete quote exceeds $5; skipping
does not record a pass. Numerical limitations of the completed run remain in the
validation summaries. No additional paid work is scheduled.

## Outputs and reruns

All new files go to the existing `prc-data` volume under
`wang_prc_detector_ablation/qwen3_8b_base/<fingerprint>/`; the short sanity cache
has its own `sanity/` subdirectory. Nothing writes into another experiment's cache.
The root appears in every quote, local receipt and run ledger. To download
completed outputs without launching a worker, substitute that root:

```bash
modal volume get prc-data \
  wang_prc_detector_ablation/qwen3_8b_base/<fingerprint>/results_summary.csv \
  wang_prc_detector_ablation/downloaded_results/results_summary.csv
```

`traces/*.npz` contain original tokens, recovered bits, raw prompt IDs (generation
provenance only), FP64 generation/replay branch probabilities and per-token entropy.
Replay token-0 entries are NaN, not fabricated probabilities. Adjacent JSON carries
sample identity, decoded text, model/code/batch metadata, key/codeword bindings and
file hashes. `keys/` and `codewords/` retain the actual secret material. Atomic
paired writes reject corrupt, incomplete or incompatible records; review any
partial write before an approved retry. Reuse is permitted only for exact identities.

The CPU stage produces:

- `primary_methods_summary.csv`, `matched_fpr_summary.csv`, `results_summary.csv`;
- `per_sample_results.csv` (800 true-key WM rows and 204800 explicit null/key rows),
  `paired_differences.csv`, `mechanism_summary.csv`, `interpretation.md`;
- `threshold_calibration.json`, cached cross-key score matrices, manifests and hashes;
- `tpr_vs_temperature`, `tpr_matched_fpr_vs_temperature`, `roc_low_temperature`,
  `score_vs_entropy`, each PDF and PNG;
- optional `oracle_diagnostic.csv`, separate from all main comparisons;
- runtime/GPU-hour estimates and a run ledger that explicitly leaves provider cost
  unset until the billing report is reconciled. Never label an estimated cost actual.

Rerunning the score stage on the same immutable setup reuses all LM traces and
cached null score matrices; it needs CPU approval but no generation/replay. No
model, dataset scoring or full analysis should run on the laptop.

## Source attribution and limits

`vendor/main.py`, `vendor/llm_prc_api.py` and `vendor/LICENSE` are unchanged copies
from [Wang's official artifact at 8593e86](https://github.com/1234wangtr/PRC_estimator/tree/8593e86aeb50b5f82d6c88e390b12a30f581dbaa).
Do not import `vendor/main.py`: it contains the original model-loading script.
The adapter uses explicitly seeded random streams with the same distribution;
it cannot reproduce unrecorded historical random draws. The short source check
validates conventions on the complete T=1.8 archive, not the incomplete Figure 5
sweep. The implemented design is documented above and in the
[final report](evidence/experiment-20260924/README.md). Superseded requests,
planning documents and verbose operational records are omitted from the PR;
historical copies remain in commit `a991c67`.
