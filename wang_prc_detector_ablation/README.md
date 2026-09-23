# Wang-channel detector ablation on Qwen3-8B-Base

**Setup implemented; no paid compute launched.** This is a controlled detector
comparison on fresh Wang-channel generations from the cached Base model. It is
not a reproduction of DeepSeek Figure 5. The earlier saved-sweep preflight and
its missing-key findings are preserved in `../cryptoanalysis_redetect/`.

The user approved code setup with “go ahead and set up.” Each paid run still
requires a separate workload/cost quote and explicit approval under the repository
`AGENTS.md`. A setup approval, a quote file, or the $5 sanity ceiling is not paid
execution approval. No approval file with `approved: true` is supplied here.

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
cluster resampling and tied AUC. Full-size checkpoint, GPU numeric, artifact-score
and report validation remain unrun pending paid approval.

## Exact staged run commands

The entry point checks approval **before importing Modal or starting a build**.
It requires this branch, committed code pushed to `origin/cryptoanalysis-redetection`,
the frozen code fingerprint, an exact quote hash, recent billing review and an
unused approval identity. A failed launch consumes that approval; retries require
a new explicit approval. `cloud.py` is an implementation module, not a supported
direct-launch entry point.

```bash
export MODAL_PROFILE=new-prc-watermark

# Free/read-only: refresh billing and print the exact next-stage quote.
modal billing report --start 2026-09-22 --resolution h --show-resources --json
python -m wang_prc_detector_ablation.launch quote --stage sanity

# Only after the user approves THIS quoted package and an approval record exists:
python -m wang_prc_detector_ablation.launch launch --stage sanity \
  --approval wang_prc_detector_ablation/approvals/sanity.json

# Each subsequent command requires its OWN prior quote, budget review and approval.
python -m wang_prc_detector_ablation.launch quote --stage prepare
python -m wang_prc_detector_ablation.launch launch --stage prepare \
  --approval wang_prc_detector_ablation/approvals/prepare.json
python -m wang_prc_detector_ablation.launch quote --stage production
python -m wang_prc_detector_ablation.launch launch --stage production \
  --approval wang_prc_detector_ablation/approvals/production.json
python -m wang_prc_detector_ablation.launch quote --stage score
python -m wang_prc_detector_ablation.launch launch --stage score \
  --approval wang_prc_detector_ablation/approvals/score.json

# If explicitly included in the score-stage approval:
python -m wang_prc_detector_ablation.launch launch --stage score \
  --oracle-prompt-context --approval wang_prc_detector_ablation/approvals/score-oracle.json
```

An approval record needs `approved`, the actual `user_approval_text`, unique
`run_id`, `stage`, `fingerprint`, `git_commit`, `profile`, `quote_sha256` (the
canonical `config.digest_json(quote)`), `max_estimated_usd`, `billing_review`, and
`oracle_prompt_context`. Production also needs `sanity_status: "passed"` or
`"skipped_cost_over_5"`; the latter requires the actual
`skipped_sanity_quote_usd > 5`. Recording a skipped check never records a pass.
Populate these only after the corresponding user instruction. Approval records
are ignored by Git and copied into the run ledger.

The first sanity package uses a 4-core/16-GiB CPU worker to compare the original
hard detector with the adapter on all ten complete DeepSeek T=1.8 files × 16
outputs. It checks original printed violation counts, individual decisions and
saved aggregate successes. The GPU portion uses **one H100, 4 CPU cores, 64 GiB
host RAM**, loads the existing Base checkpoint once, and runs one full-size key,
two prompts, T=1.0 and 1.8, batch 2: four WM + four null 64-token completions.
That is 512 generated tokens, 504 completion-only replay positions, and two
bounded prefix comparisons (cached positions 1–16 and uncached lengths 1,4,8,16).
It verifies saved random variates/probabilities reproduce all branches and captures
every replay input. BF16 prefix checks record logit error and probability total
variation, requiring TV≤.02. These short sequences do not estimate detector TPR
or establish batch-80 throughput.

Expected sanity wall time is **10–25 minutes including startup**, at **$0.50–$1.50**.
CPU/GPU function timeouts are 600/900 seconds; their base-rate resource envelope
is about $1.22, excluding image/startup/storage. The estimate reserves additional
startup/build allowance but is not a provider-enforced dollar cap. **If the concrete
sanity quote exceeds $5, skip it as requested.** No replacement benchmark or paid
retry runs automatically. A failed correctness check stops scaling.

Preparation creates the full key/codeword inventory on 4 CPU cores/16 GiB.
Production uses up to five H100s, one per temperature, each 4 cores/64 GiB, batch
80: two WM and two null generation batches, each followed by completion-only replay.
Completed batches are committed to the volume. A worker stops after saving its
current batch if its projected total exceeds 3300 seconds; a failure cancels
the other calls. Hardware, counts and batch size are never silently changed.
Score/report runs on 8 CPU cores/16 GiB, with cross-key scoring in 16-text chunks.
All functions have zero retries and use single-use containers.

Planning allowance remains **roughly $12–$25 and 1–2 hours of cloud execution**
for the experiment, subject to the first real batches. The per-stage quotes are
estimates, not additive hard spending guarantees. H100 with 4 cores/64 GiB is
about $4.65/hour at [Modal's September 23 base rates](https://modal.com/pricing).
No premium region or non-preemptible option is selected. Stored data is expected
to be about 1–2 GiB; storage is separate ($0.09/GiB/month above the included tier).
Review costs and approvals between stages; stop if the remaining budget does not fit.

The last user-stated remaining budget is $35. `billing_review.json` records the
read-only refresh: $130.36659562 gross usage across other apps since September 22,
latest returned hour September 23 12:00 UTC. It is not a remaining balance and
may omit later charges. This experiment has spent **$0**. If $35 is still available,
the proposed sanity package leaves $33.50–$34.50; do not infer current credit balance
from this usage report.

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
sweep. See [PLAN.md](PLAN.md) and the authoritative [CURRENT_REQUEST.md](CURRENT_REQUEST.md).
