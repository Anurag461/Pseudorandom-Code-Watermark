# Wang PRC detector ablation on Qwen3-8B-Base

**Approved setup plan — September 23, 2026; Base model restored, reasoning off.**

Use the same Wang-style PRC-watermarked completions and true keys to answer two
questions separately:

1. How does our standard prompt-free posterior detector compare with Wang's
   published hard detector, each using its specified threshold?
2. Does posterior scoring improve on the hard statistic when both are calibrated
   to the same nominal FPR, supported by threshold-independent ROC/AUC?

**Code setup approved** by “go ahead and set up.” The pipeline, small synthetic
checks, documentation and staged approval gate are implemented. The first paid
sanity attempt passed the CPU source check but failed the BF16 prefix tolerance.
See evidence/sanity-20260923/README.md for that attempt. The separately approved
saved-token numerical diagnostic subsequently passed its cache/FP32 controls; see
evidence/numerical-20260923/README.md. The original 2% BF16 guard remains failed,
and the remaining short smoke and production are not authorized.
Paid compute still requires separate approval for each run.

[CURRENT_REQUEST.md](CURRENT_REQUEST.md) is authoritative for the latest model,
threshold, first-token and sanity-cost instructions. [REQUEST.md](REQUEST.md)
provides compatible earlier scope. The official-Qwen-model correction is revoked
and retained only as `QWEN_CORRECTION_SUPERSEDED.md`; it must not drive execution.

## 1. Frozen scope

| Item | Setting |
| --- | --- |
| Model | Cached `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Inference | Existing BF16 Qwen backend and static KV cache; base variant, reasoning/instruct modes off |
| Generation prompts | Wang's exact 16 strings through our existing Base-model prompt/tokenization path |
| PRC | Wang fixed KeyGen/Encode/channel, t=3, eta=0.1, r=floor(0.95n) |
| Token bits | Derive ceil(log2(actual model vocabulary)); expect 18 |
| Length | Exactly 1024 generated tokens; n=18432 and r=17510 when token_bits=18 |
| Temperatures | 1.0, 1.2, 1.4, 1.6, 1.8 |
| Watermark keys / codewords | 10 keys; one noisy codeword per key/prompt, reused across temperatures |
| Full-length samples | 800 watermarked + 800 null completions |
| Main table | Wang published method versus posterior standard method, with realized FPRs |
| Separate ablation table | Empirically calibrated hard statistic versus posterior statistic at nominal FPR 0.001 |
| Null pools | 256 independent calibration keys and 256 disjoint evaluation keys |
| Initial sanity package | Estimated $0.50–$1.50; skip if its concrete quote exceeds $5 |
| Overall compute estimate | $12–$25 before credits; not an approved spending cap |
| Time estimate | 2–4 hours setup; 1–2 hours cloud execution, excluding approval/queue delays |
| Stop | Report the ten-key experiment; no automatic scaling or additional experiments |

There is no official `Qwen/Qwen3-8B` download or full-length two-group T=1.8
validation gate in this restored design. Existing Base-model results remain
separate from this new Wang-channel experiment. Do not describe this as exact
reproduction of Wang's model, prompting, Figure 5 or cryptanalytic ablation.

## 2. Source, model and generation implementation

Use Wang's official artifact commit
`8593e86aeb50b5f82d6c88e390b12a30f581dbaa` as the source for `llm_prc_api.py` and
the PRC/sampler parts of `llm/generation/main.py`. Preserve attribution and
reference implementations. Match KeyGen's default generator width, row supports,
permutation, OTP, Encode, MSB-first token bits, hierarchical sample_token and hard
Detect. Do not use our online-PRC schedule or random half-vocabulary partition.

Verify the cached Base checkpoint against the full model entry in the main
repository's `outputs/online_8b_redetect_setup/proposed_plan.json`. Its tokenizer
SHA-256 is `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
Freeze all weight-shard, index and tokenizer hashes with the code. Existing model
configuration expects vocabulary 151936; read and validate the actual loaded size.
No replacement model download is authorized by this plan.

Reasoning off means selecting the Base model and its existing raw prompt path,
not loading the non-Base chat model with an `enable_thinking=False` option. Do not
introduce a chat template, reasoning prefix or new system message for generation.
Save exact prompt IDs and verify them against the established Base tokenizer path.

Generation uses the existing batched, incremental model with KV caching. Scale
logits by each sample's temperature, top_k=0 and top_p=1. Ignore EOS stopping and
save all 1024 original token IDs, including generated special tokens.

For every token, consume token_bits consecutive noisy codeword bits. Starting
with [0,2^token_bits), use conditional upper-half mass within the current interval,
apply Wang's exact binary rule and follow the selected branch. Regions outside
vocabulary have zero mass. Use cumulative masses with a positive-sum tree fallback for tail cancellation, and stable
float32/float64 probability arithmetic; preserve the mathematical channel and
record numerical dtype choices. Validate direct-sum and cumulative implementations
with common input probabilities and fixed uniform variates, including endpoint
and small-mass cases. No unconditional bit-marginal substitution is allowed.

Use domain-separated deterministic RNG streams for keys, codeword payload/noise,
watermarked generation and null generation. Derive sample streams from source,
key/seed-group, prompt and temperature. Freeze seeds before validation. Identical
random streams are independent of scheduling; floating-point batch-shape effects
must be tested and recorded separately. Never reroll keys/seeds based on outcomes.

Save observed-path generation probabilities for both WM and null text. Ordinary
null sampling uses the full LM categorical distribution, then extracts conditional
probabilities along the observed token's path without another LM call. Nulls have
no latent codeword; mark that field absent rather than fabricate one.

## 3. Prompt-free replay and first-token handling

Replay every completion from a fresh model cache using raw preceding completion
IDs only, at its generation temperature. Neither main detector receives original
prompt IDs, prepended BOS/EOT, chat-template tokens, or generation-time probabilities.
Generated special tokens already present in a completion remain in the sequence.

- **Wang hard:** recover every token's bits, including token 1. This requires no
  LM context. Preserve the complete published hard rule.
- **Actual posterior:** set all 18 soft coordinates belonging to token 1 to zero.
  For every later token, reconstruct the observed-path conditional probabilities
  from preceding completion IDs only. This applies to both the standard method
  and the matched-FPR posterior ablation.
- **Optional oracle diagnostic:** retain generation-time probabilities and, if
  scored, allow token 1 to contribute normally because its probabilities exist.
  Label `oracle-context diagnostic`; keep it out of both comparison tables and
  main figures. It cannot support the headline prompt-free claim.

Totals excluding prompts and the short sanity test:
1,638,400 generated tokens and 1,636,800 prompt-free replay prediction positions.
Each prediction yields 18 branch probabilities without 18 transformer calls.
The revised thresholds add CPU work only; they require no further model passes.

## 4. Main comparison: fixed method definitions

### Wang hard detector (published threshold)

Let `V_hard=sum_w XOR_{j in w}(observed_bit_j XOR OTP_j)`. Declare watermarked
using exactly **`V_hard <= r/2-r**0.75`**, including equality. At r=17510 this
cutoff is approximately 7232.824913450762. Store the full-precision cutoff and
integer violation count; do not round the cutoff before deciding.

For ROC, also retain the increasing score `(r-2*V_hard)/sqrt(r)`, equivalent in
ordering to `-V_hard`. The underlying hard-sign statistic matches the principle
of our naive detector on identical bits, supports and OTP, but do not replace
Wang's published cutoff with our naive detector's cutoff.

### Posterior detector (standard threshold, FPR target 1e-3)

Apply `detectors.py::map_soft_token` unchanged to observed hierarchical bits and
prompt-free branch probabilities. Let `u_w=product_{j in w}(s_j)` and
`a_w=product_{j in w}(1-2*OTP_j)`. Save `S=sum_w(a_w*u_w)` and `V=sum_w(u_w**2)`.
Use exactly **`S >= sqrt(2*V*log(1000))`**, including equality, with no empirical
retuning. Store each sample's S-space threshold.

For V>0, canonical normalized score **`Z=S/sqrt(V)`** has standard cutoff
`sqrt(2*log(1000))`, approximately 3.71692218885. Do not substitute the former
`S/sqrt(V+1e-12)` into the standard decision: the latest exact rule supersedes that
choice. An epsilon-stabilized value may be saved as an explicitly separate
numerical diagnostic, never used silently for decisions/calibration.

For **V=0**, follow our existing completion-only wrapper: abstain/reject, rather
than letting `0 >= 0` produce a detection. Record `no_evidence=true`, false decision,
and an explicitly undefined normalized score. For ranking it may be represented
internally by negative infinity; keep its JSON/CSV serialization unambiguous.
Both standard and matched posterior decisions must reject no-evidence records.

The main table and figure compare these **actual methods at their actual operating
points**, with TPR CIs and held-out realized FPR for both. Do not claim that their
FPRs are equal, and do not rename a recalibrated statistic as the published method.

## 5. Separate matched-FPR ablation: calibration and whole ties

Retain 160 null completions per temperature. Designate null seed groups 0–4 as
calibration and 5–9 as evaluation, each with all 16 prompts: 80 nulls per split
per temperature, 400 per split pooled across temperatures. The ten WM key groups
are all used for TPR evaluation; no watermarked outcome selects a cutoff.

Generate 256 independent calibration decoding keys and a disjoint set of 256
evaluation keys, with Wang's exact key distribution. Neither set contains WM keys.
Score each calibration null against every calibration key: **102400 pairings**.
Use the held-out keys only with held-out nulls: another **102400 pairings**, or
20480 per temperature. Cache traces once; cross-key scoring is CPU-only.

Choose one global cutoff per statistic, pooled across temperatures, permitting at
most `floor(0.001*N)` calibration detections (102 when N=102400):

- **Wang hard statistic @ matched FPR:** choose the most permissive integer c_hard
  with `count(V_hard <= c_hard)/N <= 0.001`. Sort whole integer score levels in
  increasing order. Find the first level h whose inclusion would exceed the
  budget, and use **c_hard=h-1**. This includes all permissible levels without
  splitting ties; c_hard=-1 is allowed when rejecting everything is necessary.
- **Posterior statistic @ matched FPR:** use canonical Z, with inclusive
  **`Z >= c_post`**. Sort whole levels in decreasing order. Find the first score
  level z whose inclusion would exceed the budget. Use the least representable
  float64 cutoff strictly above that excluded level, **`nextafter(z,+inf)`**.
  This excludes its complete tie group while retaining permitted higher scores.
  Record the boundary score, its tie count, accepted count and exact cutoff.
  If all records have no evidence, record an explicit reject-all rule.

Do not randomly split ties; do not replace inclusive comparisons with strict
`>` decisions or an unspecified percentile convention. Tiny synthetic tests must
cover exact cutoff equality, repeated levels, gaps, no allowable positives and
no-evidence cases. Raw S alone is not interchangeable with normalized Z when V
varies across samples.

Write and hash `threshold_calibration.json` before evaluating watermarked detector
outcomes. Then report matched-FPR TPR and held-out FPR in a **separate table and
figure**. Empirical calibration targets 0.001; it does not guarantee identical
realized FPRs. Neither cutoff may be tuned separately by temperature or retuned
using WM/evaluation data.

## 6. Uncertainty, ROC and interpretation

Use 2000 seeded bootstrap replicates over the crossed 10 WM keys and 16 prompts,
resampling the two factors and applying identical indices across detectors and
temperatures. Report TPR CIs and paired detector differences for both comparisons.
Matched-FPR TPR intervals are conditional on the frozen calibration thresholds.

For null FPR, resample completion/seed-group and prompt clusters, carrying the
entire cross-key score row together; never bootstrap individual text/key pairs
as independent observations. Label inference conditional on the frozen held-out
key pool. Use paired identities across temperatures for pooled resampling.

Compute ROC/AUC from raw hard and posterior scores using held-out nulls only;
give each held-out completion equal total weight across its key pairings. Show
T=1.0, 1.2 and 1.4 prominently. Record any no-evidence posterior scores separately
rather than dropping those samples.

Report next-token entropy, hierarchical binary entropy, latent-bit agreement,
score distributions and relationships between these mechanisms and detection.
Keep three conclusions distinct:

1. Better standard-method TPR than Wang's published method is an operating-point
   comparison; inspect realized FPR before characterizing overall performance.
2. Better matched-FPR results and ROC support an advantage in posterior scoring.
3. A gain only over the published cutoff indicates threshold choice may explain
   the improvement. Neither an oracle result nor the primary TPR comparison alone
   establishes a more informative prompt-free statistic.

No outcome-dependent tuning, automatic extra experiments or broader model claims.

## 7. Setup deliverables and initial sanity-cost rule

After plan approval, implement an isolated Wang adapter/reference, efficient
hierarchical sampler, existing-Base generation/replay adapters, CPU scoring and
reporting, frozen manifests, cache validation, tests, and separately invoked Modal
stages. Work in `/private/tmp/prc-cryptoanalysis-redetection`, branch
`cryptoanalysis-redetection`; do not modify the other session's checkout/caches.
Commit and push the executable setup before any paid launch.

Locally run only syntax checks and small synthetic tests, including reference
sampler/bit/OTP tests, coordinate-1 abstention, no-evidence handling, threshold
equality/ties and split leakage. Do not run models, dataset scoring or heavy
analysis on the laptop, or start a billable image build without approval.

The proposed **initial paid sanity package** is:

1. CPU source-convention validation on the released complete T=1.8 artifact:
   ten groups × 16 WM outputs. Compare original and adapted hard statistics and
   saved decision summaries at their actual granularity. These are DeepSeek
   samples used only to validate PRC/OTP conventions, with no LM execution.
2. One H100 short smoke using one full-size key, first two Wang prompts, 64 tokens,
   and T=1.0 and 1.8: four WM and four null completions, **512 generated tokens**
   and **504 primary replay positions**, plus bounded cached/uncached prefix
   comparisons. Verify valid IDs, latent alignment, saved randomness/probabilities,
   prompt-free model input capture, and first-token abstention. A short prefix
   is not a full-length detector TPR estimate.

**Estimated combined sanity cost: $0.50–$1.50. The user ceiling is $5.** Before
launch, count CPU/GPU/runtime startup charges in one concrete package estimate.
If that estimate exceeds $5, **skip the paid initial sanity package as requested**,
record it as skipped (not passed), and do not launch a replacement benchmark.
Mandatory inexpensive synthetic checks still run locally. Production would need
its own explicit approval with the skipped-validation status clearly disclosed.
If sanity is run and finds a correctness failure, do not scale incorrect code;
fix it and request approval for any paid retry. No automatic retries are allowed.

No full-length two-group T=1.8 gate, Attack-I/II run, or non-Base model check is
part of this restored scope. Ordinary key/manifest preparation remains necessary
for production even if the optional paid sanity package is skipped.

## 8. Execution, time and cost estimate

| Stage | Proposed resources and workload | Estimated cost |
| --- | --- | --- |
| A: CPU preparation / source check | One 4-core / 16-GiB worker; verified sources, 10 WM keys, codewords, 512 null-scoring keys and manifests; released-data check if sanity retained | $0.10–$0.40 |
| B: Short GPU sanity | One H100, 4 CPU cores / 64 GiB host RAM; eight 64-token samples, replay and bounded reference checks | $0.40–$1.10 |
| C: Generation + prompt-free replay | Up to five H100s, one per temperature, each with 4 CPU cores / 64 GiB host RAM; provisional batch 80; two WM and two null batches per temperature, then corresponding replay batches; reuse model loads | Generation $5–$11; replay $4–$10 |
| D: CPU score/report | One 8-core / 16-GiB worker; all standard and matched decisions, bounded cross-key scoring, bootstrap, ROC and figures | $0.50–$2 |

**Planning allowance: $12–$25 total; 1–2 hours cloud execution**, plus approximately
**2–4 hours setup/testing**. Aggregate H100 allocation is provisionally around
2–5 GPU-hours. The restored cached Base model removes the new checkpoint download
and extra full-length validation. New table/threshold rules are CPU-only and add
no generation/replay. Estimates exclude approval delays, extraordinary retries
and any explicitly quoted storage/premium charges; they are not spending approval.

Rates checked September 23 at [Modal pricing](https://modal.com/pricing): H100
$0.001097/s, CPU $0.0000131/core/s, host memory $0.00000222/GiB/s. A worker at the
proposed allocation is approximately $4.65/hour including CPU/memory at base rates.
Avoid premium region/non-preemptible settings. Confirm additional charges before
launch. Historic Base references include modal_online_8b_runbook.md (204800 tokens
in 516.99 H100 method-seconds at batch 50), shorter batch-125 generation and native
8B replay evidence. New hierarchical-sampler throughput remains unmeasured.

Batch 80 remains provisional pending the setup's memory checks. A two-prompt
smoke does not measure its throughput precisely. Observe the first real production
batch within the approved workload, and stop for a revised quote if projections
no longer fit. Do not silently change hardware, sample counts or model. Use free
capacity without interrupting other sessions. Save completed samples atomically,
verify identities on reuse, set retries=0 and terminate workers promptly.

The latest stated budget is **$35**; if still available, this estimate leaves
**$10–$23**. A read-only refresh on September 23 at 21:30 UTC returned $130.36659562
gross usage since September 22 across other apps, with its latest returned
interval at September 23 12:00 UTC. That is neither a remaining balance nor this
experiment's cost and may omit later charges. Reconcile spending before every
paid proposal. Other experiments' approvals do not authorize these runs.

## 9. Outputs, approvals and stop conditions

Use `wang_prc_detector_ablation/qwen3_8b_base/<config-fingerprint>/`, separate from
all old online/fixed caches. Save original token IDs, prompt IDs, decoded strings,
keys/OTPs/supports, generator information, latent codewords, RNG provenance,
generation and primary replay probabilities, entropy/agreement diagnostics,
precision, model/tokenizer/source hashes, config, trace schemas and git commit.
Oracle inputs remain separate from primary detector inputs. Cache all required
traces so detector reruns and plots are CPU-only.

Deliver:

- `primary_methods_summary.csv`: the requested main five-temperature table,
  published Wang and standard posterior TPRs/CIs and realized held-out FPRs/CIs.
- `matched_fpr_summary.csv`: the separate five-temperature statistic-ablation
  table with both recalibrated TPRs/CIs and held-out FPRs/CIs.
- `results_summary.csv`: long form, one row per temperature/detector with explicit
  comparison role, exact label, unique WM N, threshold/decision rule, TPR/CI,
  realized FPR/CI and AUC. Distinguish fixed Z thresholds from sample-specific
  S thresholds. Do not mix oracle rows into primary or matched tables.
- `per_sample_results.csv`: all primary/matched scores and decisions, hard count,
  S/V/Z, no-evidence flags, sample/key/split identities and entropy/agreement.
  Identify repeated null-key pairings explicitly; N always counts unique texts.
- `threshold_calibration.json`: frozen null splits/keys, whole-tie boundary
  details, inclusive decision rules, exact cutoffs/counts and freeze hash/time.
- `tpr_vs_temperature.pdf/png` for actual methods;
  `tpr_matched_fpr_vs_temperature.pdf/png` for the separate ablation;
  `roc_low_temperature.pdf/png` and `score_vs_entropy.pdf/png`.
- Optional `oracle_diagnostic.csv`, clearly labeled, first-token-included and
  excluded from both main comparisons; no additional LM pass.
- Frozen config, sample/key inventory, validation-or-skip report, timings,
  GPU-hours, provider cost ledger and README with exact commands/commit.

| Approval / gate | Current status |
| --- | --- |
| Plan and code setup | Approved; implemented and locally checked |
| Paid CPU preparation / sanity source check | Short sanity source check approved and passed; full preparation not approved |
| Paid GPU sanity | First attempt approved; stopped at prefix TV 0.0220 > 0.02; no retry authorized |
| Paid production | Not requested or approved |
| Paid CPU score/report | Not requested or approved |

Before each new paid run, state workload, hardware, batching, time, cost and
remaining-budget effect and wait for explicit approval. The $5 skip rule does
not by itself authorize spending below $5. No automatic retries or extra
validation. If a sanity stage is skipped, report that limitation honestly.

Completion means all 160 WM and 80 held-out null texts per temperature are
accounted for; primary thresholds were never retuned; secondary cutoffs were
frozen using calibration nulls alone; paired/clustered analysis follows this
plan; caches support CPU-only reruns; costs and discrepancies are reported.
Stop after the initial ten-key result and distinguish actual-method performance
from matched-FPR scoring quality in every conclusion.
