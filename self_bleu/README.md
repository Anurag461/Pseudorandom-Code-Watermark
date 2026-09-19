# Detectability versus diversity (Self-BLEU)

This package owns the PRC/TextSeal/SynthID/Gumbel-Max diversity study. Shared
generators, upstream adapters and completion-only detectors remain in
[`baseline_comparison/`](../baseline_comparison/README.md); the study imports
them rather than duplicating their implementations.

Start with the [experiment plan](plan.md) for the rationale and the
[repeat-handling runbook](repeat_handling_ablation.md) for the current ablation.
Stage A and all three repeat-ablation stages of `setup_v4` are complete. See the
[SynthID results](../outputs/self_bleu_repeat/setup_v4/REPORT.md) and
[TextSeal/Gumbel follow-ups](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md).
All commands below run from the repository root.

**Matched top-100 batch complete:** the [report](../outputs/self_bleu_topk/matched_v2/REPORT.md)
contains exactly 500 new responses, from the same 50 prompts and two seeds for
each of five settings. At 1,024 tokens, PRC-minus-depth-2 Self-BLEU is +.00095
[−.00503, +.00734], while detection is 85/100 versus 100/100. This does not
establish a PRC diversity advantage. Depth 30 has higher Self-BLEU than PRC;
PRC-minus-depth-10 is uncertain. PRC detects 33/100 at 400 tokens, versus 100/100
for all SynthID depths. Pilot nulls are 0/100 everywhere except depth 10 at
400 tokens (1/100). All paired PRC-minus-SynthID repetition intervals include zero.

Generation had zero support violations in 512,000 checked tokens. There were
zero contradictory replay bucket endpoints. At 1,024 tokens, outside-replay-top-100
rates are .719% PRC and .770% ordinary; early positions have higher rates. The
report separates all prefix/cohort/window counts and retains the original
detector. All 34 local tests, CUDA preflight and independent diagnostic/primary
interval checks passed. Cumulative planning charge: **$9.77418 of $200**.
The batch is complete; no additional generation is queued.

The [matched top-100 runbook](topk.md) freezes exactly
500 new responses: ordinary, PRC eta .05 and native SynthID depths 2/10/30.
The primary comparison is PRC minus depth-2 Self-BLEU at 1,024 tokens. The setup
is committed before GPU validation; successful validation gates generation.
BF16 model execution is preserved, with common FP32 probability arithmetic
for all arms and PRC replay. No historical response cache is compatible.
Stop after this batch; earlier optional sweep proposals do not authorize expansion.
The completed setup is `matched_v2`: it corrects replay mismatch counts to each
prefix and reports contradictory bucket endpoints separately, with early/later
positions and PRC/null cohorts separated. Version 1 completed validation only;
no responses were generated. Neither the detector nor generation probabilities
changed in this reporting revision.

The [consolidated paired comparison](../outputs/self_bleu_repeat/paired_comparison/REPORT.md)
combines direct PRC-minus-baseline Self-BLEU intervals, repetition metrics,
detection and existing null counts, with native and fallback-on policies shown
separately. At 1,024 tokens with fallback on, the Self-BLEU differences are
−.02837 [−.03531, −.02201] versus TextSeal, −.00316 [−.00731, +.00102] versus
SynthID, and −.18829 [−.21248, −.16445] versus Gumbel. The report preserves the
distinction between 100 clustered pilot null responses and 500 historical shared
nulls, including Gumbel's 2/500 historical false positives at 400 tokens.

**SynthID depth follow-up complete:** [depths 2/30 report](../outputs/self_bleu_depth/depth2_30_v1/REPORT.md).
The 200 new full-length responses retain native fallback, fixed keys, original
prompts/seeds and completion-only detection. Both depths detect 100/100 at 400
and 1,024 tokens. Depth 2 has no established Self-BLEU difference from PRC,
depth 10 or ordinary sampling; depth 30 has higher Self-BLEU than all three at
both lengths. Both depths have 0/100 pilot and 0/500 historical false positives
at both lengths. The report includes direct paired contrasts,
absolute intervals and verification. Cumulative planning charge: **$7.71132**
including allowances; the new worker resource estimate was **$0.50548**.

The [short-prefix detection follow-up](../outputs/self_bleu_depth/short_prefixes/REPORT.md)
scores the saved depths 2/10/30 at 64, 128 and 256 tokens. Depth 2 detects
45/100, 82/100 and 100/100; depths 10 and 30 detect 100/100 at all three lengths.
The deeper-minus-depth-2 paired TPR differences are +55 pp [45, 65] at 64 and
+18 pp [11, 25] at 128. The report includes both saved null cohorts and their
nonzero false-positive counts. All 6,300 scores passed direct-prefix and
independent score checks. This analysis cost no Modal credit. The proposed
depth-20 generation was cancelled before launch; its unfinished edits were reverted.

| Files | Responsibility |
|---|---|
| `config.py`, `reference.json` | Fixed settings, keys and historical reference identity. |
| `generation.py` | Generate fixed-key response pairs using shared samplers. |
| `validation.py`, `validation_modal.py` | Prepare/collect validation locally; explicitly dispatch its GPU stages. |
| `pilot.py`, `pilot_modal.py` | Prepare, collect and analyze Stage A; dispatch missing detection replay. |
| `repeat.py`, `repeat_modal.py` | Prepare/analyze repeat-policy contrasts; explicitly dispatch each GPU stage. |
| `depth.py`, `depth_modal.py` | Run the requested SynthID depths 2/30 with native fallback and compare saved depth-10/PRC/null pairs. |

Local workflows use subcommands, for example `python -m self_bleu.pilot --help`.
Only the `*_modal.py` entrypoints dispatch GPUs. Saved data stays in the existing
`outputs/self_bleu_validation/`, `outputs/self_bleu_pilot/`, and
`outputs/self_bleu_repeat/` namespaces so historical manifests and archives keep
their original paths.

## Study controls and bounded validation

The experiment plan is [plan.md](plan.md).
`reference.json` freezes the completed comparison at commit `4696382`,
including 30 source hashes and nine artifact/provenance records. Call
`self_bleu.config.verify_reference()` locally to verify the historical git blobs
and result records without changing them. The current study source is allowed
to differ; each new generation manifest records its own implementation hashes.

`StudySetting` configures online PRC eta/key seed, TextSeal alpha, or SynthID
depth. Its 30-key SynthID bank preserves the original first ten keys, then
uses the predeclared SHA256 domain in `config.py`. Generation and
evidence extraction must use the same key list. `pilot_settings()` returns the
five Stage A configurations; `pilot_settings("B")` and `pilot_settings("depth30")`
describe the later checks. Sampling seeds are 12345 and 67890, independent of
the fixed PRC key seed 12345.

`self_bleu.generation.generate_response_batch` wraps already-loaded models;
it does not load weights, dispatch Modal jobs, or write caches. For example,
once the existing generation runtime and original artifact have been loaded:

```python
from self_bleu.config import StudySetting
from self_bleu.generation import generate_response_batch

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
`validation.py` verifies historical source bytes and freezes the
step-3 request. `validation_modal.py` explicitly dispatches either
generation validation or the subsequent TextSeal replay check; neither stage
dispatches the pilot or full sweep. Use the `new-prc-watermark` Modal profile.

```sh
python -m self_bleu.validation prepare --cache /path/to/downloaded/prc-data --output outputs/self_bleu_validation/new-run
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.validation_modal --manifest outputs/self_bleu_validation/new-run/manifest.json --stage generation
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.validation_modal --manifest outputs/self_bleu_validation/new-run/manifest.json --stage textseal
```

Generation checks the same 50-prompt batch at 1,024 tokens, seeds 12345/67890
and a replay of 12345, with a fixed key for each configuration. It includes
TextSeal alpha zero as a deterministic control and short parameter checks for
alpha .5 and SynthID depths 2/20/30. Saved first responses are compared by exact
token hashes against historical caches; mismatches require the new pair.
PRC replay observes every actual input and compares an independent token-step
reference. The separate HF worker checks seven fresh TextSeal/null records
against upstream direct-prefix detection, including alpha zero and .5.

The workers use existing offline weights and separate paths on
`prc-completion-only/self_bleu_validation/<manifest-id>`. Successful generation
batches are saved immediately for reuse. Reports and batch files are immutable;
retrieve an existing run rather than redispatching it. One H100 per stage,
no retries, 3,000/600-second timeouts and explicit resource reservations keep
validation within the initial $10 allocation. Timing-based resource estimates
exclude image/startup/storage overhead and are not exact billing totals.

**Status: step 3 completed.** See the [validation report](../outputs/self_bleu_validation/step3-v4/REPORT.md).
All six full-length replicate controls, four short parameter checks, PRC
completion-only checks and seven TextSeal direct-prefix parity checks passed.
The 20 local tests pass. The artifact collector verified 27 files containing
600 full-length and 200 short response records. Stage A subsequently reused
those pairs for the completed analysis below, with no new generation.

The initial generation run saved every pair before a BF16-to-NumPy conversion
failed in a diagnostic assertion. The `prc-replay-repair` stage recovered that
check from verified batches without regenerating responses. Its setup embeds
the original manifest and permits changes only to validation infrastructure.
The original generation source and keys are unchanged. The successful combined
request is `outputs/self_bleu_validation/step3-v4/manifest.json`.

Retrieve and verify the completed artifacts without GPU dispatch:

```sh
MODAL_PROFILE=new-prc-watermark python -m self_bleu.validation collect \
  --setup outputs/self_bleu_validation/step3-v4 \
  --raw outputs/self_bleu_validation/raw/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8 \
  --download
```

Omit `--download` to check an existing local copy. The collector verifies file
hashes, batch/response identities, fixed-key seed pairs, actual historical
token matches and every saved SynthID official score-update check.

**Status: Stage A analysis complete.** See the [pilot report](../outputs/self_bleu_pilot/stage_a_v2/REPORT.md).
PRC preserves ordinary-sampling diversity and detects 97/100 at 1,024 tokens,
but its Self-BLEU difference from default SynthID is small and uncertain.
The evidence does not justify immediate expansion to the full sweep. Through
Stage A, the cumulative planning charge, including conservative failure/overhead
allowances, was $5.51880 of the initial $10; this is not a settled invoice.

`pilot.py` freezes clean completion-only replay requests from the
verified pairs and imports 53 compatible TextSeal records. Its two Modal stages
recover 200 PRC traces and 147 missing TextSeal records; neither generates text.
`pilot.py` also computes symmetric sentence Self-BLEU and official
keyed token evidence, verifies the 153 replay files, joins scores by response
identity, computes paired prompt-bootstrap intervals, and renders the figure.

The completed request is explicitly **`stage_a_v2`**. Version 1 failed in a
verification call before saving usable PRC batches; version 2 changes that
argument to a tensor and accounts for the failed attempt. Inputs, keys and
analysis choices are identical. Do not redispatch either completed GPU stage.
Restore the raw analysis files from the archive recorded in
`outputs/self_bleu_pilot/stage_a_v2/archive.json` when using a fresh checkout.
That record gives the local path and the explicitly authorized Modal archive
location, with its checksum and transfer-verification status.
Extract its relative `outputs/` paths at the repository root. It includes the
step-3 raw generation batches and historical score/input files used below.
Then verify/reproduce locally (SacreBLEU 2.4.3, pinned SynthID package, existing
numerical environment and model tokenizer required):

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m self_bleu.pilot diversity \
  --setup outputs/self_bleu_pilot/stage_a_v2 --tokenizer /path/to/pinned/tokenizer.json
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m self_bleu.pilot token-evidence \
  --setup outputs/self_bleu_pilot/stage_a_v2
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m self_bleu.pilot summarize \
  --setup outputs/self_bleu_pilot/stage_a_v2 --output outputs/self_bleu_pilot/reanalysis
```

Artifacts are immutable: identical reruns verify existing bytes, while changed
runtime versions or results require a separate output location. The pinned
analysis source and original analysis before a figure-layout repair are retained
inside the archive. No Stage B generation was launched.

**Repeat-handling ablation: all stages complete.** The pilot used Google's repeated-context
generation fallback for SynthID and no corresponding fallback for TextSeal or
Gumbel. The [runbook](repeat_handling_ablation.md) freezes a generation-only
ablation before the parameter sweep: SynthID depth 10 with fallback off first,
then TextSeal alpha .1 and Gumbel with fallback on. Each adds 50 prompts × two
responses, and the original-policy pairs are reused. PRC is unchanged.

The first stage generated all 100 SynthID-off responses and passed the H100
forced-repeat checks, both native 64-token prefix controls, and every paired
full-trajectory check. At 1,024 tokens, mean repeated-context counts rose from
45.53 to 114.21 per response, but Self-BLEU changed by only −.00067 (95% paired
interval −.00421 to +.00309), with 100/100 detections under both policies.
The 400-token difference was also small and uncertain. This does not support
fallback as the main explanation for SynthID's between-response diversity on
this cohort. Native fallback-on SynthID remains the main comparison.
The [report](../outputs/self_bleu_repeat/setup_v4/REPORT.md) links both-policy
per-response diagnostics, checksums and retrieval instructions. The cumulative
planning charge is now **$6.24467 of $10**, including the new $0.22587 measured
worker estimate and $0.50 overhead allowance. That was the balance after SynthID;
the completed follow-ups below supersede it.

TextSeal/Gumbel fallback-on added 100 responses each, followed by 100 completion-only
TextSeal replays. At 1,024 tokens, Gumbel Self-BLEU fell from 1.00000 to .20718
(paired change −.79282, 95% interval −.81657 to −.76903). TextSeal Self-BLEU rose
from .03770 to .04727 (+.00957, interval +.00402 to +.01528), despite mean repeated
contexts falling from 346.71 to 54.86. Gumbel repeats fell from 539.00 to 55.82.
Both methods still detected 100/100 at 400 and 1,024 tokens. All 200 pairs passed
full-trajectory and native-prefix checks. These results distinguish repetition
inside responses from pairwise Self-BLEU diversity.

The [follow-up report](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md)
records both endpoints, per-response diagnostics, unchanged detector settings,
and reproducibility. The cumulative planning charge is now **$6.70584 of $10**;
the follow-ups added $0.46117 in estimated worker resources with no retries.
All repeat-policy stages were complete at this point. The separately requested
depths 2/30 follow-up above is now complete; no broader parameter sweep has run.

The [matched-policy repetition analysis](../outputs/self_bleu_repeat/matched_repetition/REPORT.md)
recomputes the earlier repeated-token-4-gram fraction and distinct-3 metrics
from saved completions. With fallback on for all contextual baselines, 1,024-token
repeated-4-gram rates are 2.38% PRC, 2.42% SynthID, 2.74% TextSeal and 2.81%
Gumbel, versus 2.92% ordinary sampling. The large earlier within-response
repetition gap shrinks substantially; paired PRC differences for both metrics
include zero at both lengths. This is distinct from between-response Self-BLEU.
The report includes paired intervals, on/off sensitivity and supplemental medians.
The analysis reproduced all 1,000 historical prompt-level metric values and
incurred no additional Modal cost.

`repeat.py` scopes the policy adapters to a single generation call;
historical generator code and defaults are unchanged. Setup and CPU analysis
share that module; explicit GPU stages remain in
`repeat_modal.py`. The current frozen request
is `outputs/self_bleu_repeat/setup_v4/manifest.json`. Preparing it dispatches
nothing. Each GPU stage has its own timeout, no retries and source/runtime
checks; generation also has forced-repeat checks and historical-prefix gates.
The analysis preserves detector settings and uses paired prompt contrasts
against Stage A.

## Source history and immutable results

The study moved out of `baseline_comparison/` into this package after its local
workflows were consolidated. Original completed manifests, scores, archives and
source snapshots remain unchanged. Their pre-consolidation source tree is
retained at commit `7bde5c6d54dee444db3b69d96bfce6b09c79ba4c`; the consolidated
layout before this move is at `7d275d710d4c7b473c9bb1304e3c26bc7cac968a`.
Historical readers verify the recorded source hashes against the original Git
bytes when files have moved. Workers require matching **current** source files.

The current repeat request is `setup_v4`, with unchanged experimental settings,
input data, upstream sources and budget. It adds full-trajectory repeat/fallback
diagnostics and a no-divergence-before-first-repeat check. Its manifest links to the superseded
`setup_v3`; earlier setups remain historical records and must not be dispatched
with the current code. Reanalysis writes a summary and figure into a separate
output directory instead of replacing the original pilot report.

SynthID scoring in `baseline_comparison.comparison_runner._score_baseline_sequence`
requires explicit `synthid_keys=setting.synthid_keys`, including for nulls. Both
full and prefix evidence use those keys; metadata records the actual key list,
depth and key domain. Omitting keys fails before scoring. The historical tuple
mask is unchanged; Stage A's primary analysis uses Google's context mask.
