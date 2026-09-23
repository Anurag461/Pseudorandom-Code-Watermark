# Repeat-handling ablation

Prepared 2026-09-18 on `comparison-with-redetect`. **All three stages completed
and analyzed.** See the [SynthID results](../outputs/self_bleu_repeat/setup_v4/REPORT.md)
and [TextSeal/Gumbel follow-ups](../outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md).
This experiment precedes the alpha/depth sweep. It isolates
the effect of generation-time fallback; it does not attribute authors' intent
or claim that repeat handling explains the observed diversity without data.

## Frozen comparison

| Arm | New full-length responses | Existing comparator |
|---|---:|---|
| SynthID depth 10, fallback off | 50 prompts × 2 seeds = 100 | Stage A SynthID, fallback on |
| TextSeal alpha .1, fallback on | 100 | Stage A TextSeal, fallback off |
| Gumbel-Max, fallback on | 100 | Stage A deterministic Gumbel, fallback off |

Use Qwen3-8B-Base, the same pinned weights/tokenizer, BF16/H100, batch 50,
prompt rows 0–49 in their original order, temperature 1, top-p 1, fixed keys,
seeds 12345/67890, and 1,024 generated tokens. Primary endpoints remain 400 and
1,024; detect at the existing six lengths. The former Gumbel point remains a
valid fallback-off reference. Fallback-on Gumbel requires two real responses;
its determinism must not be assumed. PRC and ordinary-sampling references are
reused; no PRC change or new null generation is required.

TextSeal's paper discloses omitting repeated-context fallback from its main
evaluation, while Google's released SynthID processor includes it. This is a
methodological difference, not evidence of intent. [TextSeal Remark 1](https://arxiv.org/html/2605.12456v2#S2.Thmremark1),
[pinned Google processor](https://github.com/google-deepmind/synthid-text/blob/addb4a158143c7c6851a1308f78b89fceed59683/src/synthid_text/logits_processing.py#L283).

## What changes and what stays fixed

SynthID-off uses the exact upstream call and captures its score update before
the fallback selection. It returns those scores instead of the ordinary scores
on repeated contexts. Keys, hashing, tournament updates, tensor geometry, state
updates and multinomial sampling remain the same. It does not switch to
TextSeal's different SynthID implementation. Fallback-on must exactly reproduce
the original Google call in forced-repeat probes.

TextSeal/Gumbel-on maintain an exact set of three-token contexts per response.
On later occurrences, sample from the ordinary full-vocabulary distribution.
Native sampling still runs for every row, so TextSeal's original key-routing
stream consumes the same number of draws. Fallback draws use a separate
per-prompt generator seeded from SHA256 of a predeclared domain, sampling seed
and prompt index. They cannot advance the original sampler's RNG or another
response's fallback RNG. The fallback history resets between responses.

Native initialization is preserved: Google's processor begins with a zero
context and warms up over the first three generated tokens; TextSeal/Gumbel
initially use the prompt's last three tokens. The repeated-context rule is
shared, but this is not a fully harmonized implementation comparison. Changing
initialization simultaneously would confound the within-method intervention.
SynthID retains its 1,024-entry history; the experiment does not exceed it.

**Detection is frozen within each method.** Apply completion-only raw-token
detection with the same formulas, masks and nominal p < .001 threshold as Stage
A. SynthID retains the official context mask; TextSeal/Gumbel retain their
existing tuple masks. Do not also change detector deduplication when estimating
the generation-policy effect. A detector-mask harmonization study would be a
separate experiment. Generated entropy/log-probability traces and fallback
flags are diagnostics only; TextSeal detection gets fresh direct-prefix model
forwards, and token-only detectors get raw completion IDs only.

## Validation and analysis

Before full-length generation, verify source hashes, upstream sources, offline
checkpoint identity and the original generation runtime. Force repeated
contexts through the real samplers on H100; check on/off SynthID scores,
ordinary fallback, first-occurrence behavior and the native RNG stream. Then
reproduce the saved original-policy **64-token prefixes for all 50 prompts at
both seeds**. Any mismatch stops the arm. This is a prefix reproduction gate,
not a new claim of full-length cache equivalence; the unchanged historical
source bytes remain verified in Git; current worker hashes and the original
runtime are also required.

Save each completed 50-response batch immediately under
`prc-completion-only/self_bleu_repeat/<manifest-id>/`. Never overwrite historical
data. A started-stage marker blocks redispatch after a partial failure; inspect
the artifacts and account for the failure before preparing a retry. Automatic
retries are disabled. Later stages require a successful prior-stage report and
verified hashes. Model-dependent replay is separate from generation.

Use the original SacreBLEU 2.4.3 symmetric sentence metric and 2,000 paired
prompt-bootstrap draws. Report **new policy minus original policy** separately
for each method and primary length, alongside absolute Self-BLEU and TPR.
Record both original and modified responses: repeat/fallback counts and positions,
fractions encountering either event, first repeat/fallback position, and first
token divergence. Positions are zero-based completion-token indices; null means
the event never occurs. Reconstruct original SynthID traces with upstream context
hashing, its zero-context warmup and zero-filled history. Verify reconstruction
against the native GPU control traces and full modified generation traces.
Require no token divergence before the first repeated context across all 1,024
tokens; a reference response with no repeat must remain identical throughout.
Save diagnostics before stopping if any causal/trace check fails. The generation
repeat mask is distinct from the detector mask. Rare activations can change
the subsequent trajectory; trigger counts do not by themselves measure impact.
Retain the existing FPR limitations and do not claim matched empirical FPR or
perfect detection from degenerate all-success bootstrap intervals.

The CPU suite exercises native-policy reproduction, actual upstream forced
repeats, response-local histories, private RNG streams, factory cleanup on
failure, immutable policy identities, budget rejection and paired contrasts.
The released TextSeal CPU PRF supports the test's single-row path; its batched
CUDA helper also passed the completed H100 preflight. CPU checks alone are not
8B/H100 validation. The setup's verification records describe the executed checks.

## Run sequence

All commands run from the repository root. Use the pinned numerical environment
and TextSeal/SynthID sources described in the [study README](README.md) and
[shared runtime setup](../baseline_comparison/README.md#reproducible-setup-and-checks).
Set `TEXTSEAL_SOURCE_ROOT` when TextSeal is a checkout rather than an installed package.
The supplied `setup_v4` manifest is frozen and should be used directly. It
replaces the unrun `setup_v3` to pin the added trajectory checks;
generation settings, reference data and budget are unchanged. Earlier
manifests remain for provenance. Historical source bytes are verified against
Git; dispatch verifies the current source files strictly.
For a new request, prepare locally in a fresh output directory (preparation
records the current source commit and refuses to overwrite a different request):

```sh
python -m self_bleu.repeat prepare --output outputs/self_bleu_repeat/new-setup
```

The first paid stage below **already completed**. Do not redispatch it; its
immutable started marker rejects a second run. The analysis command can retrieve
and reproduce the saved results without GPU generation:

```sh
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.repeat_modal \
  --setup outputs/self_bleu_repeat/setup_v4 --stage synthid
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 MODAL_PROFILE=new-prc-watermark \
  python -m self_bleu.repeat analyze --setup outputs/self_bleu_repeat/setup_v4 \
  --stage synthid --tokenizer /path/to/pinned/tokenizer.json --download
```

The follow-up stages below also **completed** after inspection of the SynthID
result and user authorization. These are historical dispatch commands; do not
redispatch completed workers. The final analysis command retrieves and reproduces
saved results without GPU generation:

```sh
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.repeat_modal \
  --setup outputs/self_bleu_repeat/setup_v4 --stage other_generators
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.repeat_modal \
  --setup outputs/self_bleu_repeat/setup_v4 --stage textseal_replay
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 MODAL_PROFILE=new-prc-watermark \
  python -m self_bleu.repeat analyze --setup outputs/self_bleu_repeat/setup_v4 \
  --stage all --tokenizer /path/to/pinned/tokenizer.json --download
```

The analysis creates `synthid_analysis.json` or `all_analysis.json`,
`synthid_trajectory.json`, and `synthid_response_diagnostics.json`, plus raw
prompt metrics, scores and generation-policy diagnostics in the ignored `raw/`
directory. Reuse Stage A's restored archive and original generation artifacts.
The analysis checks their frozen hashes before joining responses.

The completed TextSeal/Gumbel full-trajectory audit is reproduced with
`PYTHONPATH=. python outputs/self_bleu_repeat/setup_v4/check_followup_trajectories.py`.
This experiment-specific CPU script leaves the dispatched source bytes unchanged.
It reconstructs both policies' generation contexts, checks the native prefixes and
full modified traces, and writes `followup_trajectory.json` and
`followup_response_diagnostics.json`. It never supplies prompts to detectors.

## Budget and decisions

One H100, four CPU cores and 64 GiB per worker. Timeout reservations: 600 seconds
for SynthID, 900 for TextSeal/Gumbel generation, 300 for TextSeal replay, plus
two seconds per worker for scale-down. At the verified [Modal rates](https://modal.com/pricing)
on 2026-09-18, this reserves **$2.33241** of worker resources and **$0.50** of
additional overhead. Combined with Stage A's **$5.51880** planning charge, the
total is **$8.35121 of the initial $10**. This is a conservative planning
reservation, not new incurred spending or a guaranteed invoice cap. The full
study ceiling remains $200.

**Spending estimate after SynthID-off:** 174.895 worker seconds cost
$0.22587 in estimated resources. Adding the predeclared $0.50 overhead allowance
to the prior planning charge gives **$6.24467**, leaving **$3.75533** of the initial
$10 at that point. The timeout reservation above describes the original maximum
plan, not additional measured spending.

**Current balance after all stages:** TextSeal/Gumbel generation took 253.377
seconds ($0.32723 estimated resources), and TextSeal replay took 103.710 seconds
($0.13394). With no retries, these follow-ups added $0.46117. The cumulative
planning charge is **$6.70584**, leaving **$3.29416** of the initial $10. The
existing $0.50 overhead allowance is counted once. These are estimates, not a
settled invoice; the overall study ceiling remains $200.

If SynthID-off loses diversity, quantify the paired effect and inspect whether
TextSeal/Gumbel-on recover it without losing useful detection. If it does not,
the repeat handler is not supported as the main explanation in this cohort.
Either outcome is useful. Reassess the parameter sweep after the results;
do not silently replace the original pilot or select the repeat policy that
makes PRC look best.

The authorized SynthID-off stage and its paired inspection are complete.
Repeated contexts increased substantially, but Self-BLEU changes at 400 and
1,024 tokens were small and uncertain and TPR stayed 100/100. This does not
support fallback as the main explanation for SynthID's between-response
diversity on this cohort. Native fallback-on SynthID remains the main comparison.
The TextSeal/Gumbel-on follow-ups are now complete too. Gumbel Self-BLEU at
1,024 tokens improved from 1.00000 to .20718; TextSeal rose from .03770 to .04727
despite substantially fewer within-response repeats. Both retained 100/100
detections at both primary lengths. All 200 new pairs passed full-trajectory
checks. Keep both native and modified policies clearly labeled. The next decision
is the optional parameter pilot; no parameter sweep has been dispatched.
