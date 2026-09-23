# TextSeal/Gumbel repeated-context fallback follow-ups

Completed on `comparison-with-redetect`, 2026-09-18 (America/Los_Angeles).
All **200 new full-length responses** completed: 100 TextSeal α=.1 and 100
Gumbel-Max, each using the same 50 prompts and two seeds as Stage A, now with
repeated-context fallback enabled. The TextSeal completion-only replay and paired
analysis also completed. The prior SynthID result is unchanged.

**Fallback substantially improved Gumbel's between-response Self-BLEU diversity.
For TextSeal, it reduced within-response repeats but increased Self-BLEU at
1,024 tokens.** These are different measurements: reducing repeated contexts
inside a response does not guarantee less overlap between two responses.
Neither result alone measures overall text quality.

## Paired primary results

Lower Self-BLEU means greater diversity between responses to the same prompt.
Differences are **modified fallback on minus original fallback off**. Intervals
use the frozen 2,000 paired prompt-cluster bootstrap draws, retaining two seeds
per prompt, with the unchanged SacreBLEU 2.4.3 symmetric sentence metric on the
study's 0–1 scale.

| Method | Tokens | Original off Self-BLEU | Modified on Self-BLEU | Paired change (95% interval) | Detections, off / on |
|---|---:|---:|---:|---:|---:|
| TextSeal α=.1 | 400 | 0.056320 | 0.058209 | +0.001888 [-0.002725, +0.006752] | 100/100 / 100/100 |
| TextSeal α=.1 | 1,024 | 0.037700 | 0.047270 | +0.009570 [+0.004021, +0.015278] | 100/100 / 100/100 |
| Gumbel-Max | 400 | 1.000000 | 0.393746 | -0.606254 [-0.660016, -0.551545] | 100/100 / 100/100 |
| Gumbel-Max | 1,024 | 1.000000 | 0.207184 | -0.792816 [-0.816566, -0.769031] | 100/100 / 100/100 |

Gumbel's reduction is large at both lengths. It now produces distinct response
pairs for all 50 prompts at both lengths; the native fixed-key Gumbel responses
were identical across seeds. TextSeal's 400-token change is small and uncertain;
its 1,024-token increase has a paired interval entirely above zero. Thus the
repeat fallback does not recover TextSeal's between-response diversity under
this metric, despite substantially reducing its within-response repetitions.

Detection remained 100/100 for both policies at both primary lengths. Detectors
and nominal p < .001 thresholds were unchanged: TextSeal uses fresh raw-completion
model forwards independently at each of six lengths, and Gumbel retains its
existing token-only score and tuple mask. This is not matched empirical FPR.
The cohort contains only 50 prompt clusters; all-success detection intervals do
not establish perfect population TPR. No new null calibration was performed.

## Repeat and fallback diagnostics

Counts refer to repeated generation contexts, with the native three-token
prompt suffix as the initial context and empty per-response history. Subsequent
occurrences count as repeats. These are distinct from detector masks. The original
policy never falls back; the modified policy falls back at each repeated context.
Every response encountered a repeat under both policies at both primary lengths.
Thus fallback fractions are 0% for the original and 100% for the modified policy.

| Method | Tokens | Policy | Responses with repeat | Total repeats | Total fallbacks | Mean repeats / response | Mean fallbacks / response |
|---|---:|---|---:|---:|---:|---:|---:|
| TextSeal α=.1 | 400 | Original off | 100/100 (100%) | 4,577 | 0 | 45.77 | 0.00 |
| TextSeal α=.1 | 400 | Modified on | 100/100 (100%) | 1,159 | 1,159 | 11.59 | 11.59 |
| TextSeal α=.1 | 1,024 | Original off | 100/100 (100%) | 34,671 | 0 | 346.71 | 0.00 |
| TextSeal α=.1 | 1,024 | Modified on | 100/100 (100%) | 5,486 | 5,486 | 54.86 | 54.86 |
| Gumbel-Max | 400 | Original off | 100/100 (100%) | 10,536 | 0 | 105.36 | 0.00 |
| Gumbel-Max | 400 | Modified on | 100/100 (100%) | 1,173 | 1,173 | 11.73 | 11.73 |
| Gumbel-Max | 1,024 | Original off | 100/100 (100%) | 53,900 | 0 | 539.00 | 0.00 |
| Gumbel-Max | 1,024 | Modified on | 100/100 (100%) | 5,582 | 5,582 | 55.82 | 55.82 |

Positions below are **zero-based generated-token indices**; null in the saved
records means no event. First fallback equals first repeat for modified responses
and is null for originals. First token divergence compares each modified response
to its original at the same prompt and seed; it is a property shared by both
members of that pair.

| Method | Tokens | Median first repeat: original | Median first repeat: modified | Pairs diverged | Median first divergence among diverged |
|---|---:|---:|---:|---:|---:|
| TextSeal α=.1 | 400 | 117 | 117 | 98/100 | 127.5 |
| TextSeal α=.1 | 1,024 | 117 | 117 | 100/100 | 130 |
| Gumbel-Max | 400 | 115.5 | 115.5 | 100/100 | 121 |
| Gumbel-Max | 1,024 | 115.5 | 115.5 | 100/100 | 121 |

All 200 pairs passed **no divergence before first repeat** over the full 1,024
tokens. TextSeal diverged exactly at that repeat in 63/100 pairs; Gumbel in
62/100. The remaining pairs diverged later. Equal first-repeat positions are
expected because the paired trajectories agree until the intervention can occur.

## Controls and reproducibility

The frozen request remains `setup_v4`, manifest
`8e762224a9f7d2207e7088a2f41a0a1a93de8012fa4e9b9d83df13a961e7d158`.
Same Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16/H100,
batch 50, prompt rows 0–49, sampling seeds 12345/67890, fixed keys, temperature 1,
top-p 1 and 1,024 generated tokens. The fallback uses the predeclared private
per-prompt ordinary-sampling RNG; native sampling still runs on every row so its
RNG consumption is unchanged. Historical results and generation code remain intact.

- Both methods passed H100 forced-repeat checks: first-occurrence behavior,
  ordinary fallback, native RNG isolation and repeat masks.
- All four native-policy controls exactly reproduced the saved 64-token prefixes:
  50 prompts × two seeds × two methods. Their recorded traces also verified the
  original-context reconstruction. This is prefix reproduction, not regeneration
  of every original full-length response.
- All 200 full modified generation traces matched independent context
  reconstruction. All causal checks passed, including repeat-history agreement
  through divergence and disabled original/enabled modified fallback behavior.
- TextSeal verified completion-only model inputs for all 100 responses and
  generated 600 direct per-length results. One response additionally passed the
  upstream public detector comparison at all six lengths.
- All 112 generation/control/replay files across the three ablation stages passed
  hash verification. The restored CPU environment passed 49 existing tests and
  verified the pinned upstream source bytes and tokenizer. The combined analysis
  reproduced both prior SynthID primary results exactly.

Original repeat traces are reconstructed from saved tokens and the prompt suffix;
modified traces were recorded during generation and independently reconstructed.
Prompts are used in this generation-context audit only; detector inputs remain
completion-only. The supplemental [trajectory script](check_followup_trajectories.py)
is kept with this experiment record so dispatched source hashes remain unchanged.

## Interpretation and next decision

Gumbel's deterministic Self-BLEU=1 point is sensitive to the generation policy:
ordinary fallback introduces seed-dependent randomness at repeated contexts.
Keep the native point and label this fallback-on variant explicitly. For TextSeal,
the native/off diversity result cannot be explained by omission of the fallback
in the simple direction hypothesized here: enabling it did not reduce Self-BLEU.
The earlier SynthID-off ablation likewise did not support fallback as the main
explanation for its between-response diversity.

At 1,024 tokens, the contextual comparison is PRC .01890 / 97 detections, native
SynthID .02206 / 100, fallback-on TextSeal .04727 / 100, and fallback-on Gumbel
.20718 / 100. The PRC–SynthID diversity difference was already small and uncertain;
these follow-ups do not establish a PRC advantage over SynthID or a full frontier
comparison. Native fallback-on SynthID stays in the main comparison. The common
fallback rule retains method-native context initialization and detector masks,
so it is a within-method policy ablation, not full implementation harmonization.

All three repeat-policy stages are now complete. The next decision is whether to
run the small parameter pilot; no parameter sweep, extra prompts or additional
key/model/null study was dispatched.

## Cost, jobs and saved artifacts

Generation: **253.377 seconds**, estimated resources
**$0.32723**. TextSeal replay:
**103.710 seconds**, **$0.13394**.
Combined follow-up estimate: **$0.46117**, no retries.
The cumulative planning charge is **$6.70584**, leaving
**$3.29416** of the initial $10. This includes prior
allowances and the repeat study's existing $0.50 overhead allowance, counted once;
it is not a settled Modal invoice. The overall study ceiling remains $200.

Dispatch commit: `fddedf74a2bea74bffeb622bb3ff3c23da824433`.
[Generation job](https://modal.com/apps/new-prc-watermark/main/ap-rkNFNR4vYYyUHgxaI1nyds) · [TextSeal replay job](https://modal.com/apps/new-prc-watermark/main/ap-RWqyijtdFEonU0tByj9GrN).

- [All paired primary results](all_analysis.json), including unchanged SynthID.
- [Both-policy per-response diagnostics](followup_response_diagnostics.json):
  200 pairs with counts, first events, divergence, response hashes and seeds.
- [Trajectory summaries/checks](followup_trajectory.json),
  [independent inspection](followup_inspection.json), and
  [preflight/runtime record](followup_preflight.json).
- [Generation report](other_generators_report.json),
  [TextSeal replay report](textseal_replay_report.json), and
  [execution/cost record](followup_execution.json).
- [Artifact index](artifact_index.json) and [earlier SynthID report](REPORT.md).

New full completions, native controls and detector traces remain in the existing
`prc-completion-only` Modal volume under this manifest's `other_generators` and
`textseal_replay` directories. Original responses remain in the unchanged Stage A
archive. Full repeat/fallback positions are in local ignored
`raw/{textseal_on,gumbel_on}_trajectory_pairs.json`, reproducible from saved tokens.
Retrieve and reproduce without GPU dispatch, using the pinned CPU dependencies:

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 MODAL_PROFILE=new-prc-watermark \
  python -m self_bleu.repeat analyze --setup outputs/self_bleu_repeat/setup_v4 \
  --stage all --tokenizer outputs/self_bleu_pilot/stage_a_v2/raw/tokenizer.json --download
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  python outputs/self_bleu_repeat/setup_v4/check_followup_trajectories.py
```
