# SynthID repeated-context fallback ablation

Completed on `comparison-with-redetect`, 2026-09-18 (America/Los_Angeles).
All **100 new 1,024-token responses** completed: the same 50 prompts and two
sampling seeds as Stage A. Disabling fallback increased repeated contexts
within responses, but produced small, uncertain changes in between-response
Self-BLEU and no observed change in detection at the primary endpoints.
This ablation does **not** support fallback as the main explanation for
SynthID's diversity on this cohort. Native fallback-on SynthID remains the
main-comparison configuration.

## Paired primary results

Lower Self-BLEU means more diversity between the two responses to a prompt.
Differences are **fallback off minus native fallback on**. Intervals use 2,000
paired prompt-cluster bootstrap draws over 50 prompts, retaining both seeds
within each sampled prompt. SacreBLEU 2.4.3 and the original symmetric sentence
metric are unchanged; values below use the study's 0–1 scale.

| Tokens | Native on Self-BLEU | Off Self-BLEU | Paired difference (95% interval) | Detections, on / off |
|---|---:|---:|---:|---:|
| 400 | 0.021868 | 0.023883 | +0.002015 [−0.001995, +0.006493] | 100/100 / 100/100 |
| 1,024 | 0.022059 | 0.021385 | −0.000674 [−0.004208, +0.003095] | 100/100 / 100/100 |

These intervals allow modest effects in either direction; they do not establish
equivalence. There are only 50 independent prompt clusters. TPR uses the same
completion-only weighted-mean SynthID detector, official context mask and
nominal p < .001 rule. This is not a matched-empirical-FPR comparison. All-success
bootstrap intervals do not establish perfect population detection. Existing
Stage A null results remain applicable to the unchanged detector; no new null
generation or calibration was performed.

## Repeat, fallback and divergence diagnostics

A repeat means a repeated **generation context** under the native processor's
hashing and history, not the detector's context mask. Each subsequent occurrence
counts as a repeat. Counts cover each response's evaluated prefix.

| Tokens | Policy | Responses with repeat | Responses with fallback | Total repeats | Total fallbacks | Mean repeats / response | Mean fallbacks / response | Median first repeat among affected |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 400 | Original: on | 98/100 (98%) | 98/100 (98%) | 934 | 934 | 9.34 | 9.34 | 110.5 |
| 400 | Modified: off | 98/100 (98%) | 0/100 (0%) | 2,059 | 0 | 20.59 | 0 | 110.5 |
| 1,024 | Original: on | 100/100 (100%) | 100/100 (100%) | 4,553 | 4,553 | 45.53 | 45.53 | 111.5 |
| 1,024 | Modified: off | 100/100 (100%) | 0/100 (0%) | 11,421 | 0 | 114.21 | 0 | 111.5 |

Positions are **zero-based generated-token indices**; null denotes no event.
Under native fallback-on, first fallback equals first repeat. Under fallback-off,
there is no first fallback. The first repeated context occurs at the same
position in both policies because their trajectories agree up to that event.
Later repeat counts differ after their trajectories separate.

| Prefix | Pairs with token divergence | Median first divergence among diverged |
|---|---:|---:|
| 400 | 91/100 (91%) | 131 |
| 1,024 | 100/100 (100%) | 134.5 |

Across the full trajectories, 45 pairs first diverged at the first repeat and
55 afterward. Median delay from first repeat to first divergence was one token;
maximum delay was 528. **No pair diverged before its first repeat.** First token
divergence is a property of the pair, so both response records contain the same
index. A repeat can change sampling probabilities without changing the sampled
token immediately.

The mean increase in repeats was 11.25 per response at 400 tokens (paired 95%
interval 5.35–19.16) and 68.68 at 1,024 (36.17–106.28). These count intervals are
an additional descriptive diagnostic, not a predeclared primary endpoint.
Fallback clearly affected within-response context repetition here; that effect
did not translate into a clear change in the chosen between-response Self-BLEU
metric. This distinction limits the conclusion to the metric and cohort tested.

## Validation and provenance

- Same Qwen3-8B-Base weights/tokenizer, model revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16/H100, batch 50, prompts 0–49,
  seeds 12345/67890, fixed depth-10 keys, temperature 1 and top-p 1. Only
  generation-time repeated-context fallback changed. Generation still uses
  prompts; detection uses completion tokens only.
- Retained H100 forced-repeat probes passed, including exact native-on behavior,
  unchanged first-occurrence scores and state, and the intended off-policy update.
- Native-on 64-token controls exactly reproduced the saved originals at both
  seeds, covering all 100 responses. This is a prefix control, not a new claim
  that every original full-length response was regenerated.
- All 100 full trajectories passed the no-early-divergence check, first-repeat
  agreement, repeat-history agreement through divergence, modified trace
  reconstruction and disabled-fallback checks. The no-repeat-reference invariant
  was also checked; no full-length reference lacked a repeat in this cohort.
- Original repeat/fallback positions were reconstructed from saved tokens using
  upstream hashing, native zero-context warmup and zero-filled history; all 100
  native GPU prefix traces verified that reconstruction. Modified full generation
  traces were recorded and independently checked against token reconstruction.
- 49 local tests and the saved-reference analysis integration check passed before
  dispatch. Frozen source hashes, runtime identity and downloaded artifact hashes
  passed. No failures or retries occurred in this stage.

Dispatch commit: `536a39a29994d9e273da5c12fdb957ada18f519d`.
Frozen manifest ID:
`8e762224a9f7d2207e7088a2f41a0a1a93de8012fa4e9b9d83df13a961e7d158`.
The manifest records the preparation commit and exact source hashes; the dispatch
commit contains those prepared sources and the manifest.

Modal app: [ap-szfo8J2EeYESPjtF0AjIvJ](https://modal.com/apps/new-prc-watermark/main/ap-szfo8J2EeYESPjtF0AjIvJ).
Worker time was 174.895 seconds; estimated worker resources cost **$0.22587**.
Including the predeclared $0.50 overhead allowance and prior $5.51880 planning
charge gives **$6.24467**, leaving **$3.75533** of the initial $10 allocation.
These are resource estimates and planning allowances, not a settled invoice.
The full study ceiling remains $200.

## Saved records and next decision

- [Paired primary results](synthid_analysis.json).
- [Both-policy per-response diagnostics](synthid_response_diagnostics.json): 100
  pairs, response hashes/IDs, seeds, repeat/fallback counts at each endpoint,
  first repeat/fallback positions, and first token divergence for both policies.
- [Aggregate trajectories and checks](synthid_trajectory.json),
  [independent inspection](inspection.json), and [GPU report](synthid_report.json).
- [Artifact index and retrieval command](artifact_index.json),
  [execution record](synthid_execution.json), and [frozen request](manifest.json).

Full original and modified completions are retained: original responses in the
Stage A archive, modified responses and native-prefix controls in the existing
`prc-completion-only` Modal volume. Their locations and checksums are linked from
the artifact index. Local `raw/synthid_trajectory_pairs.json` records all
repeat/fallback positions and pair-level checks; it is reproducible from those
saved tokens without a GPU. Original results are unchanged.

TextSeal/Gumbel-on generation and TextSeal replay remain prepared but unrun.
Review whether those within-method diagnostics justify the next stage before
dispatch. No parameter sweep or broader expansion was launched. This result
does not establish a PRC advantage over native SynthID or justify replacing its
native fallback policy in the main comparison.
