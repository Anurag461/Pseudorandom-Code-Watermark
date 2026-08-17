# Online Causal PRC Experiment Plan

## Completed 0.6B eta 0.20 boundary audit and Hoeffding additions (2026-08-17)

The missing full audit at the 0.6B online boundary `T=n=3104` is complete.
It reused all 500 watermarked records as exact prefixes of the static-KV
`T=4096` online cache and all 500 null records from the compatible `T=8192`
0.6B cache. The explicit cache-only guard passed before model construction, so
no GPU generation was launched. Ten one-core CPU workers each scored 50
prompts and reproduced the boundary-sweep MAP result exactly.

- MAP: TPR `454/500 = 90.8%`; FPR `0/500 = 0.0%`.
- Entropy-aware: TPR `404/500 = 80.8%`; FPR `0/500 = 0.0%`.
- Naive: TPR `328/500 = 65.6%`; FPR `0/500 = 0.0%`.

The detector wall time was 102.1 seconds. The exact incremental Modal charge
was `$0.01466553` before credits (`$0.01348730` CPU and `$0.00117823` memory;
zero H100 charge). The ten versioned detector shards and combined result are
persisted on the `prc-data` volume. The local combined manifest is
`outputs/online_causal_n3104_t3_eta0.20_prompts500_sampler-poscdf-v1_kvcache-static-v1_from_n4096.json`.

The Hoeffding summary now includes this complete 0.6B boundary audit and the
previously completed 8B eta 0.20 passing-boundary audit at `T=n=13088` (MAP
`451/500 = 90.2%`, FPR `1/500 = 0.2%`). Both rows are explicitly labeled
`online_causal_prc` and retain their source-cache provenance in `Notes`.

## Completed cache-only 8B comparisons at the 0.6B boundaries (2026-08-17)

Two full MAP/entropy-aware/naive audits now measure the 8B model at the
previously selected 0.6B online MAP boundaries. Both jobs ran in an explicit
`--cache-only` mode that checks all 500 watermarked and null records before the
generation branch can construct a model container. They reused exact prefixes
of longer online records and launched no GPU generation.

| eta | Evaluated T=n | 0.6B online MAP at this n | 8B online MAP | 8B entropy-aware | 8B naive | 8B FPRs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 1504 | 453/500 (90.6%) | 317/500 (63.4%) | 223/500 (44.6%) | 104/500 (20.8%) | 0/500 for all three |
| 0.20 | 3104 | 454/500 (90.8%) | 303/500 (60.6%) | 218/500 (43.6%) | 111/500 (22.2%) | 0/500 for all three |

The eta 0.15 audit used the 4,096-token 8B watermarked cache and a compatible
1,625-token null cache. Ten 50-prompt CPU shards completed in 110.1 seconds of
detector wall time and cost `$0.01823568` (`$0.01679405` CPU plus `$0.00144163`
memory). Its local combined result is
`outputs/online_causal_n1504_t3_eta0.15_prompts500_gen-qwen3_8b_base_sampler-poscdf-v1_from_n4096.json`.

The eta 0.20 audit used the 14,336-token static-KV 8B watermarked cache and a
compatible 4,096-token null cache. Ten 50-prompt CPU shards completed in 120.8
seconds of detector wall time and cost `$0.01753222` (`$0.01616616` CPU plus
`$0.00136606` memory). Its local combined result is
`outputs/online_causal_n3104_t3_eta0.20_prompts500_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1_from_n14336.json`.

These rows are cross-model comparisons, not 8B boundary estimates: both are
well below 90% MAP. The 8B eta 0.20 passing boundary remains 13,088 on the
16-token grid. The current 8B eta 0.15 cache reaches only 4,096 and has 89.6%
MAP there, so finding its first strict passing point still requires longer 8B
watermarked generation.

## Completed parallel full audit: Qwen3-8B, eta 0.20, T=n=13088 (2026-08-17)

The requested selected-prefix audit is complete using the cached 14,336-token
watermarked records and the shared 13,088-token null records. No model or GPU
generation was launched. Ten one-core CPU workers each scored a stable shard of
50 prompts under MAP, entropy-aware, and naive weighting; the aggregator restored
canonical prompt order, checked complete non-overlapping coverage, and persisted
both the versioned shard results and one combined result.

- MAP: TPR `451/500 = 90.2%`; FPR `1/500 = 0.2%`.
- Entropy-aware: TPR `431/500 = 86.2%`; FPR `0/500 = 0.0%`.
- Naive: TPR `385/500 = 77.0%`; FPR `0/500 = 0.0%`.

The parallel detector took 225.9 seconds of wall time. The exact Modal charge
was `$0.02754972` before credits (`$0.02536963` CPU and `$0.00218009` memory;
zero H100 charge). The complete successful watermarked campaign including this
audit cost `$61.00515483`. Including the separately generated reusable null
cache, the end-to-end successful experiment cost `$114.39120928`; including the
aborted batch-16 smoke raises that to `$114.49698308`.

The local combined result is
`outputs/online_causal_n13088_t3_eta0.20_prompts500_gen-qwen3_8b_base_sampler-poscdf-v1_kvcache-static-v1_from_n14336.json`,
the summary row is in `online_causal_results_summary.csv`, and the exact audit
charge is appended to
`outputs/online_map_sweep_n14336_to8192_eta0.20_8b_cost_ledger.csv`. The remote
combined result and all ten prompt shards remain on the `prc-data` volume.

The full audit path now uses versioned prompt-shard caching by default in both
`main` and the selected-prefix phase of `sweep_map_prefixes`. A two-prompt Modal
A/B gate compared it with the serial reference and found identical decisions,
counts, and detailed scores. Local unit tests also enforce complete shard
coverage, stable ordering, detector completeness, and rejection of duplicated
or malformed shards.

## Completed reusable 8B null cache at T=13088 (2026-08-17)

The static-cache null campaign completed successfully at `T=13088` for all
500 canonical prompts. A reusable batch-25 smoke generated prompts 0--24 on
one H100; production reused those records and generated prompts 25--499 as 19
full batch-25 shards on at most ten H100s. A final cache-only audit verified
all 500 records, with no missing, invalid, rejected, or legacy-manifestless
records. Every record has `static/static-v1` provenance and exactly 13,088
tokens, probability values, LM-entropy values, and token-log-probability
values. The shared eta-independent cache is
`/data/_nulls/qwen3_8b_base/T13088` and can be truncated exactly for compatible
fixed or online audits at any requested length at or below 13,088.

The smoke took 2,208.2 measured H100 method-seconds and cost `$2.56828807`.
Production processed the remaining 6,216,800 token positions in 43,899.9
aggregate H100 method-seconds and 4,850.1 seconds of end-to-end wall time. Peak
memory was 70,403,537,408 bytes allocated and 70,478,987,264 bytes reserved
(about 65.6 GiB reserved). Smoke plus production generated 6,544,000 null token
positions and cost `$53.38605445` before workspace credits. Exact app-level
H100/CPU/memory charges are recorded in
`outputs/shared_null_cache_T13088_8b_cost_ledger.csv`; the reusable manifests
are `outputs/shared_null_cache_T13088_prompts25_gen-qwen3_8b_base_kvcache-static-v1.json`
and `outputs/shared_null_cache_T13088_prompts500_gen-qwen3_8b_base_kvcache-static-v1.json`.

## Completed result: Qwen3-8B, eta 0.20 (2026-08-17)

The reusable-smoke and 500-prompt online campaign completed at a ceiling of
`n=T=14336` with the static KV cache. The accepted production shape was batch
25: a one-H100, 25-prompt smoke was retained as prompts 0--24, and the remaining
475 prompts were generated as 19 full batches on at most ten H100 workers. The
strict `MAP TPR > 90%` boundary on the descending 16-token grid is:

- `n=T=13088`: `451/500 = 90.2%` (passes).
- `n=T=13072`: `450/500 = 90.0%` (fails and terminates the scan).
- Empirical boundary bracket: `(13072, 13088]`; 13088 is the smallest
  evaluated passing grid point under the requested first-failure rule.

The adaptive detector evaluated and saved 80 prefixes from 14336 through
13072. It explicitly left the 305 requested shorter prefixes from 13056
through 8192 unevaluated. Ten prompt-preparation shards covered all 500 records.
Direct local and Modal inventories verified contiguous source records 0--499,
the combined grid, all 80 production increment results, the smoke grid, and
both smoke increments. No null generation or entropy/naive final audit was run
during the boundary search.

Production completion processed 6,809,600 model token positions in 50,126.1
aggregate H100 method-seconds and 5,630.5 seconds of end-to-end wall time. The
successful reusable smoke, cache replay, and production completion cost
`$60.97760511` before workspace credits. Including the aborted batch-16 attempt
that preceded the requested switch to batch 25, total spend was `$61.08337891`.
The exact app-level H100/CPU/memory charges are saved in
`outputs/online_map_sweep_n14336_to8192_eta0.20_8b_cost_ledger.csv`.

## Completed result: 0.6B, eta 0.20 (2026-08-16)

The reusable-smoke and 500-prompt production campaign completed with the
planned static KV cache, H100 batching, prompt-sharded CPU preparation, and
adaptive descending stopping rule. The strict `MAP TPR > 90%` boundary on the
16-token grid is:

- `n=T=3104`: `454/500 = 90.8%` (passes).
- `n=T=3088`: `449/500 = 89.8%` (fails and terminates the scan).
- Empirical boundary bracket: `(3088, 3104]`; the smallest evaluated passing
  grid point is 3104.

The scan evaluated and saved 64 increments from 4096 through 3088. It did not
score the 129 requested shorter increments from 3072 through 1024; those are
explicitly recorded as unevaluated in the manifest. The production run reused
the five smoke records and generated only prompts 5--499 in four exact shards
of `125 + 125 + 125 + 120` on four H100 workers. A cache-only replay then
verified all 500 watermarked records and all ten prepared CPU shards, used zero
GPU generation, and reproduced every TP count and the same stopping boundary.

Saved local outputs are the JSON manifest, CSV, and 64 per-increment JSON files
under
`outputs/online_map_sweep_n4096_to1024_step16_t3_eta0.20_prompts500_sampler-poscdf-v1_kvcache-static-v1*`.
The reusable five-prompt smoke has the corresponding `n4096_to4080` outputs.
On the Modal volume, all 500 source records remain in the production artifact
namespace, along with the combined grid result, every evaluated increment, and
the versioned prompt-preparation shards. No null data or final entropy/naive
audit was generated during this boundary search, as planned.

Provider cost before workspace credits was `$0.19163305` for the reusable
smoke, `$2.48965899` for production completion and detection, and `$0.00758154`
for both cache-only verification replays, or `$2.68887358` in total. The exact
per-app H100/CPU/memory breakdown is saved in
`outputs/online_map_sweep_n4096_to1024_eta0.20_0p6b_cost_ledger.csv`.

## Executed experiment setup: 0.6B, eta 0.20

Use the opt-in static KV cache and generate a reusable `T=n=4096` ceiling.
The production generation is 500 prompts with `--batch 125`, exactly four
mapped shards on at most four H100s (`--max-containers 4 --gpu H100`). A small
five-prompt smoke uses one H100 and batch 5; its prompt-indexed records share
the exact production namespace and are retained. Consequently, the production
completion generates only the missing 495 records in four shards of
`125 + 125 + 125 + 120`.

Smoke only the ceiling and its next 16-token prefix:

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --source-n 4096 --floor-n 4080 --step 16 \
  --target-map-tpr 0.90 --num-prompts 5 \
  --t 3 --eta 0.20 --fpr 0.001 \
  --batch 5 --max-containers 1 --gpu H100 \
  --kv-cache-implementation static \
  --detection-shard-size 1 --detection-max-containers 5 \
  --no-final-audit --no-pin-floor-cache
```

Then complete the reusable ceiling and scan downward:

```bash
modal run modal_online_run.py::sweep_map_prefixes \
  --source-n 4096 --floor-n 1024 --step 16 \
  --target-map-tpr 0.90 --num-prompts 500 \
  --t 3 --eta 0.20 --fpr 0.001 \
  --batch 125 --max-containers 4 --gpu H100 \
  --kv-cache-implementation static \
  --detection-shard-size 50 --detection-max-containers 10 \
  --no-final-audit --no-pin-floor-cache
```

The scan is adaptive and descending. It evaluates 4096, 4080, 4064, and so
on, but stops immediately after the first point that is not strictly above
90% MAP TPR. For 500 prompts, `451/500` passes and `450/500` fails. The first
failing point is saved along with every preceding evaluated increment; all
shorter requested lengths remain explicitly marked unevaluated. Null
generation and the entropy/naive audit happen only after the boundary is known.

MAP preparation is parallelized over prompts, not lengths. Production uses ten
one-core CPU workers with 50 disjoint prompts each. Every worker loads and
validates its own records, converts each token/probability trace once, and
computes every longest-prefix signed parity-check contribution and squared
contribution once. Versioned prepared shards are cached by source artifact,
code fingerprint, maximum length, and exact prompt-index list. One one-core
aggregator validates gap-free/nonoverlapping coverage, restores canonical
prompt order, evaluates global TPR downward until the first failure, and is the
only process that writes the combined grid or individual increment results.

## Current recommended experiment: ceiling-first MAP TPR sweep

The default experiment should generate one reusable online sequence collection
to a credible maximum length, then find the MAP-TPR boundary by scoring saved
prefixes downward every 16 tokens. Stop after the first prefix that is not
strictly above the target. Do **not** physically resume generation every 16
tokens: the current continuation path must replay the saved prefix to rebuild
the model KV cache, so many small extensions would repeatedly pay for almost
the same prefix. Prefix detection is CPU-only and substantially cheaper.

For example, for the 0.6B, `eta=0.15` campaign, use a generation ceiling of
`N_max=2048`, then score

```text
L_min, L_min + 16, L_min + 32, ..., 2048
```

with `T=n=L` at every scored prefix. A reasonable initial evaluation floor is
`L_min=1024`, since the existing fixed results bracket the 90% crossing between
1024 and 2048. With 500 prompts, the strict empirical stopping condition
`MAP TPR > 90%` is `TP >= 451`; `450/500` does not pass. If no prefix passes,
extend the saved 2048-token online records once to a larger ceiling such as
2560 or 3072, then score only the new 16-token prefixes. Any generated overshoot
is retained as a source for later longer or shorter runs.

This ceiling-first procedure is exact for the online sampler-v2 construction,
not an approximation: for a fixed compatible key, experiment seed, document
seed, prompt, partition, and model, the first `L` generated token IDs, PRC bits,
and probability traces from a longer record must equal a direct online run to
length `L`. Detection must rebuild `target_r(L)`, the causal supports, OTP
prefix, `V`, and the Hoeffding threshold for that particular `L`; it must never
reuse the length-2048 detector state unchanged.

The production pipeline should therefore:

1. Accept an evaluation floor, generation ceiling, step (default 16), maximum
   permitted length, target MAP TPR, prompt count, model, `eta`, `t`, FPR, seed,
   GPU, batch size, and concurrency limit.
2. Generate directly to the ceiling, continue once from the longest compatible
   shorter cache, or use an already-compatible longer cache without generation.
3. Prepare each saved record once for MAP detection, then evaluate lengths in
   descending order. Prompt preparation may run in disjoint one-core CPU
   shards, but one aggregator must make the global stopping decision. Stop
   immediately after the first failing length; never score the remaining
   shorter grid blindly.
4. Record TP count, total, TPR, `r(L)`, support hash, threshold provenance,
   source cache/length, and pass/fail for every evaluated length in a resumable
   manifest and summary CSV. Record all skipped lengths as unevaluated.
5. Select the last passing prefix immediately before the first failing prefix.
6. At the selected length, run the full final audit, including MAP FPR with a
   compatible null cache and optional entropy/naive results. Null generation is
   not required at every exploratory prefix because the per-document threshold
   is analytic.
7. Treat the online crossing as the candidate for one independent fixed-scheme
   confirmation. Online and fixed constructions are different code ensembles,
   so the online result alone does not establish the fixed-scheme crossing.

The detector's one-shot per-length FPR threshold does not change for this
experiment. The prefix TPR estimates are paired and correlated because they use
the same documents. Selecting the first empirical crossing also introduces
selection uncertainty; use an independent fixed confirmation (or a separately
specified confidence-sequence/holdout analysis) for a population-level claim.
If a deployed detector itself tests the same document repeatedly as tokens
arrive, that is a different use case and requires the anytime/alpha-spending
policy described below.

Before production, add and test a resumable sweep entry point and a batched
`detect_map_prefix_grid` path. The acceptance tests must establish: exact
direct/extended/truncated prefix equality at multiple 16-token boundaries;
MAP grid results identical to the existing single-prefix detector; strict
`450/500` versus `451/500` stopping behavior; correct `T=n=L` and `r(L)` at
every prefix; rejection of incompatible model/key/seed/partition/prompt caches;
idempotent recovery from partial prompt batches; clean termination at the
maximum length; and model-qualified isolation between 0.6B and 8B caches. For
`t=3`, retain the startup tests with no checks at lengths 1 and 2 and exactly
the `{0,1,2}` weight-3 check at length 3. Batch size must be smoke-tested at the
largest ceiling: retain the proven 0.6B/A10G setting where it fits, but validate
a smaller 8B/H100 batch (for example 25 or 50) before scaling concurrency.

### Ceiling-first sweep implementation and first result (2026-08-16)

The 0.6B ceiling-first setup is now implemented:

- `detectors.detect_online_map_prefix_grid` converts each token/probability
  trace once, materializes the longest support/OTP prefix once, and returns the
  same MAP decision, statistic, `V`, and threshold as independent calls to the
  existing one-length detector.
- `modal_online_run.py::sweep_map_prefixes` ensures the ceiling cache exists,
  prepares the saved traces once, evaluates a descending prefix grid only
  through its first failing point, applies the strict empirical target, records
  a resumable JSON manifest and CSV, optionally runs the full detector/null
  audit at the selected length, and records measured GPU work separately from
  CPU prefix detection.
- A sweep may pin every ceiling record to a declared canonical floor cache.
  Before detection it compares token IDs, probability traces, entropy/log-prob
  traces, and causal PRC bits at the floor boundary. Incompatible ceiling
  records are moved to a timestamped, recoverable quarantine and regenerated;
  compatible records are left untouched.

#### Prompt-sharded MAP preparation (2026-08-16)

The MAP path now caches each prompt's longest-prefix signed check values and
squared check values once, then reuses initial row ranges at shorter lengths.
`sweep_map_prefixes` splits prompt indices into stable disjoint shards and maps
them to Modal functions with one physical CPU each. Prepared-shard paths include
the schema version, source artifact fingerprint, code fingerprint, maximum
length, and an exact prompt-list hash. Cache validation rejects stale metadata,
non-finite arrays, wrong row counts, reordered records, duplicate indices,
gaps, and unexpected prompts before any aggregation.

The production default is 50 prompts per shard, at most ten CPU containers for
500 prompts. Workers persist only reusable preparation caches. A single
aggregator restores requested prompt order, applies the strict global stopping
rule, and alone writes prefix increments and the combined grid. Thus a local
shard's pass/fail rate can never terminate the experiment, and no shorter
length after the first global failure is evaluated.

The full local suite passed 76 tests. A two-prompt `n=64` static-cache Modal
validation used two one-core workers and reproduced the serial rows and every
per-prompt score exactly. Since the global MAP rate failed at 64, the adaptive
aggregator saved 64 and left 48 unevaluated. In the final implementation,
supports and OTP signs were prepared once per shard in addition to once-per-
prompt check values. App `ap-OFCDhlJXLhVXr8MFMo6oDk` prepared both shards in
12.307 seconds of wall time; an identical rerun,
`ap-ZeHVKDksGKHUOWnbyypjbv`, reported `2/2` preparation cache hits and 9.908
seconds including Modal startup, volume loading, serial reference scoring, and
aggregation. The final pair cost `$0.00249032`; including the two earlier
prototype/reuse validations, total implementation validation cost was
`$0.00511536` before credits. The itemized record is
`outputs/online_map_detection_sharding_smoke_cost_ledger.csv`.

The first production validation used Qwen3-0.6B, `eta=0.05`, `t=3`, 500
prompts, a ceiling of 512, a floor of 400, and a descending step of 16:

| `T=n` | MAP detections | MAP TPR | Strictly above 90% |
|---:|---:|---:|---:|
| 512 | 468/500 | 93.6% | yes |
| 496 | 468/500 | 93.6% | yes |
| 480 | 462/500 | 92.4% | yes |
| 464 | 459/500 | 91.8% | yes |
| 448 | 455/500 | 91.0% | yes |
| 432 | 447/500 | 89.4% | no |
| 416 | 444/500 | 88.8% | no |
| 400 | 439/500 | 87.8% | no |

Thus 448 is the last passing length while descending on this 16-token grid,
and the empirical online crossing is bracketed by `(432, 448]`. The full final
audit at 448 reproduced MAP `455/500` with MAP FPR `0/500`; entropy was
`388/500` with FPR `0/500`, and naive was `260/500` with FPR `0/500`. There
were no empirical monotonicity violations on the tested grid.

The complete campaign—including the two-prompt smoke, a safety-stopped run
whose 498 valid continuations were retained, the two-record repair/final audit,
and a final cache-only persistence verification—cost `$0.21108824` according to
the Modal billing report before workspace credits (`$0.19777066` GPU,
`$0.01002224` CPU, `$0.00329534` memory). No null records were generated. The
cache-only verification launched no GPU, verified 500 watermarked and 500 null
records, and reproduced all eight MAP counts. The per-app billing and measured
token-work ledger is stored in
`outputs/online_map_sweep_n512_to400_cost_ledger.csv`; the final grid and
manifest are stored beside it as
`online_map_sweep_n512_to400_step16_t3_eta0.05_prompts500_sampler-poscdf-v1.csv`
and `.json`. Each increment also has an independently loadable local JSON and
remote `.pt` result indexed by the manifest. The large token, probability,
entropy/log-probability, and PRC-bit traces remain stored once in the canonical
512-token cache; prefix generation records are not duplicated.

The corresponding fixed-length confirmation at `T=n=448`, `eta=0.05`,
`t=3`, and 500 prompts detected MAP `466/500` (93.2%), entropy `399/500`
(79.8%), and naive `291/500` (58.2%), with `0/500` false positives for all
three detectors. At the same length, the online construction detected MAP
`455/500` (91.0%), entropy `388/500` (77.6%), and naive `260/500` (52.0%).
Thus the fixed MAP result is 11 prompts, or 2.2 percentage points, higher for
this paired parameter setting. The fixed run reused the complete 512-token
null cache, persisted all 500 watermarked records, and cost `$0.20658095`
before workspace credits.

### Eta 0.15 ceiling-first result (2026-08-16)

The second production sweep used Qwen3-0.6B, `eta=0.15`, `t=3`, 500 prompts,
an online ceiling of 2048, a floor of 1024, and a descending step of 16. Batch
64 fit on A10G, and five containers generated the 498 records not already
created by the two-prompt smoke test. A compatible fixed-length null cache at
`T=8192` supplied all 500 null traces, so no new null tokens were generated.

The strict `MAP TPR > 90%` crossing is bracketed by `(1488, 1504]` on the
16-token grid: `n=T=1504` detected `453/500` (90.6%), while `n=T=1488`
detected `446/500` (89.2%). For reference, the endpoints detected `484/500`
(96.8%) at 2048 and `342/500` (68.4%) at 1024. The full audit at 1504 matched
the grid with MAP `453/500` and FPR `0/500`; entropy detected `398/500` with
FPR `0/500`, and naive detected `279/500` with FPR `0/500`.

All 500 ceiling records were verified in the reusable Modal cache with token,
probability, entropy, token-log-probability, and PRC-bit traces. All 65 prefix
increments have separate local JSON and remote `.pt` results, in addition to
the combined grid and final audit. The compact grid is stored at
`outputs/online_map_sweep_n2048_to1024_step16_t3_eta0.15_prompts500_sampler-poscdf-v1.csv`.

The two-prompt smoke and full 500-prompt campaign cost `$0.90315289` before
workspace credits: `$0.84883360` A10G, `$0.04474380` CPU, and `$0.00957549`
memory. The full run alone cost `$0.86929309`; the reusable smoke records cost
`$0.03385980`. Exact per-app billing is recorded in
`outputs/online_map_sweep_n2048_to1024_eta0.15_cost_ledger.csv`.

The paired fixed-length Qwen3-0.6B baseline at `n=T=1504`, `eta=0.15`, `t=3`
and 500 prompts detected MAP `455/500` (91.0%), entropy `424/500` (84.8%), and
naive `341/500` (68.2%), with FPR `0/500` for every detector. Compared with the
online construction at the same length, fixed was higher by 2 MAP prompts
(`+0.4` percentage points), 26 entropy prompts (`+5.2` points), and 62 naive
prompts (`+12.4` points). Thus both constructions exceed the strict 90% MAP
target at 1504, and their observed MAP difference is small in this paired
500-prompt comparison.

The fixed run used a full-rank `r=1489` matrix, batch 64 over up to five A10Gs,
reused the complete fixed null cache at `T=8192`, and persisted all 500
watermarked records. One detector container was preempted and automatically
retried without regenerating the saved watermarked samples. The run cost
`$0.56037526` before credits: `$0.53044381` A10G, `$0.02387968` CPU, and
`$0.00605177` memory. Exact billing is recorded in
`outputs/fixed_n1504_t3_eta0.15_prompts500_cost_ledger.csv`.

### Qwen3-8B generalization (2026-08-16)

The online runner now accepts `--generation-model-size 8B`. Model choice is
part of artifact identity, watermarked-cache discovery, null-cache discovery,
continuation compatibility, record validation, result metadata, and local
filenames. The 8B path loads `Qwen3-8B-Base`, defaults to H100 and batch 25,
stores online records below `qwen3_8b_base`, and reads only model-qualified 8B
nulls below `/data/_nulls/qwen3_8b_base`. Existing 0.6B paths are unchanged.

An end-to-end two-prompt `n=T=64`, `eta=0.15`, `t=3` mechanics smoke generated
and saved 8B online records, reused the existing complete 8B null cache at
`T=768`, and completed MAP, entropy-aware, and naive detection. A cached rerun
launched no GPU generation. The smoke is not a TPR measurement; its purpose was
model loading, isolation, persistence, null compatibility, and detection. The
exact production commands and batching guidance are in
`modal_online_8b_runbook.md`.

#### Opt-in static KV cache: implementation and small validation (2026-08-16)

The custom Qwen decoder's historical `KVCache` still appends each layer's new
key and value with `torch.cat`, and it remains the default for fixed, legacy,
and existing online runs. An opt-in inference-only `StaticKVCache` now lazily
allocates one fixed-capacity K/V pair per layer, copies each new slice into its
absolute sequence range, and returns a view of the populated prefix. Online
generation allocates for `prompt_length + target_T`, checks capacity, dtype,
device, and shape on every update, and retains allocations across `reset()`.
No PRC construction, position-addressed random draw, detector, or threshold
changed.

The online CLI selects it with `--kv-cache-implementation static`. Static
artifacts and watermarked records are deliberately isolated below the suffix
`_kvcache-static-v1`; historical concatenating tags are unchanged and cache
discovery refuses to reuse watermarked records across implementations. Records,
generation segments, sweep manifests, result payloads, and cost telemetry save
the implementation/version.

As of the reusable long-null setup, null generation has an independent
`--null-kv-cache-implementation` option and can use the same static cache. New
shared null directories retain the model-qualified `T...` layout so fixed and
online detection can reuse them across eta values, but add an eta/key-independent
`_manifest.json` recording the prompt corpus, partition, forced-length policy,
null sampler/RNG policy, trace schema, and KV-cache implementation/version.
New records also save this provenance, use atomic writes, report peak CUDA
allocation/reservation, and are validated before resume. A legacy manifestless
cache remains readable, but the runner refuses to append static records to a
partially populated legacy directory. The null-only entry point
`modal_online_run.py::build_null_cache` supports a reusable smoke, missing-only
production generation, a cache-only verification pass, and a local cost/result
manifest without launching watermarked generation or detection. This execution
change does not alter the detector or its threshold.

Reusable 25-prompt 8B smoke:

```bash
modal run modal_online_run.py::build_null_cache \
  --num-prompts 25 --n 14336 --t 3 --eta 0.20 \
  --generation-model-size 8B --batch 25 --gpu H100 --max-containers 1 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static
```

Production completion reuses those exact records and schedules the remaining
19 batches over at most ten H100 containers:

```bash
modal run modal_online_run.py::build_null_cache \
  --num-prompts 500 --n 14336 --t 3 --eta 0.20 \
  --generation-model-size 8B --batch 25 --gpu H100 --max-containers 10 \
  --kv-cache-implementation static \
  --null-kv-cache-implementation static
```

Local tests use a two-layer miniature Qwen model and require exact logits and
K/V values at prompt prefill and every decode step. They also verify stable
storage pointers, allocation reuse after reset, overflow rejection, cache-tag
isolation, historical-record compatibility, and cross-implementation reuse
rejection. The full local suite passed 71 tests.

Small A10G validation used two 0.6B prompts and isolated experiment seed
`424242`. At `n=64`, separately saved concat and static direct runs matched
exactly for tokens, `p_trace`, LM entropy, token log-probability, PRC bits,
observed bucket bits, and MAP soft tokens. More importantly, an in-container
A/B utility ran both cache implementations through the same loaded model and
found exact concat-versus-static direct output and exact static direct-versus-
resumed output at both `n=80` from prefix 64 and `n=256` from prefix 128. At
`n=256`, concat direct took 10.64 seconds and static direct took 10.24 seconds;
peak reserved CUDA memory was identical at 1,637,875,712 bytes, while static
peak allocated memory was about 25 MB higher because it reserves its full
capacity immediately. These tiny runs validate correctness, not the eventual
long-context speedup.

Provider billing for all ten validation apps—including the two saved direct
runs, continuation run, two in-container A/B runs, saved `n=256` smoke, and
three CPU-only record comparisons—was `$0.09980649` before credits. The
per-app GPU, CPU, and memory charges are preserved in
`outputs/online_static_kv_cache_smoke_cost_ledger.csv`; the GPU portion was
`$0.09351127`.

A comparison between records produced in separate Modal app executions did
expose a broader reproducibility caveat: tiny GPU probability-trace differences
can occur even before token sequences diverge, including between two concat
runs. Position-addressed random uniforms remain identical, but a small logits
difference can eventually cross an inverse-CDF boundary. The in-container test
isolates cache-layout correctness; it does not establish byte-identical model
outputs across arbitrary GPU containers. This issue predates and is independent
of the static cache.

Before a large static-cache campaign, run a small reusable batch at its actual
ceiling, then verify persistence, peak memory, runtime, and a cache-only replay.
For the 0.6B eta 0.20 production completion, use batch 125 on at most four
H100s; the first production shard is also the full-batch memory validation.
The completed 8B eta 0.15 campaign remains on the concatenating cache and is
untouched.

#### Qwen3-8B eta 0.15 production result (2026-08-16)

The full online campaign used `n=T=4096`, `eta=0.15`, `t=3`, 500 prompts,
batch 50, at most five H100 containers, and exact saved-prefix evaluation from
4096 down to 3200 in steps of 16. It reused 77 watermarked and 27 null smoke
records, then generated only the missing 423 watermarked and 473 null records.
The generation plan used nine watermarked shards (`8 x 50 + 23`) and ten null
shards (`9 x 50 + 23`). One warm container later lost contact with the Modal
scheduler while requesting another input, but all mapped shards had committed
and the aggregate plan completed without regeneration or missing records.

No tested length strictly exceeded the 90% MAP target. The ceiling detected
`448/500` (89.6%) at 4096; the maximum over all 57 evaluated prefixes was also
448 detections. The floor detected `440/500` (88.0%) at 3200. Therefore the
empirical online crossing for this configuration is above 4096, rather than
within the planned `[3200, 4096]` grid. Small one- or two-document reversals
between adjacent prefixes were recorded as empirical monotonicity warnings and
do not indicate a cache or detector inconsistency.

A separate cache-only full audit at 4096 found all 500 watermarked and 500 null
records, launched no GPU, and measured MAP `448/500` (89.6%), entropy-aware
`423/500` (84.6%), and naive `335/500` (67.0%), with `0/500` false positives
for every detector. Direct volume inventory confirmed contiguous prompt-indexed
caches from 0 through 499, so all records can be reused for shorter detection
or as the complete replay source for a longer online ceiling.

The production generation and 57-prefix sweep cost `$11.12997658` before
credits (`$10.89981769` H100, `$0.19339690` CPU, `$0.03676199` memory). The
cache-only full audit cost `$0.00476049`, for an incremental production total
of `$11.13473707`. Exact billing is recorded in
`outputs/online_map_sweep_n4096_to3200_eta0.15_8b_cost_ledger.csv`.

## Side-by-side construction

| Aspect | Current fixed-length experiment | Proposed online causal experiment |
|---|---|---|
| Length | Choose `n` before generation and generate `T=n` tokens. | Extend one coordinate per generated token; when generation stops at `L`, set `T=n=L`. |
| Parity count | Pass `r=round(0.99n)` into `KeyGen`. | At every prefix, use `target_r(L) = max(0, min(round(0.99L), L-(t-1)))`. |
| Check construction | Build all `r` checks before generation, then globally permute code coordinates. | At parity coordinate `i`, introduce one row containing `i` and `t-1` distinct parents smaller than `i`; do not globally permute coordinates. |
| Free coordinates | Before permutation, the first `n-r` coordinates have no pivot rows; the permutation hides their locations. | A coordinate is free when `target_r(L)` does not increase; free coordinates are distributed through the stream so arbitrary stopping prefixes retain approximately 99% parity density. |
| Selecting the clean word | Build `G`, sample a payload `u` containing a fixed test bit and fresh random bits, and compute `c=uG^T`. | Do not use a test bit, payload vector, or `G` during generation. Sample `c_i` freshly at free coordinates and set `c_i` to the XOR of its parents at parity coordinates. |
| OTP and PRC noise | Materialize the full length-`n` OTP `z` and fresh noise `e`, then form `x=c XOR z XOR e`. | Derive `z_i` by position, sample fresh `e_i`, and form `x_i=c_i XOR z_i XOR e_i` only when coordinate `i` is generated. |
| LM sampling | Precompute the complete noisy PRC target word, then consume one target bit per LM step; token sampling uses the existing PyTorch RNG path. | Construct one noisy target bit per LM step. The current implementation uses document/position-addressed inverse-CDF draws for resumability; this is an engineering choice, not part of the paper, and should be controlled in baseline comparisons. |
| Detection | Load the fixed matrix and OTP for `n`; the general detector supports fixed blocks and blockwise FPR allocation. | Regenerate exactly the rows and OTP prefix realized by length `L`, score the prefix directly with no folding or block OR, and treat zero rows or `V=0` as insufficient evidence. |
| Hoeffding threshold | `sqrt(2 V log(1/alpha))` for the scored fixed block. | The same formula, recomputed from the checks and soft values present at the realized prefix; sequential prefix peeking requires alpha spending. |

## Implementation status (2026-08-15)

The first forced-length implementation is now present on branch
`online-causal-prc-plan`:

- `online_prc.py` implements the versioned rational row schedule, HMAC-SHA256
  support/OTP expansion, incremental per-document encoder state, sparse support
  reconstruction, and optional post-hoc generator reconstruction.
- `detectors.detect_online_hoeffding` scores the direct realized prefix with no
  folding or block OR. It supports final-only one-shot testing and an explicit
  alpha-spending policy for future prefix peeking.
- `watermark_expt.generate_batch_and_collect_online` samples one causal PRC bit
  per LM step while retaining the current bucket coupling and diagnostic
  traces. Watermarked bucket/token draws now use document-and-position-addressed
  randomness so retries, batch order, and shorter-to-longer continuation do not
  depend on process-global PyTorch RNG state. The existing fixed generator is
  unchanged.
- `modal_online_run.py` provides an isolated artifact/watermark namespace,
  compatible forced-length null reuse, batched 0.6B generation, direct CPU
  detection, separate local/remote result outputs, and automatic bidirectional
  watermarked-cache reuse:
  - requested length below a complete compatible cache: slice and detect the
    longer saved records without generation;
  - requested length above a complete compatible cache: select the longest
    shorter source, validate every source record, replay its tokens to rebuild
    the model KV cache, regenerate and verify the causal PRC prefix, and sample
    only the missing suffix;
  - require identical compact key, experiment seed, schedule, partition,
    prompts, generation model, and stopping policy before either reuse mode.
    Continued records store contiguous generation segments, source-record
    fingerprints, and source length/tag provenance.
- `tests/test_online_prc.py` covers the schedule, `t=3` startup, causality,
  prefix consistency, algebra, rank, batch isolation, detector equivalence,
  edge cases, and a strong synthetic watermark.

For the requested first experiment, `T=n=256`, `t=3`, and the causal schedule
realizes `r=253` with one-based free coordinates `{1, 2, 251}`. The intended
full command uses 500 prompts in batches of 64 on up to five A10G containers:

```bash
modal run modal_online_run.py::main --num-prompts 500 --n 256 --t 3 \
  --eta 0.05 --fpr 0.001 --batch 64 --max-containers 5 --gpu A10G
```

Independent replications pass `--experiment-seed <seed>`. Non-default seeds
are included in artifact/cache and local-result paths, so a replicate cannot
overwrite or silently reuse the first run's causal key or watermarked samples.
The partition seed remains fixed to keep the LM bucket split comparable.

## Executive summary

Set up a separate experiment in which the PRC codeword is extended one token at a time and the realized generated length is always the realized codeword length:

\[
T=n=L,
\]

where `L` is known only when generation stops. Every parity check introduced at coordinate `i` contains `i` as its unique pivot and otherwise contains only distinct earlier coordinates. For the repository's usual parity-check weight `t=3`, a check at `i` has the form `{j, k, i}` with `j < i`, `k < i`, and `j != k`.

The existing fixed-length implementation should remain intact as the baseline. The online experiment should get a separate key type, encoder state, detector entry point, cache namespace, and Modal runner/configuration mode. This prevents online artifacts from being mistaken for the existing `KeyGen`/`Encode` artifacts.

After generation stops, the experiment may optionally reconstruct a generator matrix from the recorded causal schedule. This is analysis/debugging post-processing, not part of the token-generation hot path: generation continues to sample free clean bits and derive parity clean bits directly.

The recommended online row schedule preserves the current new-run policy `r = round(0.99n)` as closely as causality permits. At prefix length `L`, define

```text
target_r(L) = max(0, min(round(0.99 * L), L - (t - 1)))
```

using a documented, versioned rounding convention. Coordinate `L - 1` is a parity coordinate exactly when `target_r(L) > target_r(L - 1)`; otherwise it is a free coordinate. This gives a prefix-consistent stream with approximately 99% parity coordinates, while reserving at least `t - 1` free coordinates so a weight-`t` causal row is possible.

For a simpler mechanics-only variant, every coordinate after the first `t - 1` coordinates can be a parity coordinate. That variant has `r(L) = L - (t - 1)`, but it should not be the primary comparison because its parity density and latent code dimension differ from the current `0.99n` runs.

## Current-code basis

The plan relies on the following behavior in the current worktree.

- `prc.KeyGen(n, ..., t, r)` samples each row using `t - 1` coordinates from the coordinates that precede that row's pivot, appends the pivot, and makes the matching generator row the XOR of the parent generator rows. Before permutation, this is already the desired causal lower-triangular construction.
- `KeyGen` then permutes the generator rows, OTP, and parity-check columns. That permutation destroys generation-order causality and must be omitted in the online path.
- `Encode` materializes the entire length-`n` codeword, OTP, and noise vector before LM generation begins.
- `generate_text_watermark_prc` and `generate_batch_and_collect` precompute length-`n` codewords, index them using `pos % n`, and refresh them at block boundaries.
- The current generation functions can collect exact noisy PRC codeword bits, base-LM entropy, and base-token log probability. The online experiment should preserve these trace fields and add causal-row metadata rather than creating an incompatible record format without diagnostics.
- `detectors.detect_hoeffding` obtains `n` from the fixed decoding key, folds or slices tokens into fixed blocks, applies a block-level FPR allocation, and calls `prc.Detect`.
- `prc.Detect` computes the Hoeffding statistic and the data-dependent threshold `sqrt(2 V log(1/fpr))` from a fixed sparse matrix and OTP.
- `modal_run.py` currently sets `T=n` through `experiment_T(n)`, enforces `r=round(0.99n)` for new runs, fingerprints key-dependent artifacts, isolates generation-model caches, reuses length-indexed null caches, and checkpoints detection results.

These are useful boundaries: the LM bucketing channel, MAP/entropy/naive soft-token functions, trace collection, model-specific cache isolation, hashing, and checkpointing can be reused. Fixed-size key construction, codeword precomputation, block folding, and `n`-keyed artifact identity cannot.

This first experiment is detection-focused, matching the current Modal campaigns that call `KeyGen` with `message_length=0`. Streaming payload recovery through `Decode` is out of scope. If message recovery later becomes a requirement, the online encoder will also need prefix-expandable generator rows and a separately reviewed streaming decoding design.

## Terminology and causal invariant

Use distinct names in new code to avoid the existing collision between `t` as parity weight and token time:

- `check_weight`: parity-check row weight; currently called `t` and normally equal to 3.
- `position`: zero-based token/code coordinate `i`.
- `length`: realized prefix length `L = i + 1`.
- `parents[i]`: the `check_weight - 1` earlier coordinates used by parity row `i`.
- `is_parity[i]`: whether coordinate `i` introduces a row.

For every parity coordinate `i`, enforce

```text
len(parents[i]) == check_weight - 1
all(parent < i)
all parents are distinct
row[i] = sorted(parents[i] union {i})
clean[i] = XOR(clean[parent] for parent in parents[i])
```

For a free coordinate, sample a fresh clean bit for each generated sequence. Never derive future clean bits from observed token buckets: observed buckets include the LM channel and do not obey the latent PRC parity equations.

Because every row has a unique pivot, the columns corresponding to parity coordinates form a triangular submatrix with ones on the diagonal. Therefore every length-`L` prefix has row rank `target_r(L)`.

## Proposed online key and state

Add an experiment-specific module, tentatively `online_prc.py`, with explicit dataclasses rather than extending the positional tuples returned by `KeyGen`.

### `OnlinePRCKey`

Suggested fields:

```text
schema_version
check_weight
noise_rate
row_rate_numerator = 99
row_rate_denominator = 100
row_schedule_version
support_seed
otp_seed
partition_seed or partition fingerprint
fpr_default
```

Use separate, domain-separated streams for row supports and OTP bits. Statistical independence of OTP row parities under the null is part of the Hoeffding argument; sharing an undifferentiated RNG stream with support generation makes both reasoning and reproducibility harder. For an initial scientific prototype, independently seeded reproducible random streams are sufficient. If the construction is later presented as cryptographic, replace them with a documented PRF construction.

The key should be compact and prefix-expandable. Given the same key and `L`, generator and detector must independently materialize exactly the same first `L` row decisions, parent sets, and OTP bits.

### `OnlineEncoderState`

Maintain, per active generated sequence:

- the clean latent-bit history;
- fresh per-document randomness for free bits and Bernoulli noise;
- current position;
- optionally the exact noisy codeword-bit trace already collected by the current pipeline.

At position `i`:

1. Evaluate the prefix row schedule at `L=i+1`.
2. If `i` is free, sample `clean[i]` independently for each sequence.
3. If `i` is a parity coordinate, deterministically regenerate `parents[i]` from the support seed and set `clean[i]` to their XOR.
4. Regenerate `otp[i]` from the OTP seed.
5. Sample fresh per-sequence noise `error[i] ~ Bernoulli(eta)`.
6. Form `xi[i] = clean[i] XOR otp[i] XOR error[i]`.
7. Pass `xi[i]` into the existing Bernoulli partition-selection and masked-token sampling logic.
8. Append `xi[i]` and any requested LM diagnostics to the trace record.

Rows and OTP bits are key-dependent and shared across documents, matching the current decoding-key semantics. Free clean bits and noise must be fresh per document. Batch order or one sequence reaching EOS must not change another sequence's online key stream.

## Row schedule and the first few tokens

For `check_weight=3`, two distinct prior coordinates are required. Under the recommended 99% schedule:

| Realized length `L` | New coordinate | Required action | Total checks |
|---:|---:|---|---:|
| 0 | - | Empty stream | 0 |
| 1 | 0 | Free clean bit | 0 |
| 2 | 1 | Free clean bit | 0 |
| 3 | 2 | First parity coordinate; parents must be `{0, 1}` | 1 |
| 4 | 3 | Parity coordinate; choose two distinct parents from `{0, 1, 2}` | 2 |
| 5 and later | `L-1` | Add a parity row or a scheduled free coordinate | `target_r(L)` |

At `L=3`, there is exactly one possible weight-3 causal row: `{0, 1, 2}`. The implementation must not call a generic sampler that tries to choose three elements from only the two previous positions. The current coordinate is the third member of the check.

The 99% target can temporarily conflict with the startup requirement. The `L - (t - 1)` cap deliberately wins. For example, at small lengths the online construction may have one fewer row than `round(0.99L)`. Store both the requested rate policy and realized `r(L)` in results rather than labeling the clamped value as exactly `0.99L`.

If `check_weight < 2`, `noise_rate` is outside `[0, 0.5)`, or the schedule requests a row before enough distinct parents exist, fail validation before generation.

## Generation integration

Keep the fixed generator functions untouched and add explicit online counterparts, for example:

```text
generate_text_watermark_prc_online(..., generation_cap, online_key, ...)
generate_batch_and_collect_online(..., generation_cap, online_key, ...)
```

`generation_cap` is only a safety/resource limit. It is not stored as `n`; each record stores `n = T = tokens.numel()` after EOS or another externally selected stop condition.

The online implementation should reuse the current code for:

- computing base-LM probabilities and `p1`;
- converting `xi` into the Bernoulli bucket probability;
- masked token sampling;
- KV-cache decoding;
- `prc_codeword_bits`, base-LM entropy, and base-token-logprob traces.

Changes needed for variable-length batches:

- Track an active mask per sequence.
- Do not advance a finished sequence's encoder RNG/state.
- Save each sequence at its actual length rather than forcing a rectangular `B x generation_cap` artifact without a length mask.
- Define whether EOS itself is included in the detected token stream. Use the same convention in generation, saved traces, and null data.

For the first controlled experiment, disabling EOS and scoring predefined prefixes of one longer causal generation is useful: it tests the online construction while allowing paired comparisons at multiple `L`. A second experiment should enable natural EOS to validate genuinely unknown realized lengths.

## Optional post-generation generator reconstruction

Do not construct or resize a generator matrix inside the token-generation loop. During generation, retain the realized schedule (free/parity decisions and the parent list for each parity coordinate) and the sampled clean bits at free coordinates. Once generation has stopped at realized length `T`, optionally reconstruct a generator matrix for analysis or debugging.

Let `r` be the number of parity coordinates and

\[
d = T-r
\]

be the number of free coordinates. Allocate

\[
G \in \mathbb F_2^{T \times d}.
\]

Process the recorded schedule in coordinate order:

1. If `i` is the `q`-th free coordinate, set \(G_i=e_q\), the `q`-th standard basis vector in \(\mathbb F_2^d\).
2. If `i` is a parity coordinate, set

   \[
   G_i = \bigoplus_{j \in \operatorname{parents}[i]} G_j.
   \]

All parent rows already exist because every recorded parent is smaller than its pivot. Define

\[
u=(c_{f_1},\ldots,c_{f_d})\in\mathbb F_2^d,
\]

where \(f_q\) is the `q`-th free coordinate and \(c_{f_q}\) is the clean bit sampled there during generation. The clean codeword already generated online should then satisfy

\[
c=Gu.
\]

This reconstruction does not change the generated word and is not required by encoding or detection. Expose it as an explicit post-processing/debug helper, for example `reconstruct_generator(recorded_schedule)`, and avoid materializing or saving the dense matrix unless the caller requests it. For large `T`, the recorded sparse parent table remains the canonical artifact.

Recommended validation checks for a reconstructed prefix are:

\[
P G = 0,
\]

where \(P\in\mathbb F_2^{r\times T}\) is the parity-check matrix materialized from the same schedule;

\[
\operatorname{rank}_{\mathbb F_2}(G)=d=T-r,
\]

when the causal triangular parity construction has full row rank; and

\[
c=Gu.
\]

These checks should operate over GF(2), report the realized dimensions and ranks, and fail loudly if the schedule, free-bit vector, or clean-codeword length is inconsistent.

## Detector and threshold

Add `detect_online_hoeffding` rather than routing online records through the fixed-block `detect_hoeffding` function.

For an observed record of realized length `L`:

1. Require `len(tokens) == len(p_trace) == L` and verify any saved codeword trace has the same length.
2. Regenerate the exact online row supports and OTP prefix for `L`.
3. Convert tokens to observed bucket bits using the existing partition map.
4. Produce one soft token per coordinate with the existing `map`, `entropy`, or `naive` weighting. Do not cyclically fold: `T=n=L`, so every generated token already corresponds to one codeword coordinate.
5. For each row `h`, compute `Q_h = product(S[j] for j in h)` and the OTP sign `A_h = product(1 - 2*z[j] for j in h)`.
6. Compute

   ```text
   statistic = sum(A_h * Q_h)
   V = sum(Q_h ** 2)
   threshold = sqrt(2 * V * log(1 / alpha))
   ```

7. Return `False` with `status="insufficient_evidence"` when there are no rows or `V <= numerical_tolerance`. In particular, do not allow `0 >= 0` to produce a positive decision.

The threshold's mathematical form does not change for a one-shot test. Its value does change with each realized prefix because `V`, row count, and soft tokens change. An old threshold JSON calibrated for a fixed `n` must never be reused for an online record.

No block-level Bonferroni correction is needed when one final `T=n` prefix is tested once. If the system exposes a detector decision repeatedly while text is arriving, use an anytime-valid policy. A simple first implementation is alpha spending, such as

\[
\alpha_L = \frac{6\alpha}{\pi^2 L^2},
\]

inside the same Hoeffding threshold. A later sequential e-process may improve power. The result metadata must distinguish `one_shot` from `alpha_spending_v1` so their thresholds cannot be compared as if identical.

With `t=3` and `L=3`, one parity row exists but normally cannot trigger the Hoeffding detector. At `alpha=10^-3`, a single row has threshold approximately `3.72 * abs(Q)`, while its maximum statistic contribution is `abs(Q)`. This is expected low sample power, not a startup bug.

The hard syndrome detector can also be adapted by applying its existing functional threshold to realized `r_eff`, but it should be secondary. Any empirical/null-calibrated detector requires new calibration indexed by online construction version, length or stopping policy, weight kind, and entropy model.

## Modal and artifact integration

Prefer a separate runner, tentatively `modal_online_run.py`, until the experiment is stable. Reuse helper code from `modal_run.py`, but use an online-specific namespace and configuration signature.

The configuration and artifact fingerprint should include at least:

```text
scheme = "online_causal_prc_v1"
check_weight
eta
row_rate policy and schedule version
support/OTP key fingerprint
generation cap
stopping policy
FPR policy (one-shot or anytime)
generation model
entropy model
partition fingerprint
trace schema version
```

Watermarked caches are key- and construction-dependent and must be isolated from fixed-`n` caches. Existing unwatermarked generations can be reused only when the generation model, prompt, sampling parameters, EOS convention, and requested prefix/length policy match. Natural-EOS null records should not be substituted with a fixed-length truncation without explicitly labeling that change in stopping policy.

Preserve current safeguards:

- generation-model-specific cache roots;
- artifact fingerprints;
- token and `p_trace` hashes;
- generation record validation;
- checkpoint identities;
- atomic result writes;
- shard aggregation consistency checks.

Extend trace/result metadata with realized `L`, realized row count, free-coordinate count, schedule version, stopping policy, and optionally a hash of the regenerated parent table. Avoid saving a dense `L x L` parity matrix; save/reconstruct the `r x t` parent/index table.

## Test plan

### 1. Pure schedule and key tests

Add `tests/test_online_prc.py` with fast, model-free cases.

1. **Startup table for weight 3:** assert no rows at `L=0,1,2`, exactly `{0,1,2}` at `L=3`, and a valid distinct-parent row at `L=4`.
2. **Row weight:** every materialized row has exactly `check_weight` distinct coordinates.
3. **Causality:** the final member is the row pivot and every other member is smaller.
4. **Schedule count:** for a length sweep, row count equals `target_r(L)` and grows by only zero or one per new coordinate.
5. **Prefix consistency:** materializing length `L` directly equals the first `L` coordinates/rows of every longer materialization.
6. **Seed reproducibility:** identical keys reproduce identical supports and OTP prefixes across processes; changing the support seed changes supports; changing only the OTP seed does not change supports.
7. **Rank:** GF(2) row rank equals the number of rows for representative lengths and randomized seeds.
8. **Validation:** reject impossible weights, duplicate parents, invalid noise, and unknown schedule versions.
9. **Rounding boundary cases:** lock expected schedule behavior around every length where `round(0.99L)` changes or ties, including the startup clamp. This prevents silent changes from floating-point or language-specific rounding.

### 2. Encoder algebra tests

1. **Clean syndrome:** materialize a clean online word and assert `H_L @ clean == 0` over GF(2).
2. **Noisy/OTP identity:** assert `H_L @ (xi XOR otp) == H_L @ error`.
3. **Free-coordinate independence:** verify scheduled free coordinates are not computed from parents and vary across document nonces/seeds.
4. **Shared key, fresh documents:** parent/OTP prefixes match across documents while free bits and noise are independently sampled.
5. **Prefix encoding:** encoding to `L` directly matches the first `L` latent/key-dependent coordinates of a longer run when all relevant RNG inputs are fixed.
6. **Batch isolation:** ending or reordering one batch member does not change another member's online codeword stream.
7. **Post-hoc generator construction:** reconstruct \(G\) from a recorded schedule and assert that each free row is the corresponding basis vector and each parity row is the XOR of its recorded parent rows.
8. **Generator/check orthogonality and rank:** materialize \(P\) from the same schedule, assert \(PG=0\), and, for full-rank causal schedules, assert \(\operatorname{rank}(G)=d=T-r\).
9. **Generated-word representation:** form \(u\) from the sampled free clean bits and assert that the already-generated clean word satisfies \(c=Gu\), including schedules with late free coordinates.

### 3. Detector unit tests

1. **Direct-score equivalence:** for a nondegenerate online prefix, compare the new detector with a temporary CSR decoding key passed to the current `Detect`; statistics, `V`, and thresholds should agree.
2. **No evidence:** `L < check_weight`, zero rows, or `V=0` always returns `False` and an explicit reason.
3. **No folding:** verify each soft token appears once and no modulo-`n` aggregation occurs.
4. **Dynamic threshold:** independently recompute `sqrt(2 V log(1/alpha))` for several lengths and weights.
5. **Weight kinds:** cover `map`, `entropy`, and `naive` using the existing soft-token implementations.
6. **One-shot versus anytime:** verify the correct alpha is recorded and that the anytime threshold is never smaller than the corresponding one-shot threshold at a monitored prefix.
7. **Malformed records:** reject mismatched token/trace lengths, unknown schema versions, wrong partition fingerprints, and mismatched online-key fingerprints.

### 4. Synthetic statistical tests

Use model-free synthetic soft tokens so FPR tests are inexpensive and repeatable.

1. For several `L`, support seeds, and weight profiles, sample independent null OTP/text inputs and verify the empirical false-positive count is consistent with the requested bound.
2. Use enough trials to make the claim meaningful. At target FPR `10^-3`, 500 samples are descriptive only; roughly 3,000 zero-failure trials are needed merely to put the rule-of-three 95% upper bound near `10^-3`. Prefer at least 100,000 synthetic trials for stable diagnostics.
3. Exercise highly overlapping parent sets, low-entropy soft tokens, `V` near zero, and maximal `|S_i|=1`.
4. Simulate repeated prefix peeking and verify that the one-shot threshold inflates family-wise FPR while the alpha-spending mode remains within its aggregate target.
5. Measure power against synthetic watermarked streams as a function of `L`, `eta`, row rate, and soft-token weight.

These tests support the implementation and threshold arithmetic; they do not replace the proof obligation that row parities of the OTP are independent. The triangular full-rank invariant and independent OTP stream should be stated explicitly in code documentation.

### 5. Generation integration tests

Use a small deterministic/mock model where possible before spending GPU time.

1. Verify the saved `prc_codeword_bits` equal the exact online `xi` values used by masking.
2. Verify token, `p_trace`, codeword, entropy, and log-probability traces have identical realized lengths.
3. Verify fixed-cap prefix generation is reproducible and prefix-consistent under controlled seeds.
4. Verify EOS handling for one sequence and mixed-length batches, including whether EOS is included.
5. Verify null generation never stores PRC codeword bits but retains the current LM diagnostics.
6. Verify the current fixed-length generator produces unchanged outputs/tests when the online module is present but unused.

### 6. Pipeline and cache tests

Extend the current cache/sharding tests to cover:

1. fixed and online schemes never sharing watermarked cache paths;
2. different schedule versions, row rates, generation models, or stopping policies producing different artifact fingerprints/paths;
3. safe reuse of compatible null prefixes;
4. rejection of natural-EOS versus forced-length cache mismatches;
5. checkpoint invalidation when the online key or parent-table hash changes;
6. shard aggregation rejection for mixed online configurations;
7. restart/resume behavior after partial generation and detection.

### 7. GPU smoke tests

Run in increasing cost order:

1. One prompt, batch size 1, small generation cap, both watermarked and null.
2. Two or more prompts with different EOS lengths to exercise active masks.
3. A small Modal shard with generation trace collection and CPU detection.
4. Re-run the same shard to confirm cache and checkpoint reuse.
5. Change one schedule/key parameter and confirm only key-dependent artifacts invalidate.

## Scientific experiment matrix

Use paired prompts and, where possible, the same null records as the current fixed-length experiment.

Primary comparison:

| Factor | Values |
|---|---|
| Construction | current fixed-`n`; online causal 99%; online every-position prototype |
| Realized/scored length | 400, 512, 1024, 2048, 4096, and any already-supported larger length |
| Check weight | 3 initially |
| Noise `eta` | match current campaigns, beginning with 0.05, 0.10, 0.15, 0.20 as available |
| Target FPR | `10^-3` initially; preserve other current campaign targets where needed |
| Soft-token weight | MAP primary; entropy and naive diagnostics |
| Stopping policy | forced prefix first; natural EOS second |
| Generation/entropy model | use the current model-isolated configurations, reported explicitly |

For each length, report:

- realized `T=n`, `r`, free-coordinate count, and `r/T`;
- TPR and observed FPR with confidence intervals;
- statistic, threshold, margin, and `V` distributions;
- fraction marked `insufficient_evidence`;
- base-LM entropy and token log-probability summaries already supported by current trace collection;
- generation throughput, detector time, and key/materialization memory;
- parent-degree and row-overlap diagnostics, because causal sampling may concentrate checks on early coordinates.

For the forced-prefix phase, generate one long online continuation and score multiple prefixes. This directly tests the required prefix property and reduces GPU cost. Treat the resulting per-length measurements as paired and correlated when estimating uncertainty. For natural EOS, evaluate each record only at its actual final length unless explicitly running the anytime detector.

## Implementation sequence

1. Implement and unit-test the model-free row schedule, key expansion, and encoder algebra in `online_prc.py`.
2. Add the optional post-generation `reconstruct_generator` helper and its GF(2) validation tests; keep it out of encoder state and the generation hot path.
3. Implement `detect_online_hoeffding` and synthetic threshold/FPR tests.
4. Add single-sequence online generation while preserving current trace details.
5. Add mixed-length batch support and integration tests.
6. Add the isolated Modal runner/configuration, artifact validation, and cache tests.
7. Run local/mock smoke tests, then a minimal GPU/Modal smoke.
8. Run the forced-prefix comparison before natural-EOS experiments.
9. Review FPR evidence, causal parent-degree behavior, and quality traces before scaling the campaign.

## Acceptance criteria

The experiment is ready to scale when all of the following hold:

- Every parity row has exact weight `t`, includes its current pivot, and otherwise refers only to distinct previous coordinates.
- Direct and truncated materialization are byte-for-byte prefix-consistent.
- Every prefix parity matrix has full row rank and every clean generated prefix satisfies its checks.
- When optional generator reconstruction is requested, the reconstructed \(G\) satisfies \(PG=0\), has \(\operatorname{rank}(G)=d=T-r\) for a full-rank causal schedule, and reproduces the generated clean word as \(c=Gu\) using the sampled free bits.
- Generator reconstruction is post-processing only and adds no dense-matrix work to the token-generation hot path.
- Generation records always satisfy `T=n=realized token count` and all trace arrays agree in length.
- The online detector performs no folding or block OR, recomputes its threshold from the realized prefix, and never detects with zero evidence.
- Repeated testing is either disabled or explicitly uses the versioned anytime policy.
- Synthetic null tests show no implementation-level FPR inflation, with sample sizes and confidence bounds reported.
- Current fixed-length tests and generation paths remain unchanged.
- Online artifacts cannot collide with fixed-length or differently configured online artifacts.
- A cached/restarted run produces identical key/check prefixes and detection results.

## Main risks and decisions to document

1. **Security interpretation:** removing the permutation is necessary for strict causality and changes the public structure of the code. The experiment should be described as an online causal PRC variant until its pseudorandomness/security argument is reviewed.
2. **Row schedule:** the 99% prefix schedule best matches current runs; the every-position schedule is only an ablation.
3. **OTP generation:** a compact PRF-derived stream gives unbounded expansion but changes a statistical random-key statement into a computational one. Keep independent domains and state which interpretation is used.
4. **Stopping:** one final score and repeated prefix monitoring require different FPR policies.
5. **Early power:** a valid weight-3 check exists at `L=3`, but useful detection requires substantially more effective checks.
6. **Early-coordinate degree:** uniformly sampling parents from the entire past may give old coordinates high degree. Measure it and consider a bounded lookback/window only as a separately labeled follow-up, since that changes the code ensemble.
