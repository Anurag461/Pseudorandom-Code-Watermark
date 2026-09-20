# Fixed PRC 4B generation: paired detector results

Completed September 20, 2026, on branch `redetection`.

| Generator → detector | MAP TPR | Entropy TPR | Watermarked N | Null N |
|---|---:|---:|---:|---:|
| Qwen3-4B-Base → Qwen3-4B-Base | **98/100 (98%)** | **93/100 (93%)** | 100 | 0 |
| Qwen3-4B-Base → Qwen3-0.6B-Base | **93/100 (93%)** | **90/100 (90%)** | 100 | 0 |

Both detectors used exactly the same 100 saved completions, generated once.
Fixed PRC: eta=.05, n=T=1024, one block, t=3, r=1014, analytical target FPR=.001.
Original key/partition seed12345 and the first 100 cached 50-token RealNews
prompts were reused. Two equal partition buckets cover 151936 output rows.

**Empirical FPR was not evaluated.** Null generations and null replays: zero;
their separate cost is $0. CSV FPR fields are `skipped`, not zero percent.
No extra validation pass, benchmark, reference replay, regeneration or retry ran.

## Model and detection provenance

- 4B checkpoint/tokenizer: `Qwen/Qwen3-4B-Base`, revision
  `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- 0.6B checkpoint/tokenizer: `Qwen/Qwen3-0.6B-Base`, revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Both pinned tokenizer.json files are byte-identical; direct token-ID replay
  is compatible. Checkpoint shards and metadata were hash-verified on cloud CPU.
- Protocol: `completion_only_raw_abstain_v1`, raw completion tokens without
  original prompt/BOS/EOT, coordinate 1 score zero, MAP and entropy weighting.
- BF16 inference, TF32 disabled; existing concat generation cache and static
  completion-only replay cache. Float32 probability traces; existing float64
  cloud CPU Hoeffding scoring with `block_or_bonferroni` FPR policy.
- Existing generation sampler retained: temperature1, no top-k/top-p, Bernoulli
  PRC bucket selection and float32 within-bucket multinomial sampling, forced
  1024 tokens with no EOS early stopping.
- Each detector independently recovered 100 x 1023 probabilities. Generation
  probabilities were retained only as provenance and never used for scoring.
- Seed12345 identifies the reused key/partition. Legacy generation does not
  fully seed GF.Random, so seed-only regeneration is not claimed. Exact realized
  codewords, tokens and available RNG states are saved.

## Actual provider-reported spending

| Stage | Hardware | Worker time | Actual cost |
|---|---|---:|---:|
| Checkpoint/key preparation | 4 CPU cores, 16 GiB | 105.68 s | $0.00988003 |
| Generate100 × 1024 tokens | H100, batch100, 4 CPU cores, 64 GiB | 130.46 s | $0.17829656 |
| Prepare paired manifests | 4 CPU cores, 16 GiB | 12.32 s | $0.00229091 |
| 4B primary replay | H100, batch100, 4 CPU cores, 64 GiB | 110.82 s | $0.14728834 |
| 0.6B primary replay | A100 80GB, batch100, 4 CPU cores, 16 GiB | 76.21 s | $0.06416753 |
| Score both traces, MAP + entropy | 4 CPU cores, 8 GiB | 9.89 s | $0.00112540 |
| **Total** | One worker at a time | **445.38 s** | **$0.40304877** |

Costs are from Modal billing by app, before workspace credits. The updated
provider totals supersede earlier partial billing snapshots. Worker times do
not include all startup or local transfer time. The original conservative
proposal was $2.50. Estimated remaining balance is **$6.10**, based on the user's
$6.52 starting estimate less this experiment and the separately identified
$0.01946503 diagnostic usage. `billing_final.json` preserves the itemized evidence.

## Saved outputs and checks

The [existing CSV](../redetection/redetection_results_summary.csv) has exactly
two added rows with explicit N=100 and /100 TPR denominators. All 299 prior rows
and the original header were preserved. Historical/prompted TPRs are
`unavailable`; naive and empirical FPR fields are `skipped`. The machine-readable
summary index includes the four detector/weight results.

`cache_index.json` identifies every saved generation, artifact, candidate,
manifest, detector input, primary trace and score report, with cloud/local
locations and checksums. The generation batch and both primary detector traces
were committed before final scoring. Full per-candidate scores and summaries
were saved in Modal and downloaded with matching checksums. No tensor scoring
or model execution occurred on the laptop. Additional source snapshots and
reproducibility material remain local; no extra archive was uploaded.

Completed output commits before scoring:

- `a9f3e42`: generation batch, original artifact and evidence.
- `1e92c25`: paired detector manifests and preparation evidence.
- `4daa761`: 4B primary trace, pause/resume record and costs.
- `20bc976`: 0.6B primary trace and costs.

The brief connectivity pause stopped the 4B app after its trace had completed
and been saved. No rerun was needed. No changes were pushed. Runtime source
hashes are authoritative because execution used the reviewed working-tree
setup; `setup.json` remains immutable and `progress.json` records completion.
