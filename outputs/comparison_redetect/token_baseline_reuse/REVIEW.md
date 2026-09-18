# SynthID and GumbelMax: cached results added

Added six rows per method to `baseline_comparisons.csv`. The table now has
24 rows: PRC, TextSeal, SynthID and GumbelMax at n = 128, 256, 400, 512, 768,
1024. All use the same 500 original shared T13088 nulls and nominal p < .001.
All 348 cells in the previous 12 PRC/TextSeal rows are unchanged.

No redetection was needed. The original scoring branches call
`official_synthid_g_values(prefix_tokens, positions)` or
`official_gumbel_scores(prefix_tokens, positions)` and apply their fixed
normal/Gamma tests. They use completion tokens and keys. The wrapper reads
prompt metadata, generation log probabilities and entropy, but prompt/log
probabilities only feed identity and quality fields; neither detector uses
them or model entropy. The original generation output stores completion IDs
separately from the prompt. The archived historical scorer and current
`comparison_runner.py` confirm this path.

SynthID retains the existing comparison's frequentist weighted normal test
on Google's g-values. GumbelMax retains the existing exact Gamma test using
TextSeal's uniform PRF. The generation variants, keys, context length 3 and
v2 context-token deduplication are unchanged. These are the same cached
comparison variants, with no new calibration or detector substitution.

| n | SynthID TPR | SynthID FPR | GumbelMax TPR | GumbelMax FPR |
|---:|---:|---:|---:|---:|
| 128 | 500/500 | 1/500 | 500/500 | 1/500 |
| 256 | 500/500 | 0/500 | 500/500 | 1/500 |
| 400 | 499/500 | 0/500 | 500/500 | 2/500 |
| 512 | 499/500 | 0/500 | 500/500 | 0/500 |
| 768 | 500/500 | 0/500 | 500/500 | 0/500 |
| 1024 | 500/500 | 0/500 | 500/500 | 0/500 |

## Verification

- SHA-verified the historical prompt-level file, aggregate, audit and artifact manifest.
- SHA-verified all ten local raw generation shards before reading them.
- Matched all 1,000 method-specific watermarked completion hashes to those shards.
- Matched all 500 shared null hashes to the frozen TextSeal input export,
  already aligned with PRC.
- Checked all 12,000 cached method/sample/prompt/prefix records for complete,
  unique coverage, model/source revisions, configuration and p < .001 decisions.
- Counts agree exactly with the independently published historical aggregate.
- Read back the CSV independently with Python and checked the artifact-tool
  roundtrip. Every prior PRC/TextSeal cell and provenance field is retained,
  apart from the necessary updated whole-CSV checksum.

Detector calls: 0. Model forwards: 0. Remote calls/jobs: 0. Additional cloud
compute cost: $0. All work used existing local files.

`Old TPR` and `TPR` are equal for these reused methods. Their entropy-model
field is `not used`; generation still used Qwen3-8B-Base. The `Score` field is
`p_value`, with the calibration stated in Notes and the method configuration
saved in provenance.

- [summary.json](summary.json): counts, source hashes, settings and identity checks.
- [verification.json](verification.json): publication and readback checks.
- [historical_scorer.py.txt](historical_scorer.py.txt): source inspected for input dependencies.
- `before.csv` and `before.provenance.json`: preserved previous comparison state.

Reproduction uses `baseline_comparison.reuse_token_baselines` to validate and
aggregate local cached records, then `publish_cached_baselines.mjs` with the
bundled artifact runtime to append the rows. The publisher can reproduce the
same final CSV from its frozen before snapshot, but refuses unrelated concurrent
changes. The preparation command refuses to overwrite its frozen snapshot
after publication. Original scores remain in the SHA-verified historical JSONL.
