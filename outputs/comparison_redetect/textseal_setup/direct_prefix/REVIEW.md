# TextSeal completion-only redetection: completed

All 500 TextSeal completions and 500 original shared nulls were processed at
n = 128, 256, 400, 512, 768, 1024. The user authorized the full cohort after the
replacement pilot passed. Six TextSeal rows are now in
`outputs/comparison_redetect/baseline_comparisons.csv`; all original PRC cells
and underlying results are unchanged.

| n | Old TPR | Completion-only TPR | Old null positives | Completion-only null positives |
|---:|---:|---:|---:|---:|
| 128 | 500/500 | 500/500 | 1/500 | 0/500 |
| 256 | 500/500 | 500/500 | 0/500 | 0/500 |
| 400 | 500/500 | 500/500 | 0/500 | 0/500 |
| 512 | 500/500 | 500/500 | 0/500 | 0/500 |
| 768 | 500/500 | 500/500 | 1/500 | 1/500 |
| 1024 | 500/500 | 500/500 | 0/500 | 0/500 |

These are separate one-shot tests using upstream `p_value_weighted < 0.001`.
TPR is 100% at every length. Empirical FPR is 0%, except n=768 at 0.2%.
There were no abstentions. The unchanged complete upstream result dictionary,
including its separate combined-score decision at 0.01, remains in the records.
Historical comparison decisions used prompt-conditioned generation entropy;
they are retained only as the labelled old results.

## Exact execution and validation

The [diagnostic](../prefix_diagnostic/REVIEW.md) showed shape-dependent BF16
outputs, so each actual prefix runs independently through original upstream
`_compute_entropies` and `_score_text`. No longest-trace slicing is used for
TextSeal. No prompt, BOS/EOT, padding, retokenization, generation entropy or
external KV state enters a model call. Repeat handling and generation are unchanged.

The ten-response pilot independently compared the adapter with upstream public
`detect` on identical IDs at each length: all 60 entropy vectors and complete
result dictionaries matched exactly. Its 120 actual forward inputs were checked.
The full run reused those ten verified records, then performed 5,940 new
forwards for 990 responses. Every actual model input was observed and checked.

Readback validated all 1,000 record checksums, execution/input identities,
6,000 prefix identities, entropy lengths and comparison decisions. The 500
null token sequences match the completed PRC shared-null alignment exactly.
A separate JavaScript aggregation of raw upstream p-values matched all counts.
The CSV roundtrip and independent Python readback preserved every one of the
132 original PRC cells. All original PRC provenance fields remain unchanged
apart from the updated whole-CSV checksum.

Local validation: 52 detector/setup/diagnostic tests previously passed, plus
six new result-publication guard tests. A local CPU rescore with different
NumPy/SciPy versions was not bitwise exact (up to 1.08e-9 on the first pilot
null); it is not used as the reference or to replace scores. Authoritative
scores and exact cache checks use the frozen runtime below.

## Frozen setup

| Item | Setting |
|---|---|
| Model | Qwen/Qwen3-8B-Base, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Runtime | H100, torch 2.4.0, Transformers 4.51.3, BF16, eager attention, batch 1 |
| Precision/state | TF32 off, BF16 reduced-precision reduction off, eval, use_cache=False |
| Upstream | Original source at `c60d0d1da2e59f09a698438e218a07ee779b4616`, SHA-checked |
| Scoring | Keys 42/12387, ngram 3, alpha .1, v2; weighted p < .001 |
| Shared nulls | Original `_nulls/qwen3_8b_base/T13088`, same 500 as PRC and the other baselines |
| Resources | One H100, 4 cores, 64 GiB, zero retries, 2-second scaledown |
| Full budget | $5 planning cap; function timeout at most 3,600 seconds |

Manifest: [native8b_manifest.json](native8b_manifest.json), canonical SHA-256
`3acd757e4f508013b4e4a617874a8366e6aac5e97a86456efd6d4278dcc82ee0`.
The input export is identical to the original setup, with all 1,000 identities
frozen in the manifest. [Destination verification](destination_verification.json)
confirmed the existing Modal workspace and data. The earlier pilot's first
dispatch was blocked by automatic review until that destination check passed;
its retry and this separately authorized full run were approved.

## Runtime and cost

Full replay: **666.489 seconds (11.11 minutes), about $0.861 measured compute**,
including 27.150 seconds loading. The conservative prelaunch projection was
$1.79; the full report's `estimated_full_usd` field is a generic projection
recomputed from full-run timings, not an additional charge or remaining work.
Pilot: 36.716 seconds, about $0.047. Diagnostic: 48.758 seconds, about $0.063.
The three measured resource totals sum to approximately $0.97, excluding
startup/image/storage overhead. These are compute estimates, not a billing invoice.

## Results and reproduction

- [Full summary](full_summary.json): counts, model/runtime identity, source hashes and checks.
- [Full execution](full_execution.json): user scope, app ID and measured runtime.
- [CSV verification](comparison_verification.json): original PRC cells preserved.
- [Result index](result_index.json): remote reports archive and raw record locations.
- [Pilot report](pilot_report.json) and [pilot readback](verification.json).
- [Full Modal app](https://modal.com/apps/new-prc-watermark/main/ap-UDzsdlQWCoJULxBP7BzZI2).

The collector `baseline_comparison.textseal_results` is a read-only Modal
client. It validates saved records and emits JSON counts; it launches no
compute. `baseline_comparison/publish_textseal_comparison.mjs` publishes those
counts with the bundled artifact tool. Generic `Method`, `Score`, `Old TPR`,
`TPR`, `FPR` and delta columns show each primary comparison. PRC's primary is
posterior; its entropy-aware results remain in their original named columns.
TextSeal's PRC-specific columns are deliberately blank.

Raw record files are stored on `prc-completion-only` at
`textseal_completion_redetect/<manifest-sha>/records/<method>/<index>.json`.
The full report's record hash map binds every file. A full-stage rerun checks
and reuses these caches; changed code, model, runtime or input identity is rejected.
Proxy-0.6B replay and adding SynthID/Gumbel rows are separate remaining steps.
