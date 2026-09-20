# Approved online 0.6B redetection

Approved by the user on 2026-09-19. Freeze and verify source data, reuse matching null traces, then run watermarked replay for the four families below.

| eta | Longest replay | Recorded lengths to report | Points | GPU / batch |
|---:|---:|---|---:|---|
| .05 | 512 | 256, then 400–512 every 16 | 9 | A10G / 100 |
| .10 | 1024 | 784–1024 every 16 | 16 | A100 80GB / 125 |
| .15 | 2048 | 1024–2048 every 16 | 65 | A100 80GB / 125 |
| .20 | 4096 | 3088–4096 every 16 | 64 | A100 80GB / 100 |

Four original seed-12345 families, 154 recorded length points. All four longest caches have their artifact and watermarked files 0000–0499 present. Every shorter length is a CPU rescore of the corresponding longest completion trace; no short-length generation or model replay is needed. The .05 n256 result comes from the canonical n400 lineage, whose prefixes were verified for all 500 n512 completions in the saved ceiling-preparation report. These identities will be checked again in preflight.

Each point uses 500 watermarked and 500 null candidates; Qwen3-0.6B-Base, BF16, raw completions, coordinate 1 score zero, t=3, target FPR=.001, the original causal row schedule, and one-shot scoring at each length. MAP and entropy TPR/FPR are included. A per-length nominal FPR is not a simultaneous guarantee across the sweep.

All four vocabulary partitions match the completed fixed-run partition. All 500 full null-source file hashes match an existing completed fixed run for each family. Reuse null traces from fixed eta=.10 n512, fixed eta=.10 n1024, fixed eta=.15 n2048, and fixed eta=.20 n4096 respectively. The required input lengths, batch sizes, GPU families and pinned BF16 model match those completed runs. This saves new null GPU inference. Retain the original source identities and trace checksums when importing caches.

New replay covers only 500 watermarked completions per family: 18 batches and 3,838,000 candidate token positions, plus the existing representative-batch validation overhead. GPU and batching choices match successful recent redetection settings. No benchmark study is planned.

Eta=.20 n3104 already has a completed redetection result and will serve as a consistency check against the 4096-derived prefix. Separate older-sampler and seed-24680/54321/67890 cohorts require their own replay and are outside this four-family proposal. Completed comparison experiments remain complete.

Append completed results to `outputs/redetection/redetection_results_summary.csv`, maintaining fixed rows first, then increasing eta and length. Preserve full-length traces, source hashes, keys/partitions, imported-cache provenance, per-candidate prefix scores and final summaries in persistent Modal storage, with local reports and cache indexes.

Exact source tags, length lists, evidence hashes and remaining preflight checks are in [proposed_plan.json](proposed_plan.json).
