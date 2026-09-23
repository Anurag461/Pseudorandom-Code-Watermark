# Redetection inventory — September 19, 2026

This is the inventory **before** the subsequent single-block batch. Since this snapshot, all 18 selected main settings and both fixed seed replicates have completed. Eta=.20 n8192 remains deferred. The main modern-ledger pending count is now 42 (previously 60); the 25 legacy multi-block rows remain pending. See the [completed results](../../outputs/fixed_0p6b_redetect_setup/RESULTS.md). Counts and classifications below preserve the original planning snapshot.

The comparison campaign is complete. The remaining correction is primarily the historical PRC length/noise, model-size, boundary, and construction results. This inventory is based on local result files, archived manifests, the current cache index, and the current runner. No model jobs, new generations, or remote cache downloads were launched; historical results remain unchanged.

**Main ledger:** `hoeffding_results_summary.csv` has **88 rows**: **63 single-block/online rows**, of which **3 have completed MAP/entropy replacements and 60 do not**, plus **25 older multi-block rows** without replacements. These are result rows, not GPU jobs. The overlapping panels below must not be added together. The machine-readable [inventory](inventory.json) records all 88 rows with original line numbers, all 28 online-ledger rows, both fixed replicates, 14 full sweep files, 14 paired arms, 56 proxy detector rows, and source hashes.

## Already complete

All six indexed result files exist locally and match their recorded SHA-256 hashes; their trace directories also exist. This verifies result-file integrity, not a fresh rehash of every tensor shard.

| Construction | Generator → detector | η | Length(s) | Coverage |
|---|---|---:|---|---|
| Fixed | 0.6B → 0.6B | .05 | 400, 448 | MAP and entropy, 500 WM + 500 null each |
| Online | 0.6B → 0.6B | .20 | 3104 | MAP and entropy, 500 + 500 |
| Online | 8B → 0.6B | .05 | 640 | MAP and entropy, 500 + 500 |
| Online | 8B → 8B | .05 | 1280, 1024 | MAP and entropy; 1024 reuses 1280 traces |

Sources: [redetection summary](../../outputs/redetection/summary.json), [cache index](../../outputs/redetection/cache_index.json), [results](../../outputs/redetection/README.md). Only fixed 400/448 and online 0.6B 3104 correspond to rows in the main 88-row ledger. The other three completed settings live outside it.

Also complete: the native-8B four-method comparison at lengths **128, 256, 400, 512, 768, 1024**, including PRC completion-only replay, corrected TextSeal, and the shared T13088 null cohort. The Self-BLEU/repeat-policy/depth/top-k/0.6B/temperature campaign is already covered by the closed [comparison report](../comparisons/REPORT.md). Its historical proxy panel is explicitly excluded from the corrected evidence. None of the completed comparison experiments needs repeating for this task.

## Main historical ledger: exact remaining rows

| Family | Generator → detector | Existing rows | MAP/entropy replacements available | Remaining |
|---|---|---:|---:|---:|
| fixed multi | 0.6B → 0.6B | 23 | 0 | 23 |
| fixed multi | 0.6B → 4B | 2 | 0 | 2 |
| fixed single | 0.6B → 0.6B | 21 | 2 | 19 |
| fixed single | 0.6B → 4B | 17 | 0 | 17 |
| fixed single | 8B → 8B | 5 | 0 | 5 |
| online | 0.6B → 0.6B | 6 | 1 | 5 |
| online | 14B → 0.6B | 2 | 0 | 2 |
| online | 14B → 14B | 4 | 0 | 4 |
| online | 8B → 8B | 8 | 0 | 8 |

Fixed single-block settings to redetect (all t=3, FPR=.001, T=n):

| Generator → detector | η | Pending n |
|---|---:|---|
| 0.6B → 0.6B | 0.05 | 256, 416, 512, 1024, 2048 |
| 0.6B → 0.6B | 0.10 | 256, 400, 512, 768, 1024 |
| 0.6B → 0.6B | 0.15 | 256, 400, 512, 1024, 1504, 2048 |
| 0.6B → 0.6B | 0.20 | 2048, 4096, 8192 |
| 0.6B → 4B | 0.05 | 256, 400, 512, 1024, 2048 |
| 0.6B → 4B | 0.10 | 256, 400, 512, 1024 |
| 0.6B → 4B | 0.15 | 256, 400, 512, 1024, 2048 |
| 0.6B → 4B | 0.20 | 2048, 4096, 8192 |
| 8B → 8B | 0.05 | 416, 749 |
| 8B → 8B | 0.10 | 768, 1382, 1625 |

This is **19 pending native-0.6B fixed settings**, **5 native-8B fixed settings**, and **17 cross-model 0.6B→4B fixed settings**. Different fixed block lengths use different keys/completions and cannot be treated as prefixes of one online run. The 4B rows are detector replays of 0.6B generations, not 4B generation experiments.

## Headline online operating points and boundary sweeps

| Generator → detector | η | Historical endpoint/ceiling | Remaining work |
|---|---:|---:|---|
| 0.6B → 0.6B | .05 | 448 | Replay original online cohort; fixed n448 completion does not cover it |
| 0.6B → 0.6B | .10 | 800 | Replay and score |
| 0.6B → 0.6B | .15 | 1504 | Replay and score |
| 0.6B → 0.6B | .20 | 3104 | Endpoint done; corrected MAP is 89.6%, so the old 90% crossing no longer holds |
| 8B → 8B | .05 | 640 | Watermarked trace reuse + CPU prefix scoring; preserve/verify historical null cohort |
| 8B → 8B | .10 | 1407 | Replay and score |
| 8B → 8B | .15 | 4096 ceiling | Replay and score; historical boundary is censored |
| 8B → 8B | .20 | 13088 | Replay and score |
| 14B → 14B | .05 | 880 | Add pinned 14B replay support, then replay and score |
| 14B → 14B | .10 | 1808 | Add pinned 14B replay support, then replay and score |

Of these ten headline endpoints, nine lack an indexed replacement. An endpoint rescore alone does not establish a new boundary. There are **10 full-cohort source families** underlying **14 saved sweep files**; the table counts unique evaluated lengths per family, including fine-grid and matched-reference files. These are historical evaluated lengths, not all lengths originally requested.

| Native model | η | Saved WM ceiling | Evaluated lengths | Count | Replay opportunity |
|---|---:|---:|---|---:|---|
| 0.6B | 0.05 | 512 | 400–512 | 8 | New replay of saved ceiling; score prefixes on CPU |
| 0.6B | 0.10 | 1024 | 784–1024 | 16 | New replay of saved ceiling; score prefixes on CPU |
| 0.6B | 0.15 | 2048 | 1024–2048 | 65 | New replay of saved ceiling; score prefixes on CPU |
| 0.6B | 0.20 | 4096 | 3088–4096 | 64 | Existing traces stop at 3104; longer saved completions need replay |
| 8B | 0.05 | 1280 | 624–1280 | 43 | Existing n1280 WM traces cover whole grid; CPU rescore |
| 8B | 0.10 | 3072 | 1392–3072 | 108 | New replay of saved ceiling; score prefixes on CPU |
| 8B | 0.15 | 4096 | 3200–4096 | 57 | New replay of saved ceiling; score prefixes on CPU |
| 8B | 0.20 | 14336 | 13072–14336 | 80 | New replay of saved ceiling; score prefixes on CPU |
| 14B | 0.05 | 1280 | 448–1280 | 28 | Requires pinned 14B detector support |
| 14B | 0.10 | 3072 | 800–3072 | 82 | Requires pinned 14B detector support |

Re-estimate the crossing using completion-only scores and the original grid/stopping convention; refine or extend within existing saved ceilings as needed. Generate new text only if a separately chosen new ceiling exceeds the existing completion cache. The historical 8B η=.20 crossing has a 16-token bracket, unlike the one-token refinements at η=.05/.10.

Additional existing matched-reference rows: **8B native** at (.05,448), (.10,800), (.15,1504), (.20,3104); **14B native** at (.05,448), (.10,800). All need corrected scores and can share watermarked replay with their larger source-family runs. The older 0.6B online η=.05 n256/n400 rows also remain prompted.

Sources: [main summary](../../hoeffding_results_summary.csv), [online ledger](../../online_causal_results_summary.csv), [14B runbook](../../modal_online_14b_runbook.md), [8B reference audits](../../outputs/online_8b_cross_model_boundary_audits_summary.csv), and exact sweep paths/lengths in [inventory.json](inventory.json).

## Fixed versus online and seeded replication

The seven existing paired cells are 0.6B (.05,448), (.15,1504), and 8B (.05,416), (.05,749), (.10,768), (.10,1382), (.10,1625). They contain **14 construction arms**. Only the fixed 0.6B n448 arm has a MAP/entropy replacement, leaving **13 arms**, with substantial overlap with the fixed grid and online replay families above. Rebuild the paired effects, intervals, and tests after rescoring; the existing 21-comparison table is historical. Five of those online 8B arms are not separate rows in the main ledger.

The **two fixed n256 η=.05 seed replicates (54321, 67890)** and their matching online replicates are also uncorrected. The online ledger additionally contains older seed-12345 runs, the poscdf-v1 n256/n400 lineage, and seed-24680 n256/n400. Preserve sampler and seed identities: equal n/η/model is insufficient for reuse. Exclude 2/27-prompt smokes, duplicate progress rows, KV-cache equivalence checks, and partial shards from production rerun totals.

Sources: [paired configuration](../../fixed_vs_online_analysis_config.json), [paired report](../../fixed_vs_online_analysis.md), [fixed replicates](../../fixed_replicate_results_summary.csv), [online ledger](../../online_causal_results_summary.csv).

## Cross-model detection

- **8B→0.6B PRC:** the proxy summary contains 28 η/length settings × 2 weights = 56 rows. Only η=.05,n640 has an indexed replacement (2 rows). **27 settings / 54 rows remain**: all six common lengths for all four η values, plus endpoints 1407/.10, 4096/.15, and 13088/.20. Existing n640 traces may cover .05 prefixes ≤640 after exact token/null checks; prefixes 768/1024 require longer replay.
- **14B→0.6B PRC:** two settings, η=.05,n880 and η=.10,n1808, both still prompted; no new detector-model implementation is needed for the 0.6B replay.
- **0.6B→4B fixed PRC:** the 17 single-block settings above, plus two legacy multi-block rows; current replay runner lacks pinned 4B support.
- **Historical TextSeal proxy sensitivity:** six common-prefix rows still use prompted entropy. Repair only if that archived proxy comparison is to be republished; the closed native comparison does not depend on this. Native rows in the old proxy table should use the corrected comparison results; SynthID/Gumbel token-only scores can be reused after cohort verification.

Sources: [proxy PRC summary](../../outputs/proxy_8b_prc_summary.csv), [proxy common-method summary](../../outputs/proxy_8b_common_method_summary.csv), [proxy report](../../proxy_8b_detector_report.md), [14B replay runbook](../../modal_online_14b_runbook.md). These are overlapping reporting views, not additional generation cohorts to sum.

## Older results and dependent figures

- **25 multi-block rows in the main ledger** (23 native 0.6B, 2 with 4B detection): all uncorrected. They include alternative t/r choices, η=.20, and FPR sweeps. Six rows are alternate FPRs for existing cohorts, so their extra work is CPU threshold/scoring work after a shared replay. Preserve original block-OR/Bonferroni semantics. Treat this as a separate legacy bucket, to revive only if these results are retained.
- **39 rows in `results_summary.csv`**, early analytic-threshold/FPR outputs, and old calibration studies use older detectors/protocols. They need a protocol/source audit if retained; this inventory does not assert all can be passed unchanged through the current runner. Do not count backups or repeated exports as new experiments.
- **Reliability/entropy explanations and paper assets:** refresh detection-linked statistics, TPRs, evidence-versus-information plots, paired comparisons, and n90 claims after replay. In particular the fixed 0.6B/8B (.05,416) and (.10,768) reliability analysis uses prompted detector evidence. Generation-conditioned entropy/NLL remains a legitimate generation measurement when labeled as such; a completion-only detector explanation needs corresponding response-only traces.
- **No redetection required merely because of this protocol change:** generated text, task utility scores (`benchmark_utility_results.csv` has 6 rows, with detection off), repetition/distinct-n/Self-BLEU, or token-only attack/key-recovery statistics. Any retained attack claim that consumes a PRC model-probability detector needs a separate audit. Unrun 14B η=.15/.20 work is new experimentation, not redetection backlog.

## Practical execution order and constraints

1. Freeze a cohort-aware manifest for the current PRC paper panels. Reuse the six verified completed results. Score the native-8B η=.05 prefixes/boundary from existing watermarked traces, with explicitly selected and verified nulls.
2. Replay the missing 0.6B and 8B online source families once to the longest required saved length; derive endpoints, grid scores, and paired online arms on CPU. Then replay the missing fixed cohorts, starting with those shared by the paired/reliability analyses.
3. Complete retained proxy panels and add validated 14B/4B detector support where needed. Rebuild statistical summaries and figures. Keep legacy multi-block/early-calibration work separate from the primary queue.

**Null cohorts are a real dependency.** The native-8B n1280 redetection uses T1382 nulls; the corrected comparison uses T13088 nulls replayed only through 1024; historical native n640/n416 audits use T768 nulls. The comparison audit found that all 500 T1382 and T13088 null completions differed within 1024 tokens. Matching prompt IDs or a larger source length cannot establish reuse. Reuse requires exact completion-prefix, partition, model, and protocol identities. See [shared-null audit](../../outputs/comparison_redetect/prc_shared_nulls/REVIEW.md).

**Current runner capability:** `modal_run.py::_redetect_model_spec` accepts pinned BF16 **0.6B and 8B** detectors; README wording claiming only 0.6B is stale. Both fixed and online constructions are supported. Manifest weights are restricted to **MAP and entropy**, so the historical naive columns are not repaired by the six completed runs. Retaining naive results requires a CPU-only token-based rescore with an explicit first-coordinate policy; no probability-model inference is necessary for naive alone.

This is a local evidence inventory, not a fresh remote storage census or an execution-ready manifest. Historical source tags are recorded, but every pending cohort still needs source/key/null integrity preflight. Cost and unique GPU-job totals are deliberately not inferred from overlapping CSV-row counts.
