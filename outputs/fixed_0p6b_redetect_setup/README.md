# Fixed 0.6B single-block redetection

**Completed:** all 18 main settings and both seed replicates. [Results and verification](RESULTS.md). Eta=.20 n8192 remains deferred.

Prepared September 19, 2026. Scope: Qwen3-0.6B-Base generation and detection, fixed PRC, T=n, t=3, target FPR=.001, 500 watermarked and 500 null candidates per setting.

The main launch contains 18 remaining settings. Previously completed eta=.05 n400/n448 are excluded. Eta=.20 n8192 is deferred at the user's request. Two n256 eta=.05 seed replicates (54321 and 67890) are prepared separately. Multi-block, online, cross-model, and completed comparison experiments are outside this launch.

| eta | Main n values |
|---:|---|
| .05 | 256, 416, 512, 1024, 2048 |
| .10 | 256, 400, 512, 768, 1024 |
| .15 | 256, 400, 512, 1024, 1504, 2048 |
| .20 | 2048, 4096 |

## Hardware

| Length | GPU | Candidates per batch |
|---|---|---:|
| 256–512 | A10G | 100 |
| 768–2048 | A100 80GB | 125 |
| 4096 | A100 80GB | 100 |

These choices reuse existing runs: the n400 replay used A10 with batch100 (14.56 GB peak reserved); n448 used A100 80GB with batch125 (8.68 GB peak live); n3104 used A100 80GB with batch125 (48.98 GB peak live). The n4096 batch100 has about 6% more token/cache capacity than the proven n3104 batch125. Replay uses BF16 and the static KV cache. No additional performance benchmark is required. Each full run validates one complete representative batch before parallel replay; the existing worker enforces its live-memory margin. At most ten workers are used per launch.

References: `outputs/redetection/current/same_0p6b_eta005_n400/inference.json`, its `run.json`, and `outputs/redetection/README.md`.

## Validation and provenance

Every source is hashed and tied to its original workspace and prompt index. Original MAP, entropy, and naive counts are reproduced exactly before the production CPU preflight. Historical reports also check candidate token hashes where available. Legacy eta=.10 n256/n400 use their original interleaved generation records; eta=.15 n1024 uses the retained lower-n cohort; eta=.20 uses its original workspace shards. The newer n416/n768/n1504 settings take their null cohorts from their original complete shard reports.

The integrated replay preserves the original key, partition, parity checks and FPR policy. It consumes only raw completion tokens, sets coordinate 1's score to zero, and recovers probabilities for coordinates 2..T. MAP and entropy results are written; naive result columns are marked skipped by the existing replay runner.

`index.json` lists all successfully frozen settings, GPU choices, historical audits, checksums, and persistent cache locations. Manifests are split by GPU. Production execution files must match their Git commit before a launch.

## Run

The runs below are complete. These commands record the original launch. Their execution commit was `23b8f280380c24a7e8899313874de70d544afc50`; cache identity includes that commit, so use the original execution checkout to resume/reproduce without creating a new inference run.

From the repository root, using a Python environment containing Modal:

```sh
bash outputs/fixed_0p6b_redetect_setup/run.sh full main
bash outputs/fixed_0p6b_redetect_setup/run.sh full replicates
```

Use `all` for both groups. `main_a10g` and `main_a100` select either hardware group independently. Omitting arguments performs CPU preflight only. The wrapper uses the `new-prc-watermark` Modal profile. Set `PRC_REDETECT_PYTHON` if needed.

Each completed setting appends to **`outputs/redetection/redetection_results_summary.csv`**. Existing rows are preserved. CPU preflight and smoke runs do not add placeholder results. Full replay is resumable with the same frozen manifests and execution commit; completed probability batches are validated and reused.

## Persistent cache

The Modal volume **`prc-completion-only`** holds:

- Frozen setup manifests, historical audit evidence and prepared run metadata under `setups/fixed_0p6b_single_block/`.
- Original scoring keys/partitions, exact completion token batches, run configuration and source hashes under each run root listed in `index.json`.
- On replay: probability tensors, tensor checksums, representative-batch validation, peak memory and timing in each `batches/*/trace.pt`.
- On scoring: `full.json` with all candidate scores/decisions and `summary.json` with counts and trace-file checksums.

Original archives remain in `prc-research-archive`; newer source generations remain in `prc-data`. Clean scoring artifacts and exact token inputs are independently copied into the prepared results cache. The pinned model checkpoint remains in `prc-hf-cache` and is checksum-verified when loaded. No original generations or caches are deleted.

Local detailed reports are saved by the runner under `outputs/redetection/.archive/runs/`; the CSV records each remote root. `cache_bundle.json` records the uploaded setup bundle and verifies its read-back checksums. The setup contains no newly inferred result rows until a full replay finishes.
