# Proposed first eta=0.20 batch: native 8B, T=14336, N=50

Prepared on `redetection`. **No paid computation launched. Approval pending for all three stages below.**

## Exact scope and reuse

- Online PRC eta=0.20. Use existing watermarked prompt IDs 0–49 from the 500-record T=14336 generation cache. This is one reusable production batch, not a disposable benchmark.
- Replay those 50 completions once with the native 8B detector, batch size 50, on one H200. Save the full T=14336 completion-only traces so this batch can be reused in a later 500-record campaign and for shorter-prefix scoring.
- New generation: **0, $0**. Null generations/replays: **0, $0**. Empirical FPR remains unmeasured. 0.6B detector work is outside this single-batch proposal.
- Read-only Modal inventory found all 500 source files; inspected 56 integrated completion-only manifests and found no matching full-length detection cache. The selected 50 records plus artifact were downloaded and SHA256-frozen. Non-executing pickle-opcode inspection confirmed prompt IDs, watermark flags, generator labels and common artifact/key fingerprints. No tensor or model work ran locally.

## Settings

Native detector: `Qwen/Qwen3-8B-Base`, model/tokenizer revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Tokenizer SHA256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`, vocabulary 151936. Cached config and weight-index hashes match the established checkpoint. Exact weight shard hashes are in setup.json.

Use BF16, TF32 disabled, static KV, raw saved completion token IDs, no prompt or synthetic prefix, coordinate-one abstention, MAP and entropy (`completion_only_raw_abstain_v1`). Target FPR=0.001, one-shot policy; t=3, seed=12345, causal row rate 99/100, r=14193 at T=14336. Reuse the exact source key and partition. Historical generation used position-addressed inverse-CDF sampling, temperature 1, unrestricted vocabulary and forced length. No resampling or retokenization.

Generation-cache provenance: original August 17, 2026 campaign, first 25 retained pilot records and the next 25 production records. Source artifact fingerprint: `ef0d12e6e60554f38c85864fcafcba18e83a6c93d7ca63f8735a400f91ef859b`. Online key SHA256: `ceb3d5de3db51e3b776ebc8296737fdf2edb1761d6e16326ee443c7611daf2dc`. The current detector revision is pinned; an unrecorded historical generation revision is not inferred.

## Stages, time and incremental cost

| Stage | Resources and workload | Expected time | Expected USD | Allowance |
|---|---|---:|---:|---:|
| Prepare | One 4-core CPU, 16 GiB; verify 50 cached sources/checkpoint and freeze one input batch | 0.5–3 min | $0.003–0.02 | $0.05 |
| Primary replay | **One H200, batch 50**, 4 CPU cores, 64 GiB host RAM; native 8B at T=14336 | 65–90 min | $5.80–8.30 | $9.40 |
| Score | One 4-core CPU, 8 GiB; saved traces, MAP and entropy at T=14336; N=50 CSV row | 0.25–2 min | $0.002–0.01 | $0.05 |
| **Total** | Dependency-ordered; maximum one GPU | **66–95 min** | **$5.81–8.33** | **$9.50** |

The H200 plus specified host resources costs $0.00145548/s ($5.239728/hour) at the [Modal rates checked for this proposal](https://modal.com/pricing). The time estimate scales the successful batch100/T6144 native replay (1642.67 seconds) by batch count and squared sequence length, giving a 4471.71-second method estimate before startup. The range includes loading, I/O and runtime uncertainty; no paid timing benchmark was run.

Work deadlines are 300 seconds for preparation, 6000 seconds for GPU replay and 180 seconds for scoring, plus bounded startup/shutdown. Retries=0; durable attempt markers prevent repeating work after a provider restart. A timeout/OOM would preserve available evidence and require a new estimate and approval before any retry or fallback.

Actual prior eta=0.15 spending including prefix scoring is **$28.68611404**. That leaves **$6.42388596** against its earlier $35.11 target. This separate proposal needs a new **$9.50** allowance; spending the entire allowance would put the combined amount at **$38.18611404**, $3.07611404 above that old target. The billing API reports usage, not the remaining account credit balance.

## Batch-50 memory and code changes

The measured B100/T6144 native replay used 117.26 GB live allocation. Replacing its 90.58 GB static KV cache with the 105.69 GB B50/T14336 cache projects 132.37 GB live allocation; allow roughly **132–135 GB** for workspace variation. The reference H200 exposed 143771 MiB (about 150.75 GB). This supports feasibility but does not establish the new shape's actual peak or speed.

The old helper rejected more than 85% live allocation after completing inference and before saving its trace. This batch proposes an explicit **95% live-memory limit**. The default remains 85% for other calls, and a completed valid trace is now saved before the margin check. The wrapper commits saved traces even on failure. Sampling and detector arithmetic are unchanged. Three pure Python mock tests passed, covering the unchanged default, the explicit 95% option, trace retention and rejection of scope expansion.

The isolated `online_8b_eta020_batch50.py` wrapper uses the existing preparation, replay and cloud scoring functions. It retains the corrected scientific/native import order before NumPy compatibility aliases and keeps open worker logs in /tmp during volume reloads. Model execution and scoring remain cloud-only.

## Output and execution safeguards

Cache inputs, prompts/provenance, exact keys/partition, manifests, primary trace, scores, file hashes, memory/time and actual cost. Download and commit primary replay evidence before CPU scoring. Append a single native detector result row to the existing CSV with **N=50**, null N=0 explicit; preserve all current rows. Do not infer N=500 from this initial batch. Keep extra source archives local; no additional archive upload or push.

All stages are separately gated on the same frozen setup hash and their own explicit approvals. No approval or attempt files exist. The proposal covers preparation, one primary GPU replay and scoring only—no independent reference replay, benchmark, null work or automatic retry.
