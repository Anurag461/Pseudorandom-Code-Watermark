**Completed September22:** all500 scored; posterior87.4%, entropy83.6%; Modal-reported cost$25.44 of$35. [Results](RESULTS.md). The following setup and intermediate updates are retained as execution history.

# 8B→0.6B η=.20/T14336 — H200 batch50 setup

Prepared September 22, 2026 on `redetection`. **Execution update: preparation passed and first50 is active. The user subsequently authorized automatic completion if total cost remains below $35, or the original A100/batch25 fallback within the same budget.** See [execution status](RUN_STATUS.md) and [conditional authorization](conditional_continuation_authorization.json).

The original first50 approval gate described below is historical. The frozen driver and setup are unchanged; [the continuation controller](execution_sources/eta020_budgeted_continue.py) records stage-specific approval from the user's latest instruction only after measured timing, cache coverage and billing checks pass. A100 fallback uses [a separate driver](execution_sources/eta020_a100_fallback.py), retaining the successful first50 H200 trace and replaying only450, in18 batches25 on9 workers. Its CPU preparation reshards existing verified inputs; scoring combines all500. With the full$4.75 first-phase allowance reserved, A100 projects$31.20 total; worker deadlines4178s plus lifecycle reserves give a$34.90 resource envelope. The exact deadline is recomputed from measured incurred costs. No additional benchmark or validation run is dispatched.

The following is the original frozen H200 proposal, retained for audit.

## Exact workload

Use all 500 saved Qwen3-8B-Base watermarked completions at η=.20, T=n=14336, prompt IDs 0–499. Detect with pinned Qwen3-0.6B-Base. Report posterior-mean (`map`) and entropy TPR at **T14336 only**; target FPR .001, `one_shot`. Empirical FPR is not evaluated. No generation, nulls, shorter-prefix sweep, native-8B replay, independent reference pass, or automatic retry.

Protocol: `completion_only_raw_abstain_v1`, original tokens/key/partition, no prompt/BOS/EOT prefix; coordinate 1 abstains. BF16, TF32 disabled, static KV, float32 saved probabilities, float64 CPU scores. Seed12345, t3, row rate99/100, r14193 with the original startup clamp. Detector revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; weights/tokenizer hashes are pinned in [setup.json](setup.json).

The refreshed [read-only inventory](inventory.json) found **500 saved source files and zero matching 0.6B caches** among 58 integrated manifests. Original artifact hash and tokenizer/config metadata were checked; both tokenizers are identical and support the token IDs and context length. The existing combined source manifest was checked against all 500 frozen record references, artifact and partition. Full source-file, token-prefix, partition/key and checkpoint-byte checks occur in the approved CPU preparation stage before GPU dispatch.

## Batching and GPU choice

Prepare ten disjoint batches of 50. Process batch0 (prompts0–49) on **one H200**, retaining its primary trace for the final N500 report. Stop for review. If separately approved, launch **nine H200 workers concurrently**, each loading the model once and processing one batch from prompts50–499. The first 50 are never replayed. Peak concurrency is nine GPUs in this staged schedule, below the user's ten-GPU ceiling.

Each GPU worker has 4 physical CPU cores and 16 GiB host RAM. No GPU remains allocated while awaiting approval; idle shutdown is 2 seconds. Every completed batch is committed to `prc-completion-only` immediately, including when the subsequent 85% memory-margin check fails.

The BF16 KV cache alone requires **38.28 GiB at batch25, 76.56 GiB at batch50, and 153.13 GiB at batch100**, using the pinned 28-layer, 8-KV-head, head-dimension128 configuration and T14336. Weights and temporary tensors need additional memory. Batch50 does not fit the existing A100 80GB safety margin; batch100 exceeds H200 memory. H200's 141GB memory supports the batch50 proposal with greater headroom; actual peak remains unmeasured. [NVIDIA specifications](https://www.nvidia.com/en-gb/data-center/h200/).

H200 costs approximately 1.73× as much per configured worker-second as A100 80GB, so it must complete the same 50 responses at least 1.73× faster to lower cost. Greater memory bandwidth makes this worth evaluating, but hardware bandwidth is not a workload benchmark. The first batch resolves runtime/memory uncertainty using useful production work. See [cost/memory basis](h200_batch50_basis.json).

## Paid stages — awaiting approval

| Stage | Exact workload and resources | Expected time | Estimate | Allowance |
|---|---|---:|---:|---:|
| Prepare | One CPU worker, 4 cores/16 GiB; verify500 cached inputs and checkpoint; prepare ten batches | 1–6 min | $0.01–0.06 | $0.15 |
| First50 (`pilot`) | One H200, batch50, 4 cores/16 GiB; one primary replay at T14336 | 25–45 min | $2.10–3.70 | $4.60 |
| **First phase** | **Prepare + first50 only** | **About30–50 min with routine overhead** | **About$2.20–3.80** | **$4.75** |
| Remaining450 (`replay`) | Nine H200 workers, one batch50 each; separate approval after measurement | Re-estimate from first50 | Not yet approved | Reserved$30.20 |
| Score/report | One CPU worker, 4 cores/8 GiB; all500, posterior and entropy at T14336 only | 0.5–3 min | $0.003–0.02 | $0.05 |

The initial H200 range is an unmeasured scenario informed by saved A100 timings and hardware capabilities. The earlier **$26–34 / 65–80min** estimate applies to A100/batch25. H200 could exceed$35 if its speedup is insufficient; continuation is blocked pending the first-batch measurement.

After the first phase, estimated budget remaining is **$31.20–32.80**, or **$30.25 if the entire$4.75 allowance is used**. Reserve$30.20 for the remaining nine workers and$0.05 for scoring. No remaining450 approval is created in advance. Stage allowances total$35.

Work deadlines: prepare600s; first50 GPU3300s; remaining GPU2390s per worker; score300s. Function timeout adds30s, startup60s, idle shutdown2s. GPU resource envelopes at those limits are about$4.576 for first50 and$30.132 for nine remaining workers. These are planning allowances and execution safeguards, **not a provider-enforced dollar cap**.

The pilot report saves GPU method time, total worker wall time, peak allocated/reserved memory, and trace hashes. It adds **15% to observed worker wall time**; the resulting projection must fit both2390s per remaining worker and the reserved$30.20. Peak memory must remain below85%. A slower or failed first batch stops the campaign for a new decision. No automatic retry or hardware fallback.

[Modal pricing](https://modal.com/pricing), checked September22: H200$0.001261/s; CPU$0.0000131/core/s; RAM$0.00000222/GiB/s. Configured GPU worker rate: **$0.00134892/s ($4.856112/hour)**. Queue/approval waits are excluded. Fresh [billing evidence](billing_before.json) records$66.48862111 for earlier native η=.20 work and its prefix scan, not this run. The current$35 budget comes from the user; the usage API does not expose available credits.

## Implementation and verification

Driver: [online_8b_eta020_0p6b.py](execution_sources/online_8b_eta020_0p6b.py). It uses the existing `_prepare_redetection`, `_recover_redetection_batch(validate=False)`, cloud CPU scoring and CSV routines. No numerical/model algorithm changed. Native/scientific imports finish before NumPy compatibility aliases. Logs stay in `/tmp` during volume reloads. Runtime source hashes and local snapshots are frozen.

Approval checks run locally **before `app.run()`**, including exact setup hash, stage, allowance, confirmed available budget, profile, branch, preceding-stage completion and one-attempt markers. Remaining450 approval must reference the first50 result hash and pass the measured cost check. **No approval or attempt files exist.**

Seven metadata-only unit tests passed: scope, disjoint first50/remaining450 coverage, shape/protocol/resource rejection, source overlap, measured cost/memory gate, source snapshots, and missing/stale/repeated approval guards. The default command imports the driver and validates local metadata successfully, without model execution or cloud dispatch.

After explicit approval is recorded in the matching stage file:

```sh
MODAL_PROFILE=new-prc-watermark /opt/anaconda3/bin/python online_8b_eta020_0p6b.py --stage prepare --approval-reference '<exact user approval reference>'
MODAL_PROFILE=new-prc-watermark /opt/anaconda3/bin/python online_8b_eta020_0p6b.py --stage pilot --approval-reference '<exact user approval reference>'
```

Stop after first50. `pilot_assessment.json` supports the next decision. Only separate approval permits `--stage replay` for remaining450. `--stage score` requires its own approval and completed collected traces. Running the script with no stage performs only local metadata checks.

Outputs: durable traces/manifests, per-record scores, native8B comparison (saved MAP452/500, entropy433/500), and one N500 row in the existing redetection CSV with FPR skipped. Primary files are committed to the volume and downloaded before scoring. No extra remote reproduction archive or unrelated result edits.
