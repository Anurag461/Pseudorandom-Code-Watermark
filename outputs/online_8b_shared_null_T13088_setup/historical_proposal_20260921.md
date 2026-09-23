# Native 8B FPR from the existing shared T=13088 null bank

September 21, 2026. **Proposal only: no paid run approved or launched.** Use 500 saved 8B-generated unwatermarked completions, prompt IDs 0–499, in `prc-data:_nulls/qwen3_8b_base/T13088`. No new generations and no 0.6B work are included.

Perform **one shared native 8B completion-only replay at T=13088**, then score its saved probabilities under both original online PRC keys:

- eta=0.15: T=6144 and all 93 previously reported shorter prefixes, T=6128 down to 4656 in steps of 16 (94 points total).
- eta=0.20: all 79 previously reported prefixes T=13088 down to 11840 in steps of 16, including the 11856/11840 MAP boundary.

This supplies MAP and entropy empirical FPR for **173 existing native-detector reporting points**, with null N=500 explicit. Existing watermarked scores are reused unchanged. The eta=0.20 points above T=13088, including T=14336, remain explicitly unmeasured for FPR. Native eta=0.05/0.10 FPR results already exist and are outside this work. The 0.6B null replay is deferred, and the separate watermarked-only 0.6B proposal is not launched by approving this null proposal.

## Reuse and protocol

Read-only inventory `../online_8b_eta020_0p6b_setup/null_fpr_inventory.json` confirms all 500 existing null files and no integrated native null traces reaching T=13088; shorter native null traces cannot supply the missing full-length probabilities. The historical source manifest verifies the shared generator, prompt corpus and partition. Approved CPU preparation will freeze individual file/token hashes and verify source compatibility. Generation-time probabilities are not completion-only detection traces.

Detector: `Qwen/Qwen3-8B-Base`, pinned model/tokenizer revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, reusing the cached checkpoint. Protocol `completion_only_raw_abstain_v1`: raw completion token IDs, no prompt or special-token prefix, coordinate-one abstention, MAP and entropy. BF16, TF32 disabled, static KV cache, float32 stored probabilities and float64 CPU scoring. Retain original eta-specific keys, seed 12345, t=3, causal 99/100 row schedule and one-shot target FPR=0.001. Partition SHA256 `503d1cf93958f0d765606ed3e25aa87a1a777cd08d8f44436b0c6ae9716aa184`.

Nulls are unwatermarked, so their generation/replay is not duplicated per eta. Score each frozen eta/key/length separately on cloud CPU. No model execution, scoring or heavy analysis runs locally. Existing null generations retain their original sampling provenance; no resampling occurs.

## Resources and incremental cost

| Named stage | Resources/work | Expected elapsed time | Estimated cost | Proposed allowance |
|---|---|---:|---:|---:|
| Prepare | One CPU worker, 4 physical cores, 16 GiB; 500 existing nulls and 10 batch manifests | 2–6 min | $0.02–0.06 | $0.15 |
| Native null replay | **10 H200s concurrently**, one batch of 50 each; each has 4 physical CPU cores and 64 GiB host RAM | 62–74 min | $54–65 | $74.70 total |
| Score/report | One CPU worker, 4 physical cores, 8 GiB; two keys, 173 frozen length points, MAP + entropy | 1–3 min | $0.005–0.02 | $0.15 |
| **Total** | One primary replay per null, no duplicate full pass | **70–90 min**, including ordinary transfers, excluding allocation queue | **About $55–65** | **$75.00** |

Generation cost is **$0**. All proposed charges above are for null/FPR work; no watermarked replay is included.

The measured native T14336/batch50 H200 inference took 4415.57–4424.30 seconds per batch. Scaling attention work by `(13088/14336)^2 = 0.83347` projects about 3684 seconds (61.4 minutes) per T13088 batch. Allowing 120 seconds per worker for loading/I/O gives a central GPU-stage estimate of **$55.37 for all ten workers**, before the small CPU stages. Exact T13088 timing is an extrapolation, not a benchmark. The ten workers run concurrently; this is about an hour of GPU wall time, not ten hours elapsed.

[Modal pricing](https://modal.com/pricing), checked September 21: H200 $0.001261/s, CPU $0.0000131/physical-core-second, host memory $0.00000222/GiB-second. Together these resources cost $0.00145548 per worker-second. The successful longer batch used 134.05 GB of 150.11 GB reported GPU memory. Keep its explicit 95% memory guard and save-before-margin-check behavior; the shorter sequence reduces the KV requirement.

Work deadlines: prepare 600s; GPU replay 4800s per worker; scoring 600s. Function timeout adds 30s, startup timeout 60s, idle shutdown 2s. Retries=0; durable attempt markers prevent duplicate dispatch on restart. Any paid retry, benchmark or additional validation needs a new estimate and explicit approval. The $75 allowance includes startup/shutdown margin but is not a provider-enforced dollar cap.

## Implementation, spending and approval

A narrow null-only driver adaptation is required before execution, reusing the successful native batch50 wrapper and existing null preparation and prefix-score routines. Freeze the 173 reporting points rather than applying a TPR stopping rule to null-only data. Reuse saved watermarked scores; record null cohort provenance and null N=500 without rewriting original watermarked artifacts. Retain scientific/native imports before NumPy pickle aliases and keep open worker logs outside Modal volumes during reload.

Save each primary null trace immediately, collect and narrowly commit primary results before CPU scoring. Then save per-record scores, FPR counts, CSV results and actual costs. Preserve unrelated changes on `redetection`; do not push or upload additional reproducibility archives. No runtime code was changed for this estimate.

Fresh read-only billing is saved in `billing_before.json`. The completed eta=.15 expansion/prefix work plus eta=.20 native/prefix work total **$95.17473515**. This shared null work is additional: approximately **$150.17–$160.17 combined related spending**, or **$170.17473515** if its entire allowance is used. These are task costs, not available credits; current account credit balance is unavailable. Previous unused allowances are not authorization for this new work.

**Await explicit approval for these three named stages and the $75 total allowance.**
