**Execution complete.** This document preserves the pre-run proposal. See [RESULTS.md](RESULTS.md) and [progress.json](progress.json) for completed results, actual scheduling and spending.

Awaiting explicit approval. All seven previously launched apps are stopped. This replaces the original sequential eta=0.15 GPU stage; it does not add a replay.

Use the prepared **eta=0.15, T=6144, N=100** saved online PRC 8B cohort. The T=4096 cohort is excluded. The eta=0.05 and eta=0.10 cohorts are already replayed, scored and committed; never replay them again for this comparison.

| Remaining stage | Workload and resources | Expected elapsed time | Expected total cost | Conservative allowance |
|---|---|---:|---:|---:|
| Parallel GPU replay | Two simultaneous A100 80GB workers; each has 4 CPU cores and 16 GiB RAM. One batch of 50 per worker: prompt indices 0–49 and 50–99. Each completion is processed once. | 11–15 min | $1.05–$1.45 combined | $1.75 combined |
| CPU scoring, after traces are saved and committed | One worker, 4 CPU cores and 8 GiB RAM; both MAP and entropy at T=6144 on the 100 saved traces | 0.25–2 min | $0.002–$0.009 | $0.035 |

Expected remaining elapsed time: about 12–17 minutes, plus provider queues and file collection. Conservative remaining cost is **$1.785, rounded up to $1.79**. This is an estimate, not a provider-enforced dollar cap. There is no overlap between GPU inference and CPU scoring. Two GPUs retain the prepared batch size of 50; there is no benchmark or change of numerical batch configuration to use all ten GPUs.

Recorded spending to date: **$1.63169696**. Estimated remaining balance before these stages: **$3.96240755**. The rounded $1.79 allowance would leave about **$2.17**. The revised total remains below the originally approved $4.30 overall estimate. Billing evidence is in billing_before_parallel_proposal.json.

The timing estimate uses the just-completed eta=0.10 replay (1691.28 worker seconds across five batches of 100 at T=3072), together with the previously recorded 0.6B references. Doubling sequence length and halving batch size projects roughly twice the per-batch replay time; two workers overlap that work. Cost includes two model loads, GPU time, four host cores and 16 GiB RAM per worker, at the established combined rate of $0.00078192 per worker-second. No timing benchmark is proposed.

Detector: **Qwen/Qwen3-0.6B-Base**, model and tokenizer revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Saved generator/reference: Qwen/Qwen3-8B-Base revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Tokenizer JSON hashes are identical; original token IDs are replayed directly. BF16, TF32 disabled, static KV cache, float32 saved probabilities, float64 CPU scoring.

Use `completion_only_raw_abstain_v1`: raw completion tokens, no prompt or BOS/EOT prefix, first-coordinate abstention, MAP and entropy weights. Reuse the original online key and partition, seed=12345, t=3, eta=0.15, row rate=99/100 with the original startup clamp and support sampler, target FPR=0.001 with one_shot policy. This matched 6144 pilot only has a saved 8B comparison at 6144. All agreed shorter eta=.05/.10 lengths have already been scored from their one-pass traces.

**No new generation, nulls, empirical FPR, duplicate replay, full reference pass, optional validation pass, benchmark or retry.** New-generation cost and separate null cost are both $0. No generation-time probabilities are substituted for completion-only traces. No extra reproducibility archive is uploaded.

The small parallel adapter reuses the unchanged corrected worker, including scientific/native imports before NumPy compatibility aliases. It supplies exactly one disjoint prepared batch to each worker, refuses existing traces or a previous attempt, and requires approval of the exact frozen parallel setup. Metadata-only checks verified N=100, both nonoverlapping ranges, T=6144 and rejection of wrong cohorts/overlap; no local model execution or scoring occurred.

Each worker is limited to 1000 seconds of child work, a 1030-second function timeout, 30-second startup limit, two-second idle shutdown and zero retries. Maximum GPU containers is two. A failure does not authorize a retry. Save each trace to Modal immediately, download with transfer checksums, and commit both primary traces before starting CPU scoring. Completed work must never be regenerated.

Exact setup: parallel_eta015_setup.json. The original setup.json remains unchanged. The adapter is online_8b_to_0p6b_parallel.py; storage-only collection is available with `python online_8b_to_0p6b_parallel.py collect`. The original CPU scoring command remains unchanged. Preserve unrelated working-tree changes on redetection; do not push.
