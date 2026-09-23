**Completed:** all500 nulls and173 scoring points; billed$55.74 of$65; all cloud apps stopped. [Final results](RESULTS.md). Setup and launch history follows.

**Launch update:** user said “go ahead”. The run is approved within the quoted **$65 maximum**, including all three stages. The older optional$75 headroom below is superseded. Frozen GPU deadline is4350s per worker, function timeout4380s; stage allowances$0.15/$64.70/$0.15. See [authorization](authorization.json) and [live status](progress.json). The earlier setup notes below are retained as history.

# Native 8B FPR, η=.15/.20 — shared T13088 null replay

Prepared September22,2026 on `redetection`. **Setup only. No paid compute has been launched, no approval files exist, and the user deferred the budget decision.** The original T13088 scope is retained; the later extension discussion was cancelled.

Use all500 saved Qwen3-8B-Base unwatermarked completions, prompt IDs0–499, from `prc-data:_nulls/qwen3_8b_base/T13088`. Run one native8B completion-only replay per null, then score those same probabilities under both original PRC keys. Existing watermarked decisions are reused unchanged.

| Key | Fixed scoring lengths | Points |
|---|---|---:|
| η=.15 | 6144 down to4656, step16 | 94 |
| η=.20 | 13088 down to11840, step16 | 79 |

Report posterior (`map`) and entropy empirical FPR for all173 points, each with nullN500. Save per-record scores and append combined TPR/FPR rows using the original saved watermarked decisions. Apply no stopping rule to the nulls. T14336 FPR, new generations, shorter-key rescoring, 0.6B inference, naive scoring, reference passes, and automatic retries are outside this run.

## Cache and protocol

The refreshed [read-only inventory](inventory.json) found all500 source files and inspected59 integrated detection manifests. No compatible native-null traces reachT13088. Frozen source references include each file's size and historical SHA256. Paid CPU preparation, after future approval, will check all current source bytes, full-token hashes, original prompt corpus, both keys and partition, and the pinned model checkpoint before GPU dispatch.

The null-bank manifest identifies Qwen3-8B-Base, forced length13088,500 canonical prompts, static KV, and the original unwatermarked sampler. Its partition matches both key families. Generation-time probabilities are not used as detection traces. Existing outputs are checked again before replay to prevent duplicate work if the cache changes.

Detector `Qwen/Qwen3-8B-Base`, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, BF16, TF32 disabled. Protocol `completion_only_raw_abstain_v1`: raw saved completion tokens, no original prompt or special-token prefix; coordinate1 abstains. Static KV, float32 saved probabilities, float64 CPU scores. Original seed12345, t3, causal99/100 row schedule and startup clamp, one-shot targetFPR.001. Model, tokenizer, artifact, source and runtime hashes are frozen in [setup.json](setup.json).

## Hardware and cost

Ten H200 workers run concurrently, each loading the model once and processing one disjoint batch of50. Each GPU worker has4 physical CPU cores and64GiB host RAM. Every finished primary trace is saved immediately. No GPU remains allocated during review; idle shutdown is2 seconds.

| Stage | Work/resources | Expected time | Expected cost | Historical proposed allowance, not approved |
|---|---|---:|---:|---:|
| Prepare | 1 CPU worker,4 cores/16GiB; verify500 nulls and create10 batches | 2–6min | $0.02–0.06 | $0.15 |
| Replay | 10 H200 workers, batch50 each,4 cores/64GiB each | 62–74min | $54–65 | $74.70 |
| Score/report | 1 CPU worker,4 cores/8GiB; two keys,173 points | 1–3min | $0.005–0.02 | $0.15 |
| Total | 500 nulls replayed once | 70–90min including ordinary transfers | **$55–65** | **$75 proposed headroom only** |

The central GPU estimate is$55.37. It uses nine measured native8B H200/batch50 timings atT14336, scaled by `(13088/14336)^2`, plus120 seconds per worker for loading/I/O. T13088 timing remains an extrapolation. The recently completed 0.6B detector run is not the timing basis for this larger model. See [timing and cost evidence](timing_cost_basis.json).

[Modal rates](https://modal.com/pricing), rechecked September22: H200$0.001261/s, CPU$0.0000131/core/s, RAM$0.00000222/GiB/s; configured worker total$0.00145548/s. Batch50 KV storage alone is89.87GiB, with additional weight/temporary memory. The wrapper retains the successful native8B run's95% memory guard and saves completed traces before checking that limit.

The frozen historical resource proposal uses work deadlines600s for preparation,4800s per GPU worker,600s for scoring; function timeout adds30s, startup60s, idle2s. Its GPU resource envelope is about$71.20, within the proposed$74.70 GPU allowance. These are execution safeguards, not a provider-enforced dollar cap. A$65-cap alternative is recorded for later budget review and would require a revised frozen plan with a shorter deadline; neither budget has been selected.

The previous separate run's fresh billed cost is$25.43796444, leaving$9.56203556 from its$35 budget. Those remaining funds do not fund this proposal. The user chose to decide this new budget later. See [read-only billing](billing_setup_20260922.json).

## Ready implementation

[Driver](execution_sources/online_8b_shared_null.py) provides explicit `prepare`, `replay`, and `score` stages, reusing the established pinned model loader, completion-only replay, prefix scoring, and CSV writer. Scientific imports precede NumPy pickle aliases; live logs stay outside mounted volumes. All500 nulls enter ten nonoverlapping batches, and both key reports reference the same ten traces. No original watermarked artifact or score is rewritten.

Before any paid stage, local guards require an approval reference, exact setup hash, that named stage's allowance, a sufficient newly approved total budget, the `redetection` branch, the correct Modal profile, prior-stage collection and absence of a previous attempt. There are no approval or attempt files. An earlier unrelated run's approval cannot launch this one.

Six metadata-only tests passed: fixed173-point scope, rejection of extra generation/replay or protocol changes, exact500-null batch coverage, correct FPR aggregation while preserving saved TPR, spending/approval/duplicate guards, and frozen source identity. The default command validates local metadata without cloud dispatch:

```sh
/opt/anaconda3/bin/python online_8b_shared_null.py
```

No numerical/model test or cloud validation has been run for this new setup. The cloud preparation checks described above remain part of the proposed paid workflow. Final outputs will include durable traces, two per-record reports,173 combined CSV rows, and timing/billing evidence. The older proposal is retained in [historical_proposal_20260921.md](historical_proposal_20260921.md).
