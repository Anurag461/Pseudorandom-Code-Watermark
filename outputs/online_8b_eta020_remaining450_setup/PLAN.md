# Online 8B eta=0.20: complete the remaining 450 watermarked prompts

Completed on `redetection`, September 21, 2026. **All three approved stages finished successfully for $59.86306135 against the $75 allowance.** All workers are stopped. The final N=500 MAP TPR is 90.4%, and entropy TPR is 86.6%. See [RESULTS.md](RESULTS.md). The approved scope and estimates are retained below.

## Scope and cache reuse

Replay **prompt IDs 50–499**, T=n=14336, with the native 8B detector: **nine batches of 50, requested concurrently on up to nine H200s**. Reuse the completed prompt IDs 0–49, including their existing scores. The final CSV row will have **N=500 explicit**, and the existing N=50 row will remain.

Read-only Modal inventory found all 500 saved generations. Of 57 integrated completion-only manifests, only the completed first-batch manifest matches this full-length source/detector setting. Its trace and score file hashes were verified through read-only storage calls. No additional compatible completed batches were found. Source paths, prompt IDs and sizes for the remaining 450 are fixed in `setup.json`; approved cloud CPU preparation will freeze their file/token hashes and validate their provenance before GPU dispatch.

New generation: **0, $0**. Null generation/replay/scoring: **0, $0**. Empirical FPR is not evaluated. 0.6B detector records: **0**. No shorter-length sweep, reference pass, benchmark or paid retry is included.

## Identical detector settings

- `Qwen/Qwen3-8B-Base`; model/tokenizer revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Checkpoint/tokenizer hashes are frozen in `setup.json`.
- `completion_only_raw_abstain_v1`: raw saved completion IDs, first-coordinate abstention, MAP and entropy scoring. No generation-time probabilities are used as detection traces.
- Online PRC eta=0.20, T=n=14336, t=3, seed=12345, row rate 99/100, r=14193, target FPR=0.001 with one-shot policy.
- BF16, TF32 disabled, static KV cache, exact existing source key and partition. Cached generations retain their original position-addressed inverse-CDF sampling, temperature 1, unrestricted vocabulary and forced length. No retokenization or resampling.
- Same explicit 95% memory guard and save-before-margin-check behavior as the successful batch. Its peak allocated memory was 134.05 GB out of PyTorch-reported 150.11 GB.

## Execution, time and incremental cost

| Stage | Resources | Expected wall time | Estimated cost | Proposed allowance |
|---|---|---:|---:|---:|
| Prepare | One CPU worker, 4 physical cores, 16 GiB | 1–6 min | $0.01–0.05 | $0.15 |
| Native replay | **9 H200 workers in parallel**, batch 50 each; each has 4 CPU cores and 64 GiB host RAM | 75–90 min if all nine start together | **$59.54 base; allow up to about $70** | **$74.70 total ($8.30 each)** |
| Score and aggregate | One CPU worker, 4 physical cores, 8 GiB; score only new 450 and merge saved first-50 scores | 0.5–3 min | $0.003–0.02 | $0.15 |
| **Total additional** | Maximum 9 GPUs | **about 80–100 min**, plus any allocation queue | **about $60 expected** | **$75.00** |

The measured first batch cost $6.61555999 for the GPU stage, including host resources; nine identical batches project **$59.54003991**. That stage took 4538.16 seconds including loading/I/O, with 4430.90 seconds in primary replay. No new timing benchmark is needed. The current [Modal rates](https://modal.com/pricing), checked September 21, remain $0.001261/H200-second, $0.0000131/physical-core-second and $0.00000222/GiB-second: $0.00145548/second per configured GPU worker.

Replay has a **5400-second (90-minute) work deadline per worker**, a 5430-second function timeout and 60-second startup timeout. This is about 19% beyond the successful worker's measured time. CPU work deadlines are 600 seconds for preparation and 300 seconds for scoring. No automatic retries; any paid retry or additional validation would require a separate estimate and approval. The $75 allowance includes startup/shutdown margin. Parallel scheduling shortens elapsed time; it does not multiply the work performed per completion.

## Spending and approval

Fresh read-only billing confirmed $6.62260577 spent on the completed eta=0.20 batch. Completing the remaining nine batches is expected to bring this eta=0.20 native N=500 task to approximately **$66.20 total**, or at most **$81.62260577** if the full new allowance is used.

Previously completed eta=0.15 expansion and prefix scoring cost $28.68611404; those tasks plus the first eta=0.20 batch total $35.30871981. A new $75 allowance would make the maximum combined related spending $110.30871981. The billing API exposes usage, not available account credits. The earlier $20/$6.52 balance estimate is not treated as current authorization or available funds.

**Approval received:** all three named stages above, nine primary GPU batches, with a new total allowance of **$75**. Separate stage approval records bind this authorization to the frozen setup hash. No paid retries or additional passes are authorized.

## Implementation and saved outputs

One isolated `online_8b_eta020_remaining450.py` wrapper adapts the successful batch-50 inference path and the established eta=0.15 saved-score aggregation pattern. Existing model, detector and scoring routines remain unchanged. Scientific/native imports still precede the NumPy pickle compatibility aliases; worker logs remain under `/tmp` during volume reloads. Each worker has a separate durable attempt marker, zero retries and separate output paths. Completed batches are downloaded as they finish and retained if another batch fails.

Preparation will stop before any GPU dispatch if another overlapping completed trace appears. Save and commit all primary traces/manifests before the CPU scoring stage. Aggregate N=500 only after exact prompt-ID coverage is verified. Append the standard CSV row, preserve every existing row, and save actual billing and timing evidence. Future shorter-length results can reuse these full traces without another GPU pass.

Additional reproducibility source archives remain local. Preserve unrelated working-tree changes; do not broadly stage or push. Nine simultaneous allocations are requested within the user's stated ten-GPU concurrency, but Modal availability may affect start times.
