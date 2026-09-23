# Completed remaining400 execution — approximately $35.11 budget

Completed: N=500 per detector, actual reported cost $28.68301936. See [RESULTS.md](RESULTS.md). The user clarified that $35.11 was an approximate target.

Online 8B eta=.15, T=6144; continue saved prompt IDs100–499 from 4096. Reuse all generation/traces/scores for IDs0–99. Native8B and0.6B raw completion detection; MAP+entropy; finalN500; null0. Exact pinned models, keys, seed12345, t3, 99/100 partition schedule, BF16 and established sampling settings are frozen in setup.json.

Preparation recovered without another paid run: the interrupted worker reached the pilot-record loop after verifying checkpoints, all400 sources, artifacts and target coverage. Read-only SHA256 checks verified all100 pilot generation files, both reports and all pilot traces, and the successful saved pilot prefix/PRC audit. No model execution or scoring occurred locally.

|Stage|Resources|Estimate|Allowance|
|---|---|---|---|
|Generate|4 concurrent H200, batch100, 4CPU/64GiB each|37–43min, $13–15.50|$15.60|
|Freeze manifests|1 CPU worker, 4CPU/16GiB|1–5min, $0.005–0.03|$0.06|
|Detect native|4 H200, batch100, 4CPU/64GiB|28–35min, $10–12.40|$12.55|
|Detect0.6B|6 A10080GB, 8 batches50, 4CPU/16GiB|12–26min, $4–6|$6.10|
|CPU scoring|1 worker,4CPU/8GiB|0.5–3min,$0.005–0.03|$0.04|

Both detectors run together, max10GPUs. Generation uses4GPUs. Remaining stage allowances total$34.35. Earlier preparation charges were$0.02187548 at the last billing snapshot. Resource/time envelope is approximately$33.95 inclusive of those charges, leaving approximately$1.16 below the user's$35.11 ceiling. Billing is checked before each stage and after completion; remaining work cannot be dispatched unless it fits the ceiling. Work deadlines: generation2550s/worker, native2040s/worker, small900s/batch, freeze600s, scoring300s. Includes startup and shutdown margin. No automatic model retries, benchmarks, nulls or independent reference passes.

The user approved the full pipeline and recovery using saved work under this total ceiling. Save and narrowly commit generation outputs before detection; save and commit primary traces before scoring; save final CSV rows, reports and billing. Preserve unrelated changes and the redetection branch. No push or extra archive upload.
