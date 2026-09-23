# 4B primary detector trace completed

The pinned Qwen/Qwen3-4B-Base detector completed one replay of the saved
100 watermarked completions, each 1024 tokens. It recovered 100 x 1023
completion-only partition probabilities for coordinates 2–1024; coordinate 1
is reserved for abstention. The protocol is `completion_only_raw_abstain_v1`,
BF16 inference and static KV cache, with no prompt or special-token prefix.
Generation-time probabilities were not used for this detector trace.

- App: [ap-lmiikeifcvSPibtJYVpRCG](https://modal.com/apps/new-prc-watermark/main/ap-lmiikeifcvSPibtJYVpRCG)
- Hardware: one H100 80GB, four CPU cores, 64 GiB host RAM; one batch of 100.
- Replay time: 60.191 seconds. Worker time including loading: 110.816 seconds.
- Peak allocated GPU memory: 23.22 GiB.
- Provider-reported cost: **$0.14728834**.
- Trace file SHA-256:
  `90190abe3534ff6a21cbc0c2f0ad85cea5ac7480684502f4d8eb706534b3c147`.
- Saved trace metadata records `full_validation=false`: no independent reference
  pass or reversed-batch pass was run. No retry or generation was launched.

The user requested a pause as connectivity was failing. The app was stopped;
the primary trace had completed, was committed to the existing Modal volume,
and was downloaded with a matching checksum. On resumption, read-only calls
confirmed the app is stopped with zero workers. No rerun is needed. The trace,
manifest, approval, timing and billing evidence are saved before further work.

`billing_after_replay_4b.json` reports total experiment spending of
**$0.33775584**. Estimated remaining budget is **$6.16**, including the previously
identified separate $0.01946503 diagnostic usage. No new paid work was launched
during the resume/reconciliation step.

The 0.6B replay and shared CPU scoring remain pending; no detection decisions
or TPR/FPR results have been computed yet. Next proposed stage: one A100 80GB,
four CPU cores, 16 GiB RAM, batch100, 1–3 minutes, estimated $0.05–0.15 with
a $0.40 allowance. It requires explicit approval before launch.
