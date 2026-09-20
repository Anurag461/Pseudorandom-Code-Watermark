# 0.6B primary detector trace completed

The pinned Qwen/Qwen3-0.6B-Base detector completed one replay of exactly the
same 100 saved 4B-generated completions used by the 4B detector. Each completion
has 1024 tokens; the trace contains 100 x 1023 partition probabilities for
coordinates 2–1024. Coordinate 1 is reserved for abstention. Protocol:
`completion_only_raw_abstain_v1`, BF16, static KV cache, no original prompts or
special-token prefix. Generation-time probabilities were not substituted.

- App: [ap-dpNpTWPbYOgXe6fHQ4loAV](https://modal.com/apps/new-prc-watermark/main/ap-dpNpTWPbYOgXe6fHQ4loAV)
- Hardware: one A100 80GB, four CPU cores, 16 GiB host RAM; batch100.
- Replay time: 50.377 seconds; worker time: 76.205 seconds.
- Peak allocated GPU memory: 12.91 GiB.
- Provider-reported cost: **$0.06416753**.
- Trace file SHA-256:
  `8c7f049b8e602b198b3191fb6adc7d66b063975d0c460eeda5185091bb50c909`.
- No reference pass or retry; saved metadata records `full_validation=false`.

The primary trace was committed to the existing Modal volume and downloaded
with a matching checksum. Both detector traces are retained locally and in
Modal; exact tokens, key, partition and input identities are shared. No
completion was regenerated, and no null cohort was added.

`billing_after_replay_0p6b.json` reports **$0.40192337** spent across all completed
stages, leaving an estimated **$6.10** including the previously identified
separate diagnostic usage. Provider accounting may settle asynchronously.

TPR is still uncomputed: the final planned cloud CPU stage will score 100
candidates x two detectors x MAP/entropy (400 decisions) and append two rows
to the existing CSV with N=100 explicit. Empirical FPR remains unevaluated,
with null N=0 and FPR cells marked skipped.

That final stage requires approval: one four-core CPU worker, 8 GiB RAM, no
GPU, estimated 1–3 minutes and $0.005–0.013 with a $0.05 allowance. No extra
model replay, generation, reference check or retry is included.
