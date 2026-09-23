# Fixed PRC 4B generation completed

Generated **100 watermarked completions exactly once**, each 1024 tokens, from
Qwen/Qwen3-4B-Base revision 906bfd4b4dc7f14ee4320094d8b41684abff8539.
Fixed PRC eta=.05, n=T=1024, t=3, r=1014; original seed12345 key/partition and
prompt indices 0–99 were reused. Null generations: zero.

- App: [ap-zYbQrA2etgC4udQJLgNtdI](https://modal.com/apps/new-prc-watermark/main/ap-zYbQrA2etgC4udQJLgNtdI)
- Hardware: NVIDIA H100 80GB HBM3; batch100, four CPU cores, 64 GiB host RAM.
- Worker time: 130.46 seconds; generation time: 80.13 seconds.
- Peak allocated GPU memory: 24.26 GiB.
- Provider-reported generation cost: **$0.17829656** (updated provider report).
- Preparation plus generation: **$0.18817659**. Remaining budget estimate:
  **$6.31**, also accounting for the previously identified separate diagnostic.

The generation batch was saved and committed to Modal volume `prc-data` before
any later stage, then downloaded and SHA-256 verified locally. It preserves
exact completion/prompt token IDs, realized PRC codewords, available RNG states,
generation probabilities and auxiliary features, pinned model information,
source hashes, original artifact hash, memory and timing evidence. The original
artifact and the primary generation batch are retained locally and included in
the narrow generation-output commit. No checkpoint weights are committed.

`cache_index.json` records local/cloud locations and hashes. The generation
batch SHA-256 is
`fef2a042c700a8a76fec2b30b273872d5506af8b4e0a18cc2333ceeaba880cd9`.

Generation-time probabilities are provenance only. Neither completion-only
detector has run, and no detection result or empirical FPR is claimed. No retry,
benchmark, reference replay, scoring, or optional validation pass ran.

The subsequent CPU manifest preparation is complete; see `MANIFEST_RESULT.md`.
The first post-completion generation billing snapshot was $0.16279311; the
updated report is $0.17829656. `billing_after_freeze.json` supersedes that first
snapshot. Both detector replays and CPU scoring still await approval.
