# Controlled baseline batch-50 validation report

Status: **passed**. The standalone validation used prompt indices `0..49`, one
H100, batch size 50, and the frozen 1,024-token settings. It did not launch the
500-prompt comparison.

All three generated methods produced 50 continuations of exactly 1,024 tokens.
The saved token IDs, base-model entropies, and selected-token log-probabilities
were complete and finite. The SynthID generation/g-value reference check agreed
exactly with Google's pinned implementation (`indices_equal=true`, maximum
absolute score difference `0.0`). The full 500-prompt online-PRC and null cache
preflight passed with zero generation attempts.

| Method | 50-prompt time (s) | Seconds/prompt | Tokens/s |
|---|---:|---:|---:|
| TextSeal | 45.5673 | 0.91135 | 1,123.61 |
| SynthID-Text | 68.6613 | 1.37323 | 745.69 |
| Gumbel-Max | 43.8909 | 0.87782 | 1,166.53 |

The complete GPU function took 166.3861 seconds, including an 8.2038-second
model load. Peak CUDA allocation was 27,704,215,040 bytes and peak reservation
was 27,839,692,800 bytes (25.93 GiB) on an NVIDIA H100 80GB HBM3. This is well
below the predeclared 70 GiB gate and leaves ample margin for production batch
50. Production must nevertheless keep the same batch size, grouping, ordering,
hardware provenance, and keys because Gumbel-Max is batch-shape-sensitive.

The finalized Modal bill was **$0.27034319**: $0.21751641 H100, $0.02631032
CPU, and $0.02651646 memory. This is below the $3 validation cap. Scaling the
entire validation app bill by ten gives a conservative **$2.70343190** estimate
for generation of all ten shards; it overcounts shared preflight work but
allows for per-worker overhead. Allow **$3–$4 total** including CPU scoring and
finalization, with a recommended $5 hard cap. With ten H100s available, the
measured generation compute floor is 2.77 minutes and the practical end-to-end
estimate is **8–15 minutes**, excluding unusual GPU queue delay.

The raw artifact is stored at
`/data/controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/generated/shard_00.pt`
and has SHA-256
`0b7326fbfa43d55a39088e19cc2d69a696b7c81e8095b3ff6d291254392716da`.
The Modal run is
<https://modal.com/apps/new-prc-watermark/main/ap-INjm5299E4tI0jNgiqOx4U>.

The batch-50 gate is cleared. The 500-prompt comparison is technically ready,
but remains unlaunched and requires explicit approval using the separate
full-run gate in `controlled_baseline_full_run_runbook.md`.
