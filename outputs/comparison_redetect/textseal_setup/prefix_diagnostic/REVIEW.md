# TextSeal prefix discrepancy: resolved

The n=128 mismatch comes from shape-dependent BF16 linear algebra in the
pinned Qwen3-8B model. Direct prefix detection is the reference. Production now
runs each requested length independently through the unmodified upstream code.
This diagnostic ran only ten responses and a ten-response replacement pilot.
The subsequently authorized [full replay is now complete](../direct_prefix/REVIEW.md).

## Evidence

The diagnostic used the original five TextSeal and five null completions,
indices 0–4, at n = 128, 256, 400, 512, 768, 1024. Same model revision, H100,
PyTorch 2.4.0, Transformers 4.51.3, BF16 eager attention, and precision settings
as the failed original pilot. Forward observers checked raw completion IDs.

| Prefix | Entropy mismatches | Upstream result mismatches | Comparison decision flips |
|---|---:|---:|---:|
| 128 | 10/10 | 10/10 | 0/10 |
| 256, 400, 512, 768, 1024 | 0/10 each | 0/10 each | 0/10 each |

At n=128, the largest entropy difference across the ten records was 0.328125;
the largest absolute weighted-p difference was 0.0018360649447360378.
For null/0000, 101/127 entropies differed (maximum 0.21875); the weighted
p-value was 0.5627631433448814 directly versus 0.5618582856755227 from reuse.
Unweighted p-values matched throughout. No decision flips in this small pilot
does not establish equivalence for all 1,000 responses or near the cutoff.

On null/0000:

- Repeating n=128 and n=1024 at the same shapes reproduced entropy exactly.
- Reversing tokens after position 128 in a 1,024-token input left the first
  127 entropies exactly unchanged. The observed difference is not future-token
  conditioning.
- Embeddings, RoPE, the first input normalization and query projection matched.
  The earliest captured difference was `model.layers.0.self_attn.k_proj`.
- Isolating that linear operation with identical weights and input prefixes
  reproduced 253/131072 differing BF16 outputs, maximum 0.001953125, solely by
  changing input shape from [1,1024,4096] to [1,128,4096]. Repeating the same
  shape was exact. FP32 diagnostic arithmetic reduced the largest shape
  difference to 3.5762786865234375e-7; production remains BF16.
- Read-only activation hooks did not change either entropy vector. Additional
  lengths 127, 129 and 255 differed from longest-prefix reuse; length 192
  matched. Thus this is not a simple length threshold.

This isolates a shape-dependent floating-point computation. We did not profile
or identify a specific cuBLAS kernel. PyTorch's [numerical accuracy documentation](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)
explains that mathematically equivalent sliced and full operations need not
be bitwise identical. Changing dtype, padding or model execution would change
the upstream reference; direct per-length execution avoids that change.

## Implemented resolution and validation

`TextSealCompletionDetector.detect_prefixes` now calls original upstream
`_compute_entropies(ids[:n])` and `_score_text` separately at each length and
stores a separate entropy vector for each. The longest-reuse method remains
explicitly diagnostic-only. Production rejects reuse manifests; cache checks
bind strategy, code, model, runtime, input hashes and per-prefix results.

The replacement H100 pilot independently called upstream public `detect` at
every record/length using an ID passthrough tokenizer. **60/60 entropy vectors
and complete result dictionaries matched exactly.** All 120 observed forwards
contained precisely the raw completion prefix with no prompt, added tokens or
external cache. This verifies upstream detection on the stored IDs; no
text decode/re-encode was introduced. All 52 focused local tests also passed.
See [the replacement setup](../direct_prefix/REVIEW.md).

## Reproduction and artifacts

- Diagnostic app: [ap-PV8k4kv8oaMpO6wFRVYGqc](https://modal.com/apps/new-prc-watermark/main/ap-PV8k4kv8oaMpO6wFRVYGqc).
- [report.json](report.json): all comparisons, activation diagnostics and controls.
- [direct_entropies.json](direct_entropies.json): direct vectors and upstream results.
- [execution.json](execution.json): runtime, source/report checksums and remote path.
- [source_before_direct.tar.gz](source_before_direct.tar.gz): exact execution source
  before the fix, verified against every original manifest code hash. To rerun
  this historical diagnostic, restore this snapshot into an isolated copy of
  the repository with the original manifest/input export and pilot report.
  Current production code intentionally rejects the old reuse manifest.

The source archive SHA-256 is
`12a7c29c734c13d913db4390fb7a54b4b09a49b5276c313e0150bb1cda40bc29`.
The diagnostic took 48.758 seconds including 38.573 seconds loading, about
$0.063 in measured resource time. The replacement pilot used about $0.047.
These exclude startup/image/storage overhead. PRC results and the comparison
CSV were unchanged. No repeat handling or generation was added.
