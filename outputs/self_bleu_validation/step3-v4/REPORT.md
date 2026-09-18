# Step 3: two-response validation completed

Completed 2026-09-17 (Pacific), on `comparison-with-redetect`.
Controls from commit `b4b8f26`; historical reference `4696382`.
This is an implementation and cache-reuse validation. Self-BLEU/TPR pilot
analysis and a frontier conclusion have not been run.

## Fixed setting and results

Qwen3-8B-Base, pinned revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`,
BF16, H100, canonical 50-token prompts 0–49 in one batch of 50,
temperature 1, top-p 1, 1,024 completion tokens. Watermark keys stay fixed.
Each configuration used sampling seeds 12345, 67890, then replayed 12345.
The full 1,024-token replay was exact in every configuration.

| Configuration | Responses changed under seed 67890 | Exact historical first-response matches |
|---|---:|---:|
| Online PRC eta .05, t=3, row rate 99/100 | 50/50 | 0/50 |
| TextSeal alpha .1 | 50/50 | 50/50 |
| SynthID depth 10 | 50/50 | 50/50 |
| Plain Gumbel-Max | 0/50 | 50/50 |
| Ordinary sampling | 50/50 | 0/50 |
| TextSeal alpha 0 | 0/50 | Not a historical setting |

The PRC key fingerprint is the historical
`ed6c81de0bc7cf35cf1c05d3ba0fc0db846e78c31241d680582d0778f7f5e783`.
All 50 historical/new first-response watermark-bit sequences match through
coordinate 1,024. Their text prefixes agree for only 0–35 tokens; the precise
cause of the text mismatch was not isolated. No historical PRC/null response
in the tested 50-prompt cohort is eligible for exact token-cache reuse. This does not
invalidate historical experiments or imply that the new replicate changes keys.

TextSeal alpha .5 and SynthID depths 2/20/30 passed 128-token, 50-prompt
parameter checks. All five saved SynthID batches (two depth-10 replicates and
three short depth checks) matched the independent official score update and
token-index ordering exactly. Short checks do not measure the full-length frontier.

## Completion-only detection checks

- PRC: the model received exactly 1,023 one-token inputs from each of three
  fresh raw completions (both PRC seeds and an ordinary-sampling response).
  Probabilities matched the independent BF16 token-step calculation exactly.
  A shorter replay with reversed batch order matched the corresponding prefix.
  The first coordinate's soft score was zero. A fresh cache started at position
  zero; no prompt, special token or generation trace was supplied.
- TextSeal: seven fresh records covered both alpha-.1 replicates, both ordinary
  replicates, both alpha-zero replicates, and the short alpha-.5 response.
  Each full-length record passed upstream public-detector parity at
  128/256/400/512/768/1,024; the short record passed at 128. Actual inputs were
  observed. Each prefix had its own forward; no longest-prefix entropy slicing.

These are targeted integration checks on three PRC and seven TextSeal records,
not pilot TPR/FPR estimates. The historical validated detector implementations
and generation-time repeat handling were preserved.

## Saved artifacts and reuse

[verification.json](verification.json) verifies all 27 remote files and their
checksums, batch/response identities, seed/key pairing and historical token matches.
There are 16 saved batches: 600 full-length response records and 200 short
records. Counts refer to response slots; deterministic controls contain duplicate
text across seeds. The 300 additional full-length replay outputs were compared
to the first responses and were not retained as extra replicate observations.

Volume: `prc-completion-only`.
Verified archive prefix:
`self_bleu_validation/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8`.
Raw files are also downloaded under `outputs/self_bleu_validation/raw/` and
excluded from git. Reports and checksums are versioned.

Stage A can use its five saved response pairs per prompt without any further
generation. Historical scores/evidence can be reused for the exact TextSeal,
SynthID and Gumbel token matches only when detector configuration and masking
also match. In particular, historical SynthID tuple-mask decisions do not
replace the planned native context-mask analysis; reuse underlying evidence
or rescore on CPU. PRC and ordinary-sampling pairs need fresh evidence. The
alpha-zero pair is available for Stage B; other Stage B settings need 1,024-token
pairs before a frontier comparison.

## Cost and repairs

| Work | Recorded seconds | Resource-time estimate |
|---|---:|---:|
| Generation, controls, short checks and original PRC validation attempt | 1,008.69 | $1.30270 |
| PRC detector-only repair | 132.02 | $0.17050 |
| TextSeal direct-prefix parity | 63.38 | $0.08185 |
| **Recorded resource time** | **1,204.08** | **$1.55505** |

Rate: one H100 + four CPU cores + 64 GiB host RAM, $0.00129148/s using
[Modal resource pricing](https://modal.com/pricing). These are timing-derived
resource estimates, not settled invoices. A separate $0.50 allowance covers
the failed startup; $2 covers image/startup/storage overhead. The resulting
planning charge is $4.05505, leaving $5.94495 of the initial $10 allocation.
All worker stages had one container, no application retries and fixed timeouts.
The $200 total study ceiling is unchanged.

The first launch failed during image construction. The next failed during
TextSeal import after legacy NumPy aliases were enabled; restoring the historical
import order resolved it. The complete generation run then encountered a
BF16-to-NumPy conversion error in the final PRC validation assertion. Casting
the binary partition to int8 before conversion fixed that check. The separate
repair verified and reused all 16 batches; it performed no generation.
Generation peak GPU memory was not recorded before that assertion failed;
the repair's peak (16,962,348,032 bytes) covers replay only.

Runs: [generation](https://modal.com/apps/new-prc-watermark/main/ap-D231o6A0BQb5LPta2xwWr3),
[PRC repair](https://modal.com/apps/new-prc-watermark/main/ap-OGbNbPrna4wxwghX8eHKHS),
[TextSeal parity](https://modal.com/apps/new-prc-watermark/main/ap-YbvQUNdrxDJUCbtedoLbF8).
[Failed startup record](../step3-v3/failed_attempts.json) and
[PRC prefix audit](../step3-v3/prc_cache_prefix_audit.json) preserve the exceptions.

Local control suite: 20 passed after the repairs. Full artifact verification
also passed. Next: Stage A Self-BLEU, missing completion-only evidence and
prompt-bootstrap uncertainty, using the frozen analysis plan.
