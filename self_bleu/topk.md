# One matched top-100 batch

Requested 2026-09-18. This is a separate decoding study with exactly **500 new
responses**, followed by analysis and a stop. Commit and push this setup before
either Modal stage. No temperature, depth, eta or prompt expansion is authorized
by this setup. No automatic retries.

Revision `matched_v2` fixes prefix-specific replay diagnostics before any
generation. `matched_v1` was committed and pushed as `539f3f3`; its H100
validation passed and cost an estimated $0.15581, with zero generated responses.
Its immutable manifest/report remain preserved. Version 2 has fresh source pins
and repeats preflight validation before its sole 500-response generation stage.

## Frozen settings

| Item | Setting |
|---|---|
| Model | Qwen3-8B-Base, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Model execution | H100 80 GB, BF16 weights/forward, original pinned Torch/CUDA runtime and static KV cache; batch size 50 |
| Prompts | Original canonical indices 0–49, 50 tokens each; original file/hash |
| Sampling | Seeds 12345 and 67890, temperature 1, top-p 1, exactly top-100 before any watermark |
| Length | Exactly 1,024 new tokens including any special tokens; no EOS stopping |
| Arms | Ordinary, PRC eta .05, native SynthID depths 2/10/30; 100 responses each |
| PRC | Original causal encoder, t=3, row rate 99/100, key seed 12345, fixed partition and position-addressed sampling streams |
| SynthID | Original fixed nested keys, ngram length 4, two leaves, context-history size 1024, native repeat fallback on |
| Primary | PRC minus SynthID depth 2 Self-BLEU at 1,024 tokens |
| Secondary | 400-token prefixes, other depths, detection and repetition diagnostics |

**Precision:** the model continues to run in BF16. All five arms convert its
logits to FP32 before truncation, softmax and probability arithmetic. PRC
completion-only replay uses this same path. This makes the base distribution
common across methods and avoids BF16 rounding of bucket masses. Historical
full-vocabulary response caches are incompatible; none are reused as responses
or nulls here. The checkpoint, tokenizer, prompts, key and partition are reused
only after identity/hash checks.

The top-100 boundary is deterministic: higher logits first, lower token ID for
ties. Excluded logits receive -1e12, giving exactly zero FP32 probability and
matching the official SynthID zero-probability representation. Watermarking
then acts on that distribution. SynthID retains the original full-vocabulary
token IDs, with its internal top-k switch off; native fallback restores this
same truncated ordinary distribution. Every generated token is checked for
support membership. PRC bucket masses include exact empty/full-support
endpoints; its original inverse-CDF sampler repairs only impossible zero-mass
draws at floating-point boundaries and records the repair count.

## Validation before generation

Local tests check stable top-100 ties, FP32 arithmetic, empty buckets, PRC's
average channel distribution, direct agreement with the existing PRC sampler
on filtered logits, fixed-key seeded reproduction, native SynthID update/fallback
support, raw-completion replay alignment, and paired bootstrap covariance.

The first H100 stage generates **no responses**. It verifies the pinned runtime,
checkpoint, keys, partition and source files; repeats the distribution checks
on CUDA; and checks both PRC codeword streams against the historical 100
responses. On saved completions it checks an independent stable-sort bucket
oracle, actual replay inputs, fresh position-zero caches, prefix/order
invariance and first-coordinate abstention. The batch stage refuses to run
unless this stage and its saved artifact hashes pass.

Completion-only PRC detection replays the saved raw completion from token one
with no prompt, BOS or supplied generation-time probabilities. Coordinate one
abstains. The trace uses the same top-100 FP32 decoder. A token outside the
replay distribution's support is recorded and retained, not selectively removed;
removing the prompt can legitimately change the conditional support.
The existing MAP/Hoeffding detector and nominal .001 threshold remain fixed.
SynthID uses raw completions, its official context mask, explicit depth-specific
keys, and the existing weighted normal test at nominal p < .001. It does not use
generation fallback diagnostics as detector evidence.

**Replay diagnostics:** each prefix uses exactly its first `n-1` saved flags,
probabilities and observed-bucket bits. Position indexing is one-based within
the completion; coordinate one abstains. Report separately for PRC and matched
ordinary nulls, for all positions 2–n, early positions 2–64 and later positions
65–n. Include event counts, evaluated-position denominators, rates with paired
prompt-bootstrap intervals, and numbers of affected responses. Count:

- The observed token lying outside replay's top-100 set.
- Contradictory saved bucket endpoints: p1=0 with observed bucket 1, or p1=1
  with observed bucket 0; retain both directions and overlap with token-support
  mismatches.

An absent token need not imply its entire bucket has zero probability. Endpoint
counts refer to the saved FP32 scalar p1, including any numerical rounding to an
endpoint. Neither diagnostic automatically indicates a generation support
violation; generation has its own per-token support checks. Preserve every
token and PRC coordinate. The existing detector clips p1 and assigns magnitude-one
soft scores to contradictory endpoints; this convention is unchanged and is
not a posterior justified for an observation assigned zero probability by replay.

## Preselected probability diagnostic

Use ordinary **saved full-vocabulary** seed-12345 trajectories only as common
histories: prompts 0, 7, 19, 31, 49 at zero-based generation positions
0, 32, 128, 400, 1023. These choices precede the new outcomes. Replay all 50
histories together to preserve model batch geometry; collect 25 selected
histories. At each one, evaluate ordinary and SynthID depths 2/10/30 under:

1. Full-vocabulary FP32 reference probabilities.
2. Matched top-100 FP32 probabilities.

Record sum of squared token probabilities (collision probability), maximum
token probability, probability mass on the ordinary top-100 support after the
watermark, original full-vocabulary mass retained before truncation, and native
fallback status. Advance processor context state along the same fixed history
for both decoders. These are fixed-history diagnostics, not a new response
cohort or a causal explanation for all observed diversity. The FP32 full-vocabulary
reference deliberately holds arithmetic fixed; it is not an exact reconstruction
of the historical BF16 SynthID probability path.

## Analysis and cost

At 1,024 and 400 tokens, compute symmetric two-response sentence Self-BLEU
(SacreBLEU 2.4.3, 13a, exp smoothing, effective order, divided by 100), repeated
four-gram fraction `1 - unique_4grams / (T-3)`, and distinct-3
`unique_3grams / (T-2)`. Repetition uses raw token IDs. Average the latter two
metrics over both responses within each prompt. Lower Self-BLEU and repeated
four-gram fraction and higher distinct-3 mean more lexical diversity.

Use 2,000 paired prompt-cluster bootstrap draws, seed 20260918, retaining both
seeds and all arms for each sampled prompt. Report absolute means and marginal
95% percentile intervals, direct PRC-minus-SynthID contrasts, and each
watermarked arm minus matched ordinary control. Report TPR with paired
intervals and pilot false-positive counts by scoring the same 100 matched
ordinary responses separately under each detector. No historical nulls are
pooled with these. Fifty clustered prompts do not establish a .001 empirical
FPR; zero-count bootstrap intervals must not be interpreted as zero population
false-positive probability. Nonprimary intervals are exploratory.

Starting cumulative planning charge is $7.86713, including the completed v1
validation. One H100 at a time, 4 CPUs,
64 GiB, timeouts 600 s (validation) and 1,800 s (generation plus replay),
retries=0, and a $0.50 overhead allowance reserve approximately $3.60 additional
credit at the frozen planning rate $0.00129148/s. The manifest records the
exact reservation and requires the total to stay below $200. Resource estimates
are not settled invoices. Successful generation shards are saved immediately,
before PRC replay; an attempted stage cannot silently restart.

## Commands and artifacts

The two source modules are `topk.py` (decoder, preparation, collection, analysis)
and `topk_modal.py` (explicit GPU stages). Historical study modules are unchanged.
Use the existing numerical environment and pinned upstream sources.

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m pytest tests/test_self_bleu_topk.py -q
python -m self_bleu.topk prepare
# Commit and push code, manifest and this runbook before dispatch.
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.topk_modal --stage validate
MODAL_PROFILE=new-prc-watermark python -m self_bleu.topk collect --stage validate --download
# Inspect the passed validation report before the only generation stage.
MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m self_bleu.topk_modal --stage batch
MODAL_PROFILE=new-prc-watermark python -m self_bleu.topk collect --stage batch --download
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m self_bleu.topk analyze
```

The immutable manifest lives at `outputs/self_bleu_topk/matched_v2/manifest.json`.
Its source hashes identify the exact code; `source_parent_commit` identifies the
prior commit used to prepare it. Raw generation and replay files live in the
existing Modal volume `prc-completion-only`, under
`self_bleu_topk/<manifest-id>/<stage>/`. Local raw copies are git-ignored. Version
the reports, prompt metrics, all 200 common-history distribution records and
their checksums. Stop after reporting this batch and every setting above.
