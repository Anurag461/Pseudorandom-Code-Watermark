# Qwen3-0.6B full-vocabulary comparison

**Complete 2026-09-19:** [results](../outputs/self_bleu_full_vocab/qwen3_0p6b_v1/REPORT.md).
Exactly 600 responses; all validation/analysis checks passed. New worker resource
estimate $0.72862; cumulative planning charge $11.00279 of $200. No additional
generation is queued. The frozen protocol follows.

Authorized scope: **600 responses**, six settings × 50 canonical prompts × seeds
12345 and 67890. Ordinary sampling; PRC eta .05; SynthID depths 2 and 10;
TextSeal alpha .1; Gumbel-max. **Repeat fallback is ON for every contextual
baseline**, as clarified by the user. No native-off TextSeal/Gumbel arms.

All responses use Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, temperature 1, top-p 1, no top-k,
batch size 50, and exactly 1,024 generated tokens without EOS stopping. Evaluate
the same responses at 400 and 1,024 tokens. Fixed keys, nested SynthID keys,
PRC eta .05/t3/row rate 99/100 and original partition remain unchanged.
The artifact's historical path mentions 8B; only its key and vocabulary partition
are reused. The 0.6B and 8B tokenizers have the same verified hash. No old
responses, nulls, or model-derived traces are reused.

BF16 model weights/forward execution are preserved. All generation paths receive
FP32 logits before probability calculations. PRC replay also uses FP32 full-vocabulary
probabilities. TextSeal detection retains its unmodified upstream **BF16 entropy
calculation**, Hugging Face eager attention, and independent full-prefix replay;
this is the previously validated detector convention, not a detector change to
make entropy arithmetic match generation. Its model checkpoint is also 0.6B.
Historical full-vocabulary 8B samplers do not all use this FP32 generation path,
so their results are not exactly matched controls for a model-size effect.

TextSeal/Gumbel use the established `SamplerRepeatPolicy` with response-local
three-token context tracking and independent fallback RNG streams. They begin
from the last three prompt tokens. SynthID keeps Google's native handler,
including its zero-context initialization and 1,024-entry context history.
All fallbacks use ordinary full-vocabulary probabilities. PRC is position based
and is unchanged. Generation records every fallback flag; analysis independently
reconstructs all four contextual arms' masks from their saved trajectories.

## Execution gates

Commit and push the setup first. Run local sampler/replay/paired-analysis checks,
then two GPU preflights, neither of which generates responses:

1. Verify generation runtime, checkpoint bytes/revision, fixed key/partition,
   original latent PRC streams, probability/channel semantics, forced repeated
   contexts and raw completion replay. Use the canonical 50-token arrays only
   as known replay fixtures; verify fresh caches, prefix/order agreement and
   first-coordinate abstention.
2. Load the same checkpoint in the pinned TextSeal HF runtime. Verify direct
   entropy/scoring against upstream public detection and observe actual raw-token
   model inputs, using two known fixtures at 32 and 50 tokens.
3. Only after both pass, generate twelve sequential 50-response batches, save
   each immediately, and replay PRC on 100 PRC and 100 ordinary completions.
4. Replay TextSeal independently at both requested lengths on 100 TextSeal
   and 100 ordinary completions; check upstream equality on the first row of
   each batch. Score SynthID/Gumbel locally from raw completion tokens.

One H100 at a time, four CPUs, 64 GiB; retries disabled. Timeouts are
600/300/2400/600 seconds, respectively. Initial cumulative planning charge is
$9.77418; timeout reservations plus $0.50 overhead allowance keep the planned
cumulative total below $15.33 of the $200 budget. Estimates are not settled bills.
An existing attempt cannot silently relaunch. Stop after this six-setting batch.

## Analysis

The primary contrast remains PRC minus SynthID depth 2 Self-BLEU at 1,024 tokens.
Report all six arms at both lengths, absolute and direct paired intervals,
PRC-minus-each-baseline and each watermark-minus-ordinary diversity differences,
repeated-four-gram fraction and distinct-3. Two-response symmetric sentence
Self-BLEU uses the existing SacreBLEU 2.4.3/13a/exp/effective-order settings and
a 0–1 scale. Use 2,000 paired prompt-cluster bootstrap resamples (seed 20260918),
keeping both seeds and all methods together. Other comparisons are exploratory.

Detection remains completion-only at nominal p < .001: original PRC MAP/Hoeffding
with coordinate one abstaining; TextSeal weighted v2; official SynthID context
mask and weighted normal test with explicit keys; Gumbel's existing deduplicated
tuple/Gamma test. Score all five detectors on the **same 100 new ordinary nulls**.
Do not pool historical nulls. These pilot counts cannot establish an empirical
.001 FPR. Preserve zero-count/perfect-detection interval limitations.

Report repeat-affected response fractions, fallback counts and first-repeat
positions. PRC diagnostics separately count zero observed-token probability and
contradictory saved bucket endpoints, using exactly n−1 positions for each prefix,
separating PRC/ordinary cohorts and positions 2–64 versus 65–n. Never discard or
reindex tokens or alter the primary detector based on these diagnostics.

## Commands and artifacts

Local module: `python -m self_bleu.full_vocab prepare|collect|analyze`.
Dispatch: `MODAL_PROFILE=new-prc-watermark python -m modal run --detach -m
self_bleu.full_vocab_modal --stage validate` (then `validate_textseal`, `batch`,
and `textseal`).
Collect each stage with `collect --stage STAGE --download` before analysis.

Setup/results: `outputs/self_bleu_full_vocab/qwen3_0p6b_v1/`.
Remote raw data: existing `prc-completion-only` volume, under
`self_bleu_full_vocab/<manifest-id>/`. Manifest and each stage bind source hashes,
runtime, model revision, settings, prompts, response identities and artifacts.
Only `full_vocab_modal.py` dispatches workers. Historical study files are untouched.
