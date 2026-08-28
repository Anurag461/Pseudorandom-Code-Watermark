# Controlled 8B baseline quality/diversity diagnostic

Date: 2026-08-22 (runs crossed 2026-08-23 UTC)

## Decision

The smoke anomaly is reproducible, but the diagnostic found no evidence that it
comes from the two suspected implementation faults. TextSeal again developed
high repeated-4-gram rates and low distinct-3 under a complete second seed, and
paper-style decoding (`temperature=0.8`, `top_p=0.9`, 400 tokens) did not remove
the effect on three of five prompts. The exact log-space Gumbel argmax produced
the same tokens as the released power-form sampler at both tested batch shapes,
so power-form underflow was not responsible for the observed Gumbel behavior.

This makes the 500-prompt frozen comparison scientifically usable as a
quality--detectability study, not as a detection-only ranking. The anomaly is a
method-by-model/prompt/decoding outcome unless larger-sample evidence says
otherwise. It must not be hidden by tuning TextSeal or Gumbel to match PRC's
diversity. The 500-prompt run remains unlaunched and approval-gated.

The diagnostic does **not** establish that Qwen3-8B has higher conditional
entropy than Qwen3.5-27B. Model size alone does not order conditional entropy,
and the unauthorized 27B comparison was not run. It does establish that, within
these 8B continuations, repeated 4-gram events occurred in substantially
lower-entropy positions than novel events. That association is compatible with
a low-entropy feedback loop but does not identify its cause.

## Scope and safeguards

- Canonical prompt rows `0..4` only.
- One H100 worker for the bounded generation matrix; model loaded once.
- No local model loading or generation.
- Cache analysis used the immutable smoke artifact plus validated online-PRC
  and null caches, with `generation_attempts=0`.
- No 500-prompt generation, 50-by-5 diversity run, 27B replication, or proxy
  experiment was launched.
- Diagnostic campaign hard cap: 2 dollars. Exact spend: `0.17113157` dollars.

## Cache-only prefix analysis

The fixed definitions were: distinct-n = unique token n-grams divided by all
n-gram positions; repetition rate = one minus the unique 4-gram fraction; and
base entropy = the Qwen distribution entropy in nats immediately before the
sampled token. Median values across five prompts were:

| Method | Prefix | distinct-3 | Repeated 4-gram rate | Base entropy |
| --- | ---: | ---: | ---: | ---: |
| Null | 128 | 0.9921 | 0.0000 | 2.9439 |
| TextSeal seed 12345 | 128 | 0.9921 | 0.0000 | 2.7744 |
| Gumbel-Max | 128 | 0.8810 | 0.1040 | 2.5461 |
| Null | 400 | 0.9899 | 0.0000 | 2.9096 |
| TextSeal seed 12345 | 400 | 0.8467 | 0.1335 | 2.0262 |
| Gumbel-Max | 400 | 0.6030 | 0.3955 | 1.9184 |
| Null | 1024 | 0.9384 | 0.0353 | 2.3476 |
| Online PRC | 1024 | 0.9814 | 0.0059 | 3.1845 |
| TextSeal seed 12345 | 1024 | 0.3503 | 0.6405 | 1.0871 |
| SynthID-Text | 1024 | 0.9795 | 0.0127 | 3.1626 |
| Gumbel-Max | 1024 | 0.4374 | 0.5357 | 0.9375 |

TextSeal was healthy at 128 tokens, showed a material median gap by 400, and
continued to deteriorate through 1024. The effect was heterogeneous: at 1024,
TextSeal prompt-level distinct-3 ranged from `0.2153` to `0.8112`, and repeated
4-gram rate ranged from `0.1528` to `0.7786`. One prompt entered an exact
period-21 run at token 248 lasting 777 tokens.

For every TextSeal prompt, mean entropy at repeated-4-gram events was lower
than at novel events. The repeat-minus-novel entropy differences were
`-1.968`, `-1.684`, `-1.901`, `-2.468`, and `-2.726` nats. This is descriptive
association along the generated trajectory, not a causal or cross-model test.

## TextSeal seed and paper-decoding checks

The complete second TextSeal seed produced different text (per-prompt token
agreement `0.0098`--`0.0322`) but reproduced the degradation:

| Frozen TextSeal run | Prefix | distinct-3 | Repeated 4-gram rate |
| --- | ---: | ---: | ---: |
| Seed 12345 | 400 | 0.8467 | 0.1335 |
| Seed 67890 | 400 | 0.7789 | 0.2116 |
| Seed 12345 | 1024 | 0.3503 | 0.6405 |
| Seed 67890 | 1024 | 0.3033 | 0.6934 |

At the paper decoding regime and 400 tokens, median vanilla distinct-3 was
`0.8920` with repetition `0.0605`; TextSeal distinct-3 was `0.6910` with
repetition `0.2821`. Prompt-level TextSeal-minus-vanilla distinct-3 differences
were `-0.342`, `-0.003`, `-0.410`, `+0.040`, and `-0.201`. Thus the decoding
change reduced neither the anomaly nor its prompt heterogeneity to zero.

Base-model NLL is not sufficient as a quality check here: repeated, predictable
loops can have low NLL. Distinct-n and repetition therefore remain necessary
co-primary quality outcomes in the full comparison.

## Gumbel-Max finding

Gumbel-Max is expected to be deterministic when the prompt/prefix, watermark
key, logits, decoding parameters, batch shape, and execution environment are
fixed. Ordinary sampling seeds are not independent Gumbel replicates.

The diagnostic compared the released `r^(1/p)` implementation with the exact
log-space argmax `argmax(log(r)/p)` through 400 tokens. Agreement was `1.0` at
batch size 1 and `1.0` at batch size 5, including identical token hashes.
Batch-1 versus batch-5 agreement was `0.09` for both formulations. Therefore:

- the tested power form and stable form are behaviorally equivalent;
- underflow is not the cause of the tested output divergence;
- batch-shape sensitivity comes from batch-dependent model logits/numerical
  execution and is amplified by deterministic autoregression;
- production must freeze batch size, grouping, ordering, hardware, model
  revision, and watermark key.

## Project-Qwen versus Hugging Face logits

The offline reference used `transformers==4.51.3`, eager attention, BF16, the
same cached Qwen revision, and no network downloads. The predeclared strict
criterion required all top-1 tokens equal, top-10 overlap at least 9, and
maximum Jensen--Shannon divergence no greater than `1e-4`.

The strict check is retained as **failed** because batch-5 prefill maximum JSD
was `8.726e-4`. All top-1 tokens were equal and top-10 overlap was 9--10.
Batch-1 prefill JSD was `7.985e-5`, and the cached next-token JSD was
`2.945e-7`. For context, project Qwen's own batch-1 versus batch-5 maximum JSD
was `5.898e-4`, while native Hugging Face's was `1.622e-3`. Native Hugging Face
therefore showed batch-shape variation of the same order and larger maximum
than the project implementation in this test.

The threshold failure is a recorded numerical discrepancy, not evidence that
the adapter applied the wrong formula. It does limit bitwise portability across
implementations and batch shapes. Internal validity of the frozen comparison
is preserved because all methods use the same project-Qwen logits path and the
production batch layout is fixed.

## Runtime, memory, and exact billing

The bounded generation worker ran for `78.2336` seconds, including a
`9.0693`-second model load. Its five generation loops took `25.7841`, `9.1059`,
`9.1070`, `8.8724`, and `8.6593` seconds. Peak CUDA memory was
`17,668,689,920` bytes allocated and `17,702,060,032` bytes reserved. The
enhanced dual-model parity check peaked at `33,229,244,928` bytes allocated and
`33,246,150,656` bytes reserved.

Exact provider billing was:

| Resource | Cost (USD) |
| --- | ---: |
| H100 | 0.14751330 |
| CPU | 0.00919542 |
| Memory | 0.01442285 |
| **Total** | **0.17113157** |

The five app URLs and every resource charge are in
`outputs/controlled_baseline_diagnostic_cost_ledger.csv`.

## Readiness and interpretation guard

No accidental regeneration, formula mismatch, power-form numerical failure,
or project-specific batch instability beyond native BF16 behavior was found.
The controlled 500-prompt run is therefore ready for a separate approval if
its claim is framed as detectability jointly with NLL, repetition, distinct-2,
and distinct-3. The diagnostic does not support a claim that TextSeal has
quality comparable to PRC on this setting, nor a detection-only winner claim.

The 14--18-dollar, 20--35-minute estimate is retained as the batch-5 upper
bound. A standalone batch-50 validation must replace it with a measured
estimate before full approval. The literal gates and commands remain in
`controlled_baseline_full_run_runbook.md`.
