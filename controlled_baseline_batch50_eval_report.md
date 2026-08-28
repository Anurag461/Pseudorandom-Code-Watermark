# Controlled baseline batch-50 evaluation

Status: **all scoring and reference checks passed, with a material quality and
diversity tradeoff**. This evaluation reused the saved batch-50 generation and
the existing online-PRC/null caches. It launched no GPU, loaded no model, and
generated no text. The remaining 450 prompts were not launched.

## Detection

| Method | T=128 | T=256 | T=400 | T=512 | T=768 | T=1024 |
|---|---:|---:|---:|---:|---:|---:|
| Online PRC | 18% | 42% | 78% | 80% | 96% | 98% |
| TextSeal | 100% | 100% | 100% | 100% | 100% | 100% |
| SynthID-Text | 100% | 100% | 100% | 100% | 100% | 100% |
| Gumbel-Max | 100% | 100% | 100% | 100% | 100% | 100% |

Every method had zero false positives among the 50 shared null texts at every
prefix. This is only a pipeline/calibration sanity check: with 50 nulls, zero
false positives is expected under a nominal 0.1% rule and cannot validate that
rate tightly. The p-values retain their distinct calibration labels and their
raw magnitudes must not be treated as directly comparable.

All 2,400 prompt-prefix records passed the shared schema. All 1,800 baseline
exact-prefix evidence comparisons had zero delta. TextSeal's official/common
decisions agreed everywhere; its maximum p-value difference was
`3.3400313e-7`, within the predeclared mixed absolute/relative numerical
tolerance. SynthID generation matched Google's pinned reference exactly on the
reference input. Online PRC and null cache generation attempts remained zero.

## Quality and diversity

At 1,024 tokens, median metrics were:

| Output | Base NLL | Repetition | Distinct-2 | Distinct-3 |
|---|---:|---:|---:|---:|
| Null | 2.846 | 0.0127 | 0.897 | 0.971 |
| Online PRC | 2.950 | 0.0118 | 0.887 | 0.972 |
| SynthID-Text | 3.011 | 0.0118 | 0.887 | 0.969 |
| TextSeal | 1.439 | 0.2752 | 0.580 | 0.692 |
| Gumbel-Max | 1.078 | 0.5323 | 0.435 | 0.463 |

The TextSeal and Gumbel issue is not confined to a few outliers. TextSeal had
repetition above 0.1 on 41/50 prompts and distinct-3 below 0.8 on 33/50.
Gumbel-Max had repetition above 0.1 on 49/50 and distinct-3 below 0.8 on 41/50.
The corresponding counts were 2/50 and 1/50 for online PRC, and 1/50 and 0/50
for SynthID.

The prefix metrics show a long-horizon collapse rather than uniformly poor
first-token behavior. TextSeal median distinct-3 was 0.992 at 128, 0.922 at
400, 0.859 at 512, and 0.692 at 1,024; median repetition rose from 0.000 to
0.053, 0.120, and 0.275. Gumbel-Max median distinct-3 was 0.992 at 128, 0.790
at 400, and 0.463 at 1,024; repetition rose from 0.000 to 0.202 and 0.532.
SynthID stayed close to the null/PRC diversity range through 1,024.

The unusually low TextSeal/Gumbel NLL does not cancel this warning. It means
their selected tokens are individually high-probability under the base model,
while the sequence-level outputs become repetitive. Detection strength must
therefore be interpreted jointly with the diversity loss.

## Decision implication

The frozen 500-prompt run remains technically valid as a joint
detectability-quality-diversity comparison. It is not a quality-matched
detector comparison: TextSeal and especially Gumbel achieve perfect detection
in this batch at substantially different output-diversity operating points.
The 500-prompt run would estimate those tradeoffs more precisely, but it will
not remove them. Changing temperature, top-p, alpha, depth, or the official
samplers would be a different sensitivity/operating-point experiment and must
not be substituted silently.

CPU scoring cost **$0.00604867** (`$0.00451978` CPU and `$0.00152889` memory),
with zero GPU cost. The Modal run is
<https://modal.com/apps/new-prc-watermark/main/ap-TiyaTghkwGCV6xEcLkk8SU>.
