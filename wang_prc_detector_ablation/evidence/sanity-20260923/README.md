# First sanity attempt — September 23, 2026

The attempt ran from pushed commit `3eaf5af3ad449e9aa14349545481b46707fe5586`,
Modal app `ap-B2uQZe8KiwVEuFJ9T2iCID`. The app is stopped, with zero tasks.
The user approved the specifically quoted $0.50–$1.50 sanity package with
“go ahead.” No retry or production run was launched.

| Check | Outcome |
| --- | --- |
| Wang complete T=1.8 original/adapted violation counts and individual decisions | All 160 match |
| Original/adapted/saved aggregate detections | 159 / 160 in all three |
| Pinned Base checkpoint hashes and revision | Passed before generation |
| T=1.0 two WM + two null, 64 tokens each | Generated and replayed; four traces saved |
| Sampler branch reconstruction, valid IDs/probabilities, completion-only input capture | Passed for the four saved traces |
| Cached versus uncached BF16 prefix check | Failed at length 4: TV 0.0220115844 > 0.02; max logit difference 0.25 |
| T=1.8 short GPU test | Not run after the failure |
| Production | Not launched |

The recorded CPU function time is 8.83 seconds; the GPU container ran for
110.82 seconds (0.03078 GPU-hours). At the quoted resource rates their combined
runtime estimate is **$0.1439**, excluding the image build and any unmeasured
overhead/storage. This is not final provider billing: the refreshed report had
no entry for this app and its newest interval was 2026-09-23 12:00 UTC.

`source_check.json` contains the original detector comparison. `report.json`
contains status, timings and the four saved trace identities/hashes. The raw
traces and secret key remain on `prc-data` under
`wang_prc_detector_ablation/qwen3_8b_base/35f5566b766f8be3aea1cef9/sanity/`.
The downloaded backup is `/private/tmp/wang-sanity-saved`; no model execution or
dataset scoring was performed locally. `failure.log` preserves the failure.

This numerical discrepancy is not yet diagnosed. BF16 matrix shapes differ
between incremental and full-prefix execution, which is a plausible explanation;
that does not establish that the cache is correct. The 2% guard is unchanged,
and this attempt remains failed. These short traces do not estimate detection TPR.

## Prepared next step — requires a new approval

One **H100 / 4 cores / 64 GiB, batch 1**, reusing only the saved failing null's
first 16 token IDs at T=1.0. Compare static cache, concatenating cache and uncached
forward passes at lengths 1, 4, 8 and 16, in BF16 and then FP32 with TF32 disabled.
This is 122 teacher-forced token positions, no generation and no repeated source
check. Save all differences and exact static/concat equality. Predeclared FP32
reference limits are TV≤1e-4 and max logit difference≤.002. Cross-precision errors
help distinguish cache handling from reduced-precision shape effects.

The diagnostic **does not change the existing BF16 tolerance or mark the sanity
gate passed**. It does not continue to T=1.8 or production automatically.
Expected time is **5–10 minutes including startup**, estimated **$0.25–$0.75**;
function timeout 420 seconds, retries zero. Its metered function-time envelope is
about $0.54, plus startup/build allowance. Total estimated CPU/GPU runtime from
the first attempt plus the diagnostic quote would be approximately $0.39–$0.89,
before unposted costs. If the previously stated $35 was still available before
this attempt, that leaves approximately $34.11–$34.61 before those costs; the
workspace billing report does not establish a remaining credit balance.

```bash
python -m wang_prc_detector_ablation.launch quote --stage numerical-diagnostic
# Only after explicit approval of this additional GPU diagnostic:
MODAL_PROFILE=new-prc-watermark python -m wang_prc_detector_ablation.launch launch \
  --stage numerical-diagnostic \
  --approval wang_prc_detector_ablation/approvals/numerical-diagnostic.json
```

No approval file authorizing this diagnostic has been created.

## Follow-up

The subsequently approved saved-token numerical diagnostic passed both cache and
FP32 controls; see [its report](../numerical-20260923/README.md). This original
smoke attempt remains failed under its unchanged BF16 guard. Billing has now
posted at $0.14605021; see billing_reconciliation.json. No production run launched.
