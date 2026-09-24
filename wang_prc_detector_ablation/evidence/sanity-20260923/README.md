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

`source_check.json` contains the original detector comparison. `report.json`
contains status, timings and the four saved trace identities/hashes. The raw
traces and secret key remain on `prc-data` under
`wang_prc_detector_ablation/qwen3_8b_base/35f5566b766f8be3aea1cef9/sanity/`.
The downloaded backup is `/private/tmp/wang-sanity-saved`; no model execution or
dataset scoring was performed locally. The failed check remains recorded above
and in `report.json`.

At this stage the numerical discrepancy was not yet diagnosed. BF16 matrix shapes differ
between incremental and full-prefix execution, which is a plausible explanation;
that does not establish that the cache is correct. The 2% guard is unchanged,
and this attempt remains failed. These short traces do not estimate detection TPR.

## Follow-up

The subsequently approved saved-token numerical diagnostic passed both cache and
FP32 controls; see [its report](../numerical-20260923/README.md). This original
smoke attempt remains failed under its unchanged BF16 guard. The completed
[full experiment](../experiment-20260924/README.md) used those follow-up controls
and disclosed the remaining numerical limitations. No additional short T=1.8
smoke was run. Historical quotes and billing records are omitted from the PR.
