# Low-entropy detector interventions

This branch keeps the first two intervention phases separate from the current
generation and Modal code paths.

## Phase 0: cache-only baseline replay

`low_entropy_replay.py` reconstructs the existing online MAP check evidence
from the compact online key, saved tokens, saved partition-probability trace,
and vocabulary partition.  It reports the same statistic, variance proxy, and
Hoeffding threshold as `detect_online_hoeffding(..., weight="map")`.

The replay adapter expects the existing artifact fields `online_key`,
`partition`, and `T`, plus record fields `tokens` and `p_trace`.  It performs no
model loading or generation.

For locally available records:

```bash
python replay_low_entropy_phase01.py \
  --artifact /path/to/artifact.pt \
  --records /path/to/wm_0000.pt /path/to/null_0000.pt \
  --fpr 1e-3 \
  --output /path/to/phase01.jsonl
```

The same functions can be imported by a short-lived Modal CPU function that
reads records directly from the existing volume.  Only the compact JSONL
results need to leave the volume.

## Phase 1: weighted-Rademacher calibration

`weighted_rademacher.py` implements the optimized conditional Chernoff bound

\[
\log \Pr(D\ge d\mid q)
\le
\inf_{\lambda\ge0}
\left\{\sum_a\log\cosh(\lambda q_a)-\lambda d\right\}.
\]

The optimized Chernoff calculation is used rather than a Gaussian or
saddlepoint approximation so the conditional false-positive guarantee remains
rigorous.  The implementation also inverts the bound to return a threshold at
the requested FPR.

Every replay result contains two calibration entries over identical check
evidence:

- `hoeffding`: the current baseline;
- `weighted_rademacher_chernoff`: the Phase 1 intervention.

The output includes their decisions, thresholds, log p-value upper bounds, and
the Rademacher-to-Hoeffding threshold ratio.  Phase 2 basis adaptation is not
part of these modules and can therefore be evaluated independently later.

## Validity requirements

The Rademacher guarantee conditions on the realized check magnitudes and
requires the OTP parity signs to be independent and uniform.  This holds for
the current full-row-rank online parity matrix.  Any later adaptive basis must
remain full rank and must be selected without using OTP parity signs or the
observed detection statistic.
