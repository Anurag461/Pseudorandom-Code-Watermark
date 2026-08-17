# Low-entropy detector interventions

This branch keeps the first three intervention phases separate from the
current generation code paths.

## Phase 0: cache-only baseline replay

`low_entropy_replay.py` reconstructs the existing MAP check evidence from saved
tokens and partition-probability traces.  It supports both the compact online
key and the fixed-PRC decoding key.  It reports the same statistic, variance
proxy, and Hoeffding threshold as the corresponding production detector.

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

For the authoritative fixed Qwen3-0.6B run at `n=T=416`, `t=3`, and
`eta=0.05`, `modal_low_entropy_replay.py` provides two CPU-only entry points:

```bash
MODAL_PROFILE=new-prc-watermark modal run \
  modal_low_entropy_replay.py::audit

MODAL_PROFILE=new-prc-watermark modal run \
  modal_low_entropy_replay.py::run \
  --output outputs/low_entropy_phase01_fixed_n416_eta005_0p6b.json
```

The audit verifies the artifact fingerprint, rank, watermarked-record count,
and compatible null caches before scoring.  The replay then asserts exact
reproduction of the authoritative Hoeffding result (`456/500` true positives
and `0/500` false positives) before returning Phase 1 results.  Neither command
requests a GPU or loads a language model.

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
the Rademacher-to-Hoeffding threshold ratio.

## Phase 2: reliability-adaptive parity basis

`adaptive_parity_basis.py` selects a new full-rank basis of the same fixed PRC
dual code.  For each cached probability trace it:

1. computes the absolute bucket reliability
   \(\rho_i=\min(p_i,1-p_i)/\max(p_i,1-p_i)\);
2. considers a fixed grid of bottom-reliability erasure quantiles, including
   the unchanged basis at quantile zero;
3. uses invertible GF(2) row operations to concentrate each candidate erasure
   set into as few checks as its column rank permits; and
4. selects the candidate maximizing
   \[
   J=\sum_a(1-2\eta)^{2|v_a|}\prod_{i:v_{a,i}=1}\rho_i.
   \]

The transformed check score includes the matching
\((1-2\eta)^{|v_a|}\) degree penalty and uses the Phase 1 conditional
weighted-Rademacher calibration.  Selection never receives the one-time pad,
parity signs, token bucket observations, or detection statistic.  The
zero-erasure candidate therefore exactly reproduces the Phase 1 decision,
while nonzero candidates can cancel unreliable coordinates shared by checks.

The CPU-only Modal adapter reuses the authoritative cached 0.6B and 8B fixed
runs and writes a separate, versioned result:

```bash
MODAL_PROFILE=new-prc-watermark modal run \
  modal_low_entropy_phase2.py::run

MODAL_PROFILE=new-prc-watermark modal run \
  modal_low_entropy_phase2.py::run_8b_n749
```

The corresponding `verify` and `verify_8b_n749` entry points independently
check the saved record manifest, result identity count, source hashes, and
soundness metadata.  Phase 0/1 caches are not overwritten.

## Validity requirements

The Rademacher guarantee conditions on the realized check magnitudes and
requires the OTP parity signs to be independent and uniform.  Phase 2 checks
this precondition and constructs every candidate by invertible row operations
from the authoritative full-row-rank matrix.  Its basis selection is a
deterministic function of the parity matrix, probability-derived reliability,
and noise rate; it does not use OTP parity signs or the observed statistic.
