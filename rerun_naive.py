"""
Re-run threshold calibration + detection on saved p-traces using fold_naive
(equal-weight fold) instead of fold_entropy_weighted.

Reads:
  calib_workdir/artifacts.pt   (decoding_key, partition)
  calib_workdir/result_NN.pt   (tokens, p_trace, job)

Mirrors fit_calibration / detect_with_threshold but swaps the fold function.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import numpy as np
from scipy.stats import norm

from watermark_expt import (
    fold_naive,
    _test_statistic,
    tokens_to_bits,
)

WORKDIR = "calib_workdir"
ART_PATH = os.path.join(WORKDIR, "artifacts.pt")
FPR = 1e-9
NUM_NULLS = 2000
SEED = 1234


def fit_calibration_naive(decoding_key, calibration_p_traces, fpr, num_nulls, seed):
    n = decoding_key[0].shape[0]
    traces = [np.asarray(p, dtype=np.float64)
              for p in calibration_p_traces if len(p) >= n]
    rng = np.random.default_rng(seed)
    null_stats = np.empty(num_nulls, dtype=np.float64)
    for i in range(num_nulls):
        p_arr = traces[rng.integers(0, len(traces))]
        T = len(p_arr)
        codeword = rng.integers(0, 2, size=n)
        xi = codeword[np.arange(T) % n]
        bern_p = np.where(p_arr <= 0.5,
                          2 * xi * p_arr,
                          1 - 2 * (1 - xi) * (1 - p_arr))
        bern_p = np.clip(bern_p, 0.0, 1.0)
        observed = rng.binomial(1, bern_p)
        post = fold_naive(observed, n)
        null_stats[i] = _test_statistic(post, decoding_key)
    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std(ddof=1))
    z = float(norm.ppf(1.0 - fpr))
    return {
        "threshold": null_mean + z * null_std,
        "null_mean": null_mean,
        "null_std": null_std,
        "z": z, "fpr": fpr, "n": n,
        "num_traces_used": len(traces),
    }


def detect_naive(decoding_key, tokens, p_trace, partition, state):
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(tokens, partition)
    posteriors = fold_naive(bits, n)
    stat = _test_statistic(posteriors, decoding_key)
    sigmas = (stat - state["null_mean"]) / state["null_std"]
    return bool(stat > state["threshold"]), stat, sigmas


def main():
    art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n_jobs = len(art["jobs"])

    results = []
    for i in range(n_jobs):
        path = os.path.join(WORKDIR, f"result_{i:02d}.pt")
        results.append(torch.load(path, weights_only=False, map_location="cpu"))

    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]
    print(f"loaded watermarked={len(watermarked)} unwatermarked={len(unwatermarked)}",
          flush=True)

    print("\n=== Fit threshold (fold_naive) ===", flush=True)
    state = fit_calibration_naive(
        decoding_key,
        [r["p_trace"] for r in watermarked],
        fpr=FPR, num_nulls=NUM_NULLS, seed=SEED,
    )
    print(f"  threshold = {state['threshold']:.4f}  "
          f"null_mean = {state['null_mean']:.4f}  "
          f"null_std  = {state['null_std']:.4f}  "
          f"traces_used = {state['num_traces_used']}", flush=True)

    print("\n=== TPR (watermarked) ===", flush=True)
    wm_stats, wm_sigmas = [], []
    tp = 0
    for r in sorted(watermarked, key=lambda r: r["job"]["prompt_idx"]):
        d, s, sg = detect_naive(decoding_key, r["tokens"], r["p_trace"],
                                partition, state)
        wm_stats.append(s); wm_sigmas.append(sg); tp += int(d)
        print(f"  prompt {r['job']['prompt_idx']:2d}: detected={d}  "
              f"stat={s:.2f}  sigmas={sg:.2f}", flush=True)

    print("\n=== FPR (unwatermarked) ===", flush=True)
    uw_stats, uw_sigmas = [], []
    fp = 0
    for r in sorted(unwatermarked, key=lambda r: r["job"]["prompt_idx"]):
        d, s, sg = detect_naive(decoding_key, r["tokens"], r["p_trace"],
                                partition, state)
        uw_stats.append(s); uw_sigmas.append(sg); fp += int(d)
        print(f"  prompt {r['job']['prompt_idx']:2d}: detected={d}  "
              f"stat={s:.2f}  sigmas={sg:.2f}", flush=True)

    n = len(watermarked)
    print("\n=== Summary (fold_naive) ===", flush=True)
    print(f"  TPR: {tp}/{n}  ({tp/n:.1%})", flush=True)
    print(f"  FPR: {fp}/{n}  ({fp/n:.1%})", flush=True)
    print(f"  threshold (FPR={FPR}): {state['threshold']:.4f}", flush=True)
    wm = np.array(wm_stats); uw = np.array(uw_stats)
    print(f"  watermarked  stat: mean={wm.mean():+.2f} std={wm.std():.2f} "
          f"min={wm.min():+.2f} max={wm.max():+.2f}", flush=True)
    print(f"  unwatermarked stat: mean={uw.mean():+.2f} std={uw.std():.2f} "
          f"min={uw.min():+.2f} max={uw.max():+.2f}", flush=True)
    # Welch's t-statistic between the two distributions
    diff = wm.mean() - uw.mean()
    se = float(np.sqrt(wm.var(ddof=1)/len(wm) + uw.var(ddof=1)/len(uw)))
    print(f"  mean(WM) - mean(UW) = {diff:+.2f}  Welch t = {diff/se:+.2f}",
          flush=True)


if __name__ == "__main__":
    main()
