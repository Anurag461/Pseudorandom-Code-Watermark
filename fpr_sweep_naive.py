"""
FPR sweep on saved n=128 calibration results, using fold_naive.

Re-derives null_mean / null_std for naive fold from the saved watermarked
p-traces, then computes empirical TPR / FPR over a range of FPR targets.
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
NUM_NULLS = 2000
SEED = 1234


def fit_null_naive(decoding_key, p_traces, num_nulls, seed):
    n = decoding_key[0].shape[0]
    traces = [np.asarray(p, dtype=np.float64) for p in p_traces if len(p) >= n]
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
    return float(null_stats.mean()), float(null_stats.std(ddof=1))


def stat_for(decoding_key, tokens, p_trace, partition):
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(tokens, partition)
    posteriors = fold_naive(bits, n)
    return _test_statistic(posteriors, decoding_key)


def main():
    art = torch.load(os.path.join(WORKDIR, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n = decoding_key[0].shape[0]
    print(f"n = {n}", flush=True)

    results = []
    for i in range(len(art["jobs"])):
        results.append(torch.load(os.path.join(WORKDIR, f"result_{i:02d}.pt"),
                                  weights_only=False, map_location="cpu"))
    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]

    null_mean, null_std = fit_null_naive(
        decoding_key, [r["p_trace"] for r in watermarked], NUM_NULLS, SEED,
    )
    print(f"null_mean = {null_mean:.4f}  null_std = {null_std:.4f} "
          f"(naive fold, {NUM_NULLS} nulls)", flush=True)

    wm_stats = np.array([stat_for(decoding_key, r["tokens"], r["p_trace"], partition)
                         for r in watermarked])
    uw_stats = np.array([stat_for(decoding_key, r["tokens"], r["p_trace"], partition)
                         for r in unwatermarked])

    print(f"\nwatermarked   stat: min={wm_stats.min():+.2f}  max={wm_stats.max():+.2f}  "
          f"mean={wm_stats.mean():+.2f}", flush=True)
    print(f"unwatermarked stat: min={uw_stats.min():+.2f}  max={uw_stats.max():+.2f}  "
          f"mean={uw_stats.mean():+.2f}", flush=True)

    fprs = [1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1]
    print("\n  FPR target |   z   | threshold |  TPR   | empirical FPR", flush=True)
    print("  -----------+-------+-----------+--------+--------------", flush=True)
    n_w = len(wm_stats)
    n_u = len(uw_stats)
    for fpr in fprs:
        z = float(norm.ppf(1.0 - fpr))
        thresh = null_mean + z * null_std
        tp = int((wm_stats > thresh).sum())
        fp = int((uw_stats > thresh).sum())
        print(f"  {fpr:>9.0e}  | {z:5.3f} | {thresh:9.4f} | {tp:2d}/{n_w} | {fp}/{n_u}",
              flush=True)


if __name__ == "__main__":
    main()
