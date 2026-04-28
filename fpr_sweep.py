"""
FPR sweep on saved n=128 calibration results.

Reuses the saved null_mean / null_std from qwen_threshold.json (entropy-weighted
fit), derives thresholds for a range of FPR targets, and computes empirical
TPR / FPR over the 30 watermarked + 30 unwatermarked traces.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json
import torch
import numpy as np
from scipy.stats import norm

from watermark_expt import (
    fold_entropy_weighted,
    _test_statistic,
    tokens_to_bits,
)

WORKDIR = "calib_workdir"
THRESHOLD_PATH = "qwen_threshold.json"


def stat_for(decoding_key, tokens, p_trace, partition):
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(tokens, partition)
    p_arr = np.asarray(p_trace, dtype=np.float64)
    posteriors = fold_entropy_weighted(bits, p_arr, n)
    return _test_statistic(posteriors, decoding_key)


def main():
    art = torch.load(os.path.join(WORKDIR, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n = decoding_key[0].shape[0]

    with open(THRESHOLD_PATH) as f:
        state = json.load(f)
    null_mean = state["null_mean"]
    null_std = state["null_std"]
    print(f"n = {n}", flush=True)
    print(f"null_mean = {null_mean:.4f}  null_std = {null_std:.4f}", flush=True)

    results = []
    for i in range(len(art["jobs"])):
        results.append(torch.load(os.path.join(WORKDIR, f"result_{i:02d}.pt"),
                                  weights_only=False, map_location="cpu"))
    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]

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
