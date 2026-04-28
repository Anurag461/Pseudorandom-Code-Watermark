"""
Repeat threshold calibration + detection but truncate every trace to the
first n tokens (one codeword cycle, no actual folding -- each codeword
slot has exactly one observation).

Runs both fold variants for comparison, on the saved multinomial p-traces.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import numpy as np
from scipy.stats import norm

from watermark_expt import (
    fold_naive,
    fold_entropy_weighted,
    _test_statistic,
    tokens_to_bits,
)

WORKDIR = "calib_workdir"
ART_PATH = os.path.join(WORKDIR, "artifacts.pt")
FPR = 1e-9
NUM_NULLS = 2000
SEED = 1234


def fit_calibration_truncated(decoding_key, calibration_p_traces, fold_fn,
                              fpr, num_nulls, seed):
    n = decoding_key[0].shape[0]
    traces = [np.asarray(p, dtype=np.float64)[:n]
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
        if fold_fn is fold_entropy_weighted:
            post = fold_fn(observed, p_arr, n)
        else:
            post = fold_fn(observed, n)
        null_stats[i] = _test_statistic(post, decoding_key)
    null_mean = float(null_stats.mean())
    null_std = float(null_stats.std(ddof=1))
    z = float(norm.ppf(1.0 - fpr))
    return {
        "threshold": null_mean + z * null_std,
        "null_mean": null_mean, "null_std": null_std,
        "z": z, "fpr": fpr, "n": n,
        "num_traces_used": len(traces),
    }


def detect_truncated(decoding_key, tokens, p_trace, partition, state, fold_fn):
    n = decoding_key[0].shape[0]
    bits = tokens_to_bits(tokens, partition)[:n]
    p_arr = np.asarray(p_trace, dtype=np.float64)[:n]
    if fold_fn is fold_entropy_weighted:
        posteriors = fold_fn(bits, p_arr, n)
    else:
        posteriors = fold_fn(bits, n)
    stat = _test_statistic(posteriors, decoding_key)
    sigmas = (stat - state["null_mean"]) / state["null_std"]
    return bool(stat > state["threshold"]), stat, sigmas


def run_one(label, fold_fn, watermarked, unwatermarked, decoding_key, partition):
    print(f"\n========== {label} (first n tokens only) ==========", flush=True)
    state = fit_calibration_truncated(
        decoding_key, [r["p_trace"] for r in watermarked],
        fold_fn, FPR, NUM_NULLS, SEED,
    )
    print(f"  threshold = {state['threshold']:.4f}  "
          f"null_mean = {state['null_mean']:.4f}  "
          f"null_std  = {state['null_std']:.4f}", flush=True)

    wm_stats = []
    tp = 0
    for r in sorted(watermarked, key=lambda r: r["job"]["prompt_idx"]):
        d, s, sg = detect_truncated(decoding_key, r["tokens"], r["p_trace"],
                                    partition, state, fold_fn)
        wm_stats.append(s); tp += int(d)

    uw_stats = []
    fp = 0
    for r in sorted(unwatermarked, key=lambda r: r["job"]["prompt_idx"]):
        d, s, sg = detect_truncated(decoding_key, r["tokens"], r["p_trace"],
                                    partition, state, fold_fn)
        uw_stats.append(s); fp += int(d)

    n = len(watermarked)
    wm = np.array(wm_stats); uw = np.array(uw_stats)
    print(f"  TPR: {tp}/{n} ({tp/n:.1%})   FPR: {fp}/{n} ({fp/n:.1%})", flush=True)
    print(f"  watermarked  stat: mean={wm.mean():+.2f} std={wm.std():.2f} "
          f"min={wm.min():+.2f} max={wm.max():+.2f}", flush=True)
    print(f"  unwatermarked stat: mean={uw.mean():+.2f} std={uw.std():.2f} "
          f"min={uw.min():+.2f} max={uw.max():+.2f}", flush=True)


def main():
    art = torch.load(ART_PATH, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n = decoding_key[0].shape[0]
    print(f"codeword length n={n}", flush=True)

    results = []
    for i in range(len(art["jobs"])):
        results.append(torch.load(os.path.join(WORKDIR, f"result_{i:02d}.pt"),
                                  weights_only=False, map_location="cpu"))
    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]
    full_lens = [len(r["p_trace"]) for r in results]
    print(f"loaded WM={len(watermarked)} UW={len(unwatermarked)} "
          f"trace_len={full_lens[0]} (truncating to {n})", flush=True)

    run_one("fold_naive",            fold_naive,            watermarked, unwatermarked, decoding_key, partition)
    run_one("fold_entropy_weighted", fold_entropy_weighted, watermarked, unwatermarked, decoding_key, partition)


if __name__ == "__main__":
    main()
