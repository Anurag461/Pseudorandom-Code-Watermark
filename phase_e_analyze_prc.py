"""
Phase E PRC parity-check recovery analysis.

Two modes:
1. Supervised: load the true parity-check matrix from PRC artifacts and
   compute, for each true row, the empirical parity bias across observed
   queries. Report what fraction of true checks are recovered at a given
   z-threshold.
2. False-positive estimate: sample random weight-3 triples NOT in the true
   H and compute their bias to estimate the false-positive rate.

Sweeps k by subsampling the observation matrix.
"""
import json
import os
import sys
from glob import glob

import numpy as np
import torch


N = 400
PRC_BASE = "calib_workdir_n400_t3_eta05"
K_SWEEP = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def parity_bias(obs_subk, indices):
    """Given obs_subk (K_sub, N) int8 with -1=unobs, return (signed_bias, n_valid)
    where bias = P[parity=0] - 0.5, and n_valid = #queries with all indices observed."""
    cols = obs_subk[:, indices]  # (K_sub, t)
    valid = (cols != -1).all(axis=1)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.0, 0
    parity = cols[valid].sum(axis=1) % 2
    p0 = float((parity == 0).mean())
    return p0 - 0.5, n_valid


def load_true_parity_check_matrix():
    art = torch.load(os.path.join(PRC_BASE, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    # decoding_key tuple: (gen_matrix, parity_check_matrix, otp, fpr, noise_rate, test_bits, g, max_bp_iter, t)
    # parity_check_matrix is a scipy.sparse.csr_matrix with exactly t ones per row.
    H = decoding_key[1]
    t = decoding_key[8]
    r = H.shape[0]
    # H.indices is the array of column indices of all nonzero entries, in row-major order
    # since each row has exactly t ones, .reshape(r, t) gives the per-row indices
    rows_arr = H.indices.reshape(r, t)
    print(f"true H shape: {H.shape}, t={t}, r={r}", flush=True)
    rows = [tuple(int(x) for x in row) for row in rows_arr]
    return rows, rows_arr


def main():
    workdir = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(workdir, "prc_recovery.json")

    bits = torch.load(os.path.join(workdir, "bits.pt"),
                      weights_only=False, map_location="cpu")
    obs = bits["obs"]  # (K, N) int8
    K = obs.shape[0]
    print(f"obs shape: {obs.shape}", flush=True)

    true_checks, H_np = load_true_parity_check_matrix()
    print(f"true checks: {len(true_checks)}", flush=True)

    # Sample random non-check triples for FPR estimation
    rng = np.random.default_rng(42)
    n_neg = 5000
    true_set = set(tuple(sorted(t)) for t in true_checks)
    neg_checks = []
    while len(neg_checks) < n_neg:
        s = tuple(sorted(rng.choice(N, size=3, replace=False).tolist()))
        if s in true_set:
            continue
        neg_checks.append(s)

    summary = {"K_sweep": [], "n_true_checks": len(true_checks)}
    rows = []
    for k_sub in K_SWEEP:
        if k_sub > K:
            break
        sub = obs[:k_sub]
        # Compute biases for true checks
        true_biases = []
        true_n_valids = []
        for s in true_checks:
            b, nv = parity_bias(sub, list(s))
            true_biases.append(b)
            true_n_valids.append(nv)
        true_biases = np.array(true_biases)
        true_n_valids = np.array(true_n_valids)
        # Compute biases for negative triples
        neg_biases = []
        neg_n_valids = []
        for s in neg_checks:
            b, nv = parity_bias(sub, list(s))
            neg_biases.append(b)
            neg_n_valids.append(nv)
        neg_biases = np.array(neg_biases)
        neg_n_valids = np.array(neg_n_valids)

        # Recovery threshold: Bonferroni-corrected z over C(N, 3)
        from math import log, sqrt, comb
        z_bonf = sqrt(2 * log(comb(N, 3) + 1))
        # SE under null is 0.5/sqrt(n_valid)
        # We use the observed n_valid per row
        # Define recovery: |bias| > z_bonf * 0.5/sqrt(n_valid)
        rec_thresh_true = z_bonf * 0.5 / np.sqrt(np.maximum(true_n_valids, 1))
        rec_thresh_neg = z_bonf * 0.5 / np.sqrt(np.maximum(neg_n_valids, 1))
        recovered_true = (np.abs(true_biases) > rec_thresh_true) & (true_n_valids > 0)
        recovered_neg = (np.abs(neg_biases) > rec_thresh_neg) & (neg_n_valids > 0)
        tpr = recovered_true.mean()  # of all true checks
        fpr = recovered_neg.mean()   # of negative samples

        # Also report mean |bias| separately
        row = {
            "k": k_sub,
            "z_bonf": z_bonf,
            "true_n_valid_mean": float(true_n_valids.mean()),
            "true_n_valid_min": int(true_n_valids.min()),
            "true_bias_mean_abs": float(np.abs(true_biases).mean()),
            "true_bias_max_abs": float(np.abs(true_biases).max()),
            "true_recovery_rate": float(tpr),
            "true_recovered": int(recovered_true.sum()),
            "true_total": int(len(recovered_true)),
            "neg_bias_mean_abs": float(np.abs(neg_biases).mean()),
            "neg_recovery_rate": float(fpr),
            "neg_recovered": int(recovered_neg.sum()),
            "neg_total": int(len(recovered_neg)),
        }
        rows.append(row)
        print(f"k={k_sub}: TPR={tpr:.3f}({recovered_true.sum()}/{len(recovered_true)}) "
              f"FPR={fpr:.4f}({recovered_neg.sum()}/{len(recovered_neg)}) "
              f"true|b|={row['true_bias_mean_abs']:.4f} "
              f"neg|b|={row['neg_bias_mean_abs']:.4f}", flush=True)

    summary["sweep"] = rows
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
