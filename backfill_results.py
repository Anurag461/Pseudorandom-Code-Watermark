"""
Backfill detection results for cached calib workdirs.

For each workdir, runs the detectors that were not previously run on those
generations and prints a single CSV-style line per (workdir, method, fpr_target).

Usage:
    python backfill_results.py
"""
import os
import sys
import torch
import numpy as np

from watermark_expt import (
    fit_calibration,
    detect_with_threshold,
    detect_syndrome,
)

# (workdir, methods_to_run, fpr_targets)
JOBS = [
    ("calib_workdir_n128_t3_eta05", ["entropy", "naive", "syn_all", "syn_ent"], [1e-9]),
    ("calib_workdir_n128_t7",       ["naive", "syn_all", "syn_ent"], [1e-9]),
    ("calib_workdir_n4096_t14",     ["syn_all", "syn_ent"], [1e-9]),
]

ENT_THR = 0.1


def load(workdir):
    art = torch.load(os.path.join(workdir, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    n_jobs = len(art["jobs"])
    results = []
    for j in range(n_jobs):
        results.append(torch.load(
            os.path.join(workdir, f"result_{j:02d}.pt"),
            weights_only=False, map_location="cpu",
        ))
    wm = [r for r in results if r["job"]["watermark"]]
    nw = [r for r in results if not r["job"]["watermark"]]
    return art, wm, nw


def run_fold(art, wm, nw, fold, fpr):
    cal_traces = [np.asarray(r["p_trace"], dtype=np.float64) for r in nw]
    state = fit_calibration(
        art["decoding_key"], cal_traces,
        fpr=fpr, num_simulated_nulls=2000, fold=fold, seed=1234,
    )
    tp = sum(int(detect_with_threshold(
        art["decoding_key"], r["tokens"], r["p_trace"],
        art["partition"], state)) for r in wm)
    fp = sum(int(detect_with_threshold(
        art["decoding_key"], r["tokens"], r["p_trace"],
        art["partition"], state)) for r in nw)
    return tp, len(wm), fp, len(nw), state["threshold"], state["null_mean"], state["null_std"]


def run_syn(art, wm, nw, ent_thr):
    tp = sum(int(detect_syndrome(
        art["decoding_key"], r["tokens"], r["p_trace"], art["partition"],
        entropy_threshold=ent_thr)) for r in wm)
    fp = sum(int(detect_syndrome(
        art["decoding_key"], r["tokens"], r["p_trace"], art["partition"],
        entropy_threshold=ent_thr)) for r in nw)
    return tp, len(wm), fp, len(nw)


def main():
    print("workdir,method,fpr_target,tp,n_wm,fp,n_nw,threshold,null_mean,null_std", flush=True)
    for workdir, methods, fprs in JOBS:
        if not os.path.exists(os.path.join(workdir, "artifacts.pt")):
            print(f"# missing {workdir}", flush=True)
            continue
        art, wm, nw = load(workdir)
        for m in methods:
            if m == "entropy":
                for fpr in fprs:
                    tp, nW, fp, nN, thr, nm, ns = run_fold(art, wm, nw, "entropy", fpr)
                    print(f"{workdir},entropy_fold,{fpr:g},{tp},{nW},{fp},{nN},{thr:.4f},{nm:.4f},{ns:.4f}", flush=True)
            elif m == "naive":
                for fpr in fprs:
                    tp, nW, fp, nN, thr, nm, ns = run_fold(art, wm, nw, "naive", fpr)
                    print(f"{workdir},naive_fold,{fpr:g},{tp},{nW},{fp},{nN},{thr:.4f},{nm:.4f},{ns:.4f}", flush=True)
            elif m == "syn_all":
                tp, nW, fp, nN = run_syn(art, wm, nw, None)
                print(f"{workdir},syndrome_all,analytical,{tp},{nW},{fp},{nN},,,", flush=True)
            elif m == "syn_ent":
                tp, nW, fp, nN = run_syn(art, wm, nw, ENT_THR)
                print(f"{workdir},syndrome_entropy,analytical,{tp},{nW},{fp},{nN},,,", flush=True)


if __name__ == "__main__":
    main()
