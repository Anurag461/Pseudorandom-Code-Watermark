"""
Entropy-threshold sweep for the syndrome detector.

Loads artifacts.pt + result_NN.pt from a workdir and runs detect_syndrome
at several entropy thresholds. Reports TPR / FPR per threshold and per-block
fire-rate stats so we can see how aggressive filtering trades off r_eff vs.
detection power.

Usage:
    python sweep_syndrome_entropy.py [WORKDIR]
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from watermark_expt import detect_syndrome

WORKDIR = sys.argv[1] if len(sys.argv) > 1 else "calib_workdir"
THRESHOLDS = [None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9]


def label(thr):
    return "all" if thr is None else f"{thr:.1f}"


def main():
    art_path = os.path.join(WORKDIR, "artifacts.pt")
    art = torch.load(art_path, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n_jobs = len(art["jobs"])
    n = art["n"]
    print(f"workdir={WORKDIR}  n={n}  jobs={n_jobs}", flush=True)

    results = []
    for j in range(n_jobs):
        results.append(torch.load(
            os.path.join(WORKDIR, f"result_{j:02d}.pt"),
            weights_only=False, map_location="cpu",
        ))
    watermarked = [r for r in results if r["job"]["watermark"]]
    unwatermarked = [r for r in results if not r["job"]["watermark"]]
    n_w, n_u = len(watermarked), len(unwatermarked)
    T_per_job = watermarked[0]["tokens"].numel() if watermarked else 0
    n_blocks_per_job = T_per_job // n
    print(f"watermarked={n_w}  unwatermarked={n_u}  "
          f"tokens/job={T_per_job}  blocks/job={n_blocks_per_job}\n",
          flush=True)

    print(f"{'thr':>6} | {'TPR':>14} | {'FPR':>14} | "
          f"{'wm blocks fired':>17} | {'null blocks fired':>18} | "
          f"{'wm avg r_eff':>13} | {'null avg r_eff':>15}", flush=True)
    print("-" * 130, flush=True)

    for thr in THRESHOLDS:
        tp = 0
        wm_block_fires = 0
        wm_block_total = 0
        wm_reff_sum = 0
        for r in watermarked:
            decision, info = detect_syndrome(
                decoding_key, r["tokens"], r["p_trace"], partition,
                entropy_threshold=thr, return_info=True,
            )
            tp += int(decision)
            wm_block_fires += info["blocks_passed"]
            wm_block_total += info["n_blocks"]
            wm_reff_sum += sum(b["r_eff"] for b in info["blocks"])

        fp = 0
        null_block_fires = 0
        null_block_total = 0
        null_reff_sum = 0
        for r in unwatermarked:
            decision, info = detect_syndrome(
                decoding_key, r["tokens"], r["p_trace"], partition,
                entropy_threshold=thr, return_info=True,
            )
            fp += int(decision)
            null_block_fires += info["blocks_passed"]
            null_block_total += info["n_blocks"]
            null_reff_sum += sum(b["r_eff"] for b in info["blocks"])

        wm_reff_avg = wm_reff_sum / max(wm_block_total, 1)
        null_reff_avg = null_reff_sum / max(null_block_total, 1)
        print(
            f"{label(thr):>6} | "
            f"{tp:>3}/{n_w} ({tp/n_w:>5.1%}) | "
            f"{fp:>3}/{n_u} ({fp/n_u:>5.1%}) | "
            f"{wm_block_fires:>4}/{wm_block_total:>4} ({wm_block_fires/max(wm_block_total,1):>5.1%}) | "
            f"{null_block_fires:>4}/{null_block_total:>4} ({null_block_fires/max(null_block_total,1):>6.1%}) | "
            f"{wm_reff_avg:>13.1f} | {null_reff_avg:>15.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
