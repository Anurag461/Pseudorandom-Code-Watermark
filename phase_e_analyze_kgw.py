"""
Phase E KGW pair classification.

For the (w1, w2) pair, the green-status of each (prev, cand) pair in
{w1, w2}^2 is determined by kgw_hash_unit(prev, cand, key) < gamma.

The attacker doesn't know the key, but observes generated bit strings.
For each consecutive (prev_token, cand_token) pair, count empirical
P[cand picked | prev = prev_token]. Compare to ground truth.

Computes recovery curve as a function of k (number of queries used).
"""
import json
import os
import sys
from glob import glob

import numpy as np
import torch

from watermark_kgw import kgw_hash_unit


KGW_BASE = "kgw_workdir_qwen06b_base"
K_SWEEP = [16, 32, 64, 128, 256, 512, 1024, 2048]


def main():
    workdir = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(workdir, "kgw_recovery.json")

    bits = torch.load(os.path.join(workdir, "bits.pt"),
                      weights_only=False, map_location="cpu")
    art = torch.load(os.path.join(workdir, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    meta = art["phase_e"]
    w1 = meta["w1"]
    w2 = meta["w2"]
    w1_ids = bits["w1_ids"]  # set
    w2_ids = bits["w2_ids"]

    # Use the union of all actual token IDs emitted (no canonicalization).
    # Each specific (prev_id, cand_id) has its own KGW green-status.
    binary_ids = w1_ids | w2_ids
    print(f"binary_ids: {sorted(binary_ids)}", flush=True)

    # Load KGW key
    kart = torch.load(os.path.join(KGW_BASE, "artifacts.pt"),
                      weights_only=False, map_location="cpu")
    kgw_key = int(kart["kgw_key"])
    gamma = float(kart["gamma"])

    # Walk each result, collect (prev_id, cand_id) sequences
    paths = sorted(glob(os.path.join(workdir, "result_*.pt")))
    K = len(paths)
    print(f"K={K} results", flush=True)
    per_query_pairs = []
    for p in paths:
        r = torch.load(p, weights_only=False, map_location="cpu")
        toks = r["tokens"].flatten().cpu().tolist()
        pairs = []
        for i in range(1, len(toks)):
            if toks[i] in binary_ids and toks[i-1] in binary_ids:
                pairs.append((toks[i-1], toks[i]))
        per_query_pairs.append(pairs)

    # Tally observed (prev, cand) pairs across all queries
    from collections import Counter
    all_pairs = Counter()
    for q in range(K):
        for pair in per_query_pairs[q]:
            all_pairs[pair] += 1
    # Keep only pairs that occur frequently enough to classify
    common_pairs = sorted(
        [p for p, c in all_pairs.items() if c >= 32],
        key=lambda p: -all_pairs[p],
    )
    print(f"common observed pairs (count >= 32, total {len(common_pairs)}):", flush=True)
    for p in common_pairs[:20]:
        print(f"  {p}: {all_pairs[p]}", flush=True)

    # Ground truth: green status of each observed (prev, cand) pair under
    # the actual KGW key.
    truth = {}
    for prev, cand in common_pairs:
        u = kgw_hash_unit(prev, cand, kgw_key)
        truth[(prev, cand)] = (u < gamma)
    n_green_of_observed = sum(1 for v in truth.values() if v)
    print(f"truth: {n_green_of_observed}/{len(truth)} observed pairs are green", flush=True)
    pair_labels = common_pairs

    # Group pairs by prev — for each prev, classify its candidates' green status
    # based on which one is more frequently chosen. Compare to truth.
    from collections import defaultdict
    by_prev = defaultdict(list)
    for prev, cand in pair_labels:
        by_prev[prev].append(cand)

    rows = []
    for k_sub in K_SWEEP:
        if k_sub > K:
            break
        cnt = {p: 0 for p in pair_labels}
        for q in range(k_sub):
            for pair in per_query_pairs[q]:
                if pair in cnt:
                    cnt[pair] += 1

        # For each prev, find the candidate with the higher relative
        # frequency. Predict GREEN for the favored candidate (and possibly
        # for the other candidate too if both rates exceed gamma=0.25).
        cls = {}
        for prev, cands in by_prev.items():
            counts = [cnt[(prev, c)] for c in cands]
            tot = sum(counts)
            if tot < 16:
                for c in cands:
                    cls[(prev, c)] = None
                continue
            rates = [cc / tot for cc in counts]
            for c, rate in zip(cands, rates):
                # Without delta-boost, each candidate's rate would be ~0.5
                # of the conditional. With +delta on green tokens, green
                # candidates get a substantially higher rate. Threshold at
                # 0.55 (chosen because under unif the SE at k_sub queries is
                # large; but the overall classification rule is "the
                # candidate with the higher rate is the green one").
                cls[(prev, c)] = (rate > 0.55)

        n_correct = 0
        n_classified = 0
        n_green_correct = 0
        n_green_total = 0
        n_red_correct = 0
        n_red_total = 0
        for pair in pair_labels:
            pred = cls[pair]
            actual = truth[pair]
            if pred is None:
                continue
            n_classified += 1
            if pred == actual:
                n_correct += 1
            if actual:
                n_green_total += 1
                if pred:
                    n_green_correct += 1
            else:
                n_red_total += 1
                if not pred:
                    n_red_correct += 1

        row = {
            "k": k_sub,
            "n_correct": n_correct,
            "n_classified": n_classified,
            "n_pairs_total": len(pair_labels),
            "accuracy": n_correct / max(n_classified, 1),
            "n_green_correct": n_green_correct,
            "n_green_total": n_green_total,
            "n_red_correct": n_red_correct,
            "n_red_total": n_red_total,
            "counts": {f"{p}_{c}": cnt[(p, c)] for (p, c) in pair_labels[:20]},
        }
        rows.append(row)
        print(f"k={k_sub}: classified={n_classified}/{len(pair_labels)} "
              f"acc={row['accuracy']:.3f} ({n_correct}/{n_classified}) "
              f"green_recall={n_green_correct}/{n_green_total} "
              f"red_recall={n_red_correct}/{n_red_total}",
              flush=True)

    out = {
        "K_sweep": K_SWEEP[:len(rows)],
        "n_observed_pairs": len(pair_labels),
        "n_green_pairs": n_green_of_observed,
        "truth": {f"{p}_{c}": v for (p, c), v in truth.items()},
        "sweep": rows,
        "kgw_key": f"0x{kgw_key:016x}",
        "gamma": gamma,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {out_json}", flush=True)


if __name__ == "__main__":
    main()
