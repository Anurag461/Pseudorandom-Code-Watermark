"""
Retrain context-conditional bias tables from the 9966-result extended
campaigns (~10x the original 1000-result Phase A data) and run the
green-rate sanity check against the actual KGW key + PRC partition.

Outputs:
  spoof_workdir/bias_prc_0.6B_extn5k.pt
  spoof_workdir/bias_kgw_0.6B_extn5k.pt
  green_rate_extn5k.json

Run via docker exec: glibc on host is too old for torch.
"""
import json
import os
import time
from glob import glob

import numpy as np
import torch

import attack_steal as atk
from watermark_kgw import kgw_hash_unit


PRC_WORKDIR = "calib_workdir_extn5k_06b"
KGW_WORKDIR = "kgw_workdir_extn5k_06b"
PRC_BASE = "calib_workdir_qwen06b_base" if os.path.isdir("calib_workdir_qwen06b_base") else "calib_workdir_n400_t3_eta05"
KGW_BASE = "kgw_workdir_qwen06b_base"
OUT_DIR = "spoof_workdir"
CONTEXT_SIZE = 2
TOP_K = 64
GAMMA = 0.25


def _load_split(workdir, name_template):
    paths = sorted(glob(os.path.join(workdir, "result_*.pt")))
    wm, nw = [], []
    for p in paths:
        r = torch.load(p, weights_only=False, map_location="cpu")
        if r["job"]["watermark"]:
            wm.append(r)
        else:
            nw.append(r)
    return wm, nw


def green_rate(bias_table, kgw_key, gamma=GAMMA):
    """Fraction of bias-table top-K entries that are KGW-green for ctx[-1]."""
    n_total = 0
    n_green = 0
    for ctx, (idx, val) in bias_table.items():
        if not ctx:
            continue
        prev = int(ctx[-1])
        for cand in idx.tolist():
            u = kgw_hash_unit(prev, int(cand), int(kgw_key))
            n_total += 1
            if u < gamma:
                n_green += 1
    return n_green, n_total


def prc_partition_rate(bias_table, partition_map):
    """Fraction of bias-table top-K entries whose partition bit is 1
    (chance baseline ~50% — PRC partition is balanced)."""
    bit_for_token = partition_map[1].long().cpu().numpy()
    n_total = 0
    n_one = 0
    for ctx, (idx, val) in bias_table.items():
        bits = bit_for_token[idx]
        n_total += len(bits)
        n_one += int(bits.sum())
    return n_one, n_total


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    out_prc = os.path.join(OUT_DIR, "bias_prc_0.6B_extn5k.pt")
    if os.path.isfile(out_prc):
        print(f"[retrain] PRC bias already saved at {out_prc}; loading", flush=True)
        bias_prc = atk.load_bias(out_prc)
        print(f"[retrain] PRC bias has {len(bias_prc)} ctx", flush=True)
    else:
        print(f"[retrain] loading PRC results from {PRC_WORKDIR} ...", flush=True)
        prc_wm, prc_nw = _load_split(PRC_WORKDIR, "result_{:02d}.pt")
        print(f"[retrain] PRC: {len(prc_wm)} wm + {len(prc_nw)} nw  ({time.time()-t0:.0f}s)", flush=True)
        t2 = time.time()
        print("[retrain] training PRC bias ...", flush=True)
        bias_prc = atk.train_bias(prc_wm, prc_nw, context_size=CONTEXT_SIZE, top_k=TOP_K)
        atk.save_bias(bias_prc, out_prc)
        print(f"[retrain] saved PRC bias ({len(bias_prc)} ctx) -> {out_prc} ({time.time()-t2:.0f}s)", flush=True)
        del prc_wm, prc_nw

    out_kgw = os.path.join(OUT_DIR, "bias_kgw_0.6B_extn5k.pt")
    if os.path.isfile(out_kgw):
        print(f"[retrain] KGW bias already saved at {out_kgw}; loading", flush=True)
        bias_kgw = atk.load_bias(out_kgw)
        print(f"[retrain] KGW bias has {len(bias_kgw)} ctx", flush=True)
    else:
        t1 = time.time()
        print(f"[retrain] loading KGW results from {KGW_WORKDIR} ...", flush=True)
        kgw_wm, kgw_nw = _load_split(KGW_WORKDIR, "result_{:04d}.pt")
        print(f"[retrain] KGW: {len(kgw_wm)} wm + {len(kgw_nw)} nw  ({time.time()-t1:.0f}s)", flush=True)
        t3 = time.time()
        print("[retrain] training KGW bias ...", flush=True)
        bias_kgw = atk.train_bias(kgw_wm, kgw_nw, context_size=CONTEXT_SIZE, top_k=TOP_K)
        atk.save_bias(bias_kgw, out_kgw)
        print(f"[retrain] saved KGW bias ({len(bias_kgw)} ctx) -> {out_kgw} ({time.time()-t3:.0f}s)", flush=True)
        del kgw_wm, kgw_nw

    print("[sanity] loading keys/partition from base artifacts ...", flush=True)
    prc_art = torch.load(os.path.join(PRC_BASE, "artifacts.pt"),
                          weights_only=False, map_location="cpu")
    kgw_art = torch.load(os.path.join(KGW_BASE, "artifacts.pt"),
                          weights_only=False, map_location="cpu")
    kgw_key = int(kgw_art["kgw_key"])
    partition = prc_art["partition"]

    sanity = {}
    print("[sanity] KGW bias vs KGW key (expect ~65%) ...", flush=True)
    n_g, n_t = green_rate(bias_kgw, kgw_key)
    sanity["kgw_bias_vs_kgw_key"] = {"n_green": n_g, "n_total": n_t,
                                       "rate": n_g / max(n_t, 1)}
    print(f"   green={n_g}/{n_t} = {n_g/max(n_t,1):.3%}", flush=True)

    print("[sanity] PRC bias vs KGW key (expect ~25% chance) ...", flush=True)
    n_g, n_t = green_rate(bias_prc, kgw_key)
    sanity["prc_bias_vs_kgw_key"] = {"n_green": n_g, "n_total": n_t,
                                       "rate": n_g / max(n_t, 1)}
    print(f"   green={n_g}/{n_t} = {n_g/max(n_t,1):.3%}", flush=True)

    print("[sanity] PRC bias vs PRC partition[1] (expect ~50% chance) ...", flush=True)
    n_o, n_t = prc_partition_rate(bias_prc, partition)
    sanity["prc_bias_vs_prc_partition"] = {"n_one": n_o, "n_total": n_t,
                                             "rate": n_o / max(n_t, 1)}
    print(f"   bit=1: {n_o}/{n_t} = {n_o/max(n_t,1):.3%}", flush=True)

    print("[sanity] KGW bias vs PRC partition[1] (expect ~50% chance) ...", flush=True)
    n_o, n_t = prc_partition_rate(bias_kgw, partition)
    sanity["kgw_bias_vs_prc_partition"] = {"n_one": n_o, "n_total": n_t,
                                             "rate": n_o / max(n_t, 1)}
    print(f"   bit=1: {n_o}/{n_t} = {n_o/max(n_t,1):.3%}", flush=True)

    sanity["meta"] = {
        "prc_results": 9966,
        "kgw_results": 9966,
        "context_size": CONTEXT_SIZE,
        "top_k": TOP_K,
        "gamma": GAMMA,
        "n_prc_contexts": len(bias_prc),
        "n_kgw_contexts": len(bias_kgw),
    }
    with open("green_rate_extn5k.json", "w") as f:
        json.dump(sanity, f, indent=2)
    print(f"[sanity] wrote green_rate_extn5k.json", flush=True)
    print(f"[done] total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
