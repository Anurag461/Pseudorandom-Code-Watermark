"""
For each candidate (w1, w2) pair from phase_e_pair_candidates.json,
compute the KGW green-status of all 4 (prev, cand) pairs in {w1_id, w2_id}^2
under the actual KGW key. Print which pairs have non-trivial signal
(at least 1 green among the 4, ideally 2).
"""
import json
import os
import torch
from watermark_kgw import kgw_hash_unit


def main():
    art = torch.load("kgw_workdir_qwen06b_base/artifacts.pt",
                      weights_only=False, map_location="cpu")
    key = int(art["kgw_key"])
    gamma = float(art["gamma"])
    print(f"KGW key=0x{key:016x}  gamma={gamma}\n", flush=True)

    candidates = json.load(open("phase_e_pair_candidates.json"))
    rows = []
    for c in candidates:
        id1 = c["id1"]
        id2 = c["id2"]
        statuses = {}
        n_green = 0
        for prev in (id1, id2):
            for cand in (id1, id2):
                u = kgw_hash_unit(prev, cand, key)
                green = (u < gamma)
                statuses[(prev, cand)] = green
                if green:
                    n_green += 1
        rows.append({
            "w1": c["w1"], "w2": c["w2"], "with_space": c["with_space"],
            "id1": id1, "id2": id2,
            "n_green_of_4": n_green,
            "statuses": {f"{p},{ca}": bool(g) for (p, ca), g in statuses.items()},
        })
        print(f"  {c['w1']!r}/{c['w2']!r} space={c['with_space']} ids={id1},{id2} "
              f"n_green/4={n_green}", flush=True)

    rows.sort(key=lambda r: -r["n_green_of_4"])
    print("\n--- ranked ---", flush=True)
    for r in rows:
        print(f"  green/4={r['n_green_of_4']}  {r['w1']}/{r['w2']} "
              f"space={r['with_space']} ids={r['id1']},{r['id2']}", flush=True)

    with open("phase_e_pair_kgw_status.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
