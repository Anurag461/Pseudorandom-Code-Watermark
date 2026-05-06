"""
Phase E bit extractor. For each result file in a Phase E workdir,
walk the generated tokens, identify those equal to w1_id_set or w2_id_set,
and record (codeword_pos = full_gen_token_index mod n, bit_value) pairs.

Outputs `bits.pt` in the workdir:
  {
    "n": 400,
    "w1": "A", "w2": "B",
    "w1_ids": set, "w2_ids": set,
    "obs": np.ndarray shape (K, n) dtype int8 with values in {-1, 0, 1}
           where -1 means unobserved at that codeword position in that query
  }

This handles K queries that each may overlap codeword positions multiple
times (T > n). When a codeword position is observed multiple times in one
query (e.g., position 50 = generation token 50 AND token 450), we keep
the first observation (oldest, before noise re-randomization).

Usage:
  python3 phase_e_extract_bits.py <workdir>
"""
import os
import sys
from glob import glob

import numpy as np
import torch


N = 400  # PRC codeword length


def get_word_ids(word, model_size="8B"):
    from transformers import AutoTokenizer
    local = f"/home/anurakas/nanochat/Qwen3-{model_size}"
    if os.path.isdir(local):
        tok = AutoTokenizer.from_pretrained(local)
    else:
        tok = AutoTokenizer.from_pretrained(f"Qwen/Qwen3-{model_size}")
    out = set()
    for txt in [word, " " + word, word.upper(), " " + word.upper(),
                word.lower(), " " + word.lower()]:
        ids = tok.encode(txt, add_special_tokens=False)
        if len(ids) == 1:
            out.add(ids[0])
    return out


def main():
    workdir = sys.argv[1]
    art = torch.load(os.path.join(workdir, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    meta = art["phase_e"]
    w1, w2 = meta["w1"], meta["w2"]
    model_size = meta.get("model_size", "8B")
    w1_ids = get_word_ids(w1, model_size=model_size)
    w2_ids = get_word_ids(w2, model_size=model_size)
    print(f"w1='{w1}' ids={w1_ids}  w2='{w2}' ids={w2_ids}", flush=True)

    paths = sorted(glob(os.path.join(workdir, "result_*.pt")))
    K = len(paths)
    print(f"K={K} results in {workdir}", flush=True)

    obs = -np.ones((K, N), dtype=np.int8)
    n_compliant_pos = np.zeros(K, dtype=np.int32)
    n_total_pos = np.zeros(K, dtype=np.int32)

    for k, p in enumerate(paths):
        r = torch.load(p, weights_only=False, map_location="cpu")
        toks = r["tokens"].flatten().cpu().tolist()
        n_total_pos[k] = len(toks)
        for pos, t in enumerate(toks):
            cp = pos % N
            if obs[k, cp] != -1:
                continue  # already filled
            if t in w1_ids:
                obs[k, cp] = 1
                n_compliant_pos[k] += 1
            elif t in w2_ids:
                obs[k, cp] = 0
                n_compliant_pos[k] += 1

    coverage = (obs != -1).mean(axis=0)  # per codeword position
    print(f"per-codeword observation rate: min={coverage.min():.3f} "
          f"mean={coverage.mean():.3f} max={coverage.max():.3f}", flush=True)
    print(f"per-query compliance: mean={n_compliant_pos.mean():.1f}/{N} "
          f"min={n_compliant_pos.min()} max={n_compliant_pos.max()}", flush=True)
    ones_at_observed = []
    for cp in range(N):
        col = obs[:, cp]
        v = col[col != -1]
        if len(v) > 0:
            ones_at_observed.append(v.mean())
    if ones_at_observed:
        ones_at_observed = np.array(ones_at_observed)
        print(f"per-position mean bit (where observed): mean={ones_at_observed.mean():.4f} "
              f"std={ones_at_observed.std():.4f} min={ones_at_observed.min():.3f} "
              f"max={ones_at_observed.max():.3f}", flush=True)

    out = {
        "n": N,
        "w1": w1, "w2": w2,
        "w1_ids": w1_ids, "w2_ids": w2_ids,
        "obs": obs,
        "K": K,
        "n_compliant_pos": n_compliant_pos,
        "n_total_pos": n_total_pos,
    }
    out_path = os.path.join(workdir, "bits.pt")
    torch.save(out, out_path)
    print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
