"""
Decode the sanity-gate generations and check binary-string compliance.
For each result, print:
 - first 100 decoded chars
 - bit-string fraction (tokens equal to w1_id_set or w2_id_set / total)
 - first 50 bits as a debug string
"""
import json
import os
import sys
from glob import glob

import torch


WORKDIR = sys.argv[1] if len(sys.argv) > 1 else "phase_e_workdir/sanity"


def main():
    art = torch.load(os.path.join(WORKDIR, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    meta = art["phase_e"]
    w1, w2 = meta["w1"], meta["w2"]
    print(f"phase_e meta: {meta}", flush=True)

    # Tokenize w1 and w2 to find ALL plausible single-token IDs (with/without space).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/home/anurakas/nanochat/Qwen3-8B")
    def get_ids(word):
        out = set()
        for txt in [word, " " + word, word.upper(), " " + word.upper(),
                    word.lower(), " " + word.lower()]:
            ids = tok.encode(txt, add_special_tokens=False)
            if len(ids) == 1:
                out.add(ids[0])
        return out
    w1_ids = get_ids(w1)
    w2_ids = get_ids(w2)
    print(f"w1='{w1}' single-token ids: {w1_ids}", flush=True)
    print(f"w2='{w2}' single-token ids: {w2_ids}", flush=True)

    paths = sorted(glob(os.path.join(WORKDIR, "result_*.pt")))
    print(f"\n{len(paths)} result files\n", flush=True)

    total_compliant = 0
    total_long = 0
    for p in paths:
        r = torch.load(p, weights_only=False, map_location="cpu")
        toks = r["tokens"].flatten().tolist()
        text = tok.decode(toks, skip_special_tokens=True)
        bits = []
        for t in toks:
            if t in w1_ids:
                bits.append(1)
            elif t in w2_ids:
                bits.append(0)
            else:
                bits.append(-1)
        n_total = len(bits)
        n_compliant = sum(1 for b in bits if b != -1)
        compliance = n_compliant / max(n_total, 1)
        bit_str = "".join("1" if b == 1 else "0" if b == 0 else "x" for b in bits[:80])
        # Find longest consecutive binary run (the "good" portion)
        max_run = 0
        cur = 0
        for b in bits:
            if b != -1:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        ones_frac = sum(1 for b in bits if b == 1) / max(n_compliant, 1)
        print(f"--- {os.path.basename(p)}  n_tokens={n_total} compliance={compliance:.3f} "
              f"max_binary_run={max_run} ones_frac={ones_frac:.3f}", flush=True)
        print(f"  first80: {bit_str}", flush=True)
        print(f"  text[:120]: {text[:120]!r}", flush=True)
        print(f"  text[-120:]: {text[-120:]!r}", flush=True)
        if compliance > 0.7:
            total_compliant += 1
        if max_run >= 400:
            total_long += 1

    print(f"\n=== compliance summary ===")
    print(f"  >70% binary tokens: {total_compliant}/{len(paths)}")
    print(f"  >=400 consecutive binary tokens: {total_long}/{len(paths)}")


if __name__ == "__main__":
    main()
