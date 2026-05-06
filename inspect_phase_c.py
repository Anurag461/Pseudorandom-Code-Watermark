"""
Inspect Phase C outputs: decode tokens, run KGW detection (z-score against
the actual KGW key), classify refusal. Print per-cell summary and sample
generations.
"""
import json
import os
from glob import glob

import numpy as np
import torch

import watermark_kgw as kgwmod
import refusal_classifier as ref


CELLS = [
    ("spoof_workdir/spoof_06b_kgw_atk0_alpha2", "vanilla"),
    ("spoof_workdir/spoof_06b_kgw_atk1_alpha2", "kgw_attack"),
    ("spoof_workdir/spoof_06b_prc_atk1_alpha2", "prc_attack"),
]

KGW_BASE = "kgw_workdir_qwen06b_base"


def load_kgw_key():
    art = torch.load(os.path.join(KGW_BASE, "artifacts.pt"),
                     weights_only=False, map_location="cpu")
    return int(art["kgw_key"]), float(art["gamma"])


def decode_text(tokens):
    """Decode via the HF tokenizer used by watermark_expt (load lazily)."""
    from transformers import AutoTokenizer
    if not hasattr(decode_text, "_tok"):
        decode_text._tok = AutoTokenizer.from_pretrained(
            "/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/"
            + os.listdir("/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots")[0]
            if os.path.isdir("/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B")
            else "Qwen/Qwen3-0.6B"
        )
    if not torch.is_tensor(tokens):
        tokens = torch.as_tensor(tokens)
    return decode_text._tok.decode(tokens.flatten().tolist(), skip_special_tokens=True)


def main():
    kgw_key, gamma = load_kgw_key()
    print(f"KGW key=0x{kgw_key:016x} gamma={gamma}\n", flush=True)

    summary = {}
    samples_to_print = 2

    for path, label in CELLS:
        result_paths = sorted(glob(os.path.join(path, "result_*.pt")))
        n = len(result_paths)
        z_scores = []
        n_green_arr = []
        T_arr = []
        refusals = []
        examples = []
        for rp in result_paths:
            r = torch.load(rp, weights_only=False, map_location="cpu")
            toks = r["tokens"]
            if toks.numel() == 0:
                continue
            text = decode_text(toks)
            is_ref = ref.is_refusal(text)
            refusals.append(is_ref)
            decision, z, ng, T = kgwmod.detect_kgw(toks, kgw_key, gamma=gamma)
            z_scores.append(float(z))
            n_green_arr.append(int(ng))
            T_arr.append(int(T))
            examples.append({
                "category": r.get("prompt_meta", {}).get("category", "?"),
                "z": float(z),
                "n_green": int(ng), "T": int(T),
                "is_refusal": bool(is_ref),
                "text": text[:600],
            })

        z_arr = np.array(z_scores) if z_scores else np.zeros(0)
        ref_rate = float(np.mean(refusals)) if refusals else 0.0
        kgw_detect = float(np.mean(z_arr > 4.0)) if len(z_arr) else 0.0
        summary[label] = {
            "n": n,
            "refusal_rate": ref_rate,
            "kgw_detect_rate(z>4)": kgw_detect,
            "mean_z": float(z_arr.mean()) if len(z_arr) else 0.0,
            "median_z": float(np.median(z_arr)) if len(z_arr) else 0.0,
            "max_z": float(z_arr.max()) if len(z_arr) else 0.0,
            "mean_green_frac": float(np.mean([g/T_ for g,T_ in zip(n_green_arr, T_arr)])) if n_green_arr else 0.0,
        }

        print(f"=== {label} ({path}) ===", flush=True)
        print(json.dumps(summary[label], indent=2), flush=True)
        # Pick one refusal + one non-refusal if available, with the highest z each.
        non_ref = sorted([e for e in examples if not e["is_refusal"]],
                         key=lambda e: -e["z"])[:samples_to_print]
        ref_ex = sorted([e for e in examples if e["is_refusal"]],
                        key=lambda e: -e["z"])[:samples_to_print]
        for tag, lst in [("non-refusal", non_ref), ("refusal", ref_ex)]:
            for e in lst:
                print(f"--- {tag}, cat={e['category']}, z={e['z']:.2f} "
                      f"(n_green={e['n_green']}/{e['T']}, "
                      f"green_frac={e['n_green']/max(e['T'],1):.3f}) ---")
                print(e["text"])
                print()
        print()

    with open("phase_c_inspect.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("=== overall summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
