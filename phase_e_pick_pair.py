"""
Find candidate (word1, word2) pairs for the structured-query attack.

Constraints:
  - both encode to single tokens with the Qwen3 tokenizer
  - word1 token is in partition[1], word2 token is in partition[0]
  - tokens with a leading space (common BPE token form for English words)

Output: prints candidate pairs to phase_e_pair_candidates.json
"""
import json
import os

import torch

PARTITION_PATH = "calib_workdir_n400_t3_eta05/artifacts.pt"

CANDIDATES = [
    ("apple", "banana"),
    ("yes", "no"),
    ("A", "B"),
    ("0", "1"),
    ("cat", "dog"),
    ("up", "down"),
    ("on", "off"),
    ("hot", "cold"),
    ("red", "blue"),
    ("left", "right"),
    ("true", "false"),
    ("good", "bad"),
    ("one", "two"),
    ("alpha", "beta"),
    ("foo", "bar"),
]


def main():
    art = torch.load(PARTITION_PATH, weights_only=False, map_location="cpu")
    partition = art["partition"]  # (2, vocab_size)
    print(f"partition shape: {tuple(partition.shape)}", flush=True)
    print(f"partition[0] sum (zeros bucket size): {int(partition[0].sum().item())}", flush=True)
    print(f"partition[1] sum (ones bucket size): {int(partition[1].sum().item())}", flush=True)

    # Use HF AutoTokenizer for the chat-template path; the codebase uses
    # this for Qwen3 (watermark_expt.py:82-122).
    from transformers import AutoTokenizer
    tok_local = "/home/anurakas/nanochat/Qwen3-8B"
    if os.path.isdir(tok_local):
        tok = AutoTokenizer.from_pretrained(tok_local)
    else:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    print(f"tokenizer.vocab_size: {tok.vocab_size}", flush=True)

    def single_tok_id(word, with_space=True):
        text = (" " + word) if with_space else word
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            return None, ids
        return ids[0], ids

    found = []
    print("\n--- candidate validation ---", flush=True)
    for w1, w2 in CANDIDATES:
        for ws in (True, False):
            id1, list1 = single_tok_id(w1, with_space=ws)
            id2, list2 = single_tok_id(w2, with_space=ws)
            if id1 is None or id2 is None:
                continue
            p1 = int(partition[1, id1].item())  # 1 if in bucket-1
            p2 = int(partition[1, id2].item())
            tag = f"space={ws}"
            row = {
                "w1": w1, "w2": w2, "with_space": ws,
                "id1": id1, "id2": id2,
                "in_bucket1_w1": p1, "in_bucket1_w2": p2,
            }
            print(f"  ({w1!r}, {w2!r}, {tag}): id={id1},{id2}  buckets={p1},{p2}",
                  flush=True)
            if p1 == 1 and p2 == 0:
                row["usable_for_prc"] = True
                found.append(row)
            elif p1 == 0 and p2 == 1:
                row["usable_for_prc_swapped"] = True
                # swap w1/w2 so w1 is bucket-1
                row["swap"] = True
                row["w1"], row["w2"] = w2, w1
                row["id1"], row["id2"] = id2, id1
                row["in_bucket1_w1"], row["in_bucket1_w2"] = p2, p1
                found.append(row)

    print(f"\n--- {len(found)} usable pairs ---", flush=True)
    for r in found:
        print(f"  {r}", flush=True)

    with open("phase_e_pair_candidates.json", "w") as f:
        json.dump(found, f, indent=2)
    print("wrote phase_e_pair_candidates.json", flush=True)


if __name__ == "__main__":
    main()
