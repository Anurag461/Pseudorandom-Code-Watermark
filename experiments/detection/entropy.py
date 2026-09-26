import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from prc_watermark.qwen import load_model, teacher_force_partition_entropy_trace_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    (model, _) = load_model(settings["model_directory"], settings["model_size"])
    artifact = torch.load(settings["artifact"], map_location="cpu", weights_only=False)
    prompts = [
        json.loads(line)["prompt_tokens"]
        for line in Path(settings["prompts"]).read_text().splitlines()
    ]
    records = [
        json.loads(line)
        for line in Path(settings["completions"]).read_text().splitlines()
    ]
    records = [r for r in records if r["source"] == "unwatermarked"]
    token_sum = bucket_sum = below = count = 0
    for offset in range(0, len(records), settings["batch_size"]):
        batch = records[offset : offset + settings["batch_size"]]
        prompt_ids = torch.tensor(
            [prompts[r["prompt_index"]] for r in batch], device="cuda"
        )
        tokens = torch.tensor([r["tokens"][:1808] for r in batch], device="cuda")
        (p, entropy) = teacher_force_partition_entropy_trace_batch(
            model,
            prompt_ids,
            tokens,
            artifact["partition"][1],
            kv_cache_implementation="static",
            chunk_size=1,
        )
        p = p.double().cpu().numpy()
        clipped = np.clip(p, np.finfo(np.float64).tiny, 1.0)
        complement = np.clip(1 - p, np.finfo(np.float64).tiny, 1.0)
        bucket = -(p * np.log2(clipped) + (1 - p) * np.log2(complement))
        bucket[(p == 0) | (p == 1)] = 0
        token_sum += float(entropy.double().sum())
        bucket_sum += float(bucket.sum())
        below += int((bucket < 0.1).sum())
        count += bucket.size
    row = {
        "model": settings["model_size"],
        "tokens": count,
        "token_entropy_bits": token_sum / count,
        "bucket_entropy_bits": bucket_sum / count,
        "bucket_below_0p1": below / count,
    }
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
