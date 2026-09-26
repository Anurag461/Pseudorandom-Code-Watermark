import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from prc_watermark import detectors, generation
from prc_watermark.prc import OnlinePRCKey, derive_document_seed
from prc_watermark.qwen import completion_only_partition_trace_batch, load_model


def run(settings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(settings["artifact"], map_location="cpu", weights_only=False)
    partition = artifact["partition"]
    prompts = [
        json.loads(line)["prompt_tokens"]
        for line in Path(settings["prompts"]).read_text().splitlines()
    ]
    indices = settings["prompt_indices"]
    (model, _) = load_model(settings["model_directory"], settings["model_size"])
    records = []
    if settings.get("sampling_seed") is not None:
        torch.manual_seed(settings["sampling_seed"])
        np.random.seed(settings["sampling_seed"])
    for offset in range(0, len(indices), settings["batch_size"]):
        chosen = indices[offset : offset + settings["batch_size"]]
        inputs = torch.tensor(
            [prompts[i] for i in chosen], device="cuda", dtype=torch.long
        )
        for source in ["watermarked", "unwatermarked"]:
            if settings["construction"] == "online":
                (tokens, _) = generation.generate_batch_and_collect_online(
                    model,
                    inputs,
                    settings["tokens"],
                    OnlinePRCKey.from_dict(artifact["online_key"]),
                    partition,
                    watermark=source == "watermarked",
                    document_seeds=[
                        derive_document_seed(
                            int(artifact.get("experiment_seed", 12345)), i
                        )
                        for i in chosen
                    ],
                    kv_cache_implementation=settings["kv_cache"],
                )
            else:
                (tokens, _) = generation.generate_batch_and_collect(
                    model,
                    inputs,
                    settings["tokens"],
                    artifact["encoding_key"],
                    partition,
                    watermark=source == "watermarked",
                )
            records.extend(
                (
                    {"prompt_index": i, "source": source, "tokens": row.tolist()}
                    for (i, row) in zip(chosen, tokens)
                )
            )
    path = output / "completions.jsonl"
    path.write_text("".join((json.dumps(row) + "\n" for row in records)))
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return str(path)


def score(settings, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(settings["artifact"], map_location="cpu", weights_only=False)
    records = [
        json.loads(line)
        for line in Path(settings["completions"]).read_text().splitlines()
    ]
    (model, _) = load_model(settings["detector_directory"], settings["detector_size"])
    partition = artifact["partition"]
    lengths = settings["lengths"]
    rows = []
    for offset in range(0, len(records), settings["batch_size"]):
        batch = records[offset : offset + settings["batch_size"]]
        tokens = torch.tensor(
            [r["tokens"][: max(lengths)] for r in batch],
            device="cuda",
            dtype=torch.long,
        )
        if tokens.shape[1] != max(lengths):
            raise ValueError("A completion is shorter than the requested prefix")
        probabilities = completion_only_partition_trace_batch(
            model, tokens, partition[1], kv_cache_implementation=settings["kv_cache"]
        )
        if torch.is_tensor(probabilities):
            probabilities = probabilities.detach().cpu().numpy()
        np.savez_compressed(
            output / f"probabilities_{offset:05d}.npz", probabilities=probabilities
        )
        for r, tokens_row, p in zip(batch, tokens.cpu(), probabilities):
            for length in lengths:
                for weight in ["map", "entropy", "naive"]:
                    detector = (
                        detectors.detect_online_hoeffding
                        if settings["construction"] == "online"
                        else detectors.detect_hoeffding
                    )
                    key = (
                        artifact["online_key"]
                        if settings["construction"] == "online"
                        else artifact["decoding_key"]
                    )
                    (detected, info) = detector(
                        key,
                        tokens_row[:length],
                        p[: length - 1],
                        partition,
                        fpr=settings["fpr"],
                        weight=weight,
                        return_info=True,
                    )
                    rows.append(
                        {
                            "prompt_index": r["prompt_index"],
                            "source": r["source"],
                            "tokens": length,
                            "detector": weight,
                            "detected": detected,
                            "statistic": info["statistic"],
                            "threshold": info["threshold"],
                            "V": info["V"],
                        }
                    )
    with (output / "decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    return str(output / "decisions.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["generate", "score"])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    print((run if args.operation == "generate" else score)(settings, args.output))


if __name__ == "__main__":
    main()
