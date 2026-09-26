import argparse
import json
from pathlib import Path
import numpy as np
import torch
from prc_watermark.generation import generate_batch_padded_and_collect, left_pad_batch
from prc_watermark.prc import KeyGen
from prc_watermark.qwen import load_model
from .tasks.registry import get_benchmark


def evaluate(
    benchmark,
    model,
    tokenizer,
    partition,
    batch_size,
    max_new_tokens,
    watermark,
    indices,
):
    (encoding_key, decoding_key) = KeyGen(
        n=800,
        message_length=0,
        false_positive_rate=0.5,
        t=3,
        noise_rate=0.1,
        r=int(0.99 * 800),
    )
    rows = []
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        conversations = [benchmark.get_example(i) for i in chosen]
        (ids, padding) = left_pad_batch(
            [tokenizer.encode(c["messages"][0]["content"]) for c in conversations],
            tokenizer.pad_token_id,
        )
        (tokens, _) = generate_batch_padded_and_collect(
            model,
            ids,
            padding,
            max_new_tokens,
            encoding_key,
            partition,
            eos_token_id=tokenizer.eos_token_id,
            watermark=watermark,
        )
        for index, token_row in zip(chosen, tokens.tolist()):
            length = (
                token_row.index(tokenizer.eos_token_id)
                if tokenizer.eos_token_id in token_row
                else len(token_row)
            )
            text = tokenizer.decode(token_row[:length])
            rows.append(
                {
                    "index": index,
                    "tokens": token_row[:length],
                    "response": text,
                    "score": benchmark.evaluate(index, text),
                    "truncated": length == max_new_tokens,
                }
            )
    return (
        rows,
        {
            "encoding_key": encoding_key,
            "decoding_key": decoding_key,
            "partition": partition.cpu(),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["arc_easy", "gsm8k", "hellaswag", "mmlu", "ifeval"],
        required=True,
    )
    parser.add_argument("--model-directory", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    settings = json.loads(Path(__file__).with_name("settings.json").read_text())
    (model, tokenizer) = load_model(args.model_directory, "0.6B", "reasoning")
    partition = torch.load(args.artifact, map_location="cpu", weights_only=False)[
        "partition"
    ]
    benchmark = get_benchmark(args.benchmark)
    args.output.mkdir(parents=True, exist_ok=True)
    limit = settings["benchmarks"][args.benchmark]["max_new_tokens"]
    summary = []
    for run in range(settings["benchmarks"][args.benchmark]["runs"]):
        for watermarked in [False, True]:
            torch.manual_seed(args.seed + run)
            np.random.seed(args.seed + run)
            (rows, artifact) = evaluate(
                benchmark,
                model,
                tokenizer,
                partition,
                args.batch_size,
                limit,
                watermarked,
                list(range(benchmark.num_examples())),
            )
            name = f"run{run}_{('watermarked' if watermarked else 'unwatermarked')}"
            (args.output / f"{name}.jsonl").write_text(
                "".join((json.dumps(row) + "\n" for row in rows))
            )
            torch.save(artifact, args.output / f"{name}_key.pt")
            summary.append(
                {
                    "run": run,
                    "watermarked": watermarked,
                    "examples": len(rows),
                    "score": float(np.mean([r["score"] for r in rows])),
                }
            )
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
