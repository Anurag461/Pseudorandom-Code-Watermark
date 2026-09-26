import argparse
import json
from pathlib import Path
import torch
from .substitution import apply_attack


def substitute(input_path, output_path, rate, seed):
    rows = [json.loads(line) for line in Path(input_path).read_text().splitlines()]
    attack = {"kind": "substitution", "rate": rate, "seed": seed, "vocab_size": 151665}
    with Path(output_path).open("w") as handle:
        for row in rows:
            tokens = apply_attack(
                torch.tensor(row["tokens"], dtype=torch.long),
                attack,
                {"watermarked": "wm", "unwatermarked": "null"}.get(
                    row["source"], row["source"]
                ),
                row["prompt_index"],
            )
            handle.write(
                json.dumps({**row, "tokens": tokens.tolist(), "attack": attack}) + "\n"
            )


def blackbox(settings, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prc_watermark.qwen import load_model
    from .blackbox import (
        Generator,
        CHAT_MODEL,
        CHAT_REVISION,
        PREFIXES,
        WORD_LISTS,
        RG_MAX_NEW,
        RG_FIRST,
        RG_TOPUP,
        RG_ROUNDS,
        RG_VALID,
        FS_N,
        FS_PROMPT,
        FS_MAX_NEW,
        rg_prompt,
        identify_fruit,
        chat_ids,
    )
    import hashlib

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scheme = settings["scheme"]
    tokenizer = AutoTokenizer.from_pretrained(
        CHAT_MODEL, revision=CHAT_REVISION, local_files_only=True
    )
    artifact = None
    if scheme == "prc":
        (model, _) = load_model(settings["model_directory"], "0.6B", "instruct")
        artifact = torch.load(
            settings["artifact"], map_location="cpu", weights_only=False
        )
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(
                settings["model_directory"],
                torch_dtype=torch.float32,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
    generate = Generator(scheme, model, tokenizer, artifact)
    (fruits, example) = WORD_LISTS["apples"]
    for h in [4, 5]:
        counts = []
        for prefix in PREFIXES:
            cells = []
            for digit in range(1, 10):
                ids = chat_ids(tokenizer, rg_prompt(prefix, digit, h, fruits, example))
                cell = [0] * len(fruits)
                for attempt in range(RG_ROUNDS):
                    n = RG_FIRST if attempt == 0 else RG_TOPUP
                    label = f"rg|{scheme}|{h}|{prefix}|{digit}|0|{attempt}"
                    for tokens in generate(ids, n, RG_MAX_NEW, label):
                        choice = identify_fruit(
                            tokenizer.decode(tokens, skip_special_tokens=True), fruits
                        )
                        if choice is not None and sum(cell) < RG_VALID:
                            cell[choice] += 1
                    if sum(cell) >= RG_VALID:
                        break
                if sum(cell) != RG_VALID:
                    raise ValueError(
                        "Insufficient parsed responses for a red-green cell"
                    )
                cells.append(cell)
            counts.append(cells)
        (output / f"red_green_h{h}.json").write_text(
            json.dumps({"scheme": scheme, "h": h, "counts": counts}) + "\n"
        )
    tokens = generate(
        chat_ids(tokenizer, FS_PROMPT), FS_N, FS_MAX_NEW, f"fs|{scheme}|0"
    )
    digests = [hashlib.sha256(json.dumps(t).encode()).hexdigest()[:16] for t in tokens]
    (output / "fixed_sampling.json").write_text(
        json.dumps({"scheme": scheme, "digests": digests}) + "\n"
    )


def analyze_blackbox(directory):
    from .blackbox import rg_pvalue, fs_pvalue

    directory = Path(directory)
    rows = []
    for h in [4, 5]:
        data = json.loads((directory / f"red_green_h{h}.json").read_text())
        (p, statistic, share) = rg_pvalue(data["counts"])
        rows.append(
            {
                "scheme": data["scheme"],
                "test": "red-green",
                "h": h,
                "p": p,
                "statistic": statistic,
            }
        )
    rows.append(
        {
            "scheme": rows[0]["scheme"],
            "test": "red-green Bonferroni H=4,5",
            "p": min(1.0, 2 * min((r["p"] for r in rows))),
        }
    )
    data = json.loads((directory / "fixed_sampling.json").read_text())
    (p, unique) = fs_pvalue(data["digests"])
    rows.append(
        {"scheme": data["scheme"], "test": "fixed-sampling", "p": p, "unique": unique}
    )
    (directory / "results.json").write_text(json.dumps(rows, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    edit = sub.add_parser("substitute")
    edit.add_argument("--input", type=Path, required=True)
    edit.add_argument("--output", type=Path, required=True)
    edit.add_argument("--rate", type=float, required=True)
    edit.add_argument("--seed", type=int, default=0)
    gen = sub.add_parser("blackbox")
    gen.add_argument("--settings", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    analyze = sub.add_parser("analyze-blackbox")
    analyze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "substitute":
        substitute(args.input, args.output, args.rate, args.seed)
    elif args.operation == "blackbox":
        blackbox(json.loads(args.settings.read_text()), args.output)
    else:
        analyze_blackbox(args.output)


if __name__ == "__main__":
    main()
