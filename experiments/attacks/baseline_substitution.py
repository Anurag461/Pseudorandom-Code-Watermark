import argparse
import json
from pathlib import Path

import torch

from baselines.kuditipudi import (
    KEY_LENGTH,
    KTH_COMMIT,
    EOS,
    NULL_SEED,
    key_seeds,
    _cpu_kgw_processor,
    synthid_processor,
    Scorer,
    exp_pvalue,
    synthid_pvalue,
    empirical_p,
)
from .substitution import apply_attack


def generate(settings, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_directory"], local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            settings["model_directory"],
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    vocab = model.get_output_embeddings().weight.shape[0]
    prompts = [
        json.loads(line)["prompt_tokens"]
        for line in Path(settings["prompts"]).read_text().splitlines()
    ]
    length = settings["tokens"]
    scheme = settings["scheme"]
    count, batch_size = (500, 25) if length == 400 else (200, 10)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, count, batch_size):
        indices = list(range(start, min(start + batch_size, count)))
        ids = torch.tensor([prompts[i] for i in indices])
        seeds = key_seeds()[indices]
        torch.manual_seed(1000 + start)
        if scheme == "exp":
            from watermarking.generation import generate as exp_generate
            from watermarking.gumbel.key import gumbel_key_func
            from watermarking.gumbel.sampler import gumbel_sampling

            generated = exp_generate(
                model,
                ids,
                vocab,
                KEY_LENGTH,
                length,
                seeds,
                gumbel_key_func,
                gumbel_sampling,
                random_offset=False,
            )
        else:
            processor = (
                _cpu_kgw_processor(list(tokenizer.get_vocab().values()))
                if scheme == "kgw2"
                else synthid_processor(torch.device("cuda"))
            )
            generated = model.generate(
                ids.cuda(),
                attention_mask=torch.ones_like(ids).cuda(),
                do_sample=True,
                max_new_tokens=length,
                min_new_tokens=length,
                top_k=0,
                top_p=1.0,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id if length == 400 else EOS,
                logits_processor=LogitsProcessorList([processor]),
            ).cpu()
        tokens = generated[:, 50 : 50 + length]
        if tokens.shape != (len(indices), length):
            raise ValueError("Unexpected generation length")
        torch.save(
            {
                "scheme": scheme,
                "prompt_idx": indices,
                "seeds": seeds,
                "tokens": tokens,
                "vocab_size": vocab,
                "kth_commit": KTH_COMMIT,
            },
            output / f"{start:04d}.pt",
        )


def reference(settings, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_directory"], local_files_only=True
    )
    scorer = Scorer(settings["scheme"], 151936, tokenizer)
    records = [
        json.loads(line)
        for line in Path(settings["human_reference"]).read_text().splitlines()
    ]
    generator = torch.Generator().manual_seed(NULL_SEED)
    seeds = torch.randint(100000, (4983,), generator=generator)
    rows = []
    for row in records:
        tokens = torch.tensor(tokenizer.encode(row["human_continuation"]))
        if len(tokens) >= 400:
            rows.append(
                {
                    "document_index": row["document_index"],
                    "statistic": scorer(
                        tokens[:400], seeds[row["document_index"]], null=True
                    ),
                }
            )
    Path(output).write_text(json.dumps(rows) + "\n")


def score(settings, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_directory"], local_files_only=True
    )
    scheme = settings["scheme"]
    length = settings["tokens"]
    scorer = Scorer(scheme, 151936, tokenizer)
    reference_values = (
        [
            row["statistic"]
            for row in json.loads(Path(settings["reference"]).read_text())
        ]
        if length == 400
        else None
    )
    nulls = {
        r["prompt_index"]: r["tokens"]
        for r in map(
            json.loads, Path(settings["null_completions"]).read_text().splitlines()
        )
        if r["source"] in ["null", "unwatermarked"]
    }
    rows = []
    for path in sorted(Path(settings["generations"]).glob("*.pt")):
        batch = torch.load(path, map_location="cpu", weights_only=False)
        for row, index in enumerate(batch["prompt_idx"]):
            for source, tokens in [
                ("wm", batch["tokens"][row]),
                ("null", torch.tensor(nulls[index])),
            ]:
                if len(tokens) < length:
                    raise ValueError("Incomplete null or watermarked completion")
                for rate in settings["rates"]:
                    text = tokens[:length].long()
                    if rate:
                        text = apply_attack(
                            text,
                            {
                                "kind": "substitution",
                                "rate": float(rate),
                                "seed": 0,
                                "vocab_size": 151665,
                            },
                            source,
                            index,
                        )
                    if length == 400:
                        statistic = scorer(text, batch["seeds"][row])
                        p = empirical_p(reference_values, statistic)
                    elif scheme == "exp":
                        p, statistic = exp_pvalue(text, batch["seeds"][row])
                    elif scheme == "synthid":
                        p, statistic = synthid_pvalue(scorer.processor, text)
                    else:
                        result = scorer.detector._score_sequence(text)
                        p, statistic = float(result["p_value"]), float(
                            result["z_score"]
                        )
                    rows.append(
                        {
                            "prompt_index": index,
                            "source": source,
                            "rate": rate,
                            "p": p,
                            "statistic": statistic,
                        }
                    )
    Path(output).write_text(json.dumps(rows) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["generate", "reference", "score"])
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if settings["scheme"] not in ["exp", "kgw2", "synthid"] or settings[
        "tokens"
    ] not in [400, 4096]:
        parser.error("Expected EXP, KGW, or SynthID at 400 or 4,096 tokens")
    {"generate": generate, "reference": reference, "score": score}[args.stage](
        settings, args.output
    )


if __name__ == "__main__":
    main()
