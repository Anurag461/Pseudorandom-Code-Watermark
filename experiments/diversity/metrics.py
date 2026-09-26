import argparse
import csv
import json
from pathlib import Path
import numpy as np
import math
from typing import Sequence


def ngram_repetition_rate(token_ids: Sequence[int], n: int = 4) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    if not grams:
        return 0.0
    return float(1.0 - len(set(grams)) / len(grams))


def distinct_n(token_ids: Sequence[int], n: int) -> float:
    grams = [tuple(token_ids[i : i + n]) for i in range(max(0, len(token_ids) - n + 1))]
    return float(len(set(grams)) / len(grams)) if grams else 0.0


def quality_metrics(
    token_ids: Sequence[int], base_token_logprobs: Sequence[float]
) -> dict:
    ids = [int(x) for x in token_ids]
    logprobs = np.asarray(base_token_logprobs, dtype=np.float64)
    if len(ids) != logprobs.size:
        raise ValueError("one base-model log-probability is required per token")
    if not np.all(np.isfinite(logprobs)):
        raise ValueError("base-model log-probabilities must be finite")
    mean_nll = float(-logprobs.mean()) if logprobs.size else 0.0
    return {
        "base_model_nll": mean_nll,
        "base_model_perplexity": float(math.exp(mean_nll)),
        "output_length": len(ids),
        "repetition_rate": ngram_repetition_rate(ids, 4),
        "repetition_metric": "repeated token 4-gram fraction: 1 - unique_4grams/total_4grams",
        "distinct_2": distinct_n(ids, 2),
        "distinct_3": distinct_n(ids, 3),
    }


def score_pairs(batches, tokenizer_path):
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    if sacrebleu.__version__ != "2.4.3":
        raise ValueError("Expected SacreBLEU 2.4.3")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    reference = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path), clean_up_tokenization_spaces=False
    )
    metric = BLEU(
        tokenize="13a", smooth_method="exp", effective_order=True, lowercase=False
    )
    pairs = {}
    for batch in batches:
        for row in batch["responses"]:
            key = (row["setting_sha256"], row["prompt_index"])
            pair = pairs.setdefault(key, {})
            if row["response_index"] in pair:
                raise ValueError("Duplicate response identity")
            pair[row["response_index"]] = row
    results = []
    for (setting, prompt), pair in sorted(pairs.items()):
        if set(pair) != {0, 1}:
            raise ValueError("Each prompt requires exactly two responses")
        rows = [pair[i] for i in [0, 1]]
        ids = [row["token_ids"][:1024] for row in rows]
        if any((len(row) != 1024 for row in ids)):
            raise ValueError("Expected 1,024-token responses")
        decoded = [tokenizer.decode(row, skip_special_tokens=True) for row in ids]
        for tokens, text in zip(ids, decoded):
            if (
                reference.decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                != text
            ):
                raise ValueError("Tokenizers disagree on decoded text")
        bleu = (
            0.5
            * (
                metric.sentence_score(decoded[0], [decoded[1]]).score
                + metric.sentence_score(decoded[1], [decoded[0]]).score
            )
            / 100
        )
        quality = [
            quality_metrics(
                tokens, row["generation_diagnostics"]["base_token_logprobs"][:1024]
            )
            for (tokens, row) in zip(ids, rows)
        ]
        results.append(
            {
                "setting": setting,
                "prompt_index": prompt,
                "self_bleu": bleu,
                **{
                    k: float(np.mean([q[k] for q in quality]))
                    for k in ["base_model_nll", "repetition_rate", "distinct_3"]
                },
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batches = [
        json.loads(path.read_text())
        for path in sorted(args.input.glob("response*.json"))
    ]
    rows = score_pairs(batches, args.tokenizer)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
