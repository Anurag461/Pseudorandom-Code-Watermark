"""
Extract evaluation prompts from the C4 RealNewsLike dataset,
following the evaluation protocol established by:

  - Kuditipudi et al. (2023), "Robust Distortion-free Watermarks for Language Models"
    * C4 RealNewsLike validation split
    * 50-token prompts, 200-token completions
    * Models: OPT-1.3B, LLaMA-7B, Alpaca-7B
    * Temperature 1, standard sampling

  - Kirchenbauer et al. (2023), "A Watermark for Large Language Models"
    * C4 RealNewsLike validation split
    * Similar prompt/completion setup

Adapted here for:
  - Tokenizer: Qwen/Qwen3-0.6B
  - Completion length: 512 tokens

Usage:
    python get_prompts.py --num_prompts 500 --save prompts.jsonl

The script saves prompts as JSONL with fields:
    - prompt_text: the raw text prefix
    - prompt_tokens: token IDs for the prompt
    - human_continuation: the next 512 tokens from the original document
    - full_text: the original document text (for human-written baseline comparisons)
    - doc_index: index into the C4 validation set
"""

import argparse
import json
import sys


def get_prompts(
    tokenizer_name: str = "Qwen/Qwen3-0.6B",
    num_prompts: int = 500,
    prompt_length: int = 50,
    min_doc_tokens: int = 562,
    save_path: str = None,
):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print("Loading C4 RealNewsLike validation split (streaming)...")
    dataset = load_dataset(
        "allenai/c4",
        "realnewslike",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )

    prompts = []
    skipped = 0

    for doc_idx, example in enumerate(dataset):
        if len(prompts) >= num_prompts:
            break

        text = example["text"]
        tokens = tokenizer.encode(text)

        # Skip documents that are too short to provide both a prompt
        # and a meaningful completion window for comparison
        if len(tokens) < min_doc_tokens:
            skipped += 1
            continue

        prompt_tokens = tokens[:prompt_length]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)

        # Also store the human continuation for baseline comparisons
        # (Kirchenbauer uses this for perplexity / quality evaluation)
        human_continuation_tokens = tokens[prompt_length : prompt_length + 512]
        human_continuation = tokenizer.decode(
            human_continuation_tokens, skip_special_tokens=True
        )

        entry = {
            "prompt_text": prompt_text,
            "prompt_tokens": prompt_tokens,
            "human_continuation": human_continuation,
            "full_text": text[:2000],  # cap storage
            "doc_index": doc_idx,
            "num_doc_tokens": len(tokens),
        }
        prompts.append(entry)

        if len(prompts) % 100 == 0:
            print(f"  collected {len(prompts)}/{num_prompts} prompts "
                  f"(scanned {doc_idx + 1} docs, skipped {skipped})")

    print(f"\nDone: {len(prompts)} prompts from {doc_idx + 1} documents "
          f"({skipped} skipped for being < {min_doc_tokens} tokens)")

    if save_path:
        with open(save_path, "w") as f:
            for entry in prompts:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved to {save_path}")

    return prompts


def print_stats(prompts, tokenizer_name):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompt_lens = [len(p["prompt_tokens"]) for p in prompts]
    doc_lens = [p["num_doc_tokens"] for p in prompts]

    print(f"\n--- Prompt Statistics ---")
    print(f"  Number of prompts: {len(prompts)}")
    print(f"  Prompt length (tokens): {prompt_lens[0]} (fixed)")
    print(f"  Document lengths: min={min(doc_lens)}, "
          f"median={sorted(doc_lens)[len(doc_lens)//2]}, "
          f"max={max(doc_lens)}")
    print(f"  Tokenizer: {tokenizer_name}")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Show a sample prompt
    print(f"\n--- Sample Prompt (index 0) ---")
    print(f"  {prompts[0]['prompt_text'][:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract C4 RealNewsLike evaluation prompts"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace tokenizer to use for tokenization "
             "(default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=500,
        help="Number of prompts to extract (default: 500)",
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=50,
        help="Number of tokens per prompt (default: 50, matching Kuditipudi et al.)",
    )
    parser.add_argument(
        "--min_doc_tokens",
        type=int,
        default=562,
        help="Minimum document length in tokens to include "
             "(default: 562, ensures room for prompt + 512-token completion)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="prompts.jsonl",
        help="Path to save prompts as JSONL (default: prompts.jsonl)",
    )

    args = parser.parse_args()

    prompts = get_prompts(
        tokenizer_name=args.tokenizer,
        num_prompts=args.num_prompts,
        prompt_length=args.prompt_length,
        min_doc_tokens=args.min_doc_tokens,
        save_path=args.save,
    )

    print_stats(prompts, args.tokenizer)
