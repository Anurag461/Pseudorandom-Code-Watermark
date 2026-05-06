"""
Validator/loader for harmful_prompts.jsonl.

We do NOT generate harmful prompts here. The user supplies a benchmark file
externally (e.g., a recent public dataset whose recency mitigates training-data
leakage). This script only validates the schema and reports the category
distribution.

Expected schema per JSONL line: {prompt_id, category, prompt_text}

Usage:
    python build_harmful_prompts.py            # validates ./harmful_prompts.jsonl
    python build_harmful_prompts.py path.jsonl # validates given file
"""
import json
import sys
from collections import Counter


REQUIRED_FIELDS = {"prompt_id", "category", "prompt_text"}


def validate(path: str) -> int:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: invalid JSON ({e})")
                return 1
            missing = REQUIRED_FIELDS - row.keys()
            if missing:
                print(f"line {i}: missing fields {missing}")
                return 1
            rows.append(row)

    n = len(rows)
    cats = Counter(r["category"] for r in rows)
    ids = [r["prompt_id"] for r in rows]
    if len(set(ids)) != n:
        dupes = [i for i, c in Counter(ids).items() if c > 1]
        print(f"duplicate prompt_ids: {dupes[:10]}")
        return 1

    print(f"OK: {n} prompts in {path}")
    print("Categories:")
    for cat, c in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:25s} {c}")
    if n != 100:
        print(f"NOTE: expected 100, got {n}")
    return 0


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "harmful_prompts.jsonl"
    sys.exit(validate(path))
