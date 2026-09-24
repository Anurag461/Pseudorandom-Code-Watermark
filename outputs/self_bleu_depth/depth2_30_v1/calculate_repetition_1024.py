"""Calculate only the two missing table cells from 100 saved responses.

Standard library only. No model, detector, bootstrap, network, or cloud call.
Local execution requires an explicit exception to the repository's scoring rule.
"""

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/self_bleu_depth/depth2_30_v1/raw/batches"
OUTPUT = Path(__file__).resolve().parent / "repetition_1024.json"


def main():
    rows = []
    sources = []
    for response_index in (0, 1):
        path = SOURCE / f"depth30_r{response_index}.json"
        content = path.read_bytes()
        batch = json.loads(content)
        manifest = batch["manifest"]
        assert manifest["model_id"] == "Qwen/Qwen3-8B-Base"
        assert manifest["setting"]["method"] == "synthid_text"
        assert manifest["setting"]["depth"] == 30
        assert manifest["generation"]["temperature"] == 1
        assert manifest["generation"]["top_p"] == 1
        assert manifest["sampling_seed"] == (12345, 67890)[response_index]
        assert len(batch["responses"]) == 50
        assert sorted(r["prompt_index"] for r in batch["responses"]) == list(range(50))
        sources.append({"path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(content).hexdigest()})
        for response in batch["responses"]:
            ids = response["token_ids"]
            assert len(ids) == 1024
            assert response["response_index"] == response_index
            unique4 = len({tuple(ids[i:i + 4]) for i in range(len(ids) - 3)})
            unique3 = len({tuple(ids[i:i + 3]) for i in range(len(ids) - 2)})
            rows.append({
                "prompt_index": response["prompt_index"],
                "response_index": response_index,
                "response_id": response["response_id"],
                "completion_sha256": response["completion_sha256"],
                "unique_4grams": unique4,
                "unique_3grams": unique3,
                "repeated_4gram_fraction": 1 - unique4 / 1021,
                "distinct_3": unique3 / 1022,
            })
    assert len(rows) == 100
    means = {key: math.fsum(row[key] for row in rows) / len(rows)
             for key in ("repeated_4gram_fraction", "distinct_3")}
    result = {
        "setting": "synthid_depth30", "tokens": 1024,
        "prompts": 50, "responses": 100, "sources": sources,
        "definitions": {
            "repeated_4gram_fraction": "1 - unique contiguous token 4-grams / 1021",
            "distinct_3": "unique contiguous token 3-grams / 1022",
            "aggregation": "Arithmetic mean across the same 100 saved responses",
        },
        "means": means,
        "table_cells": {
            "repeated_4grams_percent": f"{100 * means['repeated_4gram_fraction']:.2f}",
            "distinct_3": f"{means['distinct_3']:.4f}",
        },
        "response_metrics": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "means": means,
                      "table_cells": result["table_cells"]}, indent=2))


if __name__ == "__main__":
    main()
