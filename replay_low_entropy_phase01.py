"""Replay Phase 0/1 detector calibration on local cached ``.pt`` records."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from detectors import prepare_online_map_prefix_context
from low_entropy_replay import replay_cached_online_map_record
from online_prc import OnlinePRCKey


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the current MAP/Hoeffding score from cached records "
            "and rescore the same evidence with a weighted-Rademacher bound."
        )
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--records", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--length", type=int)
    parser.add_argument("--fpr", type=float)
    parser.add_argument(
        "--fpr-policy",
        choices=("one_shot", "alpha_spending_v1"),
        default="one_shot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = torch.load(
        args.artifact, weights_only=False, map_location="cpu"
    )
    replay_length = int(
        artifact["T"] if args.length is None else args.length
    )
    online_key = OnlinePRCKey.from_dict(artifact["online_key"])
    prepared_context = prepare_online_map_prefix_context(
        online_key, replay_length
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "records": 0,
        "hoeffding_positive": 0,
        "weighted_rademacher_positive": 0,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        for record_path in args.records:
            record = torch.load(
                record_path, weights_only=False, map_location="cpu"
            )
            result = replay_cached_online_map_record(
                artifact,
                record,
                length=replay_length,
                false_positive_rate=args.fpr,
                fpr_policy=args.fpr_policy,
                prepared_context=prepared_context,
            )
            result["record_path"] = str(record_path)
            handle.write(json.dumps(_json_safe(result), sort_keys=True) + "\n")
            counts["records"] += 1
            calibrations = result["calibrations"]
            counts["hoeffding_positive"] += int(
                calibrations["hoeffding"]["decision"]
            )
            counts["weighted_rademacher_positive"] += int(
                calibrations["weighted_rademacher_chernoff"]["decision"]
            )
    print(json.dumps({"output": str(args.output), **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
