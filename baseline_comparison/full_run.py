"""Local streaming finalization for batch validation and the later full run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .config import PREFIX_LENGTHS
from .schema import PromptLevelResult


METHODS = ("online_prc", "textseal", "synthid_text", "gumbel_max")
SAMPLE_TYPES = ("watermarked", "null")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finalize(
    scored_dir: Path,
    output_dir: Path,
    *,
    expected_shard_indices: tuple[int, ...],
    expected_prompt_indices: tuple[int, ...],
    artifact_stem: str,
) -> dict:
    shard_paths = sorted(scored_dir.glob("shard_[0-9][0-9].jsonl"))
    expected_shard_names = [
        f"shard_{index:02d}.jsonl" for index in expected_shard_indices
    ]
    if [path.name for path in shard_paths] != expected_shard_names:
        raise FileNotFoundError(
            f"expected exactly these scored shards: {expected_shard_names}"
        )
    prompt_count = len(expected_prompt_indices)
    expected_rows = prompt_count * len(METHODS) * len(SAMPLE_TYPES) * len(PREFIX_LENGTHS)
    output_dir.mkdir(parents=True, exist_ok=False)
    prompt_path = output_dir / f"{artifact_stem}_prompt_level.jsonl"
    groups: dict[tuple[str, int, str], list[dict]] = {}
    quality: dict[tuple, dict] = {}
    prompt_coverage: dict[tuple[str, str], set[int]] = {}
    unique_rows = set()
    row_count = 0
    with prompt_path.open("w") as destination:
        for shard_path in shard_paths:
            with shard_path.open() as source:
                for line in source:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    PromptLevelResult.from_dict(record)
                    identity = (
                        record["method"],
                        record["sample_type"],
                        int(record["prompt_index"]),
                        int(record["generation_seed"]),
                        int(record["prefix_length"]),
                    )
                    if identity in unique_rows:
                        raise AssertionError(f"duplicate full-run row: {identity}")
                    unique_rows.add(identity)
                    destination.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                    row_count += 1
                    groups.setdefault(
                        (record["method"], int(record["prefix_length"]), record["sample_type"]),
                        [],
                    ).append(record)
                    prompt_coverage.setdefault(
                        (record["method"], record["sample_type"]), set()
                    ).add(int(record["prompt_index"]))
                    quality_key = (
                        record["method"],
                        record["sample_type"],
                        int(record["prompt_index"]),
                        int(record["generation_seed"]),
                        record["generated_token_hash"],
                    )
                    quality.setdefault(
                        quality_key,
                        {
                            field: record[field]
                            for field in (
                                "method",
                                "sample_type",
                                "prompt_index",
                                "prompt_id",
                                "generation_seed",
                                "generated_token_hash",
                                "output_length",
                                "base_model_nll",
                                "base_model_perplexity",
                                "repetition_rate",
                                "repetition_metric",
                                "distinct_2",
                                "distinct_3",
                                "runtime_seconds",
                            )
                        },
                    )
    if row_count != expected_rows:
        raise AssertionError(f"evaluation has {row_count} rows; expected {expected_rows}")
    expected_prompts = set(expected_prompt_indices)
    for key in ((method, sample) for method in METHODS for sample in SAMPLE_TYPES):
        if prompt_coverage.get(key) != expected_prompts:
            raise AssertionError(f"full prompt coverage differs for {key}")

    summary_rows = []
    for method in METHODS:
        for prefix in PREFIX_LENGTHS:
            wm = groups[(method, prefix, "watermarked")]
            null = groups[(method, prefix, "null")]
            if len(wm) != prompt_count or len(null) != prompt_count:
                raise AssertionError(f"summary coverage differs for {method} T={prefix}")
            neglog = lambda row: -math.log10(max(float(row["p_value"]), 1e-300))
            summary_rows.append(
                {
                    "method": method,
                    "prefix_length": prefix,
                    "watermarked_prompts": prompt_count,
                    "null_prompts": prompt_count,
                    "tpr": sum(row["decision"] for row in wm) / prompt_count,
                    "observed_fpr": sum(row["decision"] for row in null) / prompt_count,
                    "false_positive_count": sum(row["decision"] for row in null),
                    "median_neg_log10_p_watermarked": statistics.median(map(neglog, wm)),
                    "median_neg_log10_p_null": statistics.median(map(neglog, null)),
                    "median_deduplicated_samples_watermarked": statistics.median(
                        row["deduplicated_sample_count"] for row in wm
                    ),
                    "median_deduplicated_samples_null": statistics.median(
                        row["deduplicated_sample_count"] for row in null
                    ),
                    "calibration_type": wm[0]["calibration_type"],
                    "nominal_decision_rule": "p < 0.001",
                }
            )
    summary_path = output_dir / f"{artifact_stem}_prefix_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    quality_path = output_dir / f"{artifact_stem}_quality.csv"
    quality_rows = list(quality.values())
    with quality_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)

    quality_summary_rows = []
    quality_fields = (
        "base_model_nll",
        "base_model_perplexity",
        "repetition_rate",
        "distinct_2",
        "distinct_3",
        "runtime_seconds",
    )
    for method in METHODS:
        for sample_type in SAMPLE_TYPES:
            rows = [
                row
                for row in quality_rows
                if row["method"] == method and row["sample_type"] == sample_type
            ]
            if len(rows) != prompt_count:
                raise AssertionError(
                    f"quality coverage differs for {(method, sample_type)}"
                )
            summary = {
                "method": method,
                "sample_type": sample_type,
                "prompt_count": prompt_count,
            }
            for field in quality_fields:
                values = [float(row[field]) for row in rows]
                summary[f"mean_{field}"] = statistics.mean(values)
                summary[f"median_{field}"] = statistics.median(values)
            quality_summary_rows.append(summary)
    quality_summary_path = output_dir / f"{artifact_stem}_quality_summary.csv"
    with quality_summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_summary_rows[0]))
        writer.writeheader()
        writer.writerows(quality_summary_rows)

    validation_path = output_dir / f"{artifact_stem}_validation.json"
    validation = {
        "status": "passed",
        "prompt_count": prompt_count,
        "prompt_indices": [expected_prompt_indices[0], expected_prompt_indices[-1]],
        "record_count": row_count,
        "schema_validation": "all PromptLevelResult.from_dict calls passed",
        "prefix_lengths": list(PREFIX_LENGTHS),
        "methods": list(METHODS),
        "sample_types": list(SAMPLE_TYPES),
        "generation_attempts": {"online_prc": 0, "null": 0},
        "calibration_warning": (
            "The four nominal p-values use different calibrations. Observed null false "
            f"positives are reported separately; {prompt_count} nulls cannot tightly "
            "validate a 0.1% FPR."
        ),
    }
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    artifact_paths = [
        prompt_path,
        summary_path,
        quality_path,
        quality_summary_path,
        validation_path,
    ]
    manifest_path = output_dir / f"{artifact_stem}_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "artifacts": [
                    {
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in artifact_paths
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "passed": True,
        "record_count": row_count,
        "quality_rows": len(quality_rows),
        "summary_rows": len(summary_rows),
        "output_dir": str(output_dir),
    }


def finalize(scored_dir: Path, output_dir: Path) -> dict:
    """Finalize the complete ten-shard, 500-prompt comparison."""
    return _finalize(
        scored_dir,
        output_dir,
        expected_shard_indices=tuple(range(10)),
        expected_prompt_indices=tuple(range(500)),
        artifact_stem="controlled_baseline_full",
    )


def finalize_batch50(scored_dir: Path, output_dir: Path) -> dict:
    """Finalize only the scored prompt-0..49 batch validation."""
    return _finalize(
        scored_dir,
        output_dir,
        expected_shard_indices=(0,),
        expected_prompt_indices=tuple(range(50)),
        artifact_stem="controlled_baseline_batch50_eval",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("finalize")
    command.add_argument("--scored-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    batch50_command = subparsers.add_parser("finalize-batch50")
    batch50_command.add_argument("--scored-dir", type=Path, required=True)
    batch50_command.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    function = finalize if args.command == "finalize" else finalize_batch50
    print(json.dumps(function(args.scored_dir, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
