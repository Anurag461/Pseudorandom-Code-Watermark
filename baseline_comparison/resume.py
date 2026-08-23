"""Fail-closed validation for reusing committed full-run shards."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .config import GENERATION_SETTINGS, MAX_NEW_TOKENS, PREFIX_LENGTHS
from .schema import PromptLevelResult
from .smoke_runner import (
    FULL_GENERATION_BATCH_SIZE,
    FULL_SHARD_SIZE,
    _model_revision,
    _numpy_pickle_compat,
    _validated_full_request,
)


METHODS = ("online_prc", "textseal", "synthid_text", "gumbel_max")
GENERATED_METHODS = ("textseal", "synthid_text", "gumbel_max")
SAMPLE_TYPES = ("watermarked", "null")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_generated_shard(
    data_volume,
    request: dict,
    *,
    expected_sha256: str | None = None,
    expected_integration_fingerprint: str | None = None,
) -> dict:
    """Validate a committed raw shard without loading Qwen or generating."""
    run_id, shard_index, prompt_indices = _validated_full_request(request)
    data_volume.reload()
    path = (
        Path("/data/controlled_baseline_full")
        / run_id
        / "generated"
        / f"shard_{shard_index:02d}.pt"
    )
    if not path.is_file():
        raise FileNotFoundError(f"generated shard is missing: {path}")
    sha256 = _sha256(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise AssertionError("reused generated shard SHA-256 differs")
    _numpy_pickle_compat()
    raw = torch.load(path, weights_only=False, map_location="cpu")
    expected = {
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "generation_settings": GENERATION_SETTINGS,
        "generation_batch_size": FULL_GENERATION_BATCH_SIZE,
        "generation_attempts": {"online_prc": 0, "null": 0},
    }
    for field, value in expected.items():
        if raw.get(field) != value:
            raise AssertionError(f"generated shard field differs: {field}")
    if raw.get("model_revision") != _model_revision():
        raise AssertionError("generated shard model revision differs")
    if (
        expected_integration_fingerprint is not None
        and raw.get("integration_code_fingerprint")
        != expected_integration_fingerprint
    ):
        raise AssertionError("reused generation integration fingerprint differs")
    if set(raw.get("sequences", {})) != set(GENERATED_METHODS):
        raise AssertionError("generated shard method coverage differs")
    for method in GENERATED_METHODS:
        rows = raw["sequences"][method]
        if len(rows) != FULL_SHARD_SIZE:
            raise AssertionError(f"{method} generated row count differs")
        for row in rows:
            for field in ("token_ids", "base_entropies", "base_token_logprobs"):
                if len(row.get(field, ())) != MAX_NEW_TOKENS:
                    raise AssertionError(f"{method} {field} length differs")
            values = np.concatenate(
                (
                    np.asarray(row["base_entropies"], dtype=np.float64),
                    np.asarray(row["base_token_logprobs"], dtype=np.float64),
                )
            )
            if not np.all(np.isfinite(values)):
                raise AssertionError(f"{method} contains non-finite base-model values")
    runtime = raw.get("runtime", {})
    for field in ("model_load_seconds", "function_seconds"):
        if not math.isfinite(float(runtime.get(field, math.nan))):
            raise AssertionError(f"generated shard runtime differs: {field}")
    if int(runtime.get("peak_cuda_reserved_bytes", -1)) > 70 * 1024**3:
        raise AssertionError("generated shard exceeds the 70 GiB memory gate")
    return {
        "passed": True,
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "path": str(path),
        "sha256": sha256,
        "integration_code_fingerprint": raw["integration_code_fingerprint"],
        "generation_attempts": raw["generation_attempts"],
        "actual_gpu": runtime["actual_gpu"],
        "peak_cuda_reserved_bytes": int(runtime["peak_cuda_reserved_bytes"]),
    }


def validate_scored_shard(
    data_volume,
    request: dict,
    *,
    expected_jsonl_sha256: str | None = None,
    expected_validation_sha256: str | None = None,
    expected_scoring_integration_fingerprint: str | None = None,
) -> dict:
    """Stream-validate a committed scored shard without rescoring it."""
    run_id, shard_index, prompt_indices = _validated_full_request(request)
    data_volume.reload()
    root = Path("/data/controlled_baseline_full") / run_id / "scored"
    jsonl_path = root / f"shard_{shard_index:02d}.jsonl"
    validation_path = root / f"shard_{shard_index:02d}_validation.json"
    if not jsonl_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError(f"scored shard {shard_index} is incomplete")
    jsonl_sha256 = _sha256(jsonl_path)
    validation_sha256 = _sha256(validation_path)
    if expected_jsonl_sha256 is not None and jsonl_sha256 != expected_jsonl_sha256:
        raise AssertionError("reused scored JSONL SHA-256 differs")
    if (
        expected_validation_sha256 is not None
        and validation_sha256 != expected_validation_sha256
    ):
        raise AssertionError("reused scored validation SHA-256 differs")

    coverage: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    identities = set()
    scoring_fingerprints = set()
    row_count = 0
    with jsonl_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            PromptLevelResult.from_dict(row)
            identity = (
                row["method"],
                row["sample_type"],
                int(row["prompt_index"]),
                int(row["generation_seed"]),
                int(row["prefix_length"]),
            )
            if identity in identities:
                raise AssertionError(f"duplicate scored row: {identity}")
            identities.add(identity)
            coverage[
                (row["method"], row["sample_type"], int(row["prefix_length"]))
            ].add(int(row["prompt_index"]))
            scoring_fingerprints.add(row["integration_code_fingerprint"])
            row_count += 1
    expected_rows = FULL_SHARD_SIZE * len(METHODS) * len(SAMPLE_TYPES) * len(PREFIX_LENGTHS)
    if row_count != expected_rows:
        raise AssertionError(f"scored shard has {row_count} rows; expected {expected_rows}")
    expected_prompts = set(prompt_indices)
    for key in (
        (method, sample_type, prefix)
        for method in METHODS
        for sample_type in SAMPLE_TYPES
        for prefix in PREFIX_LENGTHS
    ):
        if coverage.get(key) != expected_prompts:
            raise AssertionError(f"scored shard coverage differs for {key}")
    if len(scoring_fingerprints) != 1:
        raise AssertionError("scored shard contains multiple integration fingerprints")
    scoring_fingerprint = next(iter(scoring_fingerprints))
    if (
        expected_scoring_integration_fingerprint is not None
        and scoring_fingerprint != expected_scoring_integration_fingerprint
    ):
        raise AssertionError("reused scoring integration fingerprint differs")

    validation = json.loads(validation_path.read_text())
    if validation.get("passed") is not True:
        raise AssertionError("scored validation status differs")
    if validation.get("prompt_indices") != prompt_indices:
        raise AssertionError("scored validation prompt ordering differs")
    if int(validation.get("record_count", -1)) != expected_rows:
        raise AssertionError("scored validation row count differs")
    if validation.get("generation_attempts") != {"online_prc": 0, "null": 0}:
        raise AssertionError("scored validation generation attempts differ")
    validation_records = validation.get("validation_records", [])
    if len(validation_records) != FULL_SHARD_SIZE * len(METHODS) * len(SAMPLE_TYPES):
        raise AssertionError("scored validation record coverage differs")
    for record in validation_records:
        if record["method"] == "online_prc":
            if record.get("prefix_grid_equivalent_to_direct_detector") is not True:
                raise AssertionError("online PRC exact-prefix validation differs")
        else:
            checks = record.get("exact_prefix_checks", [])
            if len(checks) != len(PREFIX_LENGTHS) or any(
                float(check["max_abs_delta"]) != 0.0 for check in checks
            ):
                raise AssertionError("baseline exact-prefix validation differs")
    return {
        "passed": True,
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "record_count": row_count,
        "jsonl_path": str(jsonl_path),
        "jsonl_sha256": jsonl_sha256,
        "validation_path": str(validation_path),
        "validation_sha256": validation_sha256,
        "scoring_integration_code_fingerprint": scoring_fingerprint,
        "generation_attempts": validation["generation_attempts"],
    }
