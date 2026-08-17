"""Pure validation and aggregation helpers for parallel Phase 2 replay."""
from __future__ import annotations

import hashlib
import json
import statistics


PHASE2_SHARD_SCHEMA_VERSION = 1
PHASE2_PARALLEL_RESULT_SCHEMA_VERSION = 2

_REQUIRED_RECORD_FIELDS = (
    "prompt_idx",
    "watermark",
    "phase0_hoeffding_decision",
    "phase1_rademacher_decision",
    "phase2_hoeffding_decision",
    "phase2_adaptive_decision",
    "basis_rank",
    "tokens_sha256",
    "p_trace_sha256",
    "source_path",
)


def stable_json_sha256(value) -> str:
    """Hash a JSON-compatible value with deterministic serialization."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def phase2_prompt_shards(
    num_prompts: int,
    shard_size: int,
) -> list[list[int]]:
    """Split canonical prompt indices into stable contiguous shards."""
    count = int(num_prompts)
    size = int(shard_size)
    if count <= 0:
        raise ValueError("num_prompts must be positive")
    if size <= 0:
        raise ValueError("shard_size must be positive")
    indices = list(range(count))
    return [indices[start:start + size] for start in range(0, count, size)]


def phase2_config_fingerprint(run_identity: dict) -> str:
    """Return the cache identity for one complete parallel replay."""
    if not isinstance(run_identity, dict) or not run_identity:
        raise ValueError("run_identity must be a nonempty dictionary")
    return stable_json_sha256(run_identity)


def phase2_record_manifest(records: list[dict]) -> list[dict]:
    """Return the semantic source-record manifest used for verification."""
    return [
        {
            "prompt_idx": int(record["prompt_idx"]),
            "watermark": bool(record["watermark"]),
            "tokens_sha256": str(record["tokens_sha256"]),
            "p_trace_sha256": str(record["p_trace_sha256"]),
        }
        for record in records
    ]


def validate_phase2_shard_payload(
    payload: dict,
    *,
    expected_config_fingerprint: str | None = None,
    expected_basis_rank: int | None = None,
) -> list[int]:
    """Validate one cached prompt shard and return its prompt indices."""
    if payload.get("shard_schema_version") != PHASE2_SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported Phase 2 shard schema")
    fingerprint = str(payload.get("config_fingerprint", ""))
    if not fingerprint:
        raise ValueError("Phase 2 shard is missing its config fingerprint")
    if (
        expected_config_fingerprint is not None
        and fingerprint != expected_config_fingerprint
    ):
        raise ValueError("Phase 2 shard configuration mismatch")
    prompt_indices = [int(index) for index in payload.get("prompt_indices", [])]
    if (
        not prompt_indices
        or prompt_indices != sorted(prompt_indices)
        or len(set(prompt_indices)) != len(prompt_indices)
    ):
        raise ValueError("Phase 2 shard prompt indices are invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 2 * len(prompt_indices):
        raise ValueError("Phase 2 shard record count is inconsistent")
    if stable_json_sha256(records) != payload.get("records_sha256"):
        raise ValueError("Phase 2 shard record checksum mismatch")

    expected_identities = {
        (watermark, prompt_idx)
        for watermark in (True, False)
        for prompt_idx in prompt_indices
    }
    identities = set()
    source_paths = set()
    for record in records:
        missing = [field for field in _REQUIRED_RECORD_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"Phase 2 shard record is missing fields: {missing}"
            )
        identity = (bool(record["watermark"]), int(record["prompt_idx"]))
        if identity in identities:
            raise ValueError(f"Phase 2 shard duplicates record {identity}")
        identities.add(identity)
        source_path = str(record["source_path"])
        if source_path in source_paths:
            raise ValueError(f"Phase 2 shard duplicates source {source_path}")
        source_paths.add(source_path)
        if expected_basis_rank is not None and int(record["basis_rank"]) != int(
            expected_basis_rank
        ):
            raise ValueError("Phase 2 shard contains a rank-deficient basis")
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)
        extra = sorted(identities - expected_identities)
        raise ValueError(
            f"Phase 2 shard identity mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return prompt_indices


def merge_phase2_shard_payloads(
    shard_payloads: list[dict],
    expected_prompt_indices: list[int],
    *,
    config_fingerprint: str,
    basis_rank: int,
) -> list[dict]:
    """Merge validated shards into the canonical watermarked/null order."""
    expected = [int(index) for index in expected_prompt_indices]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected prompt indices must be nonempty and unique")
    covered = set()
    records = []
    record_identities = set()
    for payload in shard_payloads:
        prompt_indices = validate_phase2_shard_payload(
            payload,
            expected_config_fingerprint=config_fingerprint,
            expected_basis_rank=basis_rank,
        )
        overlap = covered.intersection(prompt_indices)
        if overlap:
            raise ValueError(f"Phase 2 shards overlap prompts: {sorted(overlap)}")
        covered.update(prompt_indices)
        for record in payload["records"]:
            identity = (bool(record["watermark"]), int(record["prompt_idx"]))
            if identity in record_identities:
                raise ValueError(f"Phase 2 shards duplicate record {identity}")
            record_identities.add(identity)
            records.append(record)
    expected_set = set(expected)
    if covered != expected_set:
        missing = sorted(expected_set - covered)
        extra = sorted(covered - expected_set)
        raise ValueError(
            f"Phase 2 shard coverage mismatch: missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )
    return sorted(
        records,
        key=lambda record: (
            0 if bool(record["watermark"]) else 1,
            int(record["prompt_idx"]),
        ),
    )


def _decision_count(records: list[dict], field: str) -> int:
    return int(sum(bool(record[field]) for record in records))


def _transition_counts(
    records: list[dict],
    baseline_field: str,
    candidate_field: str,
) -> dict:
    gained = int(
        sum(
            bool(record[candidate_field]) and not bool(record[baseline_field])
            for record in records
        )
    )
    lost = int(
        sum(
            bool(record[baseline_field]) and not bool(record[candidate_field])
            for record in records
        )
    )
    return {"gained": gained, "lost": lost, "net": gained - lost}


def summarize_phase2_records(
    records: list[dict],
    num_prompts: int,
    *,
    expected_hoeffding_tp: int = -1,
    expected_hoeffding_fp: int = -1,
) -> dict:
    """Summarize all four calibrator/basis combinations deterministically."""
    count = int(num_prompts)
    watermarked = [record for record in records if bool(record["watermark"])]
    null = [record for record in records if not bool(record["watermark"])]
    if len(watermarked) != count or len(null) != count:
        raise ValueError(
            "Phase 2 aggregate must contain exactly num_prompts records "
            "for each source"
        )
    methods = {
        "phase0_original_basis_hoeffding": "phase0_hoeffding_decision",
        "phase1_original_basis_weighted_rademacher": (
            "phase1_rademacher_decision"
        ),
        "phase2_adaptive_basis_hoeffding": "phase2_hoeffding_decision",
        "phase2_adaptive_basis_weighted_rademacher": (
            "phase2_adaptive_decision"
        ),
    }
    summary = {}
    for name, field in methods.items():
        tp = _decision_count(watermarked, field)
        fp = _decision_count(null, field)
        summary[name] = {
            "tp": tp,
            "tpr": tp / count,
            "fp": fp,
            "fpr": fp / count,
        }
    baseline = summary["phase0_original_basis_hoeffding"]
    if expected_hoeffding_tp >= 0 and baseline["tp"] != expected_hoeffding_tp:
        raise AssertionError(
            "Phase 0 true-positive baseline mismatch: "
            f"got {baseline['tp']}, expected {expected_hoeffding_tp}"
        )
    if expected_hoeffding_fp >= 0 and baseline["fp"] != expected_hoeffding_fp:
        raise AssertionError(
            "Phase 0 false-positive baseline mismatch: "
            f"got {baseline['fp']}, expected {expected_hoeffding_fp}"
        )

    summary["paired_transitions"] = {
        "adaptive_hoeffding_vs_original_hoeffding_watermarked": (
            _transition_counts(
                watermarked,
                "phase0_hoeffding_decision",
                "phase2_hoeffding_decision",
            )
        ),
        "adaptive_rademacher_vs_original_rademacher_watermarked": (
            _transition_counts(
                watermarked,
                "phase1_rademacher_decision",
                "phase2_adaptive_decision",
            )
        ),
        "adaptive_hoeffding_vs_original_hoeffding_null": _transition_counts(
            null,
            "phase0_hoeffding_decision",
            "phase2_hoeffding_decision",
        ),
        "adaptive_rademacher_vs_original_rademacher_null": _transition_counts(
            null,
            "phase1_rademacher_decision",
            "phase2_adaptive_decision",
        ),
    }
    quantile_counts = {}
    for record in records:
        key = f"{float(record['selected_erasure_quantile']):.12g}"
        quantile_counts[key] = quantile_counts.get(key, 0) + 1
    summary["basis_selection"] = {
        "quantile_counts": dict(sorted(quantile_counts.items())),
        "median_erased_columns": float(
            statistics.median(
                int(record["erased_columns"]) for record in records
            )
        ),
        "median_erasure_free_rows": float(
            statistics.median(
                int(record["erasure_free_rows"]) for record in records
            )
        ),
        "median_degree": float(
            statistics.median(
                float(record["degree_median"]) for record in records
            )
        ),
    }
    return summary
