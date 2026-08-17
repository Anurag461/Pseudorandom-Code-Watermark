"""Isolated Modal experiment for the causal online PRC construction.

This runner intentionally does not write to fixed-PRC watermarked caches or
the fixed experiment summary.  Compatible forced-length Qwen3-0.6B null
records are reused because null generation has no PRC key or construction;
otherwise they are generated with the same batched sampler and shared null
namespace used by ``modal_run.py``.

Example (the first full experiment):

    modal run modal_online_run.py --num-prompts 500 --n 256 --t 3 \
      --eta 0.05 --fpr 1e-3 --batch 64 --max-containers 5

8B generation and detection use the same causal construction and detector,
with model-qualified caches and 8B probability traces:

    modal run modal_online_run.py::sweep_map_prefixes \
      --generation-model-size 8B --source-n 2048 --floor-n 1024 \
      --step 16 --eta 0.15 --num-prompts 500 --batch 25 \
      --max-containers 5 --gpu H100 --no-pin-floor-cache
"""
import csv
import hashlib
import json
import os
import re
from decimal import Decimal
from datetime import datetime, timezone

import modal
from online_prc import GENERATION_SAMPLER_VERSION


SCHEME = "online_causal_prc_v1"
STOPPING_POLICY = "forced_length_v1"
FPR_POLICY = "one_shot"
SEED = 12345
MODEL_SIZE = "0.6B"
MODEL_DISPLAY = "Qwen3-0.6B-Base"
SUPPORTED_MODEL_SIZES = ("0.6B", "8B")
VOCAB = 151_936
GPU = "A10G"
DEFAULT_BATCH = 64
DEFAULT_8B_BATCH = 25
DEFAULT_MAX_CONTAINERS = 5
DEFAULT_DETECTION_SHARD_SIZE = 50
DEFAULT_DETECTION_MAX_CONTAINERS = 10
CANONICAL_NUM_PROMPTS = 500
RESULT_SCHEMA_VERSION = 2
PREPARED_MAP_SHARD_SCHEMA_VERSION = 1
LEGACY_SAMPLER_VERSION = "legacy_torch_global_v1"
ONLINE_MODEL_CACHE_NAME = "qwen3_0p6b_base"
SAMPLER_CACHE_TAG = "poscdf-v1"
DEFAULT_KV_CACHE_IMPLEMENTATION = "concat"
CONCAT_KV_CACHE_VERSION = "concat-v1"
STATIC_KV_CACHE_VERSION = "static-v1"
KV_CACHE_IMPLEMENTATIONS = ("concat", "static")
LOCAL_CSV_COLUMNS = (
    "timestamp_utc",
    "scheme",
    "eta",
    "T",
    "n",
    "r value",
    "free coordinates",
    "r setting",
    "t",
    "Target FPR",
    "Generation Model",
    "num prompts",
    "batch",
    "kv cache implementation",
    "kv cache version",
    "experiment seed",
    "Map TPR",
    "Map FPR",
    "Entropy Aware TPR",
    "Entropy FPR",
    "Naive TPR",
    "Naive FPR",
    "null cache T",
    "watermarked cache mode",
    "watermarked cache T",
    "watermarked cache tag",
    "watermarked resume source T",
    "watermarked resume source tag",
    "schedule version",
    "stopping policy",
    "FPR policy",
    "artifact fingerprint",
)


def normalize_kv_cache_implementation(implementation="concat") -> str:
    value = str(implementation or DEFAULT_KV_CACHE_IMPLEMENTATION).strip().lower()
    value = {
        "dynamic": "concat",
        "legacy": "concat",
        "preallocated": "static",
    }.get(value, value)
    if value not in KV_CACHE_IMPLEMENTATIONS:
        raise ValueError(
            f"kv cache implementation must be one of "
            f"{KV_CACHE_IMPLEMENTATIONS}; got {implementation!r}"
        )
    return value


def kv_cache_version(implementation="concat") -> str:
    implementation = normalize_kv_cache_implementation(implementation)
    return (
        STATIC_KV_CACHE_VERSION
        if implementation == "static"
        else CONCAT_KV_CACHE_VERSION
    )


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def normalize_model_size(model_size: str = MODEL_SIZE) -> str:
    value = MODEL_SIZE if model_size is None else str(model_size).strip()
    if not value:
        value = MODEL_SIZE
    upper = value.upper()
    if upper.endswith("B"):
        normalized = upper[:-1] + "B"
    elif upper.replace(".", "", 1).isdigit():
        normalized = f"{upper}B"
    else:
        normalized = value
    if normalized not in SUPPORTED_MODEL_SIZES:
        raise ValueError(
            f"online generation model must be one of "
            f"{SUPPORTED_MODEL_SIZES}; got {model_size!r}"
        )
    return normalized


def model_display(model_size: str = MODEL_SIZE) -> str:
    return f"Qwen3-{normalize_model_size(model_size)}-Base"


def model_cache_name(model_size: str = MODEL_SIZE) -> str:
    size = normalize_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def model_default_gpu(model_size: str = MODEL_SIZE) -> str:
    return "H100" if normalize_model_size(model_size) == "8B" else GPU


def model_default_batch(model_size: str = MODEL_SIZE) -> int:
    return (
        DEFAULT_8B_BATCH
        if normalize_model_size(model_size) == "8B"
        else DEFAULT_BATCH
    )


def resolve_model_runtime(model_size: str, batch: int = 0,
                          gpu: str = "") -> tuple[str, int, str]:
    """Normalize the model and choose safe model-specific CLI defaults."""
    normalized = normalize_model_size(model_size)
    requested_batch = int(batch)
    if requested_batch < 0:
        raise ValueError("batch must be nonnegative (zero selects the default)")
    resolved_batch = requested_batch or model_default_batch(normalized)
    resolved_gpu = str(gpu).strip() or model_default_gpu(normalized)
    return normalized, resolved_batch, resolved_gpu


def legacy_config_tag(n: int, t: int, eta: float,
                      experiment_seed: int = SEED,
                      generation_model_size: str = MODEL_SIZE) -> str:
    tag = (
        f"{SCHEME}/{model_cache_name(generation_model_size)}/"
        f"n{int(n)}_T{int(n)}_t{int(t)}_eta{float(eta):.2f}_rr99of100"
    )
    if int(experiment_seed) != SEED:
        tag += f"_seed{int(experiment_seed)}"
    return tag


def config_tag(n: int, t: int, eta: float,
               experiment_seed: int = SEED,
               generation_model_size: str = MODEL_SIZE,
               kv_cache_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
               ) -> str:
    """Sampler-v2 namespace; legacy caches remain readable as sources."""
    implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    tag = (
        f"{legacy_config_tag(n, t, eta, experiment_seed, generation_model_size)}"
        f"_sampler-{SAMPLER_CACHE_TAG}"
    )
    if implementation != DEFAULT_KV_CACHE_IMPLEMENTATION:
        tag += f"_kvcache-{kv_cache_version(implementation)}"
    return tag


def artifact_path(tag: str) -> str:
    return f"/data/{tag}/artifacts.pt"


def wm_dir(tag: str) -> str:
    return f"/data/{tag}/wm"


def shared_null_dir(length: int,
                    generation_model_size: str = MODEL_SIZE) -> str:
    """Use the fixed runner's model-qualified shared-null layout."""
    model_size = normalize_model_size(generation_model_size)
    root = "/data/_nulls"
    if model_size != MODEL_SIZE:
        root = f"{root}/{model_cache_name(model_size)}"
    return f"{root}/T{int(length)}"


def prepared_map_shard_path(source_tag: str, maximum_length: int,
                            prompt_indices: list[int],
                            artifact_fingerprint: str,
                            code_fingerprint: str) -> str:
    """Return a deterministic, versioned path for derived MAP preparation."""
    indices = [int(index) for index in prompt_indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("prompt_indices must be nonempty and unique")
    encoded = ",".join(str(index) for index in indices).encode()
    index_hash = hashlib.sha256(encoded).hexdigest()[:12]
    shard_label = (
        f"{min(indices):04d}-{max(indices):04d}-count{len(indices)}"
        f"-{index_hash}"
    )
    return (
        f"/data/{source_tag}/prepared_map_v"
        f"{PREPARED_MAP_SHARD_SCHEMA_VERSION}/"
        f"artifact-{str(artifact_fingerprint)[:16]}/"
        f"code-{str(code_fingerprint)[:16]}/T{int(maximum_length)}/"
        f"shard-{shard_label}.pt"
    )


def prompt_detection_shards(prompt_indices: list[int],
                            shard_size: int) -> list[list[int]]:
    """Split unique prompt indices into stable, order-preserving shards."""
    indices = [int(index) for index in prompt_indices]
    shard_size = int(shard_size)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("prompt_indices must be nonempty and unique")
    if shard_size <= 0:
        raise ValueError("detection shard_size must be positive")
    return _chunks(indices, shard_size)


def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def _format_rate(successes: int, total: int) -> str:
    return f"{int(successes)}/{int(total)} ({successes / max(total, 1):.1%})"


def descending_prefix_grid(source_n: int, floor_n: int,
                           step: int = 16) -> list[int]:
    """Return the exact descending prefix grid, including both endpoints."""
    source_n = int(source_n)
    floor_n = int(floor_n)
    step = int(step)
    if source_n <= 0 or floor_n <= 0 or floor_n > source_n:
        raise ValueError("require 0 < floor_n <= source_n")
    if step <= 0:
        raise ValueError("step must be positive")
    if (source_n - floor_n) % step:
        raise ValueError("source_n - floor_n must be divisible by step")
    return list(range(source_n, floor_n - 1, -step))


def rate_strictly_above(successes: int, total: int,
                        target_rate: float) -> bool:
    """Compare an empirical rate to its target without float-boundary drift."""
    successes = int(successes)
    total = int(total)
    target = Decimal(str(target_rate))
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes and total do not describe a valid rate")
    if not Decimal("0") < target < Decimal("1"):
        raise ValueError("target_rate must be in (0, 1)")
    return Decimal(successes) > target * Decimal(total)


def summarize_map_sweep(rows: list[dict], target_rate: float) -> dict:
    """Select the last passing length in a descending prefix sweep."""
    if not rows:
        raise ValueError("sweep rows must be nonempty")
    lengths = [int(row["n"]) for row in rows]
    if len(set(lengths)) != len(lengths):
        raise ValueError("sweep rows contain duplicate lengths")
    if lengths != sorted(lengths, reverse=True):
        raise ValueError("sweep rows must be ordered from longest to shortest")

    evaluated = []
    for row in rows:
        total = int(row["watermarked_total"])
        tp = int(row["tp"])
        passed = rate_strictly_above(tp, total, target_rate)
        evaluated.append({**row, "above_target": passed})

    passing = [row for row in evaluated if row["above_target"]]
    lowest_passing_anywhere = passing[-1] if passing else None
    last_passing = None
    next_shorter = None
    for row in evaluated:
        if row["above_target"]:
            last_passing = row
            continue
        next_shorter = row
        break

    ascending = list(reversed(evaluated))
    monotonicity_violations = []
    for lower, higher in zip(ascending, ascending[1:]):
        if float(higher["tpr"]) < float(lower["tpr"]):
            monotonicity_violations.append({
                "lower_n": int(lower["n"]),
                "lower_tpr": float(lower["tpr"]),
                "higher_n": int(higher["n"]),
                "higher_tpr": float(higher["tpr"]),
            })

    return {
        "target_map_tpr": float(target_rate),
        "comparison": "strictly_greater_than",
        "rows": evaluated,
        "last_passing_n_descending": (
            int(last_passing["n"]) if last_passing is not None else None
        ),
        "last_passing_tp": (
            int(last_passing["tp"]) if last_passing is not None else None
        ),
        "last_passing_tpr": (
            float(last_passing["tpr"]) if last_passing is not None else None
        ),
        "lowest_passing_n_anywhere": (
            int(lowest_passing_anywhere["n"])
            if lowest_passing_anywhere is not None else None
        ),
        "next_shorter_n": (
            int(next_shorter["n"]) if next_shorter is not None else None
        ),
        "next_shorter_above_target": (
            bool(next_shorter["above_target"])
            if next_shorter is not None else None
        ),
        "monotonicity_violations": monotonicity_violations,
    }


def evaluate_prepared_map_prefixes(
    prepared_records: list[dict],
    prefix_lengths: list[int],
    fpr: float,
    target_rate: float,
    stop_after_first_below: bool = True,
) -> dict:
    """Evaluate descending MAP prefixes and stop at the first failure.

    Each record must already contain the ceiling-length per-row MAP
    contributions produced by ``prepare_online_map_prefix_trace``.  Therefore
    stepping through lengths neither reloads traces nor rebuilds check values.
    """
    from detectors import score_prepared_online_map_prefix

    lengths = [int(length) for length in prefix_lengths]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("prefix_lengths must be nonempty and unique")
    if lengths != sorted(lengths, reverse=True):
        raise ValueError("prefix_lengths must be ordered longest to shortest")
    if not prepared_records:
        raise ValueError("prepared_records must be nonempty")
    rate_strictly_above(0, 1, target_rate)

    prompt_results = [{
        "prompt_idx": int(record["prompt_idx"]),
        "watermark": True,
        "map_scores": {},
    } for record in prepared_records]
    rows = []
    first_below_n = None
    for length in lengths:
        decisions = []
        scores = []
        for record in prepared_records:
            score = score_prepared_online_map_prefix(
                record["prepared"],
                length,
                fpr=fpr,
                fpr_policy=FPR_POLICY,
            )
            scores.append(score)
            decisions.append(bool(score["decision"]))
        for result, score in zip(prompt_results, scores):
            result["map_scores"][str(length)] = score

        tp = int(sum(decisions))
        total = len(decisions)
        representative = scores[0]
        rows.append({
            "n": int(length),
            "T": int(length),
            "r": int(representative["r"]),
            "free_coordinates": int(representative["free_coordinates"]),
            "tp": tp,
            "watermarked_total": total,
            "tpr": tp / total,
        })
        if (
            stop_after_first_below
            and not rate_strictly_above(tp, total, target_rate)
        ):
            first_below_n = int(length)
            break

    evaluated_lengths = [int(row["n"]) for row in rows]
    return {
        "rows": rows,
        "results": prompt_results,
        "evaluated_lengths": evaluated_lengths,
        "unevaluated_lengths": lengths[len(evaluated_lengths):],
        "stop_after_first_below": bool(stop_after_first_below),
        "first_below_n": first_below_n,
        "stopped_after_first_below": first_below_n is not None,
    }


def validate_prepared_map_shard(
    payload: dict,
    *,
    source_tag: str,
    maximum_length: int,
    artifact_fingerprint: str,
    online_key_sha256: str,
    code_fingerprint_sha256: str,
    expected_row_count: int,
) -> list[int]:
    """Validate a derived prompt-shard cache before aggregation or reuse."""
    import numpy as np

    expected = {
        "prepared_map_shard_schema_version": PREPARED_MAP_SHARD_SCHEMA_VERSION,
        "result_kind": "online_map_prepared_prompt_shard",
        "source_tag": str(source_tag),
        "maximum_length": int(maximum_length),
        "source_artifact_fingerprint": str(artifact_fingerprint),
        "online_key_sha256": str(online_key_sha256),
        "code_fingerprint_sha256": str(code_fingerprint_sha256),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"prepared MAP shard {field}={payload.get(field)!r}; "
                f"expected {value!r}"
            )

    indices = [int(index) for index in payload.get("prompt_indices", [])]
    records = payload.get("records", [])
    record_indices = [int(record["prompt_idx"]) for record in records]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("prepared MAP shard prompt indices are invalid")
    if record_indices != indices:
        raise ValueError(
            "prepared MAP shard records are not in declared prompt order"
        )
    if int(payload.get("num_prompts", -1)) != len(indices):
        raise ValueError("prepared MAP shard prompt count is inconsistent")

    row_count = int(expected_row_count)
    for record in records:
        signed = np.asarray(
            record.get("signed_check_values"), dtype=np.float64
        ).reshape(-1)
        squared = np.asarray(
            record.get("squared_check_values"), dtype=np.float64
        ).reshape(-1)
        if signed.size != row_count or squared.size != row_count:
            raise ValueError(
                f"prepared prompt {record['prompt_idx']} has "
                f"{signed.size}/{squared.size} rows; expected {row_count}"
            )
        if not np.all(np.isfinite(signed)):
            raise ValueError("prepared signed check values must be finite")
        if not np.all(np.isfinite(squared)) or np.any(squared < 0):
            raise ValueError(
                "prepared squared check values must be finite and nonnegative"
            )
    return indices


def merge_prepared_map_shards(
    shard_payloads: list[dict],
    expected_prompt_indices: list[int],
    online_key,
    maximum_length: int,
) -> list[dict]:
    """Merge validated shard records into exact requested prompt order."""
    import numpy as np
    from online_prc import materialize_supports, target_row_count

    expected = [int(index) for index in expected_prompt_indices]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_prompt_indices must be nonempty and unique")
    maximum = int(maximum_length)
    row_count = target_row_count(maximum, online_key)
    supports = materialize_supports(maximum, online_key)
    if int(supports.shape[0]) != int(row_count):
        raise AssertionError("prepared MAP support count is inconsistent")

    by_index = {}
    for payload in shard_payloads:
        for record in payload["records"]:
            index = int(record["prompt_idx"])
            if index in by_index:
                raise ValueError(
                    f"prepared MAP shards duplicate prompt index {index}"
                )
            by_index[index] = {
                "prompt_idx": index,
                "prepared": {
                    "online_key": online_key,
                    "maximum_length": maximum,
                    "supports": supports,
                    "signed_check_values": np.asarray(
                        record["signed_check_values"], dtype=np.float64
                    ).reshape(-1),
                    "squared_check_values": np.asarray(
                        record["squared_check_values"], dtype=np.float64
                    ).reshape(-1),
                },
            }

    missing = [index for index in expected if index not in by_index]
    extra = sorted(set(by_index) - set(expected))
    if missing or extra:
        raise ValueError(
            f"prepared MAP shard coverage mismatch: missing={missing}, "
            f"extra={extra}"
        )
    return [by_index[index] for index in expected]


def increment_payload_from_grid(grid_payload: dict, length: int) -> dict:
    """Extract one independently loadable MAP-prefix result from a grid."""
    length = int(length)
    matching_rows = [
        row for row in grid_payload.get("rows", [])
        if int(row["n"]) == length
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"grid must contain exactly one summary row for length {length}"
        )
    prompt_results = []
    for result in grid_payload.get("results", []):
        score = result.get("map_scores", {}).get(str(length))
        if score is None:
            raise ValueError(
                f"prompt {result.get('prompt_idx')} lacks MAP score at {length}"
            )
        prompt_results.append({
            "prompt_idx": int(result["prompt_idx"]),
            "watermark": True,
            "scores": {"map": score},
        })

    row = dict(matching_rows[0])
    observed_tp = sum(
        bool(result["scores"]["map"]["decision"])
        for result in prompt_results
    )
    if observed_tp != int(row["tp"]):
        raise AssertionError(
            f"length {length} row TP={row['tp']} but prompt scores give "
            f"{observed_tp}"
        )
    if len(prompt_results) != int(row["watermarked_total"]):
        raise AssertionError(
            f"length {length} row total does not match prompt score count"
        )

    metadata_fields = (
        "result_schema_version",
        "timestamp_utc",
        "scheme",
        "source_tag",
        "source_T",
        "t",
        "eta",
        "schedule_version",
        "support_sampler_version",
        "stopping_policy",
        "fpr_policy",
        "target_fpr",
        "generation_model",
        "generation_model_size",
        "kv_cache_implementation",
        "kv_cache_version",
        "num_prompts",
        "prompt_indices",
        "source_artifact_fingerprint",
        "online_key_sha256",
        "source_online_support_sha256",
        "code_fingerprint_sha256",
        "experiment_seed",
        "detection_strategy",
        "prepared_shard_count",
        "prepared_shard_cache_hits",
    )
    return {
        **{
            field: grid_payload[field]
            for field in metadata_fields
            if field in grid_payload
        },
        "result_kind": "saved_online_map_prefix",
        "n": length,
        "T": length,
        "r": int(row["r"]),
        "free_coordinates": int(row["free_coordinates"]),
        "prefix_online_support_sha256": row[
            "prefix_online_support_sha256"
        ],
        "counts": {
            "map": {
                "tp": int(row["tp"]),
                "watermarked_total": int(row["watermarked_total"]),
            }
        },
        "map_tpr": float(row["tpr"]),
        "results": prompt_results,
    }


def summarize_generation_cost(generation_meta: dict, source_n: int,
                              gpu: str) -> dict:
    """Aggregate measured GPU-method time and token work for one run.

    Method timers begin after a container has loaded the model, so the ledger
    intentionally separates measured GPU-method seconds from provider billing,
    which can also include image/model startup and teardown.
    """
    batches = []
    measured_gpu_seconds = 0.0
    replayed_prefix_tokens = 0
    generated_suffix_tokens = 0
    generated_null_tokens = 0
    peak_cuda_allocated_bytes = 0
    peak_cuda_reserved_bytes = 0
    for kind in ("wm", "null"):
        for record in generation_meta.get(kind, []):
            generated = int(record.get("generated", 0))
            seconds = float(record.get("seconds", 0.0))
            batch = int(record.get("batch", 0))
            resume_prefix = int(record.get("resume_prefix_T", 0) or 0)
            suffix = int(record.get("suffix_tokens_generated", 0) or 0)
            replayed = batch * resume_prefix if kind == "wm" else 0
            null_tokens = generated * int(source_n) if kind == "null" else 0
            peak_allocated = int(record.get("peak_cuda_allocated_bytes", 0) or 0)
            peak_reserved = int(record.get("peak_cuda_reserved_bytes", 0) or 0)
            if generated:
                measured_gpu_seconds += seconds
            replayed_prefix_tokens += replayed
            generated_suffix_tokens += suffix
            generated_null_tokens += null_tokens
            peak_cuda_allocated_bytes = max(
                peak_cuda_allocated_bytes, peak_allocated
            )
            peak_cuda_reserved_bytes = max(
                peak_cuda_reserved_bytes, peak_reserved
            )
            batches.append({
                "kind": kind,
                "generated_records": generated,
                "cached_records": int(record.get("cached", 0)),
                "batch": batch,
                "seconds": seconds,
                "resume_prefix_T": resume_prefix,
                "replayed_prefix_tokens": replayed,
                "generated_suffix_tokens": suffix,
                "generated_null_tokens": null_tokens,
                "kv_cache_implementation": record.get(
                    "kv_cache_implementation"
                ),
                "kv_cache_version": record.get("kv_cache_version"),
                "peak_cuda_allocated_bytes": peak_allocated,
                "peak_cuda_reserved_bytes": peak_reserved,
            })
    return {
        "gpu": str(gpu),
        "measured_gpu_method_seconds": measured_gpu_seconds,
        "measured_gpu_method_hours": measured_gpu_seconds / 3600.0,
        "replayed_prefix_tokens": replayed_prefix_tokens,
        "generated_suffix_tokens": generated_suffix_tokens,
        "generated_null_tokens": generated_null_tokens,
        "model_token_positions_processed": (
            replayed_prefix_tokens
            + generated_suffix_tokens
            + generated_null_tokens
        ),
        "peak_cuda_allocated_bytes": peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": peak_cuda_reserved_bytes,
        "batches": batches,
        "billing_note": (
            "Measured method time excludes container image/model startup; "
            "reconcile the app run URL with the Modal billing dashboard."
        ),
    }


def _append_local_csv(path: str, row: dict) -> None:
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            old_columns = tuple(reader.fieldnames or ())
            old_rows = list(reader)
        if old_columns != LOCAL_CSV_COLUMNS:
            migrated = f"{path}.schema-migration.tmp"
            with open(migrated, "w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=LOCAL_CSV_COLUMNS,
                    lineterminator="\n",
                )
                writer.writeheader()
                for old_row in old_rows:
                    writer.writerow({
                        column: old_row.get(column, "")
                        for column in LOCAL_CSV_COLUMNS
                    })
            os.replace(migrated, path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LOCAL_CSV_COLUMNS,
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in LOCAL_CSV_COLUMNS})


def online_cache_root(data_root: str = "/data",
                      generation_model_size: str = MODEL_SIZE) -> str:
    return os.path.join(
        data_root, SCHEME, model_cache_name(generation_model_size)
    )


def discover_online_cache_tags(data_root: str, requested_T: int, t: int,
                               eta: float, experiment_seed: int,
                               generation_model_size: str = MODEL_SIZE,
                               required_prompt_indices=None,
                               kv_cache_implementation: str = (
                                   DEFAULT_KV_CACHE_IMPLEMENTATION
                               )) -> list[dict]:
    """List same-configuration online cache namespaces by realized length.

    Directory names are only a first filter.  Remote planning additionally
    compares the serialized keys, partitions, prompts, and artifact metadata
    before any record is reused.
    """
    model_size = normalize_model_size(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    root = online_cache_root(data_root, model_size)
    if not os.path.isdir(root):
        return []
    candidates = []
    for name in os.listdir(root):
        match = re.match(r"^n(\d+)_T(\d+)_", name)
        if not match:
            continue
        n_value, T_value = (int(match.group(1)), int(match.group(2)))
        if n_value != T_value:
            continue
        possible_tags = [config_tag(
            T_value, t, eta, experiment_seed, model_size,
            kv_cache_implementation,
        )]
        if kv_cache_implementation == DEFAULT_KV_CACHE_IMPLEMENTATION:
            possible_tags.append(legacy_config_tag(
                T_value, t, eta, experiment_seed, model_size
            ))
        expected_tag = next(
            (candidate for candidate in possible_tags
             if name == candidate.rsplit("/", 1)[-1]),
            None,
        )
        if expected_tag is None:
            continue
        candidate_dir = os.path.join(root, name)
        if not os.path.isfile(os.path.join(candidate_dir, "artifacts.pt")):
            continue
        if required_prompt_indices is not None:
            candidate_wm_dir = os.path.join(candidate_dir, "wm")
            if not all(
                os.path.isfile(os.path.join(
                    candidate_wm_dir, f"wm_{int(index):04d}.pt"
                ))
                for index in required_prompt_indices
            ):
                continue
        candidates.append({
            "T": T_value,
            "tag": expected_tag,
            "directory": candidate_dir,
            "cache_sampler_version": (
                GENERATION_SAMPLER_VERSION
                if expected_tag == possible_tags[0]
                else LEGACY_SAMPLER_VERSION
            ),
            "kv_cache_implementation": kv_cache_implementation,
            "kv_cache_version": kv_cache_version(kv_cache_implementation),
            "relation": (
                "exact" if T_value == int(requested_T)
                else "longer" if T_value > int(requested_T)
                else "shorter"
            ),
        })
    return sorted(candidates, key=lambda candidate: candidate["T"])


def validate_generation_segments(segments: list[dict], realized_length: int) -> None:
    """Require contiguous, nonoverlapping provenance over the whole record."""
    if not isinstance(segments, list) or not segments:
        raise ValueError("generation_segments must be a nonempty list")
    cursor = 0
    for segment in segments:
        start = int(segment.get("start", -1))
        end = int(segment.get("end", -1))
        if start != cursor or end <= start:
            raise ValueError("generation_segments must be contiguous and nonempty")
        sampler = segment.get("sampler_version")
        if not isinstance(sampler, str) or not sampler:
            raise ValueError("every generation segment needs a sampler_version")
        cursor = end
    if cursor != int(realized_length):
        raise ValueError(
            f"generation_segments end at {cursor}, expected {realized_length}"
        )


def _local_code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (
        "modal_online_run.py", "online_prc.py", "watermark_expt.py",
        "detectors.py", "qwen.py", "constants.py",
    ):
        digest.update(path.encode("utf-8"))
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def artifact_compatibility_error(target: dict, source: dict) -> str | None:
    """Return why two length-specific artifacts cannot share online records."""
    import torch

    if target.get("online_key") != source.get("online_key"):
        return "online key differs"
    if int(target.get("experiment_seed", SEED)) != int(
        source.get("experiment_seed", SEED)
    ):
        return "experiment seed differs"
    if target.get("prompt_ids_list") != source.get("prompt_ids_list"):
        return "prompt corpus/order differs"
    target_partition = target.get("partition")
    source_partition = source.get("partition")
    if not isinstance(target_partition, torch.Tensor) or not isinstance(
        source_partition, torch.Tensor
    ):
        return "partition tensor is missing"
    if not torch.equal(target_partition, source_partition):
        return "token partition differs"
    if artifact_kv_cache_implementation(
        target
    ) != artifact_kv_cache_implementation(source):
        return "KV cache implementation differs"

    target_config = target.get("config_sig", {})
    source_config = source.get("config_sig", {})
    invariant_fields = (
        "scheme",
        "check_weight",
        "noise_rate",
        "row_rate_numerator",
        "row_rate_denominator",
        "schedule_version",
        "support_sampler_version",
        "stopping_policy",
        "generation_model_size",
        "generation_model",
        "keygen_seed",
        "partition_seed",
    )
    for field in invariant_fields:
        if target_config.get(field) != source_config.get(field):
            return f"artifact config field {field!r} differs"
    return None


def artifact_kv_cache_implementation(artifact: dict) -> str:
    """Read cache metadata while treating historical artifacts as concat."""
    config = artifact.get("config_sig", {})
    return normalize_kv_cache_implementation(
        artifact.get(
            "kv_cache_implementation",
            config.get(
                "kv_cache_implementation",
                DEFAULT_KV_CACHE_IMPLEMENTATION,
            ),
        )
    )


def artifact_generation_model_size(artifact: dict) -> str:
    config = artifact.get("config_sig", {})
    return normalize_model_size(
        artifact.get(
            "generation_model_size",
            config.get("generation_model_size", MODEL_SIZE),
        )
    )


def validate_generation_model_record(record: dict, model_size: str,
                                     source: str, prompt_index: int) -> None:
    """Reject cross-model records while retaining legacy 0.6B readability."""
    expected_size = normalize_model_size(model_size)
    stored_size = record.get("generation_model_size")
    stored_display = record.get("generation_model")
    if stored_size is None:
        if expected_size != MODEL_SIZE:
            raise ValueError(
                f"{source} record {prompt_index} lacks generation-model "
                f"metadata; refusing to treat it as "
                f"{model_display(expected_size)}"
            )
    elif normalize_model_size(stored_size) != expected_size:
        raise ValueError(
            f"{source} record {prompt_index} was generated by "
            f"{model_display(stored_size)}, expected "
            f"{model_display(expected_size)}"
        )
    if (
        stored_display is not None
        and str(stored_display).strip() != model_display(expected_size)
    ):
        raise ValueError(
            f"{source} record {prompt_index} has generation model label "
            f"{stored_display!r}, expected "
            f"{model_display(expected_size)!r}"
        )


def validate_online_null_record(record: dict, artifact: dict,
                                prompt_index: int,
                                required_length: int) -> None:
    """Validate a fixed-run null before using its trace for online detection."""
    import torch
    from detectors import tensor_sha256

    generation_model_size = artifact_generation_model_size(artifact)
    validate_generation_model_record(
        record, generation_model_size, "null", prompt_index
    )
    if record.get("watermark") not in (False, None):
        raise ValueError(f"null record {prompt_index} is marked watermarked")
    for field in ("tokens", "p_trace"):
        value = record.get(field)
        if value is None or len(value) < int(required_length):
            raise ValueError(
                f"null record {prompt_index} field {field!r} is shorter "
                f"than {required_length}"
            )
    expected_partition = tensor_sha256(artifact["partition"])
    stored_partition = record.get("partition_sha256")
    # Historical fixed-run nulls can predate the explicit partition hash.
    # Their model-qualified directory plus required model metadata keeps 8B
    # and 0.6B isolated; any explicit partition hash must still match.
    if stored_partition is not None and stored_partition != expected_partition:
        raise ValueError(
            f"null record {prompt_index} uses a different token partition"
        )
    stored_prompt = record.get("prompt_token_ids")
    if stored_prompt is not None:
        expected_prompt = torch.as_tensor(
            artifact["prompt_ids_list"][prompt_index], dtype=torch.long
        ).reshape(-1)
        observed_prompt = torch.as_tensor(
            stored_prompt, dtype=torch.long
        ).reshape(-1)
        if not torch.equal(observed_prompt, expected_prompt):
            raise ValueError(f"null record {prompt_index} has the wrong prompt")


def validate_online_watermarked_record(record: dict, artifact: dict,
                                       prompt_index: int) -> list[dict]:
    """Strictly validate a saved online record and normalize its provenance."""
    import numpy as np
    import torch
    from detectors import tensor_sha256
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    source_T = int(artifact["T"])
    generation_model_size = artifact_generation_model_size(artifact)
    expected_kv_cache = artifact_kv_cache_implementation(artifact)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    path_label = f"online wm record {prompt_index} at T={source_T}"
    expected_scalars = {
        "watermark": True,
        "prompt_idx": int(prompt_index),
        "scheme": SCHEME,
        "stopping_policy": STOPPING_POLICY,
        "realized_length": source_T,
        "realized_r": target_row_count(source_T, key),
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "online_key_sha256": key.fingerprint,
        "online_support_sha256": support_sha256(source_T, key),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "artifact_seed": int(artifact.get("experiment_seed", SEED)),
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "partition_sha256": tensor_sha256(artifact["partition"]),
    }
    for field, expected in expected_scalars.items():
        if record.get(field) != expected:
            raise ValueError(
                f"{path_label} has incompatible {field}: "
                f"{record.get(field)!r} != {expected!r}"
            )
    validate_generation_model_record(
        record, generation_model_size, "online watermarked", prompt_index
    )
    stored_kv_cache = normalize_kv_cache_implementation(
        record.get("kv_cache_implementation", DEFAULT_KV_CACHE_IMPLEMENTATION)
    )
    if stored_kv_cache != expected_kv_cache:
        raise ValueError(
            f"{path_label} has incompatible kv_cache_implementation: "
            f"{stored_kv_cache!r} != {expected_kv_cache!r}"
        )

    prompt = torch.as_tensor(record.get("prompt_token_ids"), dtype=torch.long)
    expected_prompt = torch.as_tensor(
        artifact["prompt_ids_list"][prompt_index], dtype=torch.long
    )
    if not torch.equal(prompt.reshape(-1), expected_prompt.reshape(-1)):
        raise ValueError(f"{path_label} has the wrong prompt token IDs")

    length_fields = (
        "tokens",
        "p_trace",
        "base_lm_entropy",
        "base_token_logprob",
        "prc_codeword_bits",
    )
    for field in length_fields:
        value = record.get(field)
        if value is None or int(np.asarray(value).size) != source_T:
            raise ValueError(
                f"{path_label} field {field!r} must have exactly {source_T} values"
            )

    codeword = np.asarray(record["prc_codeword_bits"], dtype=np.uint8)
    if np.any(codeword > 1):
        raise ValueError(f"{path_label} has non-binary PRC codeword bits")

    segments = record.get("generation_segments")
    if segments is None:
        segments = [{
            "start": 0,
            "end": source_T,
            "sampler_version": record.get(
                "online_sampler_version", LEGACY_SAMPLER_VERSION
            ),
            "legacy_inferred": True,
        }]
    else:
        segments = [dict(segment) for segment in segments]
    validate_generation_segments(segments, source_T)
    return segments


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
        "scipy",
        "galois",
        "numpy",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf",
        "PRC_MODEL_CACHE_DIR": "/cache/models",
        "PRC_MODEL_SIZE": MODEL_SIZE,
        "PRC_MODEL_VARIANT": "base",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "online_prc", "qwen", "constants", "detectors",
        "watermark_expt",
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
app = modal.App("prc-online-causal", image=image)


@app.function(volumes={"/data": data_vol}, timeout=600)
def build_artifacts(num_prompts: int, n: int, t: int, eta: float,
                    experiment_seed: int = SEED,
                    fresh: bool = False,
                    generation_model_size: str = MODEL_SIZE,
                    kv_cache_implementation: str = (
                        DEFAULT_KV_CACHE_IMPLEMENTATION
                    )) -> dict:
    import shutil

    import numpy as np
    import torch
    from detectors import semantic_sha256
    from online_prc import (
        OnlinePRCKey,
        gf2_rank,
        parity_check_dense,
        support_sha256,
        target_row_count,
    )

    if int(n) <= 0:
        raise ValueError("n must be positive")
    if int(num_prompts) <= 0 or int(num_prompts) > CANONICAL_NUM_PROMPTS:
        raise ValueError(f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]")
    if int(experiment_seed) < 0:
        raise ValueError("experiment_seed must be nonnegative")
    generation_model_size = normalize_model_size(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    key = OnlinePRCKey.from_seed(
        experiment_seed, check_weight=int(t), noise_rate=float(eta)
    )
    tag = config_tag(
        n, t, eta, experiment_seed, generation_model_size,
        kv_cache_implementation,
    )
    path = artifact_path(tag)
    config = {
        "scheme": SCHEME,
        "n": int(n),
        "T": int(n),
        "check_weight": int(t),
        "noise_rate": float(eta),
        "row_rate_numerator": key.row_rate_numerator,
        "row_rate_denominator": key.row_rate_denominator,
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "generation_cap": int(n),
        "stopping_policy": STOPPING_POLICY,
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "keygen_seed": int(experiment_seed),
        "partition_seed": SEED,
    }
    # Keep the historical concat artifact byte-for-byte addressable while
    # giving the opt-in static implementation an explicit, isolated identity.
    if kv_cache_implementation != DEFAULT_KV_CACHE_IMPLEMENTATION:
        config.update({
            "kv_cache_implementation": kv_cache_implementation,
            "kv_cache_version": kv_cache_version(kv_cache_implementation),
        })

    data_vol.reload()
    if not fresh and os.path.exists(path):
        previous = torch.load(path, weights_only=False, map_location="cpu")
        if previous.get("config_sig") == config:
            return {
                "tag": tag,
                "artifact_fingerprint": previous["artifact_fingerprint"],
                "reused": True,
            }
    if os.path.isdir(wm_dir(tag)):
        shutil.rmtree(wm_dir(tag))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(VOCAB, generator=generator)
    bucket_zero = torch.zeros(VOCAB, dtype=torch.bfloat16)
    bucket_zero[permutation[:VOCAB // 2]] = 1.0
    partition = torch.stack([bucket_zero, 1 - bucket_zero], dim=0)

    rows = []
    with open("/root/prompts.jsonl") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= CANONICAL_NUM_PROMPTS:
                break
    if len(rows) < CANONICAL_NUM_PROMPTS:
        raise RuntimeError(
            f"prompts.jsonl has {len(rows)} rows, need {CANONICAL_NUM_PROMPTS}"
        )

    checks = parity_check_dense(n, key)
    rank = gf2_rank(checks)
    realized_r = target_row_count(n, key)
    if rank != realized_r:
        raise RuntimeError(f"online parity rank {rank} != row count {realized_r}")
    artifact = {
        "online_key": key.to_dict(),
        "partition": partition,
        "prompt_ids_list": [row["prompt_tokens"] for row in rows],
        "num_prompts": CANONICAL_NUM_PROMPTS,
        "n": int(n),
        "T": int(n),
        "r": int(realized_r),
        "free_coordinates": int(n - realized_r),
        "support_sha256": support_sha256(n, key),
        "rank": int(rank),
        "config_sig": config,
        "experiment_seed": int(experiment_seed),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "kv_cache_implementation": kv_cache_implementation,
        "kv_cache_version": kv_cache_version(kv_cache_implementation),
    }
    artifact["artifact_fingerprint"] = semantic_sha256(artifact)
    torch.save(artifact, path)
    data_vol.commit()
    print(
        f"[build] {tag}: T=n={n}, t={t}, r={realized_r}, "
        f"free={n - realized_r}, rank={rank}, "
        f"kv_cache={kv_cache_implementation}", flush=True,
    )
    return {
        "tag": tag,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "reused": False,
    }


def _find_compatible_null_T(prompt_indices: list[int], requested_T: int,
                            generation_model_size: str = MODEL_SIZE):
    model_size = normalize_model_size(generation_model_size)
    null_root = os.path.dirname(shared_null_dir(0, model_size))
    if not os.path.isdir(null_root):
        return None
    candidates = []
    for name in os.listdir(null_root):
        match = re.fullmatch(r"T(\d+)", name)
        if match and int(match.group(1)) >= int(requested_T):
            candidates.append(int(match.group(1)))
    for length in sorted(candidates):
        directory = shared_null_dir(length, model_size)
        if all(
            os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
            for index in prompt_indices
        ):
            return length
    return None


@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(tag: str, prompt_indices: list[int], T: int,
                    allow_wm_reuse: bool = True) -> dict:
    import torch

    data_vol.reload()
    requested_artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(
        requested_artifact
    )
    kv_cache_implementation = artifact_kv_cache_implementation(
        requested_artifact
    )
    watermarked_missing = [
        index for index in prompt_indices
        if not os.path.exists(os.path.join(wm_dir(tag), f"wm_{index:04d}.pt"))
    ]

    wm_mode = "exact_cache" if not watermarked_missing else "fresh_generation"
    wm_source_tag = tag
    wm_source_T = int(T)
    wm_resume_source_tag = ""
    wm_resume_source_T = 0
    rejected_candidates = []
    if watermarked_missing and allow_wm_reuse:
        key_dict = requested_artifact["online_key"]
        check_weight = int(key_dict["check_weight"])
        noise_rate = float(key_dict["noise_rate"])
        experiment_seed = int(requested_artifact.get("experiment_seed", SEED))
        compatible = []
        for candidate in discover_online_cache_tags(
            "/data", T, check_weight, noise_rate, experiment_seed,
            generation_model_size,
            kv_cache_implementation=kv_cache_implementation,
        ):
            if candidate["tag"] == tag:
                continue
            candidate_wm_dir = os.path.join(candidate["directory"], "wm")
            if not all(
                os.path.isfile(os.path.join(
                    candidate_wm_dir, f"wm_{index:04d}.pt"
                ))
                for index in prompt_indices
            ):
                continue
            source_artifact = torch.load(
                os.path.join(candidate["directory"], "artifacts.pt"),
                weights_only=False,
                map_location="cpu",
            )
            incompatibility = artifact_compatibility_error(
                requested_artifact, source_artifact
            )
            if incompatibility:
                rejected_candidates.append({
                    "tag": candidate["tag"],
                    "reason": incompatibility,
                })
                continue
            compatible.append(candidate)

        longer = sorted(
            (candidate for candidate in compatible if candidate["T"] > int(T)),
            key=lambda candidate: (
                candidate["T"],
                candidate["cache_sampler_version"] == LEGACY_SAMPLER_VERSION,
            ),
        )
        shorter = sorted(
            (candidate for candidate in compatible if candidate["T"] < int(T)),
            key=lambda candidate: (
                candidate["T"],
                candidate["cache_sampler_version"] != LEGACY_SAMPLER_VERSION,
            ),
            reverse=True,
        )
        if longer:
            # Prefixing the smallest sufficient cache is generation-free.
            selected = longer[0]
            watermarked_missing = []
            wm_mode = "prefix_from_longer"
            wm_source_tag = selected["tag"]
            wm_source_T = int(selected["T"])
        elif shorter:
            # Continue the longest available common prefix into exact target
            # records. Existing exact target records remain untouched.
            selected = shorter[0]
            wm_mode = "continue_from_shorter"
            wm_resume_source_tag = selected["tag"]
            wm_resume_source_T = int(selected["T"])

    null_T = _find_compatible_null_T(
        prompt_indices, T, generation_model_size
    )
    if null_T is None:
        null_T = int(T)
        null_missing = [
            index for index in prompt_indices
            if not os.path.exists(
                os.path.join(
                    shared_null_dir(T, generation_model_size),
                    f"null_{index:04d}.pt",
                )
            )
        ]
    else:
        null_missing = []
    return {
        "wm_missing": watermarked_missing,
        "wm_mode": wm_mode,
        "wm_source_tag": wm_source_tag,
        "wm_source_T": wm_source_T,
        "wm_resume_source_tag": wm_resume_source_tag,
        "wm_resume_source_T": wm_resume_source_T,
        "wm_rejected_candidates": rejected_candidates,
        "null_missing": null_missing,
        "null_T": int(null_T),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "kv_cache_implementation": kv_cache_implementation,
        "kv_cache_version": kv_cache_version(kv_cache_implementation),
    }


@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=7200,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class OnlineModel:
    tag: str = modal.parameter()
    model_size: str = modal.parameter()
    code_fingerprint_sha256: str = modal.parameter()
    kv_cache_implementation: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        import torch
        from detectors import tensor_sha256
        from online_prc import OnlinePRCKey

        self.model_size = normalize_model_size(self.model_size)
        self.kv_cache_implementation = normalize_kv_cache_implementation(
            self.kv_cache_implementation
        )
        os.environ["PRC_MODEL_SIZE"] = self.model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"
        data_vol.reload()
        artifact = torch.load(
            artifact_path(self.tag), weights_only=False, map_location="cpu"
        )
        artifact_model_size = artifact_generation_model_size(artifact)
        if artifact_model_size != self.model_size:
            raise ValueError(
                f"artifact generation model {artifact_model_size} does not "
                f"match requested model {self.model_size}"
            )
        artifact_kv_cache = artifact_kv_cache_implementation(artifact)
        if artifact_kv_cache != self.kv_cache_implementation:
            raise ValueError(
                f"artifact KV cache implementation {artifact_kv_cache} does "
                f"not match requested {self.kv_cache_implementation}"
            )
        self.artifact = artifact
        self.key_dict = artifact["online_key"]
        self.key = OnlinePRCKey.from_dict(self.key_dict)
        self.partition_cpu = artifact["partition"]
        self.partition_fingerprint = tensor_sha256(self.partition_cpu)
        self.key_fingerprint = self.key.fingerprint
        self.artifact_fingerprint = artifact["artifact_fingerprint"]
        self.support_fingerprint = artifact["support_sha256"]
        # Artifacts from the first run predate the explicit replicate field.
        self.experiment_seed = int(artifact.get("experiment_seed", SEED))
        self.prompts = artifact["prompt_ids_list"]
        self.n = int(artifact["n"])
        self.T = int(artifact["T"])

        import watermark_expt as we
        self.we = we
        we.partition = self.partition_cpu.to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch
        return torch.tensor(
            [self.prompts[index] for index in indices],
            dtype=torch.long,
            device=self.we.device,
        )

    @modal.method()
    def ready(self) -> dict:
        return {
            "model": model_display(self.model_size),
            "model_size": self.model_size,
            "model_cache_dir": os.environ.get("PRC_MODEL_CACHE_DIR", ""),
            "T": self.T,
            "n": self.n,
            "kv_cache_implementation": self.kv_cache_implementation,
            "kv_cache_version": kv_cache_version(
                self.kv_cache_implementation
            ),
        }

    @modal.method()
    def generate_wm(self, request) -> dict:
        import time

        import numpy as np
        import torch
        from detectors import semantic_sha256
        from online_prc import derive_document_seed, target_row_count

        if isinstance(request, dict):
            prompt_indices = [int(index) for index in request["prompt_indices"]]
            resume_source_tag = str(request.get("resume_source_tag", ""))
        else:
            prompt_indices = [int(index) for index in request]
            resume_source_tag = ""
        data_vol.reload()
        directory = wm_dir(self.tag)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"wm_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}

        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        prompt_batch = self._prompt_batch(todo)
        document_seeds = [
            derive_document_seed(self.experiment_seed, index) for index in todo
        ]
        resume_source_T = 0
        source_records = []
        source_segments = []
        source_record_fingerprints = []
        prefix_tokens = None
        prefix_codeword = None
        if resume_source_tag:
            source_artifact = torch.load(
                artifact_path(resume_source_tag),
                weights_only=False,
                map_location="cpu",
            )
            incompatibility = artifact_compatibility_error(
                self.artifact, source_artifact
            )
            if incompatibility:
                raise ValueError(
                    f"resume source {resume_source_tag} is incompatible: "
                    f"{incompatibility}"
                )
            resume_source_T = int(source_artifact["T"])
            if not 0 < resume_source_T < self.T:
                raise ValueError(
                    f"resume source T={resume_source_T} must be shorter than "
                    f"target T={self.T}"
                )
            for index in todo:
                source_path = os.path.join(
                    wm_dir(resume_source_tag), f"wm_{index:04d}.pt"
                )
                source_record = torch.load(
                    source_path, weights_only=False, map_location="cpu"
                )
                segments = validate_online_watermarked_record(
                    source_record, source_artifact, index
                )
                source_records.append(source_record)
                source_segments.append(segments)
                source_record_fingerprints.append(
                    semantic_sha256(source_record)
                )
            prefix_tokens = torch.stack([
                torch.as_tensor(record["tokens"], dtype=torch.long).reshape(-1)
                for record in source_records
            ])
            prefix_codeword = np.stack([
                np.asarray(record["prc_codeword_bits"], dtype=np.uint8)
                for record in source_records
            ])

        suffix_tokens, suffix_p_traces, details = (
            self.we.generate_batch_and_collect_online(
                self.we.model,
                prompt_batch,
                self.T,
                self.key,
                self.partition,
                watermark=True,
                return_trace_details=True,
                document_seeds=document_seeds,
                prefix_tokens_batch=prefix_tokens,
                prefix_codeword_bits_batch=prefix_codeword,
                kv_cache_implementation=self.kv_cache_implementation,
            )
        )
        if resume_source_tag:
            tokens = torch.cat([prefix_tokens, suffix_tokens], dim=1)
            p_traces = np.concatenate([
                np.stack([
                    np.asarray(record["p_trace"], dtype=np.float64)
                    for record in source_records
                ]),
                suffix_p_traces,
            ], axis=1)
            full_codeword = np.concatenate([
                prefix_codeword,
                details["prc_codeword_bits"],
            ], axis=1)
            full_entropy = np.concatenate([
                np.stack([
                    np.asarray(record["base_lm_entropy"], dtype=np.float32)
                    for record in source_records
                ]),
                details["base_lm_entropy"],
            ], axis=1)
            full_logprob = np.concatenate([
                np.stack([
                    np.asarray(record["base_token_logprob"], dtype=np.float32)
                    for record in source_records
                ]),
                details["base_token_logprob"],
            ], axis=1)
        else:
            tokens = suffix_tokens
            p_traces = suffix_p_traces
            full_codeword = details["prc_codeword_bits"]
            full_entropy = details["base_lm_entropy"]
            full_logprob = details["base_token_logprob"]

        if int(tokens.shape[1]) != self.T:
            raise AssertionError(
                f"assembled online record length {tokens.shape[1]} != {self.T}"
            )
        realized_r = target_row_count(self.T, self.key)
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row],
                tokens[row],
                p_traces[row],
                self.partition_cpu,
                self.T,
                True,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=full_codeword[row],
                base_lm_entropy=full_entropy[row],
                base_token_logprob=full_logprob[row],
                partition_fingerprint=self.partition_fingerprint,
            )
            if resume_source_tag:
                segments = [dict(segment) for segment in source_segments[row]]
                segments.append({
                    "start": resume_source_T,
                    "end": self.T,
                    "sampler_version": GENERATION_SAMPLER_VERSION,
                    "mode": "continued_suffix",
                    "kv_cache_implementation": self.kv_cache_implementation,
                    "kv_cache_version": kv_cache_version(
                        self.kv_cache_implementation
                    ),
                    "source_tag": resume_source_tag,
                    "source_T": resume_source_T,
                    "source_record_sha256": source_record_fingerprints[row],
                })
                reuse_mode = "continued_from_shorter"
                sampler_label = "segmented_v1"
            else:
                segments = [{
                    "start": 0,
                    "end": self.T,
                    "sampler_version": GENERATION_SAMPLER_VERSION,
                    "mode": "fresh",
                    "kv_cache_implementation": self.kv_cache_implementation,
                    "kv_cache_version": kv_cache_version(
                        self.kv_cache_implementation
                    ),
                }]
                reuse_mode = "fresh"
                sampler_label = GENERATION_SAMPLER_VERSION
            validate_generation_segments(segments, self.T)
            record.update({
                "prompt_idx": int(index),
                "scheme": SCHEME,
                "stopping_policy": STOPPING_POLICY,
                "realized_length": self.T,
                "realized_r": int(realized_r),
                "free_coordinates": int(self.T - realized_r),
                "schedule_version": self.key.schedule_version,
                "support_sampler_version": self.key.support_sampler_version,
                "online_key_sha256": self.key_fingerprint,
                "online_support_sha256": self.support_fingerprint,
                "generation_model_size": self.model_size,
                "generation_model": model_display(self.model_size),
                "generation_model_variant": "base",
                "kv_cache_implementation": self.kv_cache_implementation,
                "kv_cache_version": kv_cache_version(
                    self.kv_cache_implementation
                ),
                "artifact_seed": self.experiment_seed,
                "artifact_fingerprint": self.artifact_fingerprint,
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
                "online_sampler_version": sampler_label,
                "generation_segments": segments,
                "watermarked_cache_mode": reuse_mode,
                "resume_source_tag": resume_source_tag or None,
                "resume_source_T": resume_source_T or None,
            })
            record["trace_semantics"]["prc_codeword_bits"] = (
                "exact noisy causal online PRC bits sampled per coordinate"
            )
            torch.save(record, os.path.join(directory, f"wm_{index:04d}.pt"))
        data_vol.commit()
        peak_allocated_bytes = 0
        peak_reserved_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
        return {
            "generated": len(todo),
            "cached": len(prompt_indices) - len(todo),
            "batch": len(todo),
            "resumed": bool(resume_source_tag),
            "resume_prefix_T": resume_source_T,
            "suffix_tokens_generated": len(todo) * (self.T - resume_source_T),
            "kv_cache_implementation": self.kv_cache_implementation,
            "kv_cache_version": kv_cache_version(
                self.kv_cache_implementation
            ),
            "peak_cuda_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_reserved_bytes": peak_reserved_bytes,
            "seconds": time.time() - started,
        }

    @modal.method()
    def validate_kv_cache_runtime(self, prompt_indices: list[int],
                                  prefix_T: int) -> dict:
        """A/B both cache paths in one loaded model without saving records."""
        import time

        import numpy as np
        import torch
        from online_prc import derive_document_seed

        prefix_T = int(prefix_T)
        if not 0 < prefix_T < self.T:
            raise ValueError("prefix_T must be strictly between zero and T")
        prompt_indices = [int(index) for index in prompt_indices]
        prompt_batch = self._prompt_batch(prompt_indices)
        document_seeds = [
            derive_document_seed(self.experiment_seed, index)
            for index in prompt_indices
        ]

        def generate(implementation, prefix_tokens=None, prefix_bits=None):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            started = time.time()
            tokens, p_traces, details = (
                self.we.generate_batch_and_collect_online(
                    self.we.model,
                    prompt_batch,
                    self.T,
                    self.key,
                    self.partition,
                    watermark=True,
                    return_trace_details=True,
                    document_seeds=document_seeds,
                    prefix_tokens_batch=prefix_tokens,
                    prefix_codeword_bits_batch=prefix_bits,
                    kv_cache_implementation=implementation,
                )
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return {
                "tokens": tokens,
                "p_trace": p_traces,
                "details": details,
                "seconds": time.time() - started,
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated())
                    if torch.cuda.is_available() else 0
                ),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available() else 0
                ),
            }

        concat = generate("concat")
        static = generate("static")
        resumed_suffix = generate(
            "static",
            prefix_tokens=static["tokens"][:, :prefix_T],
            prefix_bits=static["details"]["prc_codeword_bits"][:, :prefix_T],
        )
        resumed = {
            "tokens": torch.cat([
                static["tokens"][:, :prefix_T],
                resumed_suffix["tokens"],
            ], dim=1),
            "p_trace": np.concatenate([
                static["p_trace"][:, :prefix_T],
                resumed_suffix["p_trace"],
            ], axis=1),
            "details": {},
        }
        for field in (
            "prc_codeword_bits", "base_lm_entropy", "base_token_logprob"
        ):
            resumed["details"][field] = np.concatenate([
                static["details"][field][:, :prefix_T],
                resumed_suffix["details"][field],
            ], axis=1)

        def exact_comparison(left, right):
            values = {
                "tokens": (
                    np.asarray(left["tokens"]), np.asarray(right["tokens"])
                ),
                "p_trace": (left["p_trace"], right["p_trace"]),
            }
            for field in (
                "prc_codeword_bits", "base_lm_entropy", "base_token_logprob"
            ):
                values[field] = (
                    left["details"][field], right["details"][field]
                )
            fields = {}
            for field, (left_values, right_values) in values.items():
                left_values = np.asarray(left_values)
                right_values = np.asarray(right_values)
                exact = bool(np.array_equal(left_values, right_values))
                result = {"exact_equal": exact}
                if not exact and left_values.shape == right_values.shape:
                    unequal = np.flatnonzero(
                        left_values.reshape(-1) != right_values.reshape(-1)
                    )
                    result["first_mismatch_flat_index"] = int(unequal[0])
                    if np.issubdtype(left_values.dtype, np.number):
                        result["max_abs_difference"] = float(np.max(np.abs(
                            left_values.astype(np.float64)
                            - right_values.astype(np.float64)
                        )))
                fields[field] = result
            return {
                "all_exact": all(
                    result["exact_equal"] for result in fields.values()
                ),
                "fields": fields,
            }

        return {
            "T": self.T,
            "prefix_T": prefix_T,
            "prompt_indices": prompt_indices,
            "concat_vs_static_direct": exact_comparison(concat, static),
            "static_direct_vs_resumed": exact_comparison(static, resumed),
            "metrics": {
                "concat_direct": {
                    key: concat[key] for key in (
                        "seconds", "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    )
                },
                "static_direct": {
                    key: static[key] for key in (
                        "seconds", "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    )
                },
                "static_resumed_suffix": {
                    key: resumed_suffix[key] for key in (
                        "seconds", "peak_cuda_allocated_bytes",
                        "peak_cuda_reserved_bytes",
                    )
                },
            },
        }

    @modal.method()
    def generate_null(self, prompt_indices: list[int]) -> dict:
        import time

        import torch

        data_vol.reload()
        directory = shared_null_dir(self.T, self.model_size)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}
        started = time.time()
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect_online(
            self.we.model,
            prompt_batch,
            self.T,
            self.key,
            self.partition,
            watermark=False,
            return_trace_details=True,
            # Shared null caches retain the established concatenating path;
            # static-cache validation is isolated to online watermarked data.
            kv_cache_implementation=DEFAULT_KV_CACHE_IMPLEMENTATION,
        )
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row],
                tokens[row],
                p_traces[row],
                self.partition_cpu,
                self.T,
                False,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=None,
                base_lm_entropy=details["base_lm_entropy"][row],
                base_token_logprob=details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": int(index),
                "generation_model_size": self.model_size,
                "generation_model": model_display(self.model_size),
                "generation_model_variant": "base",
            })
            torch.save(record, os.path.join(directory, f"null_{index:04d}.pt"))
        data_vol.commit()
        return {
            "generated": len(todo),
            "cached": len(prompt_indices) - len(todo),
            "batch": len(todo),
            "seconds": time.time() - started,
        }


@app.function(volumes={"/data": data_vol}, timeout=600)
def audit_continuation(target_tag: str, source_tag: str,
                       num_prompts: int = 2) -> dict:
    """Independently audit a shorter-to-longer saved continuation."""
    import numpy as np
    import torch
    from detectors import semantic_sha256
    from online_prc import (
        OnlinePRCEncoder,
        OnlinePRCKey,
        derive_document_seed,
    )

    data_vol.reload()
    target_artifact = torch.load(
        artifact_path(target_tag), weights_only=False, map_location="cpu"
    )
    source_artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    incompatibility = artifact_compatibility_error(
        target_artifact, source_artifact
    )
    if incompatibility:
        raise ValueError(f"audit artifacts are incompatible: {incompatibility}")
    target_T = int(target_artifact["T"])
    source_T = int(source_artifact["T"])
    if not 0 < source_T < target_T:
        raise ValueError("audit requires 0 < source_T < target_T")

    key = OnlinePRCKey.from_dict(target_artifact["online_key"])
    experiment_seed = int(target_artifact.get("experiment_seed", SEED))
    if int(num_prompts) <= 0 or int(num_prompts) > CANONICAL_NUM_PROMPTS:
        raise ValueError("num_prompts is outside the canonical prompt range")
    prompt_indices = list(range(int(num_prompts)))
    audited = []
    fields = (
        "tokens",
        "p_trace",
        "base_lm_entropy",
        "base_token_logprob",
        "prc_codeword_bits",
    )
    for index in prompt_indices:
        source_record = torch.load(
            os.path.join(wm_dir(source_tag), f"wm_{index:04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        target_record = torch.load(
            os.path.join(wm_dir(target_tag), f"wm_{index:04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        validate_online_watermarked_record(
            source_record, source_artifact, index
        )
        target_segments = validate_online_watermarked_record(
            target_record, target_artifact, index
        )
        for field in fields:
            source_values = np.asarray(source_record[field]).reshape(-1)
            target_values = np.asarray(target_record[field]).reshape(-1)
            if not np.array_equal(source_values, target_values[:source_T]):
                raise AssertionError(
                    f"prompt {index} field {field!r} did not preserve its prefix"
                )

        expected_bits = OnlinePRCEncoder(
            key, [derive_document_seed(experiment_seed, index)]
        ).encode_to_length(target_T)[0]
        actual_bits = np.asarray(
            target_record["prc_codeword_bits"], dtype=np.uint8
        )
        if not np.array_equal(expected_bits, actual_bits):
            raise AssertionError(
                f"prompt {index} target causal bitstream is not reproducible"
            )
        final_segment = target_segments[-1]
        if (
            int(final_segment["start"]) != source_T
            or int(final_segment["end"]) != target_T
            or final_segment.get("sampler_version")
            != GENERATION_SAMPLER_VERSION
            or final_segment.get("source_record_sha256")
            != semantic_sha256(source_record)
        ):
            raise AssertionError(
                f"prompt {index} continuation provenance is incomplete"
            )
        audited.append({
            "prompt_idx": int(index),
            "prefix_fields_equal": list(fields),
            "causal_bits_reproduced": True,
            "source_record_sha256": semantic_sha256(source_record),
        })
    return {
        "target_tag": target_tag,
        "source_tag": source_tag,
        "target_T": target_T,
        "source_T": source_T,
        "suffix_length": target_T - source_T,
        "audited": audited,
    }


@app.function(volumes={"/data": data_vol}, timeout=600)
def prepare_sweep_ceiling(target_tag: str, reference_tag: str,
                          prompt_indices: list[int]) -> dict:
    """Pin a sweep ceiling to one canonical shorter sampler-v2 cache.

    Existing target records with a different realized prefix are moved into a
    recoverable quarantine directory. Missing/quarantined records can then be
    regenerated from ``reference_tag`` without touching compatible records.
    """
    import numpy as np
    import torch

    data_vol.reload()
    target_artifact = torch.load(
        artifact_path(target_tag), weights_only=False, map_location="cpu"
    )
    reference_artifact = torch.load(
        artifact_path(reference_tag), weights_only=False, map_location="cpu"
    )
    incompatibility = artifact_compatibility_error(
        target_artifact, reference_artifact
    )
    if incompatibility:
        raise ValueError(
            f"sweep reference {reference_tag} is incompatible with "
            f"{target_tag}: {incompatibility}"
        )
    target_T = int(target_artifact["T"])
    reference_T = int(reference_artifact["T"])
    if not 0 < reference_T < target_T:
        raise ValueError(
            "sweep reference must be strictly shorter than the ceiling"
        )

    fields = (
        "tokens",
        "p_trace",
        "base_lm_entropy",
        "base_token_logprob",
        "prc_codeword_bits",
    )
    compatible = []
    missing = []
    quarantined = []
    quarantine_dir = os.path.join(
        os.path.dirname(wm_dir(target_tag)),
        "quarantine",
        f"prefix-mismatch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    for index in prompt_indices:
        reference_path = os.path.join(
            wm_dir(reference_tag), f"wm_{index:04d}.pt"
        )
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(
                f"canonical sweep reference is missing prompt {index}: "
                f"{reference_path}"
            )
        reference_record = torch.load(
            reference_path, weights_only=False, map_location="cpu"
        )
        reference_segments = validate_online_watermarked_record(
            reference_record, reference_artifact, index
        )
        if any(
            segment.get("sampler_version") != GENERATION_SAMPLER_VERSION
            for segment in reference_segments
        ):
            raise ValueError(
                f"canonical sweep reference prompt {index} contains a "
                "non-sampler-v2 generation segment"
            )

        target_path = os.path.join(wm_dir(target_tag), f"wm_{index:04d}.pt")
        if not os.path.isfile(target_path):
            missing.append(int(index))
            continue
        mismatch = None
        try:
            target_record = torch.load(
                target_path, weights_only=False, map_location="cpu"
            )
            validate_online_watermarked_record(
                target_record, target_artifact, index
            )
            for field in fields:
                reference_values = np.asarray(
                    reference_record[field]
                ).reshape(-1)
                target_values = np.asarray(target_record[field]).reshape(-1)
                if not np.array_equal(
                    reference_values, target_values[:reference_T]
                ):
                    mismatch = field
                    break
        except Exception as exc:
            mismatch = f"validation:{type(exc).__name__}:{exc}"

        if mismatch is None:
            compatible.append(int(index))
            continue
        os.makedirs(quarantine_dir, exist_ok=True)
        quarantined_path = os.path.join(
            quarantine_dir, f"wm_{index:04d}.pt"
        )
        os.replace(target_path, quarantined_path)
        quarantined.append({
            "prompt_idx": int(index),
            "reason": mismatch,
            "original_path": target_path,
            "quarantined_path": quarantined_path,
        })
        missing.append(int(index))

    if quarantined:
        data_vol.commit()
    return {
        "target_tag": target_tag,
        "target_T": target_T,
        "reference_tag": reference_tag,
        "reference_T": reference_T,
        "compatible_count": len(compatible),
        "missing_prompt_indices": missing,
        "quarantined": quarantined,
        "quarantine_dir": quarantine_dir if quarantined else None,
    }


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(tag: str, prompt_indices: list[int], null_T: int, fpr: float,
               batch: int, code_fingerprint_sha256: str,
               wm_source_tag: str = "",
               wm_reuse_mode: str = "exact_cache",
               wm_resume_source_tag: str = "",
               wm_resume_source_T: int = 0) -> dict:
    import torch
    from detectors import detect_online_hoeffding
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    T = int(artifact["T"])
    expected_support = support_sha256(T, key)
    wm_source_tag = wm_source_tag or tag
    source_artifact = torch.load(
        artifact_path(wm_source_tag), weights_only=False, map_location="cpu"
    )
    incompatibility = artifact_compatibility_error(artifact, source_artifact)
    if incompatibility:
        raise ValueError(
            f"watermarked cache {wm_source_tag} is incompatible with {tag}: "
            f"{incompatibility}"
        )
    wm_source_T = int(source_artifact["T"])
    if wm_source_T < T:
        raise ValueError(
            f"watermarked detection cache T={wm_source_T} is shorter than T={T}"
        )
    results = []

    for watermark in (True, False):
        directory = (
            wm_dir(wm_source_tag)
            if watermark
            else shared_null_dir(null_T, generation_model_size)
        )
        prefix = "wm" if watermark else "null"
        for index in prompt_indices:
            path = os.path.join(directory, f"{prefix}_{index:04d}.pt")
            record = torch.load(path, weights_only=False, map_location="cpu")
            if len(record["tokens"]) < T or len(record["p_trace"]) < T:
                raise ValueError(f"record {path} is shorter than T={T}")
            if watermark:
                validate_online_watermarked_record(
                    record, source_artifact, index
                )
            else:
                validate_online_null_record(
                    record, artifact, index, T
                )

            tokens = record["tokens"][:T]
            probabilities = record["p_trace"][:T]
            scored = {}
            for weight in ("map", "entropy", "naive"):
                decision, info = detect_online_hoeffding(
                    key,
                    tokens,
                    probabilities,
                    partition,
                    fpr=fpr,
                    weight=weight,
                    fpr_policy=FPR_POLICY,
                    return_info=True,
                )
                scored[weight] = {"decision": bool(decision), **info}
            results.append({
                "prompt_idx": int(index),
                "watermark": watermark,
                "scores": scored,
            })

    wm = [result for result in results if result["watermark"]]
    null = [result for result in results if not result["watermark"]]
    counts = {}
    for weight in ("map", "entropy", "naive"):
        counts[weight] = {
            "tp": sum(result["scores"][weight]["decision"] for result in wm),
            "fp": sum(result["scores"][weight]["decision"] for result in null),
            "watermarked_total": len(wm),
            "null_total": len(null),
        }
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "tag": tag,
        "n": T,
        "T": T,
        "t": key.check_weight,
        "eta": key.noise_rate,
        "r": target_row_count(T, key),
        "free_coordinates": T - target_row_count(T, key),
        "row_rate_numerator": key.row_rate_numerator,
        "row_rate_denominator": key.row_rate_denominator,
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "stopping_policy": STOPPING_POLICY,
        "fpr_policy": FPR_POLICY,
        "target_fpr": float(fpr),
        "generation_model": model_display(generation_model_size),
        "generation_model_size": generation_model_size,
        "kv_cache_implementation": artifact_kv_cache_implementation(artifact),
        "kv_cache_version": kv_cache_version(
            artifact_kv_cache_implementation(artifact)
        ),
        "num_prompts": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "batch": int(batch),
        "null_cache_T": int(null_T),
        "watermarked_cache_mode": wm_reuse_mode,
        "watermarked_cache_T": wm_source_T,
        "watermarked_cache_tag": wm_source_tag,
        "watermarked_resume_source_tag": wm_resume_source_tag or None,
        "watermarked_resume_source_T": int(wm_resume_source_T) or None,
        "watermarked_source_artifact_fingerprint": source_artifact[
            "artifact_fingerprint"
        ],
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "online_support_sha256": expected_support,
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "counts": counts,
        "results": results,
    }
    output_dir = f"/data/{tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"fpr-{_slug(f'{float(fpr):.12g}')}_prompts-{len(prompt_indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_saved_prefix(source_tag: str, prefix_T: int,
                        prompt_indices: list[int], null_T: int,
                        fpr: float, code_fingerprint_sha256: str) -> dict:
    """Detect a shorter causal prefix from already-saved longer records.

    No text is generated here.  Prefix consistency lets the detector rebuild
    the length-``prefix_T`` supports and OTP from the source artifact's compact
    online key, while tokens and probability traces are sliced from disk.
    """
    import torch
    from detectors import detect_online_hoeffding
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    data_vol.reload()
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    source_T = int(artifact["T"])
    prefix_T = int(prefix_T)
    if prefix_T <= 0 or prefix_T > source_T:
        raise ValueError(
            f"prefix_T must be in [1, source_T={source_T}], got {prefix_T}"
        )
    if int(null_T) < prefix_T:
        raise ValueError(
            f"null cache T={null_T} is shorter than prefix_T={prefix_T}"
        )

    expected_source_support = support_sha256(source_T, key)
    prefix_support = support_sha256(prefix_T, key)
    results = []
    for watermark in (True, False):
        directory = (
            wm_dir(source_tag)
            if watermark
            else shared_null_dir(null_T, generation_model_size)
        )
        record_prefix = "wm" if watermark else "null"
        for index in prompt_indices:
            path = os.path.join(
                directory, f"{record_prefix}_{index:04d}.pt"
            )
            record = torch.load(path, weights_only=False, map_location="cpu")
            if len(record["tokens"]) < prefix_T:
                raise ValueError(
                    f"record {path} has {len(record['tokens'])} tokens; "
                    f"need prefix_T={prefix_T}"
                )
            if len(record["p_trace"]) < prefix_T:
                raise ValueError(
                    f"record {path} has {len(record['p_trace'])} p-trace "
                    f"values; need prefix_T={prefix_T}"
                )
            if watermark:
                validate_online_watermarked_record(record, artifact, index)
            else:
                validate_online_null_record(
                    record, artifact, index, prefix_T
                )

            tokens = record["tokens"][:prefix_T]
            probabilities = record["p_trace"][:prefix_T]
            scored = {}
            for weight in ("map", "entropy", "naive"):
                decision, info = detect_online_hoeffding(
                    key,
                    tokens,
                    probabilities,
                    partition,
                    fpr=fpr,
                    weight=weight,
                    fpr_policy=FPR_POLICY,
                    return_info=True,
                )
                if int(info["length"]) != prefix_T:
                    raise AssertionError("prefix detector used the wrong length")
                scored[weight] = {"decision": bool(decision), **info}
            results.append({
                "prompt_idx": int(index),
                "watermark": watermark,
                "scores": scored,
            })

    wm_results = [result for result in results if result["watermark"]]
    null_results = [result for result in results if not result["watermark"]]
    counts = {}
    for weight in ("map", "entropy", "naive"):
        counts[weight] = {
            "tp": sum(
                result["scores"][weight]["decision"]
                for result in wm_results
            ),
            "fp": sum(
                result["scores"][weight]["decision"]
                for result in null_results
            ),
            "watermarked_total": len(wm_results),
            "null_total": len(null_results),
        }

    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "saved_longer_prefix_redetect",
        "source_tag": source_tag,
        "source_T": source_T,
        "n": prefix_T,
        "T": prefix_T,
        "t": key.check_weight,
        "eta": key.noise_rate,
        "r": target_row_count(prefix_T, key),
        "free_coordinates": prefix_T - target_row_count(prefix_T, key),
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "stopping_policy": STOPPING_POLICY,
        "fpr_policy": FPR_POLICY,
        "target_fpr": float(fpr),
        "generation_model": model_display(generation_model_size),
        "generation_model_size": generation_model_size,
        "kv_cache_implementation": artifact_kv_cache_implementation(artifact),
        "kv_cache_version": kv_cache_version(
            artifact_kv_cache_implementation(artifact)
        ),
        "num_prompts": len(prompt_indices),
        "prompt_indices": prompt_indices,
        "null_cache_T": int(null_T),
        "source_artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "source_online_support_sha256": expected_source_support,
        "prefix_online_support_sha256": prefix_support,
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "counts": counts,
        "results": results,
    }
    output_dir = f"/data/{source_tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"prefix-T{prefix_T}_fpr-{_slug(f'{float(fpr):.12g}')}"
        f"_prompts-{len(prompt_indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.function(
    cpu=1.0,
    volumes={"/data": data_vol},
    timeout=1800,
)
def prepare_map_prefix_shard(request: dict) -> dict:
    """Prepare longest-prefix MAP contributions for one prompt shard."""
    import time

    import numpy as np
    import torch
    from detectors import (
        prepare_online_map_prefix_context,
        prepare_online_map_prefix_trace,
    )
    from online_prc import OnlinePRCKey, target_row_count

    started = time.time()
    source_tag = str(request["source_tag"])
    prompt_indices = [int(index) for index in request["prompt_indices"]]
    maximum = int(request["maximum_length"])
    code_fingerprint = str(request["code_fingerprint_sha256"])
    if not prompt_indices or len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("prompt shard indices must be nonempty and unique")

    data_vol.reload()
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    source_T = int(artifact["T"])
    if maximum <= 0 or maximum > source_T:
        raise ValueError(
            f"maximum_length must be in [1, source_T={source_T}]"
        )
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    row_count = target_row_count(maximum, key)
    output_path = prepared_map_shard_path(
        source_tag,
        maximum,
        prompt_indices,
        artifact["artifact_fingerprint"],
        code_fingerprint,
    )

    if os.path.isfile(output_path):
        cached = torch.load(
            output_path, weights_only=False, map_location="cpu"
        )
        cached_indices = validate_prepared_map_shard(
            cached,
            source_tag=source_tag,
            maximum_length=maximum,
            artifact_fingerprint=artifact["artifact_fingerprint"],
            online_key_sha256=key.fingerprint,
            code_fingerprint_sha256=code_fingerprint,
            expected_row_count=row_count,
        )
        if cached_indices != prompt_indices:
            raise ValueError("cached prepared shard prompt order is incorrect")
        return {
            "remote_output_path": output_path,
            "prompt_indices": prompt_indices,
            "num_prompts": len(prompt_indices),
            "cached": True,
            "seconds": time.time() - started,
        }

    partition = artifact["partition"]
    prepared_context = prepare_online_map_prefix_context(key, maximum)
    records = []
    for index in prompt_indices:
        path = os.path.join(wm_dir(source_tag), f"wm_{index:04d}.pt")
        record = torch.load(path, weights_only=False, map_location="cpu")
        validate_online_watermarked_record(record, artifact, index)
        prepared = prepare_online_map_prefix_trace(
            key,
            record["tokens"],
            record["p_trace"],
            partition,
            maximum,
            prepared_context=prepared_context,
        )
        records.append({
            "prompt_idx": int(index),
            "signed_check_values": np.asarray(
                prepared["signed_check_values"], dtype=np.float64
            ).copy(),
            "squared_check_values": np.asarray(
                prepared["squared_check_values"], dtype=np.float64
            ).copy(),
        })

    elapsed = time.time() - started
    payload = {
        "prepared_map_shard_schema_version": (
            PREPARED_MAP_SHARD_SCHEMA_VERSION
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "online_map_prepared_prompt_shard",
        "source_tag": source_tag,
        "source_T": source_T,
        "maximum_length": maximum,
        "prompt_indices": prompt_indices,
        "num_prompts": len(prompt_indices),
        "source_artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
        "row_count": int(row_count),
        "cpu_preparation_seconds": elapsed,
        "records": records,
    }
    validate_prepared_map_shard(
        payload,
        source_tag=source_tag,
        maximum_length=maximum,
        artifact_fingerprint=artifact["artifact_fingerprint"],
        online_key_sha256=key.fingerprint,
        code_fingerprint_sha256=code_fingerprint,
        expected_row_count=row_count,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    if not os.path.isfile(output_path):
        raise IOError(f"failed to persist prepared MAP shard {output_path}")
    data_vol.commit()
    return {
        "remote_output_path": output_path,
        "prompt_indices": prompt_indices,
        "num_prompts": len(prompt_indices),
        "cached": False,
        "seconds": elapsed,
    }


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def detect_map_prefix_grid_serial(source_tag: str,
                                  prefix_lengths: list[int],
                                  prompt_indices: list[int], fpr: float,
                                  code_fingerprint_sha256: str,
                                  target_map_tpr: float,
                                  stop_after_first_below: bool = True,
                                  persist_results: bool = True) -> dict:
    """Serial reference path retained for exact sharding validation."""
    import time

    import torch
    from detectors import (
        prepare_online_map_prefix_context,
        prepare_online_map_prefix_trace,
    )
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    started = time.time()
    data_vol.reload()
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    source_T = int(artifact["T"])
    lengths = [int(length) for length in prefix_lengths]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("prefix_lengths must be nonempty and unique")
    if any(length <= 0 or length > source_T for length in lengths):
        raise ValueError(
            f"every prefix length must be in [1, source_T={source_T}]"
        )
    if not prompt_indices:
        raise ValueError("prompt_indices must be nonempty")
    if stop_after_first_below and lengths != sorted(lengths, reverse=True):
        raise ValueError(
            "adaptive prefix lengths must be ordered longest to shortest"
        )
    rate_strictly_above(0, 1, target_map_tpr)

    expected_source_support = support_sha256(source_T, key)
    prepared_records = []
    maximum = max(lengths)
    prepared_context = prepare_online_map_prefix_context(key, maximum)
    for index in prompt_indices:
        path = os.path.join(wm_dir(source_tag), f"wm_{index:04d}.pt")
        record = torch.load(path, weights_only=False, map_location="cpu")
        validate_online_watermarked_record(record, artifact, index)
        prepared = prepare_online_map_prefix_trace(
            key,
            record["tokens"],
            record["p_trace"],
            partition,
            maximum,
            prepared_context=prepared_context,
        )
        prepared_records.append({
            "prompt_idx": int(index),
            "prepared": prepared,
        })

    adaptive = evaluate_prepared_map_prefixes(
        prepared_records,
        lengths,
        fpr,
        target_map_tpr,
        stop_after_first_below=stop_after_first_below,
    )
    rows = adaptive["rows"]
    for row in rows:
        length = int(row["n"])
        row_count = target_row_count(length, key)
        if int(row["r"]) != int(row_count):
            raise AssertionError("prepared detector used the wrong row count")
        row["prefix_online_support_sha256"] = support_sha256(length, key)
    prompt_results = adaptive["results"]
    evaluated_lengths = adaptive["evaluated_lengths"]

    elapsed = time.time() - started
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "saved_online_map_prefix_grid",
        "source_tag": source_tag,
        "source_T": source_T,
        "requested_prefix_lengths": lengths,
        "prefix_lengths": evaluated_lengths,
        "unevaluated_prefix_lengths": adaptive["unevaluated_lengths"],
        "target_map_tpr": float(target_map_tpr),
        "target_comparison": "strictly_greater_than",
        "stop_after_first_below": adaptive["stop_after_first_below"],
        "stopped_after_first_below": adaptive["stopped_after_first_below"],
        "first_below_n": adaptive["first_below_n"],
        "t": key.check_weight,
        "eta": key.noise_rate,
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "stopping_policy": STOPPING_POLICY,
        "fpr_policy": FPR_POLICY,
        "target_fpr": float(fpr),
        "generation_model": model_display(generation_model_size),
        "generation_model_size": generation_model_size,
        "kv_cache_implementation": artifact_kv_cache_implementation(artifact),
        "kv_cache_version": kv_cache_version(
            artifact_kv_cache_implementation(artifact)
        ),
        "num_prompts": len(prompt_indices),
        "prompt_indices": [int(index) for index in prompt_indices],
        "source_artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "source_online_support_sha256": expected_source_support,
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "cpu_detection_seconds": elapsed,
        "rows": rows,
        "results": prompt_results,
    }
    if not persist_results:
        payload["prefix_result_paths"] = {}
        return {"payload": payload, "remote_output_path": None}
    output_dir = f"/data/{source_tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    prefix_result_paths = {}
    for length in evaluated_lengths:
        increment_payload = increment_payload_from_grid(payload, length)
        increment_path = os.path.join(
            output_dir,
            f"map-prefix-T{length}_fpr-{_slug(f'{float(fpr):.12g}')}"
            f"_prompts-{len(prompt_indices)}.pt",
        )
        torch.save(increment_payload, increment_path)
        if not os.path.isfile(increment_path):
            raise IOError(
                f"failed to persist MAP prefix result for T={length}: "
                f"{increment_path}"
            )
        prefix_result_paths[str(length)] = increment_path
    payload["prefix_result_paths"] = prefix_result_paths
    grid_label = (
        f"{min(evaluated_lengths)}-{max(evaluated_lengths)}"
        f"-count{len(evaluated_lengths)}"
    )
    output_path = os.path.join(
        output_dir,
        f"map-prefix-grid-{grid_label}_fpr-"
        f"{_slug(f'{float(fpr):.12g}')}_prompts-{len(prompt_indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def aggregate_map_prefix_shards(
    source_tag: str,
    prefix_lengths: list[int],
    prompt_indices: list[int],
    fpr: float,
    code_fingerprint_sha256: str,
    target_map_tpr: float,
    prepared_shards: list[dict],
    preparation_wall_seconds: float,
    stop_after_first_below: bool = True,
) -> dict:
    """Merge prepared prompt shards, adaptively score, and persist once."""
    import time

    import torch
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    started = time.time()
    data_vol.reload()
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    source_T = int(artifact["T"])
    lengths = [int(length) for length in prefix_lengths]
    indices = [int(index) for index in prompt_indices]
    if not lengths or len(set(lengths)) != len(lengths):
        raise ValueError("prefix_lengths must be nonempty and unique")
    if any(length <= 0 or length > source_T for length in lengths):
        raise ValueError(
            f"every prefix length must be in [1, source_T={source_T}]"
        )
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("prompt_indices must be nonempty and unique")
    if stop_after_first_below and lengths != sorted(lengths, reverse=True):
        raise ValueError(
            "adaptive prefix lengths must be ordered longest to shortest"
        )
    rate_strictly_above(0, 1, target_map_tpr)

    maximum = max(lengths)
    row_count = target_row_count(maximum, key)
    shard_payloads = []
    shard_inventory = []
    for summary in prepared_shards:
        path = str(summary["remote_output_path"])
        payload = torch.load(path, weights_only=False, map_location="cpu")
        validated_indices = validate_prepared_map_shard(
            payload,
            source_tag=source_tag,
            maximum_length=maximum,
            artifact_fingerprint=artifact["artifact_fingerprint"],
            online_key_sha256=key.fingerprint,
            code_fingerprint_sha256=code_fingerprint_sha256,
            expected_row_count=row_count,
        )
        if validated_indices != [
            int(index) for index in summary["prompt_indices"]
        ]:
            raise ValueError(
                f"prepared shard summary disagrees with payload at {path}"
            )
        shard_payloads.append(payload)
        shard_inventory.append({
            "remote_output_path": path,
            "prompt_indices": validated_indices,
            "num_prompts": len(validated_indices),
            "cached_this_invocation": bool(summary.get("cached", False)),
            "invocation_seconds": float(summary.get("seconds", 0.0)),
            "original_cpu_preparation_seconds": float(
                payload.get("cpu_preparation_seconds", 0.0)
            ),
        })

    prepared_records = merge_prepared_map_shards(
        shard_payloads,
        indices,
        key,
        maximum,
    )
    adaptive = evaluate_prepared_map_prefixes(
        prepared_records,
        lengths,
        fpr,
        target_map_tpr,
        stop_after_first_below=stop_after_first_below,
    )
    rows = adaptive["rows"]
    for row in rows:
        length = int(row["n"])
        expected_rows = target_row_count(length, key)
        if int(row["r"]) != int(expected_rows):
            raise AssertionError("prepared detector used the wrong row count")
        row["prefix_online_support_sha256"] = support_sha256(length, key)
    prompt_results = adaptive["results"]
    evaluated_lengths = adaptive["evaluated_lengths"]

    aggregation_seconds = time.time() - started
    expected_source_support = support_sha256(source_T, key)
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "saved_online_map_prefix_grid",
        "detection_strategy": "prompt_sharded_precompute_v1",
        "source_tag": source_tag,
        "source_T": source_T,
        "requested_prefix_lengths": lengths,
        "prefix_lengths": evaluated_lengths,
        "unevaluated_prefix_lengths": adaptive["unevaluated_lengths"],
        "target_map_tpr": float(target_map_tpr),
        "target_comparison": "strictly_greater_than",
        "stop_after_first_below": adaptive["stop_after_first_below"],
        "stopped_after_first_below": adaptive["stopped_after_first_below"],
        "first_below_n": adaptive["first_below_n"],
        "t": key.check_weight,
        "eta": key.noise_rate,
        "schedule_version": key.schedule_version,
        "support_sampler_version": key.support_sampler_version,
        "stopping_policy": STOPPING_POLICY,
        "fpr_policy": FPR_POLICY,
        "target_fpr": float(fpr),
        "generation_model": model_display(generation_model_size),
        "generation_model_size": generation_model_size,
        "kv_cache_implementation": artifact_kv_cache_implementation(artifact),
        "kv_cache_version": kv_cache_version(
            artifact_kv_cache_implementation(artifact)
        ),
        "num_prompts": len(indices),
        "prompt_indices": indices,
        "source_artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "source_online_support_sha256": expected_source_support,
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "cpu_detection_seconds": (
            float(preparation_wall_seconds) + aggregation_seconds
        ),
        "cpu_preparation_wall_seconds": float(preparation_wall_seconds),
        "cpu_preparation_invocation_seconds": sum(
            item["invocation_seconds"] for item in shard_inventory
        ),
        "cpu_aggregation_seconds": aggregation_seconds,
        "prepared_shard_count": len(shard_inventory),
        "prepared_shard_cache_hits": sum(
            item["cached_this_invocation"] for item in shard_inventory
        ),
        "prepared_shards": shard_inventory,
        "rows": rows,
        "results": prompt_results,
    }

    output_dir = f"/data/{source_tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    prefix_result_paths = {}
    for length in evaluated_lengths:
        increment_payload = increment_payload_from_grid(payload, length)
        increment_path = os.path.join(
            output_dir,
            f"map-prefix-T{length}_fpr-{_slug(f'{float(fpr):.12g}')}"
            f"_prompts-{len(indices)}.pt",
        )
        torch.save(increment_payload, increment_path)
        if not os.path.isfile(increment_path):
            raise IOError(
                f"failed to persist MAP prefix result for T={length}: "
                f"{increment_path}"
            )
        prefix_result_paths[str(length)] = increment_path
    payload["prefix_result_paths"] = prefix_result_paths
    grid_label = (
        f"{min(evaluated_lengths)}-{max(evaluated_lengths)}"
        f"-count{len(evaluated_lengths)}"
    )
    output_path = os.path.join(
        output_dir,
        f"map-prefix-grid-{grid_label}_fpr-"
        f"{_slug(f'{float(fpr):.12g}')}_prompts-{len(indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


def _execute_generation_plan(tag: str, plan: dict, batch: int,
                             max_containers: int, gpu: str,
                             code_fingerprint: str,
                             generation_model_size: str = MODEL_SIZE,
                             kv_cache_implementation: str = (
                                 DEFAULT_KV_CACHE_IMPLEMENTATION
                             ),
                             include_null: bool = True,
                             log_prefix: str = "sweep") -> dict:
    """Execute missing generation work selected by ``plan_generation``."""
    generation_meta = {"wm": [], "null": []}
    needs_wm = bool(plan["wm_missing"])
    needs_null = bool(include_null and plan["null_missing"])
    if not needs_wm and not needs_null:
        print(f"[{log_prefix}] all requested generation records cached", flush=True)
        return generation_meta

    from concurrent.futures import ThreadPoolExecutor

    model = OnlineModel.with_options(
        gpu=gpu, max_containers=max_containers
    )(
        tag=tag,
        model_size=normalize_model_size(generation_model_size),
        code_fingerprint_sha256=code_fingerprint,
        kv_cache_implementation=normalize_kv_cache_implementation(
            kv_cache_implementation
        ),
    )
    print(f"[{log_prefix}] model ready: {model.ready.remote()}", flush=True)
    work = []
    if needs_wm:
        requests = [
            {
                "prompt_indices": chunk,
                "resume_source_tag": plan["wm_resume_source_tag"],
            }
            for chunk in _chunks(plan["wm_missing"], batch)
        ]
        work.append(("wm", model.generate_wm, requests))
    if needs_null:
        work.append((
            "null",
            model.generate_null,
            _chunks(plan["null_missing"], batch),
        ))

    def run_map(item):
        name, method, chunks = item
        return name, list(method.map(chunks))

    with ThreadPoolExecutor(max_workers=len(work)) as pool:
        mapped = list(pool.map(run_map, work))
    for name, records in mapped:
        generation_meta[name] = records
        generated = sum(int(record["generated"]) for record in records)
        actual_batches = [
            int(record["batch"]) for record in records if record["batch"]
        ]
        seconds = sum(float(record.get("seconds", 0.0)) for record in records)
        print(
            f"[{log_prefix}] {name}: generated={generated}, "
            f"batch_sizes={actual_batches}, measured_gpu_seconds={seconds:.1f}",
            flush=True,
        )
    return generation_meta


@app.function(volumes={"/data": data_vol}, timeout=600)
def compare_kv_cache_records(n: int, t: int, eta: float,
                             experiment_seed: int,
                             prompt_indices: list[int],
                             generation_model_size: str = MODEL_SIZE) -> dict:
    """Compare isolated concat/static online records field-for-field."""
    import numpy as np
    import torch

    generation_model_size = normalize_model_size(generation_model_size)
    concat_tag = config_tag(
        n, t, eta, experiment_seed, generation_model_size, "concat"
    )
    static_tag = config_tag(
        n, t, eta, experiment_seed, generation_model_size, "static"
    )
    data_vol.reload()
    concat_artifact = torch.load(
        artifact_path(concat_tag), weights_only=False, map_location="cpu"
    )
    static_artifact = torch.load(
        artifact_path(static_tag), weights_only=False, map_location="cpu"
    )
    artifact_checks = {
        "online_key": concat_artifact["online_key"] == static_artifact["online_key"],
        "partition": torch.equal(
            concat_artifact["partition"], static_artifact["partition"]
        ),
        "prompt_corpus": (
            concat_artifact["prompt_ids_list"]
            == static_artifact["prompt_ids_list"]
        ),
        "length": int(concat_artifact["T"]) == int(static_artifact["T"]) == int(n),
    }
    fields = (
        "tokens",
        "p_trace",
        "base_lm_entropy",
        "base_token_logprob",
        "prc_codeword_bits",
        "observed_bucket_bits",
        "map_soft_tokens",
    )
    comparisons = []
    for index in prompt_indices:
        concat_record = torch.load(
            os.path.join(wm_dir(concat_tag), f"wm_{int(index):04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        static_record = torch.load(
            os.path.join(wm_dir(static_tag), f"wm_{int(index):04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        validate_online_watermarked_record(
            concat_record, concat_artifact, int(index)
        )
        validate_online_watermarked_record(
            static_record, static_artifact, int(index)
        )
        field_results = {}
        for field in fields:
            concat_values = np.asarray(concat_record[field])
            static_values = np.asarray(static_record[field])
            equal = bool(np.array_equal(concat_values, static_values))
            result = {"exact_equal": equal}
            if not equal and concat_values.shape == static_values.shape:
                unequal = np.flatnonzero(
                    concat_values.reshape(-1) != static_values.reshape(-1)
                )
                result["first_mismatch_flat_index"] = int(unequal[0])
                result["equal_flat_prefix_length"] = int(unequal[0])
            if (
                not equal
                and np.issubdtype(concat_values.dtype, np.number)
                and concat_values.shape == static_values.shape
            ):
                result["max_abs_difference"] = float(np.max(np.abs(
                    concat_values.astype(np.float64)
                    - static_values.astype(np.float64)
                )))
            field_results[field] = result
        comparisons.append({
            "prompt_idx": int(index),
            "fields": field_results,
            "all_fields_exact": all(
                result["exact_equal"] for result in field_results.values()
            ),
            "concat_cache_mode": concat_record.get("watermarked_cache_mode"),
            "static_cache_mode": static_record.get("watermarked_cache_mode"),
        })

    payload = {
        "comparison_schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n": int(n),
        "t": int(t),
        "eta": float(eta),
        "experiment_seed": int(experiment_seed),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "concat_tag": concat_tag,
        "static_tag": static_tag,
        "prompt_indices": [int(index) for index in prompt_indices],
        "artifact_checks": artifact_checks,
        "comparisons": comparisons,
        "all_exact": (
            all(artifact_checks.values())
            and all(item["all_fields_exact"] for item in comparisons)
        ),
    }
    output_dir = os.path.join("/data", static_tag, "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"kv-cache-equivalence-prompts-{len(prompt_indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.local_entrypoint()
def compare_kv_caches(n: int = 64, num_prompts: int = 2,
                      t: int = 3, eta: float = 0.05,
                      experiment_seed: int = SEED,
                      generation_model_size: str = MODEL_SIZE):
    if n <= 0 or num_prompts <= 0 or num_prompts > CANONICAL_NUM_PROMPTS:
        raise ValueError("n or num_prompts is invalid")
    result = compare_kv_cache_records.remote(
        n,
        t,
        eta,
        experiment_seed,
        list(range(int(num_prompts))),
        normalize_model_size(generation_model_size),
    )
    payload = result["payload"]
    print(
        f"[kv-cache-compare] n={n}, prompts={num_prompts}, "
        f"all_exact={payload['all_exact']}",
        flush=True,
    )
    for comparison in payload["comparisons"]:
        print(
            f"[kv-cache-compare] prompt={comparison['prompt_idx']} "
            f"all_fields_exact={comparison['all_fields_exact']}",
            flush=True,
        )
    print(
        f"[kv-cache-compare] remote result: {result['remote_output_path']}",
        flush=True,
    )
    os.makedirs("outputs", exist_ok=True)
    local_path = os.path.join(
        "outputs",
        f"online_kv_cache_equivalence_n{n}_t{t}_eta{eta:.2f}_"
        f"prompts{num_prompts}_seed{experiment_seed}_"
        f"gen-{model_cache_name(generation_model_size)}.json",
    )
    with open(local_path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"[kv-cache-compare] local result: {local_path}", flush=True)


@app.local_entrypoint()
def validate_kv_cache_runtime_smoke(n: int = 80, prefix_n: int = 64,
                                    num_prompts: int = 2,
                                    t: int = 3, eta: float = 0.05,
                                    experiment_seed: int = SEED,
                                    generation_model_size: str = MODEL_SIZE,
                                    gpu: str = ""):
    generation_model_size, _, gpu = resolve_model_runtime(
        generation_model_size, 1, gpu
    )
    if not 0 < prefix_n < n:
        raise ValueError("require 0 < prefix_n < n")
    tag = config_tag(
        n, t, eta, experiment_seed, generation_model_size, "static"
    )
    model = OnlineModel.with_options(gpu=gpu, max_containers=1)(
        tag=tag,
        model_size=generation_model_size,
        code_fingerprint_sha256=_local_code_fingerprint(),
        kv_cache_implementation="static",
    )
    print(f"[kv-cache-runtime] model ready: {model.ready.remote()}", flush=True)
    payload = model.validate_kv_cache_runtime.remote(
        list(range(int(num_prompts))), prefix_n
    )
    direct_exact = payload["concat_vs_static_direct"]["all_exact"]
    resumed_exact = payload["static_direct_vs_resumed"]["all_exact"]
    print(
        f"[kv-cache-runtime] concat_vs_static_direct={direct_exact}; "
        f"static_direct_vs_resumed={resumed_exact}",
        flush=True,
    )
    print(
        f"[kv-cache-runtime] metrics={payload['metrics']}", flush=True
    )
    os.makedirs("outputs", exist_ok=True)
    local_path = os.path.join(
        "outputs",
        f"online_kv_cache_runtime_n{n}_from_n{prefix_n}_t{t}_"
        f"eta{eta:.2f}_prompts{num_prompts}_seed{experiment_seed}_"
        f"gen-{model_cache_name(generation_model_size)}.json",
    )
    with open(local_path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"[kv-cache-runtime] local result: {local_path}", flush=True)


@app.local_entrypoint()
def validate_map_detection_sharding(
    source_n: int = 64,
    floor_n: int = 48,
    step: int = 16,
    num_prompts: int = 2,
    t: int = 3,
    eta: float = 0.05,
    fpr: float = 1e-3,
    target_map_tpr: float = 0.90,
    experiment_seed: int = 424242,
    generation_model_size: str = MODEL_SIZE,
    kv_cache_implementation: str = "static",
    detection_shard_size: int = 1,
    detection_max_containers: int = 2,
):
    """Compare serial and prompt-sharded MAP detection on saved records."""
    import time

    generation_model_size = normalize_model_size(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    indices = list(range(int(num_prompts)))
    lengths = descending_prefix_grid(source_n, floor_n, step)
    shards = prompt_detection_shards(indices, detection_shard_size)
    code_fingerprint = _local_code_fingerprint()
    tag = config_tag(
        source_n,
        t,
        eta,
        experiment_seed,
        generation_model_size,
        kv_cache_implementation,
    )

    serial = detect_map_prefix_grid_serial.remote(
        tag,
        lengths,
        indices,
        fpr,
        code_fingerprint,
        target_map_tpr,
        True,
        False,
    )
    requests = [{
        "source_tag": tag,
        "maximum_length": max(lengths),
        "prompt_indices": shard,
        "code_fingerprint_sha256": code_fingerprint,
    } for shard in shards]
    started = time.time()
    prepare_function = prepare_map_prefix_shard.with_options(
        cpu=1.0,
        max_containers=min(detection_max_containers, len(shards)),
    )
    prepared = list(prepare_function.map(requests))
    preparation_wall_seconds = time.time() - started
    sharded = aggregate_map_prefix_shards.remote(
        tag,
        lengths,
        indices,
        fpr,
        code_fingerprint,
        target_map_tpr,
        prepared,
        preparation_wall_seconds,
        True,
    )

    serial_payload = serial["payload"]
    sharded_payload = sharded["payload"]
    compared_fields = (
        "prefix_lengths",
        "unevaluated_prefix_lengths",
        "first_below_n",
        "rows",
        "results",
    )
    mismatches = [
        field for field in compared_fields
        if serial_payload[field] != sharded_payload[field]
    ]
    if mismatches:
        raise AssertionError(
            f"serial and prompt-sharded MAP detection differ: {mismatches}"
        )
    print(
        f"[map-sharding] exact serial equivalence for {num_prompts} prompts, "
        f"{len(shards)} shards, evaluated={sharded_payload['prefix_lengths']}",
        flush=True,
    )
    print(
        f"[map-sharding] preparation wall={preparation_wall_seconds:.3f}s, "
        f"cache_hits={sharded_payload['prepared_shard_cache_hits']}/"
        f"{sharded_payload['prepared_shard_count']}",
        flush=True,
    )
    print(
        f"[map-sharding] remote result: {sharded['remote_output_path']}",
        flush=True,
    )


@app.local_entrypoint()
def main(num_prompts: int = CANONICAL_NUM_PROMPTS,
         n: int = 256, t: int = 3, eta: float = 0.05,
         fpr: float = 1e-3, batch: int = 0,
         experiment_seed: int = SEED,
         max_containers: int = DEFAULT_MAX_CONTAINERS,
         gpu: str = "", fresh: bool = False,
         generation_model_size: str = MODEL_SIZE,
         kv_cache_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
         csv_out: str = "online_causal_results_summary.csv"):
    generation_model_size, batch, gpu = resolve_model_runtime(
        generation_model_size, batch, gpu
    )
    generation_model = model_display(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    if num_prompts <= 0 or num_prompts > CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if n <= 0 or t < 2 or batch <= 0 or max_containers <= 0:
        raise ValueError("n, t, batch, and max_containers are invalid")
    if experiment_seed < 0:
        raise ValueError("experiment_seed must be nonnegative")
    if not 0 <= eta < 0.5 or not 0 < fpr < 1:
        raise ValueError("eta must be in [0,.5) and fpr in (0,1)")
    prompt_indices = list(range(int(num_prompts)))
    code_fingerprint = _local_code_fingerprint()
    tag = config_tag(
        n, t, eta, experiment_seed, generation_model_size,
        kv_cache_implementation,
    )
    print(
        f"[main] {SCHEME}: T=n={n}, t={t}, eta={eta}, fpr={fpr:g}, "
        f"model={generation_model}, prompts={num_prompts}, batch={batch}, "
        f"kv_cache={kv_cache_implementation}, "
        f"experiment_seed={experiment_seed}, GPU={gpu}, "
        f"max_containers={max_containers}", flush=True,
    )
    build = build_artifacts.remote(
        num_prompts, n, t, eta, experiment_seed, fresh,
        generation_model_size, kv_cache_implementation,
    )
    print(
        f"[main] artifact {'reused' if build['reused'] else 'built'}: "
        f"{build['artifact_fingerprint']}", flush=True,
    )
    plan = plan_generation.remote(
        tag, prompt_indices, n, allow_wm_reuse=not fresh
    )
    print(
        f"[main] generation plan: wm_missing={len(plan['wm_missing'])}, "
        f"wm_mode={plan['wm_mode']}, "
        f"wm_source_T={plan['wm_source_T']}, "
        f"wm_resume_source_T={plan['wm_resume_source_T']}, "
        f"null_missing={len(plan['null_missing'])}, "
        f"compatible_null_T={plan['null_T']}", flush=True,
    )
    if plan["wm_rejected_candidates"]:
        print(
            f"[main] rejected incompatible wm candidates: "
            f"{plan['wm_rejected_candidates']}",
            flush=True,
        )

    generation_meta = {"wm": [], "null": []}
    if plan["wm_missing"] or plan["null_missing"]:
        from concurrent.futures import ThreadPoolExecutor

        model = OnlineModel.with_options(
            gpu=gpu, max_containers=max_containers
        )(
            tag=tag,
            model_size=generation_model_size,
            code_fingerprint_sha256=code_fingerprint,
            kv_cache_implementation=kv_cache_implementation,
        )
        print(f"[main] model ready: {model.ready.remote()}", flush=True)
        work = []
        if plan["wm_missing"]:
            wm_requests = [
                {
                    "prompt_indices": chunk,
                    "resume_source_tag": plan["wm_resume_source_tag"],
                }
                for chunk in _chunks(plan["wm_missing"], batch)
            ]
            work.append((
                "wm", model.generate_wm,
                wm_requests,
            ))
        if plan["null_missing"]:
            work.append((
                "null", model.generate_null,
                _chunks(plan["null_missing"], batch),
            ))

        def run_map(item):
            name, method, chunks = item
            return name, list(method.map(chunks))

        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            mapped = list(pool.map(run_map, work))
        for name, records in mapped:
            generation_meta[name] = records
            generated = sum(record["generated"] for record in records)
            actual_batches = [record["batch"] for record in records if record["batch"]]
            print(
                f"[main] {name}: generated={generated}, "
                f"batch_sizes={actual_batches}", flush=True,
            )
    else:
        print("[main] all generation records cached", flush=True)

    detected = detect_all.remote(
        tag,
        prompt_indices,
        plan["null_T"],
        fpr,
        batch,
        code_fingerprint,
        plan["wm_source_tag"],
        plan["wm_mode"],
        plan["wm_resume_source_tag"],
        plan["wm_resume_source_T"],
    )
    payload = detected["payload"]
    counts = payload["counts"]
    print("\n=== Online causal Hoeffding summary ===", flush=True)
    print(
        f"T=n={n}, t={t}, r={payload['r']}, "
        f"free={payload['free_coordinates']}, eta={eta}", flush=True,
    )
    for weight in ("map", "entropy", "naive"):
        count = counts[weight]
        print(
            f"{weight:>7}: TPR {_format_rate(count['tp'], count['watermarked_total'])}  "
            f"FPR {_format_rate(count['fp'], count['null_total'])}", flush=True,
        )
    print(f"[main] remote result: {detected['remote_output_path']}", flush=True)

    os.makedirs("outputs", exist_ok=True)
    seed_suffix = "" if experiment_seed == SEED else f"_seed{experiment_seed}"
    model_suffix = (
        "" if generation_model_size == MODEL_SIZE
        else f"_gen-{model_cache_name(generation_model_size)}"
    )
    kv_cache_suffix = (
        "" if kv_cache_implementation == DEFAULT_KV_CACHE_IMPLEMENTATION
        else f"_kvcache-{kv_cache_version(kv_cache_implementation)}"
    )
    reuse_suffix = (
        f"_from_n{plan['wm_source_T']}"
        if plan["wm_mode"] == "prefix_from_longer" else ""
    )
    local_json = os.path.join(
        "outputs",
        f"online_causal_n{n}_t{t}_eta{eta:.2f}_prompts{num_prompts}"
        f"{seed_suffix}{model_suffix}_sampler-{SAMPLER_CACHE_TAG}"
        f"{kv_cache_suffix}{reuse_suffix}.json",
    )
    local_payload = {**payload, "results": payload["results"]}
    with open(local_json, "w") as handle:
        json.dump(local_payload, handle, indent=2, allow_nan=False)
    row = {
        "timestamp_utc": payload["timestamp_utc"],
        "scheme": SCHEME,
        "eta": eta,
        "T": n,
        "n": n,
        "r value": payload["r"],
        "free coordinates": payload["free_coordinates"],
        "r setting": "causal round(0.99L), startup-clamped",
        "t": t,
        "Target FPR": f"{fpr:.0e}",
        "Generation Model": generation_model,
        "num prompts": num_prompts,
        "batch": batch,
        "kv cache implementation": kv_cache_implementation,
        "kv cache version": kv_cache_version(kv_cache_implementation),
        "experiment seed": experiment_seed,
        "Map TPR": _format_rate(counts["map"]["tp"], num_prompts),
        "Map FPR": _format_rate(counts["map"]["fp"], num_prompts),
        "Entropy Aware TPR": _format_rate(counts["entropy"]["tp"], num_prompts),
        "Entropy FPR": _format_rate(counts["entropy"]["fp"], num_prompts),
        "Naive TPR": _format_rate(counts["naive"]["tp"], num_prompts),
        "Naive FPR": _format_rate(counts["naive"]["fp"], num_prompts),
        "null cache T": payload["null_cache_T"],
        "watermarked cache mode": payload["watermarked_cache_mode"],
        "watermarked cache T": payload["watermarked_cache_T"],
        "watermarked cache tag": payload["watermarked_cache_tag"],
        "watermarked resume source T": (
            payload["watermarked_resume_source_T"] or ""
        ),
        "watermarked resume source tag": (
            payload["watermarked_resume_source_tag"] or ""
        ),
        "schedule version": payload["schedule_version"],
        "stopping policy": STOPPING_POLICY,
        "FPR policy": FPR_POLICY,
        "artifact fingerprint": payload["artifact_fingerprint"],
    }
    _append_local_csv(csv_out, row)
    print(f"[main] local result: {local_json}", flush=True)
    print(f"[main] local summary: {csv_out}", flush=True)


@app.local_entrypoint()
def sweep_map_prefixes(source_n: int = 512, floor_n: int = 400,
                       step: int = 16, target_map_tpr: float = 0.90,
                       num_prompts: int = CANONICAL_NUM_PROMPTS,
                       t: int = 3, eta: float = 0.05,
                       fpr: float = 1e-3, batch: int = 0,
                       experiment_seed: int = SEED,
                       max_containers: int = DEFAULT_MAX_CONTAINERS,
                       gpu: str = "", fresh: bool = False,
                       generation_model_size: str = MODEL_SIZE,
                       kv_cache_implementation: str = (
                           DEFAULT_KV_CACHE_IMPLEMENTATION
                       ),
                       detection_shard_size: int = (
                           DEFAULT_DETECTION_SHARD_SIZE
                       ),
                       detection_max_containers: int = (
                           DEFAULT_DETECTION_MAX_CONTAINERS
                       ),
                       final_audit: bool = True,
                       pin_floor_cache: bool = True):
    """Generate one ceiling cache and scan exact MAP prefixes downward."""
    import time

    started = time.time()
    generation_model_size, batch, gpu = resolve_model_runtime(
        generation_model_size, batch, gpu
    )
    generation_model = model_display(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    lengths = descending_prefix_grid(source_n, floor_n, step)
    if num_prompts <= 0 or num_prompts > CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if (
        t < 2 or batch <= 0 or max_containers <= 0
        or detection_shard_size <= 0 or detection_max_containers <= 0
    ):
        raise ValueError("generation or detection runtime settings are invalid")
    if experiment_seed < 0:
        raise ValueError("experiment_seed must be nonnegative")
    if not 0 <= eta < 0.5 or not 0 < fpr < 1:
        raise ValueError("eta must be in [0,.5) and fpr in (0,1)")
    # Validate the target separately so errors occur before any remote work.
    rate_strictly_above(0, 1, target_map_tpr)

    prompt_indices = list(range(int(num_prompts)))
    code_fingerprint = _local_code_fingerprint()
    requested_tag = config_tag(
        source_n, t, eta, experiment_seed, generation_model_size,
        kv_cache_implementation,
    )
    print(
        f"[sweep] {SCHEME}: ceiling T=n={source_n}, floor={floor_n}, "
        f"step={step}, target_map_tpr>{target_map_tpr:.1%}, t={t}, "
        f"eta={eta}, model={generation_model}, prompts={num_prompts}, "
        f"batch={batch}, kv_cache={kv_cache_implementation}, "
        f"experiment_seed={experiment_seed}, GPU={gpu}, "
        f"max_containers={max_containers}, "
        f"detection_shard_size={detection_shard_size}, "
        f"detection_max_containers={detection_max_containers}",
        flush=True,
    )

    build = build_artifacts.remote(
        num_prompts, source_n, t, eta, experiment_seed, fresh,
        generation_model_size, kv_cache_implementation,
    )
    print(
        f"[sweep] ceiling artifact "
        f"{'reused' if build['reused'] else 'built'}: "
        f"{build['artifact_fingerprint']}",
        flush=True,
    )
    ceiling_preparation = None
    if pin_floor_cache:
        reference_tag = config_tag(
            floor_n, t, eta, experiment_seed, generation_model_size,
            kv_cache_implementation,
        )
        ceiling_preparation = prepare_sweep_ceiling.remote(
            requested_tag,
            reference_tag,
            prompt_indices,
        )
        if ceiling_preparation["quarantined"]:
            print(
                f"[sweep] quarantined "
                f"{len(ceiling_preparation['quarantined'])} ceiling records "
                f"whose T={floor_n} prefix did not match the canonical "
                f"sampler-v2 cache: {ceiling_preparation['quarantine_dir']}",
                flush=True,
            )
    plan = plan_generation.remote(
        requested_tag,
        prompt_indices,
        source_n,
        allow_wm_reuse=not fresh and not pin_floor_cache,
    )
    if pin_floor_cache and plan["wm_missing"]:
        plan.update({
            "wm_mode": "continue_from_shorter",
            "wm_source_tag": requested_tag,
            "wm_source_T": int(source_n),
            "wm_resume_source_tag": ceiling_preparation["reference_tag"],
            "wm_resume_source_T": int(ceiling_preparation["reference_T"]),
        })
    print(
        f"[sweep] generation plan: wm_missing={len(plan['wm_missing'])}, "
        f"wm_mode={plan['wm_mode']}, wm_source_T={plan['wm_source_T']}, "
        f"wm_resume_source_T={plan['wm_resume_source_T']}, "
        f"null_missing={len(plan['null_missing']) if final_audit else 0}, "
        f"compatible_null_T={plan['null_T'] if final_audit else 'not requested'}",
        flush=True,
    )
    if plan["wm_rejected_candidates"]:
        print(
            f"[sweep] rejected incompatible wm candidates: "
            f"{plan['wm_rejected_candidates']}",
            flush=True,
        )

    generation_meta = _execute_generation_plan(
        requested_tag,
        plan,
        batch,
        max_containers,
        gpu,
        code_fingerprint,
        generation_model_size,
        kv_cache_implementation,
        include_null=final_audit,
        log_prefix="sweep",
    )
    source_tag = plan["wm_source_tag"]
    source_T = int(plan["wm_source_T"])
    continuation_audit = None
    if plan["wm_mode"] == "continue_from_shorter":
        continuation_audit = audit_continuation.remote(
            requested_tag,
            plan["wm_resume_source_tag"],
            num_prompts,
        )
        source_tag = requested_tag
        source_T = int(source_n)
        print(
            f"[sweep] audited all {len(continuation_audit['audited'])} "
            f"continued records from T={continuation_audit['source_T']} "
            f"to T={continuation_audit['target_T']}",
            flush=True,
        )

    detection_shards = prompt_detection_shards(
        prompt_indices, detection_shard_size
    )
    preparation_requests = [{
        "source_tag": source_tag,
        "maximum_length": max(lengths),
        "prompt_indices": shard,
        "code_fingerprint_sha256": code_fingerprint,
    } for shard in detection_shards]
    preparation_started = time.time()
    preparation_function = prepare_map_prefix_shard.with_options(
        cpu=1.0,
        max_containers=min(
            int(detection_max_containers), len(detection_shards)
        ),
    )
    prepared_shards = list(preparation_function.map(preparation_requests))
    preparation_wall_seconds = time.time() - preparation_started
    if len(prepared_shards) != len(detection_shards):
        raise AssertionError("MAP prompt-shard preparation result count changed")
    print(
        f"[sweep] MAP preparation: shards={len(prepared_shards)}, "
        f"cache_hits={sum(item['cached'] for item in prepared_shards)}, "
        f"wall_seconds={preparation_wall_seconds:.1f}",
        flush=True,
    )
    detected = aggregate_map_prefix_shards.remote(
        source_tag,
        lengths,
        prompt_indices,
        fpr,
        code_fingerprint,
        target_map_tpr,
        prepared_shards,
        preparation_wall_seconds,
        True,
    )
    grid_payload = detected["payload"]
    summary = summarize_map_sweep(grid_payload["rows"], target_map_tpr)
    evaluated_lengths = [
        int(length) for length in grid_payload["prefix_lengths"]
    ]

    print("\n=== Online MAP descending prefix sweep ===", flush=True)
    print("     n       r       MAP TPR   > target", flush=True)
    for row in summary["rows"]:
        print(
            f"{int(row['n']):6d}  {int(row['r']):6d}  "
            f"{_format_rate(row['tp'], row['watermarked_total']):>16}  "
            f"{'yes' if row['above_target'] else 'no'}",
            flush=True,
        )

    selected_n = summary["last_passing_n_descending"]
    final_result = None
    if selected_n is not None and final_audit:
        final_result = detect_saved_prefix.remote(
            source_tag,
            selected_n,
            prompt_indices,
            plan["null_T"],
            fpr,
            code_fingerprint,
        )
        final_map = final_result["payload"]["counts"]["map"]
        if (
            int(final_map["tp"]) != int(summary["last_passing_tp"])
            or int(final_map["watermarked_total"]) != num_prompts
        ):
            raise AssertionError(
                "final full detector disagrees with the MAP prefix-grid result"
            )

    cost = summarize_generation_cost(generation_meta, source_n, gpu)
    cost.update({
        "prefix_grid_cpu_detection_seconds": float(
            grid_payload["cpu_detection_seconds"]
        ),
        "local_end_to_end_wall_seconds": time.time() - started,
    })
    manifest = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "online_map_descending_prefix_sweep",
        "requested_source_n": int(source_n),
        "actual_source_T": source_T,
        "actual_source_tag": source_tag,
        "floor_n": int(floor_n),
        "requested_prefix_lengths": lengths,
        "evaluated_prefix_lengths": evaluated_lengths,
        "unevaluated_prefix_lengths": grid_payload[
            "unevaluated_prefix_lengths"
        ],
        "stop_after_first_below": True,
        "stopped_after_first_below": bool(
            grid_payload["stopped_after_first_below"]
        ),
        "first_below_n": grid_payload["first_below_n"],
        "step": int(step),
        "target_map_tpr": float(target_map_tpr),
        "t": int(t),
        "eta": float(eta),
        "target_fpr": float(fpr),
        "generation_model": generation_model,
        "generation_model_size": generation_model_size,
        "kv_cache_implementation": kv_cache_implementation,
        "kv_cache_version": kv_cache_version(kv_cache_implementation),
        "num_prompts": int(num_prompts),
        "batch": int(batch),
        "experiment_seed": int(experiment_seed),
        "gpu": str(gpu),
        "max_containers": int(max_containers),
        "detection_shard_size": int(detection_shard_size),
        "detection_max_containers": int(detection_max_containers),
        "detection_shard_count": len(detection_shards),
        "detection_strategy": grid_payload["detection_strategy"],
        "generation_plan": plan,
        "ceiling_preparation": ceiling_preparation,
        "generation_cost": cost,
        "continuation_audit": continuation_audit,
        "grid_remote_output_path": detected["remote_output_path"],
        "increment_remote_output_paths": grid_payload[
            "prefix_result_paths"
        ],
        "summary": summary,
        "grid_results": grid_payload["results"],
        "final_audit_remote_output_path": (
            final_result["remote_output_path"] if final_result else None
        ),
        "final_audit": final_result["payload"] if final_result else None,
        "cache_inventory": {
            "strategy": "single_ceiling_cache_with_exact_prefix_views",
            "watermarked_source_tag": source_tag,
            "watermarked_source_T": source_T,
            "watermarked_records_verified": len(grid_payload["results"]),
            "cached_watermarked_fields": [
                "tokens",
                "p_trace",
                "base_lm_entropy",
                "base_token_logprob",
                "prc_codeword_bits",
            ],
            "null_source_T": int(plan["null_T"]) if final_audit else None,
            "null_records_verified": (
                int(final_result["payload"]["counts"]["map"]["null_total"])
                if final_result else 0
            ),
            "prefix_generation_records_duplicated": False,
        },
        "code_fingerprint_sha256": code_fingerprint,
    }

    os.makedirs("outputs", exist_ok=True)
    seed_suffix = "" if experiment_seed == SEED else f"_seed{experiment_seed}"
    model_suffix = (
        "" if generation_model_size == MODEL_SIZE
        else f"_gen-{model_cache_name(generation_model_size)}"
    )
    kv_cache_suffix = (
        "" if kv_cache_implementation == DEFAULT_KV_CACHE_IMPLEMENTATION
        else f"_kvcache-{kv_cache_version(kv_cache_implementation)}"
    )
    stem = (
        f"online_map_sweep_n{source_n}_to{floor_n}_step{step}_t{t}_"
        f"eta{eta:.2f}_prompts{num_prompts}{seed_suffix}{model_suffix}_"
        f"sampler-{SAMPLER_CACHE_TAG}{kv_cache_suffix}"
    )
    local_json = os.path.join("outputs", f"{stem}.json")
    local_csv = os.path.join("outputs", f"{stem}.csv")
    increment_dir = os.path.join("outputs", f"{stem}_increments")
    os.makedirs(increment_dir, exist_ok=True)
    increment_result_index = {}
    rows_by_length = {int(row["n"]): row for row in summary["rows"]}
    for length in evaluated_lengths:
        increment_payload = increment_payload_from_grid(grid_payload, length)
        increment_payload.update({
            "target_map_tpr": float(target_map_tpr),
            "above_target": bool(rows_by_length[length]["above_target"]),
            "selected_last_passing_n": selected_n,
            "remote_output_path": grid_payload["prefix_result_paths"][
                str(length)
            ],
        })
        increment_path = os.path.join(
            increment_dir, f"map_prefix_n{length}.json"
        )
        with open(increment_path, "w") as handle:
            json.dump(increment_payload, handle, indent=2, allow_nan=False)
        increment_result_index[str(length)] = {
            "local_output_path": increment_path,
            "remote_output_path": increment_payload["remote_output_path"],
            "source_tag": source_tag,
            "source_T": source_T,
            "tp": int(rows_by_length[length]["tp"]),
            "watermarked_total": int(
                rows_by_length[length]["watermarked_total"]
            ),
            "map_tpr": float(rows_by_length[length]["tpr"]),
            "above_target": bool(rows_by_length[length]["above_target"]),
        }
    manifest["increment_result_index"] = increment_result_index
    with open(local_json, "w") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
    csv_columns = (
        "n", "T", "r", "free_coordinates", "tp", "watermarked_total",
        "tpr", "above_target", "target_map_tpr",
        "prefix_online_support_sha256", "source_T", "source_tag",
    )
    with open(local_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=csv_columns, lineterminator="\n"
        )
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({
                **{column: row.get(column, "") for column in csv_columns},
                "target_map_tpr": float(target_map_tpr),
                "source_T": source_T,
                "source_tag": source_tag,
            })

    if selected_n is None:
        print(
            f"[sweep] no contiguous descending prefix above "
            f"{target_map_tpr:.1%}; ceiling={source_n}",
            flush=True,
        )
    else:
        print(
            f"[sweep] last passing n while descending: {selected_n} "
            f"({_format_rate(summary['last_passing_tp'], num_prompts)})",
            flush=True,
        )
        if summary["next_shorter_n"] is not None:
            print(
                f"[sweep] next shorter grid point: "
                f"n={summary['next_shorter_n']} "
                f"above_target={summary['next_shorter_above_target']}",
                flush=True,
            )
    if summary["monotonicity_violations"]:
        print(
            f"[sweep] empirical monotonicity warnings: "
            f"{summary['monotonicity_violations']}",
            flush=True,
        )
    print(
        f"[sweep] measured GPU-method time: "
        f"{cost['measured_gpu_method_seconds']:.1f}s; "
        f"model token positions: {cost['model_token_positions_processed']}",
        flush=True,
    )
    print(f"[sweep] local manifest: {local_json}", flush=True)
    print(f"[sweep] local grid CSV: {local_csv}", flush=True)
    print(
        f"[sweep] per-increment results: {increment_dir} "
        f"({len(increment_result_index)} files)",
        flush=True,
    )
    if grid_payload["unevaluated_prefix_lengths"]:
        print(
            f"[sweep] early stop left "
            f"{len(grid_payload['unevaluated_prefix_lengths'])} shorter "
            f"increments unevaluated",
            flush=True,
        )


@app.local_entrypoint()
def redetect_prefix(source_n: int = 400, prefix_n: int = 256,
                    num_prompts: int = CANONICAL_NUM_PROMPTS,
                    t: int = 3, eta: float = 0.05,
                    fpr: float = 1e-3, experiment_seed: int = SEED,
                    generation_model_size: str = MODEL_SIZE,
                    kv_cache_implementation: str = (
                        DEFAULT_KV_CACHE_IMPLEMENTATION
                    )):
    """Score a shorter prefix directly from a saved longer online run."""
    if source_n <= 0 or prefix_n <= 0 or prefix_n > source_n:
        raise ValueError("require 0 < prefix_n <= source_n")
    if num_prompts <= 0 or num_prompts > CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if t < 2 or experiment_seed < 0:
        raise ValueError("t or experiment_seed is invalid")
    if not 0 <= eta < 0.5 or not 0 < fpr < 1:
        raise ValueError("eta must be in [0,.5) and fpr in (0,1)")
    generation_model_size = normalize_model_size(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )

    prompt_indices = list(range(int(num_prompts)))
    source_tag = config_tag(
        source_n, t, eta, experiment_seed, generation_model_size,
        kv_cache_implementation,
    )
    plan = plan_generation.remote(source_tag, prompt_indices, prefix_n)
    if plan["wm_missing"]:
        raise FileNotFoundError(
            f"source run {source_tag} is missing "
            f"{len(plan['wm_missing'])} watermarked records"
        )
    if plan["null_missing"]:
        raise FileNotFoundError(
            f"no complete null cache found for prefix_n={prefix_n}; "
            f"missing {len(plan['null_missing'])} records"
        )
    print(
        f"[redetect_prefix] source_T={source_n} -> T=n={prefix_n}; "
        f"model={model_display(generation_model_size)}; "
        f"kv_cache={kv_cache_implementation}; "
        f"prompts={num_prompts}; null_cache_T={plan['null_T']}; "
        "generation=none",
        flush=True,
    )
    detected = detect_saved_prefix.remote(
        source_tag,
        prefix_n,
        prompt_indices,
        plan["null_T"],
        fpr,
        _local_code_fingerprint(),
    )
    payload = detected["payload"]
    counts = payload["counts"]
    print("\n=== Saved online-prefix Hoeffding summary ===", flush=True)
    for weight in ("map", "entropy", "naive"):
        count = counts[weight]
        print(
            f"{weight:>7}: "
            f"TPR {_format_rate(count['tp'], count['watermarked_total'])}  "
            f"FPR {_format_rate(count['fp'], count['null_total'])}",
            flush=True,
        )
    print(
        f"[redetect_prefix] remote result: {detected['remote_output_path']}",
        flush=True,
    )

    os.makedirs("outputs", exist_ok=True)
    seed_suffix = "" if experiment_seed == SEED else f"_seed{experiment_seed}"
    model_suffix = (
        "" if generation_model_size == MODEL_SIZE
        else f"_gen-{model_cache_name(generation_model_size)}"
    )
    kv_cache_suffix = (
        "" if kv_cache_implementation == DEFAULT_KV_CACHE_IMPLEMENTATION
        else f"_kvcache-{kv_cache_version(kv_cache_implementation)}"
    )
    local_json = os.path.join(
        "outputs",
        f"online_causal_prefix_n{prefix_n}_from_n{source_n}_t{t}_"
        f"eta{eta:.2f}_prompts{num_prompts}{seed_suffix}{model_suffix}_"
        f"sampler-{SAMPLER_CACHE_TAG}{kv_cache_suffix}.json",
    )
    with open(local_json, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(f"[redetect_prefix] local result: {local_json}", flush=True)
