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
SUPPORTED_MODEL_SIZES = ("0.6B", "8B", "14B")
VOCAB = 151_936
GPU = "A10G"
DEFAULT_BATCH = 64
DEFAULT_8B_BATCH = 25
DEFAULT_14B_BATCH = 10
DEFAULT_14B_MEMORY_MIB = 65_536
DEFAULT_MAX_CONTAINERS = 5
DEFAULT_DETECTION_SHARD_SIZE = 50
DEFAULT_DETECTION_MAX_CONTAINERS = 10
CANONICAL_NUM_PROMPTS = 500
RESULT_SCHEMA_VERSION = 3
PREPARED_MAP_SHARD_SCHEMA_VERSION = 1
FULL_AUDIT_SHARD_SCHEMA_VERSION = 1
CROSS_MODEL_ENTROPY_TRACE_SCHEMA_VERSION = 1
CROSS_MODEL_ENTROPY_AUDIT_SHARD_SCHEMA_VERSION = 2
CROSS_MODEL_ENTROPY_RESULT_SCHEMA_VERSION = 2
NULL_CACHE_MANIFEST_SCHEMA_VERSION = 1
NULL_CACHE_MANIFEST_FILENAME = "_manifest.json"
NULL_GENERATION_SAMPLER_VERSION = "torch_multinomial_global_v1"
LEGACY_SAMPLER_VERSION = "legacy_torch_global_v1"
ONLINE_MODEL_CACHE_NAME = "qwen3_0p6b_base"
SAMPLER_CACHE_TAG = "poscdf-v1"
DEFAULT_KV_CACHE_IMPLEMENTATION = "concat"
DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION = "static"
DEFAULT_ENTROPY_BATCH = 50
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
    "null kv cache implementation",
    "null kv cache version",
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


def resolve_null_kv_cache_implementation(
    null_implementation: str = "",
    watermarked_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
) -> str:
    """Resolve an explicit null cache choice, inheriting WM when omitted."""
    return normalize_kv_cache_implementation(
        null_implementation or watermarked_implementation
    )


def _slug(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _numpy_pickle_compat() -> None:
    """Read NumPy-2-authored torch payloads in the pinned NumPy 1.26 image."""
    import sys
    import numpy as np

    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


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
    return (
        "H100"
        if normalize_model_size(model_size) in ("8B", "14B")
        else GPU
    )


def model_default_batch(model_size: str = MODEL_SIZE) -> int:
    normalized = normalize_model_size(model_size)
    if normalized == "14B":
        return DEFAULT_14B_BATCH
    if normalized == "8B":
        return DEFAULT_8B_BATCH
    return DEFAULT_BATCH


def model_default_memory_mib(model_size: str = MODEL_SIZE) -> int:
    """Guaranteed host RAM for model loading; zero keeps legacy defaults."""
    return (
        DEFAULT_14B_MEMORY_MIB
        if normalize_model_size(model_size) == "14B"
        else 0
    )


def model_cls_options(model_size: str, gpu: str,
                      max_containers: int) -> dict:
    """Build model-specific Modal resource overrides."""
    options = {
        "gpu": str(gpu),
        "max_containers": int(max_containers),
    }
    memory_mib = model_default_memory_mib(model_size)
    if memory_mib:
        options["memory"] = memory_mib
    return options


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


def cross_model_entropy_trace_source(
    entropy_model_size: str,
    generation_model_size: str,
) -> str:
    """Stable label for teacher-forced cross-model probability traces."""
    return (
        f"teacher_forced_{model_cache_name(entropy_model_size)}_on_"
        f"{model_cache_name(generation_model_size)}"
    )


def cross_model_wm_entropy_dir(
    source_tag: str,
    trace_T: int,
    entropy_model_size: str,
    estimator_chunk_size: int = 1,
) -> str:
    directory = (
        f"/data/{source_tag}/cross_model_entropy_v"
        f"{CROSS_MODEL_ENTROPY_TRACE_SCHEMA_VERSION}/"
        f"{model_cache_name(entropy_model_size)}"
    )
    if int(estimator_chunk_size) != 1:
        directory += f"/chunk{int(estimator_chunk_size)}"
    return f"{directory}/T{int(trace_T)}/wm"


def cross_model_null_entropy_dir(
    trace_T: int,
    entropy_model_size: str,
    generation_model_size: str,
    estimator_chunk_size: int = 1,
) -> str:
    directory = (
        f"/data/_online_null_cross_model_entropy/"
        f"{model_cache_name(generation_model_size)}/"
        f"{model_cache_name(entropy_model_size)}"
    )
    if int(estimator_chunk_size) != 1:
        directory += f"/chunk{int(estimator_chunk_size)}"
    return f"{directory}/T{int(trace_T)}"


def cross_model_entropy_trace_path(
    source: str,
    prompt_index: int,
    trace_T: int,
    entropy_model_size: str,
    generation_model_size: str,
    source_tag: str = "",
    estimator_chunk_size: int = 1,
) -> str:
    source = str(source)
    if source == "wm":
        if not source_tag:
            raise ValueError("watermarked entropy traces require source_tag")
        directory = cross_model_wm_entropy_dir(
            source_tag, trace_T, entropy_model_size, estimator_chunk_size
        )
        prefix = "wm"
    elif source == "null":
        directory = cross_model_null_entropy_dir(
            trace_T,
            entropy_model_size,
            generation_model_size,
            estimator_chunk_size,
        )
        prefix = "null"
    else:
        raise ValueError(f"unknown cross-model entropy source {source!r}")
    return os.path.join(directory, f"{prefix}_{int(prompt_index):04d}.pt")


def cross_model_entropy_trace_identity(
    *,
    source: str,
    prompt_index: int,
    trace_T: int,
    generation_model_size: str,
    entropy_model_size: str,
    partition_sha256: str,
    prompt_sha256: str,
    tokens_sha256: str,
    source_artifact_fingerprint: str = "",
    estimator_chunk_size: int = 1,
) -> dict:
    """Fields that make an alternate entropy trace safe to reuse."""
    generation_model_size = normalize_model_size(generation_model_size)
    entropy_model_size = normalize_model_size(entropy_model_size)
    identity = {
        "cross_model_entropy_trace_schema_version": (
            CROSS_MODEL_ENTROPY_TRACE_SCHEMA_VERSION
        ),
        "trace_kind": "online_cross_model_partition_probability",
        "source": str(source),
        "prompt_idx": int(prompt_index),
        "trace_T": int(trace_T),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "entropy_model_size": entropy_model_size,
        "entropy_model": model_display(entropy_model_size),
        "entropy_trace_source": cross_model_entropy_trace_source(
            entropy_model_size, generation_model_size
        ),
        "partition_sha256": str(partition_sha256),
        "prompt_sha256": str(prompt_sha256),
        "tokens_sha256": str(tokens_sha256),
        "estimator_kv_cache_implementation": (
            DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION
        ),
        "estimator_kv_cache_version": kv_cache_version(
            DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION
        ),
    }
    if int(estimator_chunk_size) != 1:
        identity["estimator_chunk_size"] = int(estimator_chunk_size)
        identity["estimator_execution"] = "causal_multi_token_chunks_v1"
    if str(source) == "wm":
        if not source_artifact_fingerprint:
            raise ValueError(
                "watermarked entropy traces require an artifact fingerprint"
            )
        identity["source_artifact_fingerprint"] = str(
            source_artifact_fingerprint
        )
    elif str(source) != "null":
        raise ValueError(f"unknown cross-model entropy source {source!r}")
    return identity


def validate_cross_model_entropy_trace(
    payload: dict,
    require_full_entropy: bool = False,
    **identity_kwargs,
):
    """Validate trace identity, length, range, and serialized hash."""
    import numpy as np
    from detectors import semantic_sha256

    expected = cross_model_entropy_trace_identity(**identity_kwargs)
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"cross-model entropy trace {field}="
                f"{payload.get(field)!r}; expected {value!r}"
            )
    p_trace = np.asarray(payload.get("p_trace"), dtype=np.float64).reshape(-1)
    if p_trace.size != int(expected["trace_T"]):
        raise ValueError(
            f"cross-model entropy trace has {p_trace.size} probabilities; "
            f"expected {expected['trace_T']}"
        )
    if not np.all(np.isfinite(p_trace)):
        raise ValueError("cross-model entropy trace contains non-finite values")
    if np.any((p_trace < 0.0) | (p_trace > 1.0)):
        raise ValueError("cross-model entropy probabilities must be in [0, 1]")
    observed_hash = semantic_sha256(p_trace)
    if payload.get("p_trace_sha256") != observed_hash:
        raise ValueError("cross-model entropy p_trace hash is inconsistent")
    entropy_value = payload.get("full_entropy_trace")
    if require_full_entropy and entropy_value is None:
        raise ValueError("cross-model trace is missing full-vocabulary entropy")
    if entropy_value is not None:
        entropy_trace = np.asarray(
            entropy_value, dtype=np.float64
        ).reshape(-1)
        if entropy_trace.size != int(expected["trace_T"]):
            raise ValueError(
                "cross-model entropy trace has "
                f"{entropy_trace.size} full entropies; expected "
                f"{expected['trace_T']}"
            )
        if not np.all(np.isfinite(entropy_trace)) or np.any(entropy_trace < 0):
            raise ValueError(
                "cross-model full-vocabulary entropies must be finite and "
                "nonnegative"
            )
        if payload.get("full_entropy_trace_sha256") != semantic_sha256(
            entropy_trace
        ):
            raise ValueError(
                "cross-model full_entropy_trace hash is inconsistent"
            )
    return p_trace


def null_cache_manifest_path(length: int,
                             generation_model_size: str = MODEL_SIZE) -> str:
    return os.path.join(
        shared_null_dir(length, generation_model_size),
        NULL_CACHE_MANIFEST_FILENAME,
    )


def expected_null_cache_manifest(
    artifact: dict,
    length: int,
    null_kv_cache_implementation: str,
) -> dict:
    """Build eta/key-independent provenance for a shared null cache."""
    from detectors import (
        GENERATION_TRACE_SCHEMA_VERSION,
        semantic_sha256,
        tensor_sha256,
    )

    model_size = artifact_generation_model_size(artifact)
    implementation = normalize_kv_cache_implementation(
        null_kv_cache_implementation
    )
    prompts = artifact["prompt_ids_list"]
    return {
        "schema_version": NULL_CACHE_MANIFEST_SCHEMA_VERSION,
        "cache_kind": "unwatermarked_generation",
        "T": int(length),
        "forced_length": True,
        "stopping_policy": STOPPING_POLICY,
        "generation_model_size": model_size,
        "generation_model": model_display(model_size),
        "generation_model_variant": "base",
        "prompt_count": len(prompts),
        "prompt_corpus_sha256": semantic_sha256(prompts),
        "partition_sha256": tensor_sha256(artifact["partition"]),
        "generation_trace_schema_version": GENERATION_TRACE_SCHEMA_VERSION,
        "generation_sampler_version": NULL_GENERATION_SAMPLER_VERSION,
        "generation_rng_policy": NULL_GENERATION_SAMPLER_VERSION,
        "kv_cache_implementation": implementation,
        "kv_cache_version": kv_cache_version(implementation),
    }


def null_cache_manifest_compatibility_error(
    manifest: dict,
    artifact: dict,
    length: int,
    null_kv_cache_implementation: str | None = None,
) -> str | None:
    """Return why a shared null manifest is incompatible, if applicable."""
    expected_implementation = (
        normalize_kv_cache_implementation(null_kv_cache_implementation)
        if null_kv_cache_implementation is not None else
        normalize_kv_cache_implementation(
            manifest.get("kv_cache_implementation")
        )
    )
    expected = expected_null_cache_manifest(
        artifact, length, expected_implementation
    )
    for field, expected_value in expected.items():
        observed = manifest.get(field)
        if observed != expected_value:
            return f"{field} differs: {observed!r} != {expected_value!r}"
    return None


def load_null_cache_manifest(
    length: int,
    generation_model_size: str = MODEL_SIZE,
) -> dict | None:
    path = null_cache_manifest_path(length, generation_model_size)
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"null cache manifest {path} is not a mapping")
    return manifest


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


def require_complete_cache_plan(plan: dict) -> None:
    """Refuse cache-only detection before any model container can launch."""
    missing_wm = [int(index) for index in plan.get("wm_missing", [])]
    missing_null = [int(index) for index in plan.get("null_missing", [])]
    if missing_wm or missing_null:
        raise FileNotFoundError(
            "cache-only audit requires complete generation caches; "
            f"missing watermarked={missing_wm}, missing null={missing_null}"
        )


def full_audit_shard_path(
    tag: str,
    prefix_T: int,
    null_T: int,
    prompt_indices: list[int],
    artifact_fingerprint: str,
    watermarked_source_fingerprint: str,
    code_fingerprint: str,
    fpr: float,
) -> str:
    """Return a deterministic path for one full-detector prompt shard."""
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
        f"/data/{tag}/full_audit_shards_v"
        f"{FULL_AUDIT_SHARD_SCHEMA_VERSION}/"
        f"artifact-{str(artifact_fingerprint)[:16]}/"
        f"wm-{str(watermarked_source_fingerprint)[:16]}/"
        f"code-{str(code_fingerprint)[:16]}/T{int(prefix_T)}-"
        f"nullT{int(null_T)}-fpr{_slug(f'{float(fpr):.12g}')}/"
        f"shard-{shard_label}.pt"
    )


def validate_full_audit_shard(
    payload: dict,
    *,
    tag: str,
    watermarked_source_tag: str,
    prefix_T: int,
    null_T: int,
    fpr: float,
    artifact_fingerprint: str,
    watermarked_source_fingerprint: str,
    online_key_sha256: str,
    code_fingerprint_sha256: str,
) -> list[int]:
    """Validate one cached MAP/entropy/naive prompt-shard result."""
    expected = {
        "full_audit_shard_schema_version": FULL_AUDIT_SHARD_SCHEMA_VERSION,
        "result_kind": "online_full_audit_prompt_shard",
        "tag": str(tag),
        "watermarked_source_tag": str(watermarked_source_tag),
        "T": int(prefix_T),
        "null_T": int(null_T),
        "target_fpr": float(fpr),
        "fpr_policy": FPR_POLICY,
        "artifact_fingerprint": str(artifact_fingerprint),
        "watermarked_source_artifact_fingerprint": str(
            watermarked_source_fingerprint
        ),
        "online_key_sha256": str(online_key_sha256),
        "code_fingerprint_sha256": str(code_fingerprint_sha256),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"full-audit shard {field}={payload.get(field)!r}; "
                f"expected {value!r}"
            )

    indices = [int(index) for index in payload.get("prompt_indices", [])]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("full-audit shard prompt indices are invalid")
    if int(payload.get("num_prompts", -1)) != len(indices):
        raise ValueError("full-audit shard prompt count is inconsistent")

    results = payload.get("results", [])
    expected_order = [
        (watermark, index)
        for watermark in (True, False)
        for index in indices
    ]
    observed_order = [
        (bool(result.get("watermark")), int(result.get("prompt_idx", -1)))
        for result in results
    ]
    if observed_order != expected_order:
        raise ValueError(
            "full-audit shard results are not in watermark/prompt order"
        )
    for result in results:
        scores = result.get("scores", {})
        if set(scores) != {"map", "entropy", "naive"}:
            raise ValueError("full-audit shard detector set is incomplete")
        for score in scores.values():
            if not isinstance(score.get("decision"), bool):
                raise ValueError("full-audit decision must be boolean")
            if int(score.get("length", -1)) != int(prefix_T):
                raise ValueError("full-audit shard score used the wrong length")
    return indices


def merge_full_audit_shards(
    shard_payloads: list[dict],
    expected_prompt_indices: list[int],
) -> list[dict]:
    """Merge full-audit shards in serial detector result order."""
    expected = [int(index) for index in expected_prompt_indices]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_prompt_indices must be nonempty and unique")
    by_kind = {True: {}, False: {}}
    for payload in shard_payloads:
        for result in payload.get("results", []):
            watermark = bool(result["watermark"])
            index = int(result["prompt_idx"])
            if index in by_kind[watermark]:
                raise ValueError(
                    f"full-audit shards duplicate watermark={watermark} "
                    f"prompt index {index}"
                )
            by_kind[watermark][index] = result

    for watermark in (True, False):
        missing = [index for index in expected if index not in by_kind[watermark]]
        extra = sorted(set(by_kind[watermark]) - set(expected))
        if missing or extra:
            raise ValueError(
                f"full-audit shard coverage mismatch for "
                f"watermark={watermark}: missing={missing}, extra={extra}"
            )
    return [
        by_kind[watermark][index]
        for watermark in (True, False)
        for index in expected
    ]


def cross_model_entropy_audit_shard_path(
    source_tag: str,
    prefix_T: int,
    null_trace_T: int,
    prompt_indices: list[int],
    entropy_model_size: str,
    artifact_fingerprint: str,
    code_fingerprint: str,
    fpr: float,
) -> str:
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
        f"/data/{source_tag}/cross_model_entropy_audit_shards_v"
        f"{CROSS_MODEL_ENTROPY_AUDIT_SHARD_SCHEMA_VERSION}/"
        f"{model_cache_name(entropy_model_size)}/"
        f"artifact-{str(artifact_fingerprint)[:16]}/"
        f"code-{str(code_fingerprint)[:16]}/T{int(prefix_T)}-"
        f"nullTraceT{int(null_trace_T)}-"
        f"fpr{_slug(f'{float(fpr):.12g}')}/shard-{shard_label}.pt"
    )


def validate_cross_model_entropy_audit_shard(
    payload: dict,
    *,
    source_tag: str,
    prefix_T: int,
    null_T: int,
    null_trace_T: int,
    fpr: float,
    generation_model_size: str,
    entropy_model_size: str,
    artifact_fingerprint: str,
    online_key_sha256: str,
    code_fingerprint_sha256: str,
) -> list[int]:
    expected = {
        "cross_model_entropy_audit_shard_schema_version": (
            CROSS_MODEL_ENTROPY_AUDIT_SHARD_SCHEMA_VERSION
        ),
        "result_kind": "online_cross_model_map_entropy_prompt_shard",
        "source_tag": str(source_tag),
        "T": int(prefix_T),
        "null_T": int(null_T),
        "null_trace_T": int(null_trace_T),
        "target_fpr": float(fpr),
        "fpr_policy": FPR_POLICY,
        "generation_model_size": normalize_model_size(
            generation_model_size
        ),
        "entropy_model_size": normalize_model_size(entropy_model_size),
        "artifact_fingerprint": str(artifact_fingerprint),
        "online_key_sha256": str(online_key_sha256),
        "code_fingerprint_sha256": str(code_fingerprint_sha256),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"cross-model audit shard {field}={payload.get(field)!r}; "
                f"expected {value!r}"
            )
    indices = [int(index) for index in payload.get("prompt_indices", [])]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("cross-model audit shard prompt indices are invalid")
    if int(payload.get("num_prompts", -1)) != len(indices):
        raise ValueError("cross-model audit shard prompt count is inconsistent")
    expected_order = [
        (watermark, index)
        for watermark in (True, False)
        for index in indices
    ]
    results = payload.get("results", [])
    observed_order = [
        (bool(result.get("watermark")), int(result.get("prompt_idx", -1)))
        for result in results
    ]
    if observed_order != expected_order:
        raise ValueError(
            "cross-model audit results are not in watermark/prompt order"
        )
    for result in results:
        scores = result.get("scores", {})
        if set(scores) != {"map", "entropy"}:
            raise ValueError(
                "cross-model audit must contain MAP and entropy scores"
            )
        for weight, score in scores.items():
            if not isinstance(score.get("decision"), bool):
                raise ValueError(
                    f"cross-model {weight} decision must be boolean"
                )
            if int(score.get("length", -1)) != int(prefix_T):
                raise ValueError(
                    f"cross-model {weight} score used the wrong length"
                )
    return indices


def merge_cross_model_entropy_audit_shards(
    shard_payloads: list[dict],
    expected_prompt_indices: list[int],
) -> list[dict]:
    expected = [int(index) for index in expected_prompt_indices]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_prompt_indices must be nonempty and unique")
    by_kind = {True: {}, False: {}}
    for payload in shard_payloads:
        for result in payload.get("results", []):
            watermark = bool(result["watermark"])
            index = int(result["prompt_idx"])
            if index in by_kind[watermark]:
                raise ValueError(
                    f"cross-model audit shards duplicate watermark="
                    f"{watermark} prompt index {index}"
                )
            by_kind[watermark][index] = result
    for watermark in (True, False):
        missing = [index for index in expected if index not in by_kind[watermark]]
        extra = sorted(set(by_kind[watermark]) - set(expected))
        if missing or extra:
            raise ValueError(
                f"cross-model audit shard coverage mismatch for "
                f"watermark={watermark}: missing={missing}, extra={extra}"
            )
    return [
        by_kind[watermark][index]
        for watermark in (True, False)
        for index in expected
    ]


def compare_full_audit_results(
    left: list[dict],
    right: list[dict],
    float_atol: float = 1e-12,
) -> dict:
    """Compare audit payloads exactly except for CPU float roundoff."""
    import math

    mismatches = []
    max_abs_float_difference = 0.0

    def compare(left_value, right_value, path):
        nonlocal max_abs_float_difference
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            if set(left_value) != set(right_value):
                mismatches.append(f"{path}:keys")
                return
            for key in left_value:
                compare(left_value[key], right_value[key], f"{path}.{key}")
            return
        if isinstance(left_value, list) and isinstance(right_value, list):
            if len(left_value) != len(right_value):
                mismatches.append(f"{path}:length")
                return
            for index, (left_item, right_item) in enumerate(
                zip(left_value, right_value)
            ):
                compare(left_item, right_item, f"{path}[{index}]")
            return
        if isinstance(left_value, float) and isinstance(right_value, float):
            difference = abs(left_value - right_value)
            if math.isfinite(difference):
                max_abs_float_difference = max(
                    max_abs_float_difference, difference
                )
            if not math.isclose(
                left_value, right_value, rel_tol=0.0, abs_tol=float_atol
            ):
                mismatches.append(path)
            return
        if left_value != right_value:
            mismatches.append(path)

    compare(left, right, "results")
    return {
        "equivalent": not mismatches,
        "float_atol": float(float_atol),
        "max_abs_float_difference": max_abs_float_difference,
        "mismatches": mismatches,
    }


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


def validate_online_null_record(
    record: dict,
    artifact: dict,
    prompt_index: int,
    required_length: int,
    *,
    source_length: int | None = None,
    expected_kv_cache_implementation: str | None = None,
    require_provenance: bool = False,
) -> None:
    """Validate a shared null while retaining legacy-cache readability."""
    import numpy as np
    import torch
    from detectors import tensor_sha256

    generation_model_size = artifact_generation_model_size(artifact)
    validate_generation_model_record(
        record, generation_model_size, "null", prompt_index
    )
    if record.get("watermark") not in (False, None):
        raise ValueError(f"null record {prompt_index} is marked watermarked")
    if record.get("prc_codeword_bits") is not None:
        raise ValueError(f"null record {prompt_index} stores PRC codeword bits")
    expected_source_length = (
        int(source_length) if source_length is not None else None
    )
    length_fields = ["tokens", "p_trace"]
    for field in ("base_lm_entropy", "base_token_logprob"):
        if require_provenance or record.get(field) is not None:
            length_fields.append(field)
    for field in length_fields:
        value = record.get(field)
        observed_length = 0 if value is None else int(np.asarray(value).size)
        if observed_length < int(required_length):
            raise ValueError(
                f"null record {prompt_index} field {field!r} is shorter "
                f"than {required_length}"
            )
        if (
            expected_source_length is not None
            and observed_length != expected_source_length
        ):
            raise ValueError(
                f"null record {prompt_index} field {field!r} has length "
                f"{observed_length}, expected source length "
                f"{expected_source_length}"
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

    optional_scalars = {
        "prompt_idx": int(prompt_index),
        "stopping_policy": STOPPING_POLICY,
        "generation_sampler_version": NULL_GENERATION_SAMPLER_VERSION,
        "generation_rng_policy": NULL_GENERATION_SAMPLER_VERSION,
    }
    if expected_source_length is not None:
        optional_scalars.update({
            "source_T": expected_source_length,
            "realized_length": expected_source_length,
        })
    for field, expected in optional_scalars.items():
        observed = record.get(field)
        if observed is None and not require_provenance:
            continue
        if observed != expected:
            raise ValueError(
                f"null record {prompt_index} has incompatible {field}: "
                f"{observed!r} != {expected!r}"
            )

    stored_implementation = record.get("kv_cache_implementation")
    if expected_kv_cache_implementation is not None:
        expected_implementation = normalize_kv_cache_implementation(
            expected_kv_cache_implementation
        )
        if stored_implementation is None and require_provenance:
            raise ValueError(
                f"null record {prompt_index} lacks kv cache provenance"
            )
        if stored_implementation is not None:
            observed_implementation = normalize_kv_cache_implementation(
                stored_implementation
            )
            if observed_implementation != expected_implementation:
                raise ValueError(
                    f"null record {prompt_index} has incompatible "
                    f"kv_cache_implementation: "
                    f"{observed_implementation!r} != "
                    f"{expected_implementation!r}"
                )
            observed_version = record.get("kv_cache_version")
            expected_version = kv_cache_version(expected_implementation)
            if observed_version != expected_version:
                raise ValueError(
                    f"null record {prompt_index} has incompatible "
                    f"kv_cache_version: {observed_version!r} != "
                    f"{expected_version!r}"
                )


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
        "torch==2.4.0",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "safetensors==0.4.5",
        "huggingface_hub==0.30.2",
        "scipy==1.14.1",
        "galois==0.4.2",
        "numba==0.59.1",
        "numpy==1.26.0",
        "pytest==8.3.3",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PRC_MODEL_CACHE_DIR": "/cache/models",
        "PRC_MODEL_SIZE": MODEL_SIZE,
        "PRC_MODEL_VARIANT": "base",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl", copy=True)
    .add_local_dir("tests", "/root/tests", copy=True)
    .add_local_python_source(
        "prc", "online_prc", "qwen", "constants", "detectors",
        "watermark_expt", "proxy_8b_analysis",
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
app = modal.App("prc-online-causal", image=image)


@app.function(cpu=2.0, timeout=900)
def proxy_unit_tests_remote() -> dict:
    """Run only the focused, dependency-heavy proxy replay tests on Modal CPU."""
    import subprocess

    command = [
        "python",
        "-m",
        "pytest",
        "-q",
        "/root/tests/test_qwen_kv_cache.py",
        "/root/tests/test_proxy_8b_analysis.py",
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"proxy Modal CPU tests failed\n{completed.stdout}\n{completed.stderr}"
        )
    return {
        "passed": True,
        "command": command,
        "stdout": completed.stdout,
        "generation_attempts": 0,
        "gpu_workers": 0,
    }


@app.function(cpu=2.0, volumes={"/data": data_vol}, timeout=1800)
def proxy_8b_native_quality_shard(prompt_indices: list[int]) -> dict:
    """Read cached 8B traces and return exact T=1024 quality fields only."""
    import torch
    from detectors import tensor_sha256
    from proxy_8b_analysis import (
        NULL_TRACE_T,
        PRC_AUDITS,
        cached_quality_metrics,
    )

    _numpy_pickle_compat()
    indices = [int(index) for index in prompt_indices]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("quality shard indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= CANONICAL_NUM_PROMPTS:
        raise ValueError("quality shard index is outside the canonical corpus")
    data_vol.reload()
    rows = []
    for audit in PRC_AUDITS:
        artifact = torch.load(
            artifact_path(audit["source_tag"]),
            weights_only=False,
            map_location="cpu",
        )
        for index in indices:
            record = torch.load(
                os.path.join(wm_dir(audit["source_tag"]), f"wm_{index:04d}.pt"),
                weights_only=False,
                map_location="cpu",
            )
            validate_online_watermarked_record(record, artifact, index)
            tokens = torch.as_tensor(record["tokens"], dtype=torch.long)[:1024]
            rows.append({
                "prompt_index": index,
                "prompt_id": f"prompt-{index}",
                "method": "online_prc",
                "eta": float(audit["eta"]),
                "sample_type": "watermarked",
                "prefix_length": 1024,
                "boundary_status": audit["boundary_status"],
                "quality_likelihood_model": "Qwen3-8B-Base",
                "generated_token_hash": tensor_sha256(tokens.contiguous()),
                "source_tag": audit["source_tag"],
                "generation_attempts": 0,
                **cached_quality_metrics(
                    tokens,
                    record["base_token_logprob"],
                    prefix_length=1024,
                ),
            })

    # The same canonical null texts are shared across all eta values, so emit
    # each null exactly once rather than pretending there are four samples.
    reference_artifact = torch.load(
        artifact_path(PRC_AUDITS[-1]["source_tag"]),
        weights_only=False,
        map_location="cpu",
    )
    null_manifest = load_null_cache_manifest(NULL_TRACE_T, "8B")
    if null_manifest is None:
        raise FileNotFoundError("the canonical 8B null cache manifest is missing")
    for index in indices:
        record = torch.load(
            os.path.join(
                shared_null_dir(NULL_TRACE_T, "8B"),
                f"null_{index:04d}.pt",
            ),
            weights_only=False,
            map_location="cpu",
        )
        validate_online_null_record(
            record,
            reference_artifact,
            index,
            1024,
            source_length=NULL_TRACE_T,
            expected_kv_cache_implementation=null_manifest[
                "kv_cache_implementation"
            ],
            require_provenance=True,
        )
        tokens = torch.as_tensor(record["tokens"], dtype=torch.long)[:1024]
        rows.append({
            "prompt_index": index,
            "prompt_id": f"prompt-{index}",
            "method": "null",
            "eta": None,
            "sample_type": "null",
            "prefix_length": 1024,
            "boundary_status": "shared_across_detector_methods",
            "quality_likelihood_model": "Qwen3-8B-Base",
            "generated_token_hash": tensor_sha256(tokens.contiguous()),
            "source_tag": f"shared_null_qwen3_8b_base_T{NULL_TRACE_T}",
            "generation_attempts": 0,
            **cached_quality_metrics(
                tokens,
                record["base_token_logprob"],
                prefix_length=1024,
            ),
        })
    return {
        "prompt_indices": indices,
        "rows": rows,
        "generation_attempts": 0,
        "model_loads": 0,
    }


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


def _find_compatible_null_T(
    prompt_indices: list[int],
    requested_T: int,
    generation_model_size: str = MODEL_SIZE,
    artifact: dict | None = None,
):
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
        if artifact is not None:
            try:
                manifest = load_null_cache_manifest(length, model_size)
                if manifest is not None:
                    incompatibility = null_cache_manifest_compatibility_error(
                        manifest, artifact, length
                    )
                    if incompatibility:
                        continue
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        if all(
            os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
            for index in prompt_indices
        ):
            return length
    return None


@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(tag: str, prompt_indices: list[int], T: int,
                    allow_wm_reuse: bool = True,
                    null_kv_cache_implementation: str = "",
                    include_null: bool = True) -> dict:
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
    null_kv_cache_implementation = resolve_null_kv_cache_implementation(
        null_kv_cache_implementation, kv_cache_implementation
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

    null_T = None
    null_missing = []
    if include_null:
        null_T = _find_compatible_null_T(
            prompt_indices, T, generation_model_size, requested_artifact
        )
    if include_null and null_T is None:
        null_T = int(T)
        target_manifest = load_null_cache_manifest(
            null_T, generation_model_size
        )
        if target_manifest is not None:
            incompatibility = null_cache_manifest_compatibility_error(
                target_manifest,
                requested_artifact,
                null_T,
                null_kv_cache_implementation,
            )
            if incompatibility:
                raise ValueError(
                    f"cannot append to incompatible null cache T={null_T}: "
                    f"{incompatibility}"
                )
        elif (
            null_kv_cache_implementation != DEFAULT_KV_CACHE_IMPLEMENTATION
            and os.path.isdir(shared_null_dir(null_T, generation_model_size))
            and any(
                name.startswith("null_") and name.endswith(".pt")
                for name in os.listdir(
                    shared_null_dir(null_T, generation_model_size)
                )
            )
        ):
            raise ValueError(
                f"cannot mix {null_kv_cache_implementation} records into "
                f"legacy manifestless null cache T={null_T}"
            )
        null_missing = [
            index for index in prompt_indices
            if not os.path.exists(
                os.path.join(
                    shared_null_dir(T, generation_model_size),
                    f"null_{index:04d}.pt",
                )
            )
        ]
    elif include_null:
        null_missing = []
    else:
        null_T = int(T)
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
        "null_kv_cache_implementation": null_kv_cache_implementation,
        "null_kv_cache_version": kv_cache_version(
            null_kv_cache_implementation
        ),
    }


@app.function(volumes={"/data": data_vol}, timeout=1800)
def plan_null_cache_generation(
    tag: str,
    prompt_indices: list[int],
    requested_T: int,
    null_kv_cache_implementation: str,
) -> dict:
    """Validate reusable null records and identify only genuine missing work."""
    import uuid

    import torch

    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    model_size = artifact_generation_model_size(artifact)
    requested_T = int(requested_T)
    implementation = normalize_kv_cache_implementation(
        null_kv_cache_implementation
    )
    null_root = os.path.dirname(shared_null_dir(0, model_size))
    candidates = []
    if os.path.isdir(null_root):
        for name in os.listdir(null_root):
            match = re.fullmatch(r"T(\d+)", name)
            if match and int(match.group(1)) >= requested_T:
                candidates.append(int(match.group(1)))

    rejected_candidates = []
    for length in sorted(candidates):
        directory = shared_null_dir(length, model_size)
        paths = {
            index: os.path.join(directory, f"null_{index:04d}.pt")
            for index in prompt_indices
        }
        if not all(os.path.isfile(path) for path in paths.values()):
            continue
        try:
            manifest = load_null_cache_manifest(length, model_size)
            if manifest is not None:
                incompatibility = null_cache_manifest_compatibility_error(
                    manifest, artifact, length
                )
                if incompatibility:
                    raise ValueError(incompatibility)
            for index, path in paths.items():
                record = torch.load(
                    path, weights_only=False, map_location="cpu"
                )
                validate_online_null_record(
                    record,
                    artifact,
                    index,
                    requested_T,
                    source_length=length,
                    expected_kv_cache_implementation=(
                        manifest.get("kv_cache_implementation")
                        if manifest is not None else None
                    ),
                    require_provenance=manifest is not None,
                )
        except Exception as exc:
            rejected_candidates.append({
                "T": int(length),
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue
        return {
            "null_T": int(length),
            "null_missing": [],
            "null_invalid": [],
            "null_rejected_candidates": rejected_candidates,
            "null_kv_cache_implementation": (
                manifest.get("kv_cache_implementation")
                if manifest is not None else None
            ),
            "null_kv_cache_version": (
                manifest.get("kv_cache_version")
                if manifest is not None else None
            ),
            "legacy_manifestless": manifest is None,
        }

    directory = shared_null_dir(requested_T, model_size)
    os.makedirs(directory, exist_ok=True)
    manifest = load_null_cache_manifest(requested_T, model_size)
    legacy_manifestless = False
    if manifest is not None:
        incompatibility = null_cache_manifest_compatibility_error(
            manifest, artifact, requested_T, implementation
        )
        if incompatibility:
            raise ValueError(
                f"cannot append to incompatible null cache T={requested_T}: "
                f"{incompatibility}"
            )
    else:
        existing = [
            name for name in os.listdir(directory)
            if name.startswith("null_") and name.endswith(".pt")
        ]
        legacy_manifestless = bool(existing)
        if existing and implementation != DEFAULT_KV_CACHE_IMPLEMENTATION:
            raise ValueError(
                f"cannot mix {implementation} records into legacy "
                f"manifestless null cache T={requested_T}"
            )

    missing = []
    invalid = []
    quarantine_dir = os.path.join(directory, "_quarantine")
    for index in prompt_indices:
        path = os.path.join(directory, f"null_{index:04d}.pt")
        if not os.path.isfile(path):
            missing.append(int(index))
            continue
        try:
            record = torch.load(path, weights_only=False, map_location="cpu")
            validate_online_null_record(
                record,
                artifact,
                index,
                requested_T,
                source_length=requested_T,
                expected_kv_cache_implementation=(
                    manifest.get("kv_cache_implementation")
                    if manifest is not None else None
                ),
                require_provenance=manifest is not None,
            )
        except Exception as exc:
            os.makedirs(quarantine_dir, exist_ok=True)
            quarantined_path = os.path.join(
                quarantine_dir,
                f"null_{index:04d}-{uuid.uuid4().hex}.pt",
            )
            os.replace(path, quarantined_path)
            invalid.append({
                "prompt_idx": int(index),
                "reason": f"{type(exc).__name__}: {exc}",
                "quarantined_path": quarantined_path,
            })
            missing.append(int(index))
    if invalid:
        data_vol.commit()
    return {
        "null_T": requested_T,
        "null_missing": missing,
        "null_invalid": invalid,
        "null_rejected_candidates": rejected_candidates,
        "null_kv_cache_implementation": implementation,
        "null_kv_cache_version": kv_cache_version(implementation),
        "legacy_manifestless": legacy_manifestless,
    }


@app.function(volumes={"/data": data_vol}, timeout=1800)
def verify_shared_null_cache(
    tag: str,
    prompt_indices: list[int],
    null_T: int,
) -> dict:
    """Perform the cache-only acceptance audit after null generation."""
    import torch

    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    model_size = artifact_generation_model_size(artifact)
    null_T = int(null_T)
    manifest = load_null_cache_manifest(null_T, model_size)
    if manifest is not None:
        incompatibility = null_cache_manifest_compatibility_error(
            manifest, artifact, null_T
        )
        if incompatibility:
            raise ValueError(
                f"null cache T={null_T} manifest is incompatible: "
                f"{incompatibility}"
            )

    provenance_counts = {}
    for index in prompt_indices:
        path = os.path.join(
            shared_null_dir(null_T, model_size),
            f"null_{index:04d}.pt",
        )
        record = torch.load(path, weights_only=False, map_location="cpu")
        validate_online_null_record(
            record,
            artifact,
            index,
            null_T,
            source_length=null_T,
            expected_kv_cache_implementation=(
                manifest.get("kv_cache_implementation")
                if manifest is not None else None
            ),
            require_provenance=manifest is not None,
        )
        provenance = (
            str(
                record.get("kv_cache_implementation")
                or "legacy-unversioned"
            ),
            str(record.get("kv_cache_version") or "legacy-unversioned"),
        )
        label = "/".join(provenance)
        provenance_counts[label] = provenance_counts.get(label, 0) + 1
    return {
        "verified": len(prompt_indices),
        "prompt_indices": [int(index) for index in prompt_indices],
        "null_T": null_T,
        "generation_model_size": model_size,
        "generation_model": model_display(model_size),
        "manifest": manifest,
        "legacy_manifestless": manifest is None,
        "provenance_counts": provenance_counts,
        "model_token_positions_processed": 0,
    }


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def plan_cross_model_entropy_audits(
    audits: list[dict],
    prompt_indices: list[int],
    entropy_model_size: str,
    null_T: int,
) -> dict:
    """Validate generation inputs and inventory reusable derived traces."""
    import torch
    from detectors import semantic_sha256, tensor_sha256

    _numpy_pickle_compat()
    data_vol.reload()
    indices = [int(index) for index in prompt_indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("cross-model audit prompt indices are invalid")
    entropy_model_size = normalize_model_size(entropy_model_size)
    null_T = int(null_T)
    plans = []
    reference_artifact = None
    shared_null_identity = None

    def trace_status(path, identity, require_full_entropy=False):
        if not os.path.isfile(path):
            return False, None
        try:
            payload = torch.load(path, weights_only=False, map_location="cpu")
            validate_cross_model_entropy_trace(
                payload,
                require_full_entropy=bool(require_full_entropy),
                **identity,
            )
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    for audit in audits:
        source_tag = str(audit["source_tag"])
        prefix_T = int(audit["prefix_T"])
        trace_T = int(audit.get("trace_T", prefix_T))
        estimator_chunk_size = int(audit.get("estimator_chunk_size", 1))
        if estimator_chunk_size <= 0:
            raise ValueError("cross-model estimator chunk size must be positive")
        require_full_entropy = bool(audit.get("require_full_entropy", False))
        artifact = torch.load(
            artifact_path(source_tag), weights_only=False, map_location="cpu"
        )
        generation_model_size = artifact_generation_model_size(artifact)
        if prefix_T <= 0 or trace_T < prefix_T or trace_T > int(artifact["T"]):
            raise ValueError(
                f"audit prefix/trace T={prefix_T}/{trace_T} is incompatible "
                "with source artifact T="
                f"{artifact['T']} for {source_tag}"
            )
        partition_hash = tensor_sha256(artifact["partition"])
        prompt_corpus_hash = semantic_sha256(artifact["prompt_ids_list"])
        null_identity = (
            generation_model_size,
            partition_hash,
            prompt_corpus_hash,
        )
        if shared_null_identity is None:
            shared_null_identity = null_identity
            reference_artifact = artifact
        elif null_identity != shared_null_identity:
            raise ValueError(
                "audits cannot share null entropy traces because their "
                "generation model, partition, or prompt corpus differs"
            )

        missing = []
        invalid = []
        for index in indices:
            record_path = os.path.join(
                wm_dir(source_tag), f"wm_{index:04d}.pt"
            )
            if not os.path.isfile(record_path):
                raise FileNotFoundError(
                    f"cache-only cross-model audit is missing {record_path}"
                )
            record = torch.load(
                record_path, weights_only=False, map_location="cpu"
            )
            validate_online_watermarked_record(record, artifact, index)
            tokens = torch.as_tensor(record["tokens"], dtype=torch.long)[
                :trace_T
            ].contiguous()
            identity = {
                "source": "wm",
                "prompt_index": index,
                "trace_T": trace_T,
                "generation_model_size": generation_model_size,
                "entropy_model_size": entropy_model_size,
                "partition_sha256": partition_hash,
                "prompt_sha256": tensor_sha256(torch.as_tensor(
                    artifact["prompt_ids_list"][index], dtype=torch.long
                )),
                "tokens_sha256": tensor_sha256(tokens),
                "source_artifact_fingerprint": artifact[
                    "artifact_fingerprint"
                ],
                "estimator_chunk_size": estimator_chunk_size,
            }
            path = cross_model_entropy_trace_path(
                "wm",
                index,
                trace_T,
                entropy_model_size,
                generation_model_size,
                source_tag,
                estimator_chunk_size,
            )
            valid, reason = trace_status(
                path, identity, require_full_entropy=require_full_entropy
            )
            if not valid:
                missing.append(index)
                if reason is not None:
                    invalid.append({
                        "prompt_idx": index,
                        "path": path,
                        "reason": reason,
                    })
        plans.append({
            **dict(audit),
            "source_tag": source_tag,
            "prefix_T": prefix_T,
            "trace_T": trace_T,
            "require_full_entropy": require_full_entropy,
            "estimator_chunk_size": estimator_chunk_size,
            "source_T": int(artifact["T"]),
            "generation_model_size": generation_model_size,
            "artifact_fingerprint": artifact["artifact_fingerprint"],
            "wm_trace_missing": missing,
            "wm_trace_invalid": invalid,
            "wm_trace_cached": len(indices) - len(missing),
        })

    if reference_artifact is None:
        raise ValueError("at least one cross-model audit is required")
    generation_model_size = artifact_generation_model_size(reference_artifact)
    if null_T < max(int(plan["prefix_T"]) for plan in plans):
        raise ValueError("shared null trace is shorter than an audit prefix")
    null_manifest = load_null_cache_manifest(null_T, generation_model_size)
    if null_manifest is None:
        raise FileNotFoundError(
            f"shared null cache T={null_T} has no provenance manifest"
        )
    incompatibility = null_cache_manifest_compatibility_error(
        null_manifest, reference_artifact, null_T
    )
    if incompatibility:
        raise ValueError(
            f"shared null cache T={null_T} is incompatible: {incompatibility}"
        )

    partition_hash = tensor_sha256(reference_artifact["partition"])
    require_null_full_entropy = any(
        bool(plan.get("require_full_entropy", False)) for plan in plans
    )
    null_chunk_sizes = {
        int(plan.get("estimator_chunk_size", 1)) for plan in plans
    }
    if len(null_chunk_sizes) != 1:
        raise ValueError("shared null proxy trace requires one estimator chunk size")
    null_estimator_chunk_size = next(iter(null_chunk_sizes))
    null_missing = []
    null_invalid = []
    for index in indices:
        record_path = os.path.join(
            shared_null_dir(null_T, generation_model_size),
            f"null_{index:04d}.pt",
        )
        if not os.path.isfile(record_path):
            raise FileNotFoundError(
                f"cache-only cross-model audit is missing {record_path}"
            )
        record = torch.load(
            record_path, weights_only=False, map_location="cpu"
        )
        validate_online_null_record(
            record,
            reference_artifact,
            index,
            null_T,
            source_length=null_T,
            expected_kv_cache_implementation=null_manifest[
                "kv_cache_implementation"
            ],
            require_provenance=True,
        )
        tokens = torch.as_tensor(record["tokens"], dtype=torch.long)[
            :null_T
        ].contiguous()
        identity = {
            "source": "null",
            "prompt_index": index,
            "trace_T": null_T,
            "generation_model_size": generation_model_size,
            "entropy_model_size": entropy_model_size,
            "partition_sha256": partition_hash,
            "prompt_sha256": tensor_sha256(torch.as_tensor(
                reference_artifact["prompt_ids_list"][index],
                dtype=torch.long,
            )),
            "tokens_sha256": tensor_sha256(tokens),
            "estimator_chunk_size": null_estimator_chunk_size,
        }
        path = cross_model_entropy_trace_path(
            "null",
            index,
            null_T,
            entropy_model_size,
            generation_model_size,
            estimator_chunk_size=null_estimator_chunk_size,
        )
        valid, reason = trace_status(
            path,
            identity,
            require_full_entropy=require_null_full_entropy,
        )
        if not valid:
            null_missing.append(index)
            if reason is not None:
                null_invalid.append({
                    "prompt_idx": index,
                    "path": path,
                    "reason": reason,
                })
    return {
        "audits": plans,
        "prompt_indices": indices,
        "entropy_model_size": entropy_model_size,
        "entropy_model": model_display(entropy_model_size),
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "null_T": null_T,
        "null_trace_missing": null_missing,
        "null_trace_invalid": null_invalid,
        "null_trace_cached": len(indices) - len(null_missing),
        "require_null_full_entropy": require_null_full_entropy,
        "null_estimator_chunk_size": null_estimator_chunk_size,
        "null_manifest": null_manifest,
        "generation_records_verified": len(indices) * (len(plans) + 1),
    }


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def plan_textseal_proxy_entropy(prompt_indices: list[int]) -> dict:
    """Validate committed TextSeal artifacts and inventory proxy traces."""
    import torch
    from detectors import tensor_sha256
    from proxy_8b_analysis import (
        BASELINE_RUN_ID,
        PRC_AUDITS,
        textseal_proxy_trace_identity,
        textseal_proxy_trace_path,
        validate_textseal_proxy_trace,
    )

    _numpy_pickle_compat()
    indices = [int(index) for index in prompt_indices]
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("TextSeal proxy prompt indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= CANONICAL_NUM_PROMPTS:
        raise ValueError("TextSeal proxy prompt index is outside the canonical corpus")
    data_vol.reload()
    artifact = torch.load(
        artifact_path(PRC_AUDITS[-1]["source_tag"]),
        weights_only=False,
        map_location="cpu",
    )
    if artifact_generation_model_size(artifact) != "8B":
        raise ValueError("canonical proxy prompt artifact is not Qwen3-8B")

    missing_by_shard = {}
    invalid = []
    verified = 0
    for shard_index in sorted({index // 50 for index in indices}):
        requested = [index for index in indices if index // 50 == shard_index]
        raw_path = (
            f"/data/controlled_baseline_full/{BASELINE_RUN_ID}/generated/"
            f"shard_{shard_index:02d}.pt"
        )
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(raw_path)
        raw = torch.load(raw_path, weights_only=False, map_location="cpu")
        shard_indices = [int(index) for index in raw.get("prompt_indices", [])]
        if shard_indices != list(range(shard_index * 50, shard_index * 50 + 50)):
            raise ValueError(f"controlled-baseline shard {shard_index} ordering differs")
        outputs = raw.get("sequences", {}).get("textseal")
        if outputs is None or len(outputs) != 50:
            raise ValueError(f"TextSeal shard {shard_index} coverage differs")
        missing = []
        for index in requested:
            tokens = torch.as_tensor(
                outputs[index - shard_index * 50]["token_ids"], dtype=torch.long
            )[:1024].contiguous()
            if tokens.numel() != 1024:
                raise ValueError(f"TextSeal prompt {index} is not 1024 tokens")
            identity = {
                "prompt_index": index,
                "prompt_sha256": tensor_sha256(torch.as_tensor(
                    artifact["prompt_ids_list"][index], dtype=torch.long
                )),
                "tokens_sha256": tensor_sha256(tokens),
            }
            path = textseal_proxy_trace_path(index)
            try:
                if not os.path.isfile(path):
                    raise FileNotFoundError(path)
                payload = torch.load(path, weights_only=False, map_location="cpu")
                validate_textseal_proxy_trace(payload, **identity)
            except Exception as exc:
                missing.append(index)
                if not isinstance(exc, FileNotFoundError):
                    invalid.append({
                        "prompt_index": index,
                        "path": path,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
            verified += 1
        if missing:
            missing_by_shard[shard_index] = missing
    return {
        "baseline_run_id": BASELINE_RUN_ID,
        "prompt_indices": indices,
        "generation_records_verified": verified,
        "missing_by_shard": missing_by_shard,
        "missing_trace_records": sum(map(len, missing_by_shard.values())),
        "cached_trace_records": len(indices) - sum(map(len, missing_by_shard.values())),
        "invalid_traces": invalid,
        "teacher_forced_token_positions": 1024 * sum(
            map(len, missing_by_shard.values())
        ),
        "generation_attempts": 0,
    }


@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=7200,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class CrossModelEntropyModel:
    """Teacher-force cached tokens with an alternate, smaller LM."""
    entropy_model_size: str = modal.parameter()
    generation_model_size: str = modal.parameter()
    trace_kv_cache_implementation: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        self.entropy_model_size = normalize_model_size(
            self.entropy_model_size
        )
        self.generation_model_size = normalize_model_size(
            self.generation_model_size
        )
        self.trace_kv_cache_implementation = (
            normalize_kv_cache_implementation(
                self.trace_kv_cache_implementation
            )
        )
        if self.trace_kv_cache_implementation != "static":
            raise ValueError(
                "cross-model entropy replay requires the optimized static "
                "KV cache"
            )
        os.environ["PRC_MODEL_SIZE"] = self.entropy_model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"
        import watermark_expt as we
        self.we = we
        hf_cache.commit()

    @modal.method()
    def ready(self) -> dict:
        return {
            "entropy_model_size": self.entropy_model_size,
            "entropy_model": model_display(self.entropy_model_size),
            "generation_model_size": self.generation_model_size,
            "generation_model": model_display(self.generation_model_size),
            "trace_kv_cache_implementation": (
                self.trace_kv_cache_implementation
            ),
            "trace_kv_cache_version": kv_cache_version(
                self.trace_kv_cache_implementation
            ),
            "model_cache_dir": os.environ.get("PRC_MODEL_CACHE_DIR", ""),
        }

    @modal.method()
    def estimate(self, request: dict) -> dict:
        import time

        import numpy as np
        import torch
        from detectors import semantic_sha256, tensor_sha256

        _numpy_pickle_compat()
        started = time.time()
        source = str(request["source"])
        artifact_tag = str(request["artifact_tag"])
        source_tag = str(request.get("source_tag", ""))
        trace_T = int(request["trace_T"])
        estimator_chunk_size = int(request.get("estimator_chunk_size", 1))
        require_full_entropy = bool(request.get("require_full_entropy", False))
        indices = [int(index) for index in request["prompt_indices"]]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("entropy replay batch indices are invalid")

        data_vol.reload()
        artifact = torch.load(
            artifact_path(artifact_tag),
            weights_only=False,
            map_location="cpu",
        )
        observed_generation_model = artifact_generation_model_size(artifact)
        if observed_generation_model != self.generation_model_size:
            raise ValueError(
                f"artifact model {observed_generation_model} does not match "
                f"{self.generation_model_size}"
            )
        partition_cpu = artifact["partition"]
        partition_hash = tensor_sha256(partition_cpu)
        null_manifest = None
        if source == "wm":
            source_artifact = torch.load(
                artifact_path(source_tag),
                weights_only=False,
                map_location="cpu",
            )
            if source_artifact["artifact_fingerprint"] != artifact[
                "artifact_fingerprint"
            ]:
                # Current campaign passes the source artifact as both values.
                # Retain an explicit check in case a future caller separates
                # detector and cache artifacts.
                incompatibility = artifact_compatibility_error(
                    artifact, source_artifact
                )
                if incompatibility:
                    raise ValueError(incompatibility)
            directory = wm_dir(source_tag)
            record_prefix = "wm"
        elif source == "null":
            source_artifact = artifact
            directory = shared_null_dir(
                trace_T, self.generation_model_size
            )
            record_prefix = "null"
            null_manifest = load_null_cache_manifest(
                trace_T, self.generation_model_size
            )
            if null_manifest is None:
                raise FileNotFoundError(
                    f"null cache T={trace_T} has no manifest"
                )
        else:
            raise ValueError(f"unknown entropy replay source {source!r}")

        records = []
        todo = []
        invalid_cached = 0
        for index in indices:
            record = torch.load(
                os.path.join(directory, f"{record_prefix}_{index:04d}.pt"),
                weights_only=False,
                map_location="cpu",
            )
            if source == "wm":
                validate_online_watermarked_record(
                    record, source_artifact, index
                )
            else:
                validate_online_null_record(
                    record,
                    artifact,
                    index,
                    trace_T,
                    source_length=trace_T,
                    expected_kv_cache_implementation=null_manifest[
                        "kv_cache_implementation"
                    ],
                    require_provenance=True,
                )
            tokens = torch.as_tensor(
                record["tokens"], dtype=torch.long
            )[:trace_T].contiguous()
            if tokens.numel() != trace_T:
                raise ValueError(
                    f"{source} record {index} is shorter than T={trace_T}"
                )
            identity = {
                "source": source,
                "prompt_index": index,
                "trace_T": trace_T,
                "generation_model_size": self.generation_model_size,
                "entropy_model_size": self.entropy_model_size,
                "partition_sha256": partition_hash,
                "prompt_sha256": tensor_sha256(torch.as_tensor(
                    artifact["prompt_ids_list"][index], dtype=torch.long
                )),
                "tokens_sha256": tensor_sha256(tokens),
                "estimator_chunk_size": estimator_chunk_size,
            }
            if source == "wm":
                identity["source_artifact_fingerprint"] = source_artifact[
                    "artifact_fingerprint"
                ]
            path = cross_model_entropy_trace_path(
                source,
                index,
                trace_T,
                self.entropy_model_size,
                self.generation_model_size,
                source_tag,
                estimator_chunk_size,
            )
            if os.path.isfile(path):
                try:
                    existing = torch.load(
                        path, weights_only=False, map_location="cpu"
                    )
                    validate_cross_model_entropy_trace(
                        existing,
                        require_full_entropy=require_full_entropy,
                        **identity,
                    )
                    continue
                except Exception:
                    invalid_cached += 1
            records.append((index, tokens, identity, path))
            todo.append(index)

        if not todo:
            return {
                "source": source,
                "trace_T": trace_T,
                "prompt_indices": indices,
                "estimated": 0,
                "cached": len(indices),
                "invalid_cached_replaced": 0,
                "seconds": time.time() - started,
                "teacher_forced_token_positions": 0,
                "model_forward_positions": 0,
                "peak_cuda_allocated_bytes": 0,
                "peak_cuda_reserved_bytes": 0,
            }

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        prompt_batch = torch.tensor(
            [artifact["prompt_ids_list"][index] for index in todo],
            dtype=torch.long,
            device=self.we.device,
        )
        token_batch = torch.stack(
            [tokens for _, tokens, _, _ in records]
        ).to(self.we.device)
        p_traces, full_entropy_traces = (
            self.we.estimate_partition_entropy_trace_batch(
            self.we.model,
            prompt_batch,
            token_batch,
            partition_cpu,
            kv_cache_implementation=self.trace_kv_cache_implementation,
            chunk_size=estimator_chunk_size,
            )
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        else:
            peak_allocated = 0
            peak_reserved = 0

        for row, (index, _, identity, path) in enumerate(records):
            p_trace = np.asarray(p_traces[row], dtype=np.float64)
            full_entropy_trace = np.asarray(
                full_entropy_traces[row], dtype=np.float64
            )
            payload = {
                **cross_model_entropy_trace_identity(**identity),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "p_trace": p_trace,
                "p_trace_sha256": semantic_sha256(p_trace),
                "full_entropy_trace": full_entropy_trace,
                "full_entropy_trace_sha256": semantic_sha256(
                    full_entropy_trace
                ),
            }
            validate_cross_model_entropy_trace(
                payload,
                require_full_entropy=require_full_entropy,
                **identity,
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(payload, path)
        data_vol.commit()
        prompt_length = int(prompt_batch.shape[1])
        return {
            "source": source,
            "trace_T": trace_T,
            "prompt_indices": indices,
            "estimated": len(todo),
            "cached": len(indices) - len(todo),
            "invalid_cached_replaced": invalid_cached,
            "seconds": time.time() - started,
            "teacher_forced_token_positions": len(todo) * trace_T,
            "model_forward_positions": len(todo) * (
                prompt_length + max(trace_T - 1, 0)
            ),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "trace_kv_cache_implementation": (
                self.trace_kv_cache_implementation
            ),
            "trace_kv_cache_version": kv_cache_version(
                self.trace_kv_cache_implementation
            ),
            "estimator_chunk_size": estimator_chunk_size,
        }

    @modal.method()
    def estimate_textseal(self, request: dict) -> dict:
        """Teacher-force committed TextSeal tokens; never sample new text."""
        import time

        import numpy as np
        import torch
        from detectors import semantic_sha256, tensor_sha256
        from proxy_8b_analysis import (
            BASELINE_RUN_ID,
            ESTIMATOR_CHUNK_SIZE,
            textseal_proxy_trace_identity,
            textseal_proxy_trace_path,
            validate_textseal_proxy_trace,
        )

        _numpy_pickle_compat()
        started = time.time()
        if self.generation_model_size != "8B" or self.entropy_model_size != "0.6B":
            raise ValueError("TextSeal proxy replay is frozen to 8B -> 0.6B")
        if str(request.get("run_id")) != BASELINE_RUN_ID:
            raise ValueError("unexpected controlled-baseline run ID")
        shard_index = int(request["shard_index"])
        indices = [int(index) for index in request["prompt_indices"]]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError("TextSeal replay indices must be nonempty and unique")

        data_vol.reload()
        raw_path = (
            f"/data/controlled_baseline_full/{BASELINE_RUN_ID}/generated/"
            f"shard_{shard_index:02d}.pt"
        )
        if not os.path.isfile(raw_path):
            raise FileNotFoundError(raw_path)
        raw = torch.load(raw_path, weights_only=False, map_location="cpu")
        shard_indices = [int(index) for index in raw.get("prompt_indices", [])]
        if len(shard_indices) != len(set(shard_indices)):
            raise ValueError("controlled-baseline shard prompt indices are invalid")
        row_by_index = {index: row for row, index in enumerate(shard_indices)}
        if any(index not in row_by_index for index in indices):
            raise ValueError("TextSeal replay request crosses its raw shard")
        outputs = raw.get("sequences", {}).get("textseal")
        if outputs is None or len(outputs) != len(shard_indices):
            raise ValueError("controlled-baseline TextSeal output coverage differs")

        # The eta=.20 artifact is used only as the validated canonical prompt
        # token corpus and fixed vocabulary partition. No PRC generation occurs.
        artifact_tag = str(request["artifact_tag"])
        artifact = torch.load(
            artifact_path(artifact_tag), weights_only=False, map_location="cpu"
        )
        if artifact_generation_model_size(artifact) != "8B":
            raise ValueError("canonical prompt artifact is not Qwen3-8B")
        partition_cpu = artifact["partition"]

        records = []
        invalid_cached = 0
        for index in indices:
            output = outputs[row_by_index[index]]
            tokens = torch.as_tensor(output["token_ids"], dtype=torch.long)[
                :1024
            ].contiguous()
            if tokens.numel() != 1024:
                raise ValueError(f"TextSeal prompt {index} is not 1024 tokens")
            prompt = torch.as_tensor(
                artifact["prompt_ids_list"][index], dtype=torch.long
            ).contiguous()
            identity = {
                "prompt_index": index,
                "prompt_sha256": tensor_sha256(prompt),
                "tokens_sha256": tensor_sha256(tokens),
            }
            path = textseal_proxy_trace_path(index)
            if os.path.isfile(path):
                try:
                    cached = torch.load(path, weights_only=False, map_location="cpu")
                    validate_textseal_proxy_trace(cached, **identity)
                    continue
                except Exception:
                    invalid_cached += 1
            records.append((index, prompt, tokens, identity, path))

        if not records:
            return {
                "source": "textseal",
                "shard_index": shard_index,
                "prompt_indices": indices,
                "estimated": 0,
                "cached": len(indices),
                "invalid_cached_replaced": 0,
                "seconds": time.time() - started,
                "teacher_forced_token_positions": 0,
                "model_forward_positions": 0,
                "peak_cuda_allocated_bytes": 0,
                "peak_cuda_reserved_bytes": 0,
            }

        prompt_batch = torch.stack([item[1] for item in records]).to(self.we.device)
        token_batch = torch.stack([item[2] for item in records]).to(self.we.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        _, entropy_traces = self.we.estimate_partition_entropy_trace_batch(
            self.we.model,
            prompt_batch,
            token_batch,
            partition_cpu,
            kv_cache_implementation=self.trace_kv_cache_implementation,
            chunk_size=ESTIMATOR_CHUNK_SIZE,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_allocated = int(torch.cuda.max_memory_allocated())
            peak_reserved = int(torch.cuda.max_memory_reserved())
        else:
            peak_allocated = peak_reserved = 0

        for row, (index, _, _, identity, path) in enumerate(records):
            values = np.asarray(entropy_traces[row], dtype=np.float64)
            payload = {
                **textseal_proxy_trace_identity(**identity),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "full_entropy_trace": values,
                "full_entropy_trace_sha256": semantic_sha256(values),
                "source_generation_artifact": raw_path,
            }
            validate_textseal_proxy_trace(payload, **identity)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(payload, path)
        data_vol.commit()
        prompt_length = int(prompt_batch.shape[1])
        return {
            "source": "textseal",
            "shard_index": shard_index,
            "prompt_indices": indices,
            "estimated": len(records),
            "cached": len(indices) - len(records),
            "invalid_cached_replaced": invalid_cached,
            "seconds": time.time() - started,
            "teacher_forced_token_positions": len(records) * 1024,
            "model_forward_positions": len(records) * (prompt_length + 1023),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
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
    null_kv_cache_implementation: str = modal.parameter()

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
        self.null_kv_cache_implementation = (
            resolve_null_kv_cache_implementation(
                self.null_kv_cache_implementation,
                self.kv_cache_implementation,
            )
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
            "null_kv_cache_implementation": (
                self.null_kv_cache_implementation
            ),
            "null_kv_cache_version": kv_cache_version(
                self.null_kv_cache_implementation
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
        import uuid

        import torch

        data_vol.reload()
        directory = shared_null_dir(self.T, self.model_size)
        os.makedirs(directory, exist_ok=True)
        manifest = load_null_cache_manifest(self.T, self.model_size)
        if manifest is not None:
            incompatibility = null_cache_manifest_compatibility_error(
                manifest,
                self.artifact,
                self.T,
                self.null_kv_cache_implementation,
            )
            if incompatibility:
                raise ValueError(
                    f"null cache T={self.T} has incompatible manifest: "
                    f"{incompatibility}"
                )

        todo = []
        quarantined = []
        quarantine_dir = os.path.join(directory, "_quarantine")
        for index in prompt_indices:
            path = os.path.join(directory, f"null_{index:04d}.pt")
            if not os.path.isfile(path):
                todo.append(index)
                continue
            try:
                record = torch.load(
                    path, weights_only=False, map_location="cpu"
                )
                validate_online_null_record(
                    record,
                    self.artifact,
                    index,
                    self.T,
                    source_length=self.T,
                    expected_kv_cache_implementation=(
                        manifest.get("kv_cache_implementation")
                        if manifest is not None else None
                    ),
                    require_provenance=manifest is not None,
                )
            except Exception as exc:
                os.makedirs(quarantine_dir, exist_ok=True)
                quarantined_path = os.path.join(
                    quarantine_dir,
                    f"null_{index:04d}-{uuid.uuid4().hex}.pt",
                )
                os.replace(path, quarantined_path)
                quarantined.append({
                    "prompt_idx": int(index),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "quarantined_path": quarantined_path,
                })
                todo.append(index)
        if not todo:
            return {
                "generated": 0,
                "cached": len(prompt_indices),
                "batch": 0,
                "seconds": 0.0,
                "kv_cache_implementation": (
                    manifest.get("kv_cache_implementation")
                    if manifest is not None else None
                ),
                "kv_cache_version": (
                    manifest.get("kv_cache_version")
                    if manifest is not None else None
                ),
                "quarantined": quarantined,
            }

        existing_records = [
            name for name in os.listdir(directory)
            if name.startswith("null_") and name.endswith(".pt")
        ]
        if manifest is None and existing_records:
            if (
                self.null_kv_cache_implementation
                != DEFAULT_KV_CACHE_IMPLEMENTATION
            ):
                raise ValueError(
                    f"cannot mix {self.null_kv_cache_implementation} records "
                    f"into legacy manifestless null cache T={self.T}"
                )
        elif manifest is None:
            manifest = expected_null_cache_manifest(
                self.artifact,
                self.T,
                self.null_kv_cache_implementation,
            )
            manifest_path = null_cache_manifest_path(self.T, self.model_size)
            temporary_manifest = (
                f"{manifest_path}.tmp-{uuid.uuid4().hex}"
            )
            with open(temporary_manifest, "w") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
            os.replace(temporary_manifest, manifest_path)

        started = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect_online(
            self.we.model,
            prompt_batch,
            self.T,
            self.key,
            self.partition,
            watermark=False,
            return_trace_details=True,
            kv_cache_implementation=self.null_kv_cache_implementation,
        )
        if int(tokens.shape[1]) != self.T:
            raise AssertionError(
                f"generated null record length {tokens.shape[1]} != {self.T}"
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
                "stopping_policy": STOPPING_POLICY,
                "source_T": self.T,
                "realized_length": self.T,
                "generation_sampler_version": (
                    NULL_GENERATION_SAMPLER_VERSION
                ),
                "generation_rng_policy": NULL_GENERATION_SAMPLER_VERSION,
                "kv_cache_implementation": (
                    self.null_kv_cache_implementation
                ),
                "kv_cache_version": kv_cache_version(
                    self.null_kv_cache_implementation
                ),
                "generation_segments": [{
                    "start": 0,
                    "end": self.T,
                    "sampler_version": NULL_GENERATION_SAMPLER_VERSION,
                    "mode": "fresh_null",
                    "kv_cache_implementation": (
                        self.null_kv_cache_implementation
                    ),
                    "kv_cache_version": kv_cache_version(
                        self.null_kv_cache_implementation
                    ),
                }],
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
            })
            path = os.path.join(directory, f"null_{index:04d}.pt")
            temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
            torch.save(record, temporary_path)
            os.replace(temporary_path, path)
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
            "seconds": time.time() - started,
            "kv_cache_implementation": self.null_kv_cache_implementation,
            "kv_cache_version": kv_cache_version(
                self.null_kv_cache_implementation
            ),
            "peak_cuda_allocated_bytes": peak_allocated_bytes,
            "peak_cuda_reserved_bytes": peak_reserved_bytes,
            "quarantined": quarantined,
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


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def detect_full_audit_prompt_shard(request: dict) -> dict:
    """Score MAP, entropy, and naive detectors for one prompt shard."""
    import time

    import torch
    from detectors import detect_online_hoeffding
    from online_prc import OnlinePRCKey

    started = time.time()
    tag = str(request["tag"])
    watermarked_source_tag = str(request["watermarked_source_tag"])
    prefix_T = int(request["prefix_T"])
    null_T = int(request["null_T"])
    fpr = float(request["fpr"])
    code_fingerprint = str(request["code_fingerprint_sha256"])
    prompt_indices = [int(index) for index in request["prompt_indices"]]
    if not prompt_indices or len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("full-audit prompt shard must be nonempty and unique")

    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    source_artifact = torch.load(
        artifact_path(watermarked_source_tag),
        weights_only=False,
        map_location="cpu",
    )
    incompatibility = artifact_compatibility_error(artifact, source_artifact)
    if incompatibility:
        raise ValueError(
            f"watermarked cache {watermarked_source_tag} is incompatible "
            f"with {tag}: {incompatibility}"
        )
    artifact_T = int(artifact["T"])
    source_T = int(source_artifact["T"])
    if prefix_T <= 0 or prefix_T > artifact_T or prefix_T > source_T:
        raise ValueError(
            f"prefix_T={prefix_T} exceeds artifact/source lengths "
            f"{artifact_T}/{source_T}"
        )

    generation_model_size = artifact_generation_model_size(artifact)
    null_manifest = load_null_cache_manifest(null_T, generation_model_size)
    if null_T < prefix_T:
        raise ValueError(
            f"null cache T={null_T} is shorter than prefix T={prefix_T}"
        )
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    output_path = full_audit_shard_path(
        tag,
        prefix_T,
        null_T,
        prompt_indices,
        artifact["artifact_fingerprint"],
        source_artifact["artifact_fingerprint"],
        code_fingerprint,
        fpr,
    )
    validation_kwargs = {
        "tag": tag,
        "watermarked_source_tag": watermarked_source_tag,
        "prefix_T": prefix_T,
        "null_T": null_T,
        "fpr": fpr,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "watermarked_source_fingerprint": source_artifact[
            "artifact_fingerprint"
        ],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
    }
    if os.path.isfile(output_path):
        cached = torch.load(
            output_path, weights_only=False, map_location="cpu"
        )
        cached_indices = validate_full_audit_shard(
            cached, **validation_kwargs
        )
        if cached_indices != prompt_indices:
            raise ValueError("cached full-audit shard prompt order is wrong")
        return {
            "remote_output_path": output_path,
            "prompt_indices": prompt_indices,
            "num_prompts": len(prompt_indices),
            "cached": True,
            "seconds": time.time() - started,
        }

    results = []
    for watermark in (True, False):
        directory = (
            wm_dir(watermarked_source_tag)
            if watermark
            else shared_null_dir(null_T, generation_model_size)
        )
        record_prefix = "wm" if watermark else "null"
        source = source_artifact if watermark else artifact
        for index in prompt_indices:
            path = os.path.join(
                directory, f"{record_prefix}_{index:04d}.pt"
            )
            record = torch.load(path, weights_only=False, map_location="cpu")
            if len(record["tokens"]) < prefix_T:
                raise ValueError(f"record {path} is shorter than T={prefix_T}")
            if len(record["p_trace"]) < prefix_T:
                raise ValueError(
                    f"record {path} has a short probability trace"
                )
            if watermark:
                validate_online_watermarked_record(record, source, index)
            else:
                validate_online_null_record(
                    record,
                    artifact,
                    index,
                    prefix_T,
                    source_length=null_T,
                    expected_kv_cache_implementation=(
                        null_manifest.get("kv_cache_implementation")
                        if null_manifest is not None else None
                    ),
                    require_provenance=null_manifest is not None,
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
                    raise AssertionError(
                        "full-audit shard detector used the wrong length"
                    )
                scored[weight] = {"decision": bool(decision), **info}
            results.append({
                "prompt_idx": int(index),
                "watermark": watermark,
                "scores": scored,
            })

    elapsed = time.time() - started
    payload = {
        "full_audit_shard_schema_version": FULL_AUDIT_SHARD_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "online_full_audit_prompt_shard",
        "tag": tag,
        "watermarked_source_tag": watermarked_source_tag,
        "source_T": source_T,
        "T": prefix_T,
        "null_T": null_T,
        "target_fpr": fpr,
        "fpr_policy": FPR_POLICY,
        "prompt_indices": prompt_indices,
        "num_prompts": len(prompt_indices),
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "watermarked_source_artifact_fingerprint": source_artifact[
            "artifact_fingerprint"
        ],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
        "cpu_detection_seconds": elapsed,
        "results": results,
    }
    validate_full_audit_shard(payload, **validation_kwargs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    if not os.path.isfile(output_path):
        raise IOError(f"failed to persist full-audit shard {output_path}")
    data_vol.commit()
    return {
        "remote_output_path": output_path,
        "prompt_indices": prompt_indices,
        "num_prompts": len(prompt_indices),
        "cached": False,
        "seconds": elapsed,
    }


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def aggregate_full_audit_shards(
    tag: str,
    prefix_T: int,
    prompt_indices: list[int],
    null_T: int,
    fpr: float,
    batch: int,
    code_fingerprint_sha256: str,
    watermarked_source_tag: str,
    watermarked_cache_mode: str,
    watermarked_resume_source_tag: str,
    watermarked_resume_source_T: int,
    shard_summaries: list[dict],
    detection_wall_seconds: float,
) -> dict:
    """Validate, merge, count, and persist full-detector prompt shards."""
    import time

    import torch
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    started = time.time()
    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    source_artifact = torch.load(
        artifact_path(watermarked_source_tag),
        weights_only=False,
        map_location="cpu",
    )
    incompatibility = artifact_compatibility_error(artifact, source_artifact)
    if incompatibility:
        raise ValueError(
            f"watermarked cache {watermarked_source_tag} is incompatible "
            f"with {tag}: {incompatibility}"
        )
    generation_model_size = artifact_generation_model_size(artifact)
    null_manifest = load_null_cache_manifest(null_T, generation_model_size)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    prefix_T = int(prefix_T)
    indices = [int(index) for index in prompt_indices]
    validation_kwargs = {
        "tag": tag,
        "watermarked_source_tag": watermarked_source_tag,
        "prefix_T": prefix_T,
        "null_T": int(null_T),
        "fpr": float(fpr),
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "watermarked_source_fingerprint": source_artifact[
            "artifact_fingerprint"
        ],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint_sha256,
    }
    shard_payloads = []
    shard_inventory = []
    for summary in shard_summaries:
        path = str(summary["remote_output_path"])
        payload = torch.load(path, weights_only=False, map_location="cpu")
        validated_indices = validate_full_audit_shard(
            payload, **validation_kwargs
        )
        declared_indices = [
            int(index) for index in summary["prompt_indices"]
        ]
        if validated_indices != declared_indices:
            raise ValueError(
                f"full-audit shard summary disagrees with payload at {path}"
            )
        shard_payloads.append(payload)
        shard_inventory.append({
            "remote_output_path": path,
            "prompt_indices": validated_indices,
            "num_prompts": len(validated_indices),
            "cached_this_invocation": bool(summary.get("cached", False)),
            "invocation_seconds": float(summary.get("seconds", 0.0)),
            "original_cpu_detection_seconds": float(
                payload.get("cpu_detection_seconds", 0.0)
            ),
        })

    results = merge_full_audit_shards(shard_payloads, indices)
    wm = [result for result in results if result["watermark"]]
    null = [result for result in results if not result["watermark"]]
    counts = {}
    for weight in ("map", "entropy", "naive"):
        counts[weight] = {
            "tp": sum(result["scores"][weight]["decision"] for result in wm),
            "fp": sum(
                result["scores"][weight]["decision"] for result in null
            ),
            "watermarked_total": len(wm),
            "null_total": len(null),
        }

    aggregation_seconds = time.time() - started
    source_T = int(source_artifact["T"])
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "prompt_sharded_online_full_audit",
        "detection_strategy": "prompt_sharded_full_audit_v1",
        "tag": tag,
        "n": prefix_T,
        "T": prefix_T,
        "t": key.check_weight,
        "eta": key.noise_rate,
        "r": target_row_count(prefix_T, key),
        "free_coordinates": prefix_T - target_row_count(prefix_T, key),
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
        "num_prompts": len(indices),
        "prompt_indices": indices,
        "batch": int(batch),
        "null_cache_T": int(null_T),
        "null_cache_manifest": null_manifest,
        "null_kv_cache_implementation": (
            null_manifest.get("kv_cache_implementation")
            if null_manifest is not None else None
        ),
        "null_kv_cache_version": (
            null_manifest.get("kv_cache_version")
            if null_manifest is not None else None
        ),
        "watermarked_cache_mode": watermarked_cache_mode,
        "watermarked_cache_T": source_T,
        "watermarked_cache_tag": watermarked_source_tag,
        "watermarked_resume_source_tag": (
            watermarked_resume_source_tag or None
        ),
        "watermarked_resume_source_T": (
            int(watermarked_resume_source_T) or None
        ),
        "watermarked_source_artifact_fingerprint": source_artifact[
            "artifact_fingerprint"
        ],
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "online_support_sha256": support_sha256(prefix_T, key),
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "cpu_detection_wall_seconds": float(detection_wall_seconds),
        "cpu_aggregation_seconds": aggregation_seconds,
        "cpu_detection_invocation_seconds": sum(
            item["invocation_seconds"] for item in shard_inventory
        ),
        "detection_shard_count": len(shard_inventory),
        "detection_shard_cache_hits": sum(
            item["cached_this_invocation"] for item in shard_inventory
        ),
        "detection_shards": shard_inventory,
        "counts": counts,
        "results": results,
    }
    output_dir = f"/data/{tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"full-audit-prefix-T{prefix_T}_fpr-"
        f"{_slug(f'{float(fpr):.12g}')}_prompts-{len(indices)}.pt",
    )
    torch.save(payload, output_path)
    if not os.path.isfile(output_path):
        raise IOError(f"failed to persist full audit {output_path}")
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def detect_cross_model_entropy_prompt_shard(request: dict) -> dict:
    """Run MAP and entropy detectors from alternate-model traces."""
    import time

    import torch
    from detectors import detect_online_hoeffding, tensor_sha256
    from online_prc import OnlinePRCKey

    _numpy_pickle_compat()
    started = time.time()
    source_tag = str(request["source_tag"])
    prefix_T = int(request["prefix_T"])
    watermarked_trace_T = int(
        request.get("watermarked_trace_T", prefix_T)
    )
    estimator_chunk_size = int(request.get("estimator_chunk_size", 1))
    null_T = int(request["null_T"])
    null_trace_T = int(request["null_trace_T"])
    fpr = float(request["fpr"])
    entropy_model_size = normalize_model_size(
        request["entropy_model_size"]
    )
    code_fingerprint = str(request["code_fingerprint_sha256"])
    indices = [int(index) for index in request["prompt_indices"]]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("cross-model detector shard indices are invalid")

    data_vol.reload()
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    if not 0 < prefix_T <= watermarked_trace_T <= int(artifact["T"]):
        raise ValueError("cross-model detector prefix exceeds source artifact")
    if null_T < prefix_T or null_trace_T < prefix_T:
        raise ValueError("null generation/trace cache is shorter than prefix")
    null_manifest = load_null_cache_manifest(null_T, generation_model_size)
    if null_manifest is None:
        raise FileNotFoundError(f"null cache T={null_T} has no manifest")
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    partition_hash = tensor_sha256(partition)
    output_path = cross_model_entropy_audit_shard_path(
        source_tag,
        prefix_T,
        null_trace_T,
        indices,
        entropy_model_size,
        artifact["artifact_fingerprint"],
        code_fingerprint,
        fpr,
    )
    validation_kwargs = {
        "source_tag": source_tag,
        "prefix_T": prefix_T,
        "null_T": null_T,
        "null_trace_T": null_trace_T,
        "fpr": fpr,
        "generation_model_size": generation_model_size,
        "entropy_model_size": entropy_model_size,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
    }
    if os.path.isfile(output_path):
        cached = torch.load(
            output_path, weights_only=False, map_location="cpu"
        )
        cached_indices = validate_cross_model_entropy_audit_shard(
            cached, **validation_kwargs
        )
        if cached_indices != indices:
            raise ValueError("cached cross-model shard prompt order is wrong")
        return {
            "remote_output_path": output_path,
            "prompt_indices": indices,
            "num_prompts": len(indices),
            "cached": True,
            "seconds": time.time() - started,
        }

    results = []
    for watermark in (True, False):
        source = "wm" if watermark else "null"
        record_prefix = source
        directory = (
            wm_dir(source_tag)
            if watermark
            else shared_null_dir(null_T, generation_model_size)
        )
        trace_T = watermarked_trace_T if watermark else null_trace_T
        for index in indices:
            record = torch.load(
                os.path.join(directory, f"{record_prefix}_{index:04d}.pt"),
                weights_only=False,
                map_location="cpu",
            )
            if watermark:
                validate_online_watermarked_record(record, artifact, index)
            else:
                validate_online_null_record(
                    record,
                    artifact,
                    index,
                    prefix_T,
                    source_length=null_T,
                    expected_kv_cache_implementation=null_manifest[
                        "kv_cache_implementation"
                    ],
                    require_provenance=True,
                )
            full_tokens = torch.as_tensor(
                record["tokens"], dtype=torch.long
            )[:trace_T].contiguous()
            identity = {
                "source": source,
                "prompt_index": index,
                "trace_T": trace_T,
                "generation_model_size": generation_model_size,
                "entropy_model_size": entropy_model_size,
                "partition_sha256": partition_hash,
                "prompt_sha256": tensor_sha256(torch.as_tensor(
                    artifact["prompt_ids_list"][index], dtype=torch.long
                )),
                "tokens_sha256": tensor_sha256(full_tokens),
                "estimator_chunk_size": estimator_chunk_size,
            }
            if watermark:
                identity["source_artifact_fingerprint"] = artifact[
                    "artifact_fingerprint"
                ]
            trace_path = cross_model_entropy_trace_path(
                source,
                index,
                trace_T,
                entropy_model_size,
                generation_model_size,
                source_tag,
                estimator_chunk_size,
            )
            trace_payload = torch.load(
                trace_path, weights_only=False, map_location="cpu"
            )
            probabilities = validate_cross_model_entropy_trace(
                trace_payload, **identity
            )[:prefix_T]
            tokens = full_tokens[:prefix_T]
            scores = {}
            for weight in ("map", "entropy"):
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
                    raise AssertionError(
                        f"cross-model {weight} detector used the wrong length"
                    )
                scores[weight] = {"decision": bool(decision), **info}
            results.append({
                "prompt_idx": index,
                "watermark": watermark,
                "scores": scores,
            })

    elapsed = time.time() - started
    payload = {
        "cross_model_entropy_audit_shard_schema_version": (
            CROSS_MODEL_ENTROPY_AUDIT_SHARD_SCHEMA_VERSION
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": "online_cross_model_map_entropy_prompt_shard",
        "source_tag": source_tag,
        "T": prefix_T,
        "null_T": null_T,
        "null_trace_T": null_trace_T,
        "watermarked_trace_T": watermarked_trace_T,
        "estimator_chunk_size": estimator_chunk_size,
        "target_fpr": fpr,
        "fpr_policy": FPR_POLICY,
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "entropy_model_size": entropy_model_size,
        "entropy_model": model_display(entropy_model_size),
        "entropy_trace_source": cross_model_entropy_trace_source(
            entropy_model_size, generation_model_size
        ),
        "prompt_indices": indices,
        "num_prompts": len(indices),
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
        "cpu_detection_seconds": elapsed,
        "results": results,
    }
    validate_cross_model_entropy_audit_shard(
        payload, **validation_kwargs
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(payload, output_path)
    data_vol.commit()
    return {
        "remote_output_path": output_path,
        "prompt_indices": indices,
        "num_prompts": len(indices),
        "cached": False,
        "seconds": elapsed,
    }


@app.function(cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
def aggregate_cross_model_entropy_audit_shards(
    audit: dict,
    prompt_indices: list[int],
    entropy_model_size: str,
    null_T: int,
    null_trace_T: int,
    fpr: float,
    code_fingerprint_sha256: str,
    shard_summaries: list[dict],
    detection_wall_seconds: float,
) -> dict:
    """Merge and persist one cross-model MAP/entropy audit."""
    import time

    import torch
    from online_prc import OnlinePRCKey, support_sha256, target_row_count

    _numpy_pickle_compat()
    started = time.time()
    data_vol.reload()
    source_tag = str(audit["source_tag"])
    prefix_T = int(audit["prefix_T"])
    artifact = torch.load(
        artifact_path(source_tag), weights_only=False, map_location="cpu"
    )
    generation_model_size = artifact_generation_model_size(artifact)
    entropy_model_size = normalize_model_size(entropy_model_size)
    key = OnlinePRCKey.from_dict(artifact["online_key"])
    indices = [int(index) for index in prompt_indices]
    validation_kwargs = {
        "source_tag": source_tag,
        "prefix_T": prefix_T,
        "null_T": int(null_T),
        "null_trace_T": int(null_trace_T),
        "fpr": float(fpr),
        "generation_model_size": generation_model_size,
        "entropy_model_size": entropy_model_size,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "code_fingerprint_sha256": code_fingerprint_sha256,
    }
    shard_payloads = []
    inventory = []
    for summary in shard_summaries:
        path = str(summary["remote_output_path"])
        payload = torch.load(path, weights_only=False, map_location="cpu")
        validated = validate_cross_model_entropy_audit_shard(
            payload, **validation_kwargs
        )
        declared = [int(index) for index in summary["prompt_indices"]]
        if validated != declared:
            raise ValueError(
                f"cross-model shard summary disagrees with {path}"
            )
        shard_payloads.append(payload)
        inventory.append({
            "remote_output_path": path,
            "prompt_indices": validated,
            "num_prompts": len(validated),
            "cached_this_invocation": bool(summary.get("cached", False)),
            "invocation_seconds": float(summary.get("seconds", 0.0)),
            "original_cpu_detection_seconds": float(
                payload.get("cpu_detection_seconds", 0.0)
            ),
        })
    results = merge_cross_model_entropy_audit_shards(
        shard_payloads, indices
    )
    wm = [result for result in results if result["watermark"]]
    null = [result for result in results if not result["watermark"]]
    counts = {}
    for weight in ("map", "entropy"):
        counts[weight] = {
            "tp": sum(
                result["scores"][weight]["decision"] for result in wm
            ),
            "fp": sum(
                result["scores"][weight]["decision"] for result in null
            ),
            "watermarked_total": len(wm),
            "null_total": len(null),
        }
    payload = {
        "cross_model_entropy_result_schema_version": (
            CROSS_MODEL_ENTROPY_RESULT_SCHEMA_VERSION
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME,
        "result_kind": (
            "prompt_sharded_online_cross_model_map_entropy_audit"
        ),
        "detection_strategy": (
            "static_kv_teacher_forcing_then_prompt_sharded_map_entropy_v2"
        ),
        "source_tag": source_tag,
        "source_T": int(artifact["T"]),
        "n": prefix_T,
        "T": prefix_T,
        "t": key.check_weight,
        "eta": key.noise_rate,
        "r": target_row_count(prefix_T, key),
        "free_coordinates": prefix_T - target_row_count(prefix_T, key),
        "target_fpr": float(fpr),
        "fpr_policy": FPR_POLICY,
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "entropy_model_size": entropy_model_size,
        "entropy_model": model_display(entropy_model_size),
        "entropy_trace_source": cross_model_entropy_trace_source(
            entropy_model_size, generation_model_size
        ),
        "entropy_trace_kv_cache_implementation": (
            DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION
        ),
        "entropy_trace_kv_cache_version": kv_cache_version(
            DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION
        ),
        "num_prompts": len(indices),
        "prompt_indices": indices,
        "null_cache_T": int(null_T),
        "null_entropy_trace_T": int(null_trace_T),
        "watermarked_cache_mode": "prefix_from_longer",
        "watermarked_cache_T": int(artifact["T"]),
        "watermarked_cache_tag": source_tag,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "online_key_sha256": key.fingerprint,
        "online_support_sha256": support_sha256(prefix_T, key),
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "experiment_seed": int(artifact.get("experiment_seed", SEED)),
        "cpu_detection_wall_seconds": float(detection_wall_seconds),
        "cpu_aggregation_seconds": time.time() - started,
        "detection_shard_count": len(inventory),
        "detection_shard_cache_hits": sum(
            item["cached_this_invocation"] for item in inventory
        ),
        "detection_shards": inventory,
        "counts": counts,
        "results": results,
    }
    output_dir = f"/data/{source_tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"cross-model-map-entropy-{model_cache_name(entropy_model_size)}-"
        f"prefix-T{prefix_T}-fpr-{_slug(f'{float(fpr):.12g}')}-"
        f"prompts-{len(indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


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
    null_manifest = load_null_cache_manifest(
        null_T, generation_model_size
    )
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
                    record,
                    artifact,
                    index,
                    T,
                    source_length=int(null_T),
                    expected_kv_cache_implementation=(
                        null_manifest.get("kv_cache_implementation")
                        if null_manifest is not None else None
                    ),
                    require_provenance=null_manifest is not None,
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
        "null_cache_manifest": null_manifest,
        "null_kv_cache_implementation": (
            null_manifest.get("kv_cache_implementation")
            if null_manifest is not None else None
        ),
        "null_kv_cache_version": (
            null_manifest.get("kv_cache_version")
            if null_manifest is not None else None
        ),
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
    null_manifest = load_null_cache_manifest(
        null_T, generation_model_size
    )
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
                    record,
                    artifact,
                    index,
                    prefix_T,
                    source_length=int(null_T),
                    expected_kv_cache_implementation=(
                        null_manifest.get("kv_cache_implementation")
                        if null_manifest is not None else None
                    ),
                    require_provenance=null_manifest is not None,
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
        "null_cache_manifest": null_manifest,
        "null_kv_cache_implementation": (
            null_manifest.get("kv_cache_implementation")
            if null_manifest is not None else None
        ),
        "null_kv_cache_version": (
            null_manifest.get("kv_cache_version")
            if null_manifest is not None else None
        ),
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
                             null_kv_cache_implementation: str = "",
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

    model = OnlineModel.with_options(**model_cls_options(
        generation_model_size, gpu, max_containers
    ))(
        tag=tag,
        model_size=normalize_model_size(generation_model_size),
        code_fingerprint_sha256=code_fingerprint,
        kv_cache_implementation=normalize_kv_cache_implementation(
            kv_cache_implementation
        ),
        null_kv_cache_implementation=(
            resolve_null_kv_cache_implementation(
                null_kv_cache_implementation,
                kv_cache_implementation,
            )
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


def _execute_parallel_full_audit(
    *,
    tag: str,
    prefix_T: int,
    prompt_indices: list[int],
    null_T: int,
    fpr: float,
    batch: int,
    code_fingerprint: str,
    watermarked_source_tag: str,
    watermarked_cache_mode: str,
    watermarked_resume_source_tag: str = "",
    watermarked_resume_source_T: int = 0,
    detection_shard_size: int = DEFAULT_DETECTION_SHARD_SIZE,
    detection_max_containers: int = DEFAULT_DETECTION_MAX_CONTAINERS,
    log_prefix: str = "audit",
) -> dict:
    """Run the complete three-detector audit across prompt CPU shards."""
    import time

    shards = prompt_detection_shards(prompt_indices, detection_shard_size)
    requests = [{
        "tag": tag,
        "watermarked_source_tag": watermarked_source_tag,
        "prefix_T": int(prefix_T),
        "null_T": int(null_T),
        "fpr": float(fpr),
        "code_fingerprint_sha256": code_fingerprint,
        "prompt_indices": shard,
    } for shard in shards]
    worker = detect_full_audit_prompt_shard.with_options(
        cpu=1.0,
        max_containers=min(int(detection_max_containers), len(shards)),
    )
    started = time.time()
    summaries = list(worker.map(requests))
    wall_seconds = time.time() - started
    if len(summaries) != len(shards):
        raise AssertionError("full-audit prompt-shard result count changed")
    print(
        f"[{log_prefix}] full detector: shards={len(shards)}, "
        f"cache_hits={sum(item['cached'] for item in summaries)}, "
        f"wall_seconds={wall_seconds:.1f}",
        flush=True,
    )
    return aggregate_full_audit_shards.remote(
        tag,
        int(prefix_T),
        [int(index) for index in prompt_indices],
        int(null_T),
        float(fpr),
        int(batch),
        code_fingerprint,
        watermarked_source_tag,
        watermarked_cache_mode,
        watermarked_resume_source_tag,
        int(watermarked_resume_source_T),
        summaries,
        wall_seconds,
    )


def cross_model_entropy_estimation_requests(
    plan: dict,
    entropy_batch: int,
) -> list[dict]:
    """Build one combined request queue for WM runs and the shared null."""
    entropy_batch = int(entropy_batch)
    if entropy_batch <= 0:
        raise ValueError("entropy_batch must be positive")
    requests = []
    for audit in plan["audits"]:
        for chunk in _chunks(audit["wm_trace_missing"], entropy_batch):
            requests.append({
                "source": "wm",
                "artifact_tag": audit["source_tag"],
                "source_tag": audit["source_tag"],
                "trace_T": int(audit.get("trace_T", audit["prefix_T"])),
                "require_full_entropy": bool(
                    audit.get("require_full_entropy", False)
                ),
                "estimator_chunk_size": int(
                    audit.get("estimator_chunk_size", 1)
                ),
                "prompt_indices": chunk,
                "audit_label": str(audit.get("label", "")),
            })
    reference_tag = str(plan["audits"][-1]["source_tag"])
    for chunk in _chunks(plan["null_trace_missing"], entropy_batch):
        requests.append({
            "source": "null",
            "artifact_tag": reference_tag,
            "source_tag": "",
            "trace_T": int(plan["null_T"]),
            "require_full_entropy": bool(
                plan.get("require_null_full_entropy", False)
            ),
            "estimator_chunk_size": int(
                plan.get("null_estimator_chunk_size", 1)
            ),
            "prompt_indices": chunk,
            "audit_label": "shared-null",
        })
    return requests


def summarize_cross_model_entropy_workload(plan: dict) -> dict:
    wm_positions = sum(
        len(audit["wm_trace_missing"]) * int(
            audit.get("trace_T", audit["prefix_T"])
        )
        for audit in plan["audits"]
    )
    null_positions = (
        len(plan["null_trace_missing"]) * int(plan["null_T"])
    )
    return {
        "watermarked_teacher_forced_token_positions": wm_positions,
        "null_teacher_forced_token_positions": null_positions,
        "teacher_forced_token_positions": wm_positions + null_positions,
        "watermarked_trace_records_missing": sum(
            len(audit["wm_trace_missing"]) for audit in plan["audits"]
        ),
        "null_trace_records_missing": len(plan["null_trace_missing"]),
    }


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
    model = OnlineModel.with_options(**model_cls_options(
        generation_model_size, gpu, 1
    ))(
        tag=tag,
        model_size=generation_model_size,
        code_fingerprint_sha256=_local_code_fingerprint(),
        kv_cache_implementation="static",
        null_kv_cache_implementation="static",
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
def validate_full_audit_sharding(
    n: int = 256,
    num_prompts: int = 2,
    t: int = 3,
    eta: float = 0.05,
    fpr: float = 1e-3,
    batch: int = 2,
    experiment_seed: int = SEED,
    generation_model_size: str = MODEL_SIZE,
    kv_cache_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
    null_kv_cache_implementation: str = "",
    detection_shard_size: int = 1,
    detection_max_containers: int = 2,
):
    """Prove full-detector prompt sharding matches the serial reference."""
    generation_model_size = normalize_model_size(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    null_kv_cache_implementation = resolve_null_kv_cache_implementation(
        null_kv_cache_implementation, kv_cache_implementation
    )
    indices = list(range(int(num_prompts)))
    code_fingerprint = _local_code_fingerprint()
    tag = config_tag(
        n,
        t,
        eta,
        experiment_seed,
        generation_model_size,
        kv_cache_implementation,
    )
    build_artifacts.remote(
        num_prompts,
        n,
        t,
        eta,
        experiment_seed,
        False,
        generation_model_size,
        kv_cache_implementation,
    )
    plan = plan_generation.remote(
        tag,
        indices,
        n,
        True,
        null_kv_cache_implementation,
        True,
    )
    if plan["wm_missing"] or plan["null_missing"]:
        raise FileNotFoundError(
            "full-audit sharding validation is cache-only; missing "
            f"wm={plan['wm_missing']}, null={plan['null_missing']}"
        )

    serial = detect_all.remote(
        tag,
        indices,
        plan["null_T"],
        fpr,
        batch,
        code_fingerprint,
        plan["wm_source_tag"],
        plan["wm_mode"],
        plan["wm_resume_source_tag"],
        plan["wm_resume_source_T"],
    )
    sharded = _execute_parallel_full_audit(
        tag=tag,
        prefix_T=n,
        prompt_indices=indices,
        null_T=plan["null_T"],
        fpr=fpr,
        batch=batch,
        code_fingerprint=code_fingerprint,
        watermarked_source_tag=plan["wm_source_tag"],
        watermarked_cache_mode=plan["wm_mode"],
        watermarked_resume_source_tag=plan["wm_resume_source_tag"],
        watermarked_resume_source_T=plan["wm_resume_source_T"],
        detection_shard_size=detection_shard_size,
        detection_max_containers=detection_max_containers,
        log_prefix="full-audit-sharding",
    )
    serial_payload = serial["payload"]
    sharded_payload = sharded["payload"]
    compared_fields = (
        "n",
        "T",
        "r",
        "free_coordinates",
        "target_fpr",
        "counts",
    )
    mismatches = [
        field for field in compared_fields
        if serial_payload[field] != sharded_payload[field]
    ]
    result_comparison = compare_full_audit_results(
        serial_payload["results"], sharded_payload["results"]
    )
    if not result_comparison["equivalent"]:
        mismatches.append("results")
    if mismatches:
        raise AssertionError(
            "serial and prompt-sharded full detection differ: "
            f"{mismatches}; result_comparison={result_comparison}"
        )
    print(
        f"[full-audit-sharding] exact serial equivalence for "
        f"{num_prompts} prompts across "
        f"{sharded_payload['detection_shard_count']} shards",
        flush=True,
    )
    print(
        f"[full-audit-sharding] decisions/counts exact; maximum float "
        f"difference={result_comparison['max_abs_float_difference']:.3g}",
        flush=True,
    )
    print(
        f"[full-audit-sharding] counts={sharded_payload['counts']}",
        flush=True,
    )
    print(
        f"[full-audit-sharding] remote result: "
        f"{sharded['remote_output_path']}",
        flush=True,
    )


@app.local_entrypoint()
def build_null_cache(
    num_prompts: int = 5,
    n: int = 64,
    t: int = 3,
    eta: float = 0.05,
    batch: int = 0,
    experiment_seed: int = SEED,
    max_containers: int = 1,
    gpu: str = "",
    generation_model_size: str = MODEL_SIZE,
    kv_cache_implementation: str = "static",
    null_kv_cache_implementation: str = "static",
):
    """Build or verify a reusable shared null cache without WM generation."""
    import time

    started = time.time()
    generation_model_size, batch, gpu = resolve_model_runtime(
        generation_model_size, batch, gpu
    )
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    null_kv_cache_implementation = resolve_null_kv_cache_implementation(
        null_kv_cache_implementation, kv_cache_implementation
    )
    if not 0 < int(num_prompts) <= CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if n <= 0 or t < 2 or batch <= 0 or max_containers <= 0:
        raise ValueError("n, t, batch, and max_containers must be positive")
    if experiment_seed < 0 or not 0 <= eta < 0.5:
        raise ValueError("experiment_seed or eta is invalid")

    prompt_indices = list(range(int(num_prompts)))
    code_fingerprint = _local_code_fingerprint()
    tag = config_tag(
        n,
        t,
        eta,
        experiment_seed,
        generation_model_size,
        kv_cache_implementation,
    )
    print(
        f"[null-cache] target T={n}, model="
        f"{model_display(generation_model_size)}, prompts={num_prompts}, "
        f"batch={batch}, null_kv_cache={null_kv_cache_implementation}, "
        f"GPU={gpu}, max_containers={max_containers}",
        flush=True,
    )
    build = build_artifacts.remote(
        num_prompts,
        n,
        t,
        eta,
        experiment_seed,
        False,
        generation_model_size,
        kv_cache_implementation,
    )
    plan = plan_null_cache_generation.remote(
        tag,
        prompt_indices,
        n,
        null_kv_cache_implementation,
    )
    print(
        f"[null-cache] plan: source_T={plan['null_T']}, "
        f"missing={len(plan['null_missing'])}, "
        f"invalid={len(plan['null_invalid'])}, "
        f"legacy_manifestless={plan['legacy_manifestless']}",
        flush=True,
    )
    if plan["null_rejected_candidates"]:
        print(
            f"[null-cache] rejected candidates: "
            f"{plan['null_rejected_candidates']}",
            flush=True,
        )

    generation_records = []
    if plan["null_missing"]:
        if int(plan["null_T"]) != int(n):
            raise AssertionError("missing work must target the requested T")
        model = OnlineModel.with_options(**model_cls_options(
            generation_model_size, gpu, max_containers
        ))(
            tag=tag,
            model_size=generation_model_size,
            code_fingerprint_sha256=code_fingerprint,
            kv_cache_implementation=kv_cache_implementation,
            null_kv_cache_implementation=null_kv_cache_implementation,
        )
        print(f"[null-cache] model ready: {model.ready.remote()}", flush=True)
        generation_records = list(model.generate_null.map(
            _chunks(plan["null_missing"], batch)
        ))
        print(
            f"[null-cache] generated="
            f"{sum(item['generated'] for item in generation_records)}, "
            f"batch_sizes="
            f"{[item['batch'] for item in generation_records if item['batch']]}",
            flush=True,
        )
    else:
        print("[null-cache] cache-only; no GPU generation launched", flush=True)

    verification = verify_shared_null_cache.remote(
        tag, prompt_indices, plan["null_T"]
    )
    generation_cost = summarize_generation_cost(
        {"wm": [], "null": generation_records}, n, gpu
    )
    generation_cost["local_end_to_end_wall_seconds"] = time.time() - started
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result_kind": "shared_null_cache_build",
        "requested_T": int(n),
        "actual_null_T": int(plan["null_T"]),
        "num_prompts": int(num_prompts),
        "prompt_indices": prompt_indices,
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "batch": int(batch),
        "gpu": str(gpu),
        "max_containers": int(max_containers),
        "watermarked_kv_cache_implementation": kv_cache_implementation,
        "null_kv_cache_implementation": (
            verification["manifest"].get("kv_cache_implementation")
            if verification["manifest"] is not None else None
        ),
        "null_kv_cache_version": (
            verification["manifest"].get("kv_cache_version")
            if verification["manifest"] is not None else None
        ),
        "artifact_tag": tag,
        "artifact_fingerprint": build["artifact_fingerprint"],
        "generation_plan": plan,
        "generation_batches": generation_records,
        "generation_cost": generation_cost,
        "verification": verification,
    }
    os.makedirs("outputs", exist_ok=True)
    replay_suffix = "_cache-replay" if not generation_records else ""
    output_path = os.path.join(
        "outputs",
        f"shared_null_cache_T{n}_prompts{num_prompts}_"
        f"gen-{model_cache_name(generation_model_size)}_"
        f"kvcache-{kv_cache_version(null_kv_cache_implementation)}"
        f"{replay_suffix}.json",
    )
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    print(
        f"[null-cache] verified={verification['verified']}, "
        f"provenance={verification['provenance_counts']}",
        flush=True,
    )
    print(f"[null-cache] local manifest: {output_path}", flush=True)


@app.local_entrypoint()
def main(num_prompts: int = CANONICAL_NUM_PROMPTS,
         n: int = 256, t: int = 3, eta: float = 0.05,
         fpr: float = 1e-3, batch: int = 0,
         experiment_seed: int = SEED,
         max_containers: int = DEFAULT_MAX_CONTAINERS,
         gpu: str = "", fresh: bool = False,
         generation_model_size: str = MODEL_SIZE,
         kv_cache_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
         null_kv_cache_implementation: str = "",
         detection_shard_size: int = DEFAULT_DETECTION_SHARD_SIZE,
         detection_max_containers: int = DEFAULT_DETECTION_MAX_CONTAINERS,
         cache_only: bool = False,
         csv_out: str = "online_causal_results_summary.csv"):
    generation_model_size, batch, gpu = resolve_model_runtime(
        generation_model_size, batch, gpu
    )
    generation_model = model_display(generation_model_size)
    kv_cache_implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    null_kv_cache_implementation = resolve_null_kv_cache_implementation(
        null_kv_cache_implementation, kv_cache_implementation
    )
    if num_prompts <= 0 or num_prompts > CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if (
        n <= 0 or t < 2 or batch <= 0 or max_containers <= 0
        or detection_shard_size <= 0 or detection_max_containers <= 0
    ):
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
        f"null_kv_cache={null_kv_cache_implementation}, "
        f"experiment_seed={experiment_seed}, GPU={gpu}, "
        f"max_containers={max_containers}, "
        f"detection_shard_size={detection_shard_size}, "
        f"detection_max_containers={detection_max_containers}, "
        f"cache_only={cache_only}", flush=True,
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
        tag, prompt_indices, n, not fresh, null_kv_cache_implementation
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

    if cache_only:
        require_complete_cache_plan(plan)
        print(
            "[main] cache-only guard passed; GPU generation is disabled",
            flush=True,
        )

    generation_meta = {"wm": [], "null": []}
    if plan["wm_missing"] or plan["null_missing"]:
        if cache_only:
            raise AssertionError(
                "cache-only guard allowed missing generation records"
            )
        from concurrent.futures import ThreadPoolExecutor

        model = OnlineModel.with_options(**model_cls_options(
            generation_model_size, gpu, max_containers
        ))(
            tag=tag,
            model_size=generation_model_size,
            code_fingerprint_sha256=code_fingerprint,
            kv_cache_implementation=kv_cache_implementation,
            null_kv_cache_implementation=null_kv_cache_implementation,
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

    detected = _execute_parallel_full_audit(
        tag=tag,
        prefix_T=n,
        prompt_indices=prompt_indices,
        null_T=plan["null_T"],
        fpr=fpr,
        batch=batch,
        code_fingerprint=code_fingerprint,
        watermarked_source_tag=plan["wm_source_tag"],
        watermarked_cache_mode=plan["wm_mode"],
        watermarked_resume_source_tag=plan["wm_resume_source_tag"],
        watermarked_resume_source_T=plan["wm_resume_source_T"],
        detection_shard_size=detection_shard_size,
        detection_max_containers=detection_max_containers,
        log_prefix="main",
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
    local_payload = {
        **payload,
        "results": payload["results"],
        "execution_mode": "cache_only" if cache_only else "generate_or_reuse",
    }
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
        "null kv cache implementation": (
            payload.get("null_kv_cache_implementation")
            or "legacy-unversioned"
        ),
        "null kv cache version": (
            payload.get("null_kv_cache_version")
            or "legacy-unversioned"
        ),
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
                       null_kv_cache_implementation: str = "",
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
    null_kv_cache_implementation = resolve_null_kv_cache_implementation(
        null_kv_cache_implementation, kv_cache_implementation
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
        f"null_kv_cache={null_kv_cache_implementation}, "
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
        not fresh and not pin_floor_cache,
        null_kv_cache_implementation,
        final_audit,
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
        null_kv_cache_implementation,
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
        final_result = _execute_parallel_full_audit(
            tag=source_tag,
            prefix_T=selected_n,
            prompt_indices=prompt_indices,
            null_T=plan["null_T"],
            fpr=fpr,
            batch=batch,
            code_fingerprint=code_fingerprint,
            watermarked_source_tag=source_tag,
            watermarked_cache_mode=(
                "exact_cache" if int(selected_n) == source_T
                else "prefix_from_longer"
            ),
            watermarked_resume_source_tag=(
                plan["wm_resume_source_tag"] or ""
            ),
            watermarked_resume_source_T=(
                plan["wm_resume_source_T"] or 0
            ),
            detection_shard_size=detection_shard_size,
            detection_max_containers=detection_max_containers,
            log_prefix="sweep-final-audit",
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
        "null_kv_cache_implementation": (
            plan["null_kv_cache_implementation"] if final_audit else None
        ),
        "null_kv_cache_version": (
            plan["null_kv_cache_version"] if final_audit else None
        ),
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


@app.local_entrypoint()
def detect_14b_lower_eta_with_0p6b_map_entropy(
    num_prompts: int = CANONICAL_NUM_PROMPTS,
    fpr: float = 1e-3,
    entropy_batch: int = DEFAULT_ENTROPY_BATCH,
    entropy_max_containers: int = DEFAULT_MAX_CONTAINERS,
    entropy_gpu: str = GPU,
    detection_shard_size: int = DEFAULT_DETECTION_SHARD_SIZE,
    detection_max_containers: int = DEFAULT_DETECTION_MAX_CONTAINERS,
    experiment_seed: int = SEED,
    plan_only: bool = False,
):
    """Cache-only 0.6B MAP/entropy audit for 14B eta .05/.10 runs.

    The GPU stage uses one combined, bounded request queue across both
    watermarked runs and one shared T=1808 null trace.  The CPU stage combines
    both audits into one prompt-sharded queue.  It never generates text.
    """
    import time

    if not 0 < int(num_prompts) <= CANONICAL_NUM_PROMPTS:
        raise ValueError(
            f"num_prompts must be in [1, {CANONICAL_NUM_PROMPTS}]"
        )
    if (
        entropy_batch <= 0
        or entropy_max_containers <= 0
        or detection_shard_size <= 0
        or detection_max_containers <= 0
    ):
        raise ValueError("entropy/detection parallelism settings are invalid")
    if experiment_seed < 0 or not 0 < fpr < 1:
        raise ValueError("experiment_seed or fpr is invalid")

    generation_model_size = "14B"
    entropy_model_size = "0.6B"
    null_T = 1808
    audits = [
        {
            "label": "eta0.05-n880",
            "eta": 0.05,
            "prefix_T": 880,
            "source_T": 1280,
            "source_tag": config_tag(
                1280,
                3,
                0.05,
                experiment_seed,
                generation_model_size,
                "static",
            ),
        },
        {
            "label": "eta0.10-n1808",
            "eta": 0.10,
            "prefix_T": 1808,
            "source_T": 3072,
            "source_tag": config_tag(
                3072,
                3,
                0.10,
                experiment_seed,
                generation_model_size,
                "static",
            ),
        },
    ]
    indices = list(range(int(num_prompts)))
    code_fingerprint = _local_code_fingerprint()
    print(
        "[cross-model-map-entropy] cache-only campaign: 14B generations, "
        "0.6B detector model, eta={0.05,0.10}, n={880,1808}, "
        f"prompts={num_prompts}, null_T={null_T}",
        flush=True,
    )
    print(
        "[cross-model-map-entropy] optimized replay: static/static-v1 KV; "
        f"batch={entropy_batch}; GPU={entropy_gpu}; "
        f"max GPU containers={entropy_max_containers}",
        flush=True,
    )
    print(
        "[cross-model-map-entropy] parallel MAP+entropy detector: "
        f"shard_size={detection_shard_size}; "
        f"max CPU containers={detection_max_containers}",
        flush=True,
    )

    plan = plan_cross_model_entropy_audits.remote(
        audits, indices, entropy_model_size, null_T
    )
    workload = summarize_cross_model_entropy_workload(plan)
    print(
        "[cross-model-map-entropy] preflight verified "
        f"{plan['generation_records_verified']} generation records; "
        f"missing WM traces="
        f"{workload['watermarked_trace_records_missing']}; "
        f"missing shared-null traces="
        f"{workload['null_trace_records_missing']}; "
        f"remaining teacher-forced positions="
        f"{workload['teacher_forced_token_positions']:,}",
        flush=True,
    )
    if plan_only:
        print(
            "[cross-model-map-entropy] plan-only: no GPU replay or detector "
            "workers launched",
            flush=True,
        )
        return

    campaign_started = time.time()
    estimation_requests = cross_model_entropy_estimation_requests(
        plan, entropy_batch
    )
    trace_summaries = []
    entropy_wall_seconds = 0.0
    if estimation_requests:
        estimator = CrossModelEntropyModel.with_options(
            **model_cls_options(
                entropy_model_size,
                entropy_gpu,
                entropy_max_containers,
            )
        )(
            entropy_model_size=entropy_model_size,
            generation_model_size=generation_model_size,
            trace_kv_cache_implementation=(
                DEFAULT_ENTROPY_KV_CACHE_IMPLEMENTATION
            ),
        )
        print(
            f"[cross-model-map-entropy] detector model ready: "
            f"{estimator.ready.remote()}",
            flush=True,
        )
        entropy_started = time.time()
        trace_summaries = list(estimator.estimate.map(estimation_requests))
        entropy_wall_seconds = time.time() - entropy_started
        print(
            f"[cross-model-map-entropy] replay batches="
            f"{len(trace_summaries)}; estimated records="
            f"{sum(item['estimated'] for item in trace_summaries)}; "
            f"wall_seconds={entropy_wall_seconds:.1f}",
            flush=True,
        )
    else:
        print(
            "[cross-model-map-entropy] all derived traces cached; skipping GPU",
            flush=True,
        )

    verified_plan = plan_cross_model_entropy_audits.remote(
        audits, indices, entropy_model_size, null_T
    )
    remaining = summarize_cross_model_entropy_workload(verified_plan)
    if remaining["teacher_forced_token_positions"]:
        raise RuntimeError(
            "cross-model trace verification still found missing work: "
            f"{remaining}"
        )

    detection_requests = []
    request_labels = []
    for audit in verified_plan["audits"]:
        for shard in prompt_detection_shards(indices, detection_shard_size):
            detection_requests.append({
                "source_tag": audit["source_tag"],
                "prefix_T": int(audit["prefix_T"]),
                "null_T": null_T,
                "null_trace_T": null_T,
                "fpr": float(fpr),
                "entropy_model_size": entropy_model_size,
                "code_fingerprint_sha256": code_fingerprint,
                "prompt_indices": shard,
            })
            request_labels.append(str(audit["label"]))
    detector = detect_cross_model_entropy_prompt_shard.with_options(
        cpu=1.0,
        max_containers=min(
            int(detection_max_containers), len(detection_requests)
        ),
    )
    detection_started = time.time()
    detection_summaries = list(detector.map(detection_requests))
    detection_wall_seconds = time.time() - detection_started
    by_label = {str(audit["label"]): [] for audit in verified_plan["audits"]}
    for label, summary in zip(request_labels, detection_summaries):
        by_label[label].append(summary)
    print(
        f"[cross-model-map-entropy] detector shards="
        f"{len(detection_summaries)}; cache_hits="
        f"{sum(item['cached'] for item in detection_summaries)}; "
        f"wall_seconds={detection_wall_seconds:.1f}",
        flush=True,
    )

    results = []
    os.makedirs("outputs", exist_ok=True)
    replay_metrics = {
        "entropy_gpu": entropy_gpu,
        "entropy_batch": int(entropy_batch),
        "entropy_max_containers": int(entropy_max_containers),
        "entropy_replay_wall_seconds": entropy_wall_seconds,
        "measured_gpu_method_seconds": sum(
            float(item.get("seconds", 0.0)) for item in trace_summaries
        ),
        "teacher_forced_token_positions": sum(
            int(item.get("teacher_forced_token_positions", 0))
            for item in trace_summaries
        ),
        "model_forward_positions": sum(
            int(item.get("model_forward_positions", 0))
            for item in trace_summaries
        ),
        "peak_cuda_allocated_bytes": max(
            [int(item.get("peak_cuda_allocated_bytes", 0))
             for item in trace_summaries] or [0]
        ),
        "peak_cuda_reserved_bytes": max(
            [int(item.get("peak_cuda_reserved_bytes", 0))
             for item in trace_summaries] or [0]
        ),
        "detection_shard_size": int(detection_shard_size),
        "detection_max_containers": int(detection_max_containers),
        "detection_wall_seconds": detection_wall_seconds,
        "trace_batches": trace_summaries,
    }
    for audit in verified_plan["audits"]:
        aggregated = aggregate_cross_model_entropy_audit_shards.remote(
            audit,
            indices,
            entropy_model_size,
            null_T,
            null_T,
            fpr,
            code_fingerprint,
            by_label[str(audit["label"])],
            detection_wall_seconds,
        )
        payload = {
            **aggregated["payload"],
            "replay_execution": replay_metrics,
        }
        local_path = os.path.join(
            "outputs",
            f"online_cross_model_map_entropy_n{int(audit['prefix_T'])}_"
            f"eta{float(audit['eta']):.2f}_prompts{num_prompts}_"
            f"gen-{model_cache_name(generation_model_size)}_"
            f"entropy-{model_cache_name(entropy_model_size)}.json",
        )
        with open(local_path, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        counts = payload["counts"]
        for weight in ("map", "entropy"):
            weight_counts = counts[weight]
            print(
                f"[cross-model-map-entropy] eta={audit['eta']:.2f}, "
                f"n={audit['prefix_T']}, {weight}: TPR "
                f"{_format_rate(weight_counts['tp'], weight_counts['watermarked_total'])}; "
                f"FPR {_format_rate(weight_counts['fp'], weight_counts['null_total'])}",
                flush=True,
            )
        print(
            f"[cross-model-map-entropy] local result: {local_path}",
            flush=True,
        )
        print(
            f"[cross-model-map-entropy] remote result: "
            f"{aggregated['remote_output_path']}",
            flush=True,
        )
        results.append({
            "label": audit["label"],
            "local_output_path": local_path,
            "remote_output_path": aggregated["remote_output_path"],
            "counts": counts,
        })

    campaign_manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "online_14b_lower_eta_cross_model_map_entropy_0p6b",
        "generation_mode": "cache_only_no_text_generation",
        "generation_model": model_display(generation_model_size),
        "entropy_model": model_display(entropy_model_size),
        "audits": audits,
        "num_prompts": int(num_prompts),
        "shared_null_T": null_T,
        "preflight_plan": plan,
        "verified_plan": verified_plan,
        "initial_workload": workload,
        "replay_execution": replay_metrics,
        "local_end_to_end_wall_seconds": time.time() - campaign_started,
        "results": results,
        "billing_note": (
            "Measured method time excludes container/model startup; use the "
            "Modal app run and billing dashboard for settled cost."
        ),
    }
    manifest_path = os.path.join(
        "outputs",
        f"online_14b_lower_eta_cross_model_map_entropy_0p6b_"
        f"prompts{num_prompts}_manifest.json",
    )
    with open(manifest_path, "w") as handle:
        json.dump(campaign_manifest, handle, indent=2, allow_nan=False)
    print(
        f"[cross-model-map-entropy] campaign manifest: {manifest_path}",
        flush=True,
    )


def _proxy_8b_variable_batch_requests(plan: dict) -> list[dict]:
    """Use the proven long-context batch 10 and larger short-context batches."""
    requests = []
    for audit in plan["audits"]:
        trace_T = int(audit.get("trace_T", audit["prefix_T"]))
        batch = 10 if trace_T >= 8192 else 25 if trace_T >= 4096 else 50
        for chunk in _chunks(audit["wm_trace_missing"], batch):
            requests.append({
                "source": "wm",
                "artifact_tag": audit["source_tag"],
                "source_tag": audit["source_tag"],
                "trace_T": trace_T,
                "require_full_entropy": True,
                "estimator_chunk_size": int(
                    audit.get("estimator_chunk_size", 1)
                ),
                "prompt_indices": chunk,
                "audit_label": str(audit["label"]),
            })
    reference_tag = str(plan["audits"][-1]["source_tag"])
    for chunk in _chunks(plan["null_trace_missing"], 10):
        requests.append({
            "source": "null",
            "artifact_tag": reference_tag,
            "source_tag": "",
            "trace_T": int(plan["null_T"]),
            "require_full_entropy": True,
            "estimator_chunk_size": int(
                plan.get("null_estimator_chunk_size", 1)
            ),
            "prompt_indices": chunk,
            "audit_label": "shared-null",
        })
    return requests


@app.local_entrypoint(name="proxy-8b")
def proxy_8b_entrypoint(
    mode: str = "plan",
    approval_token: str = "",
    gpu: str = "A10G",
    max_containers: int = 5,
    detection_max_containers: int = 10,
):
    """Plan, benchmark, or run the approved cache-only 8B proxy analysis."""
    import time
    from proxy_8b_analysis import (
        APPROVAL_TOKEN,
        BASELINE_RUN_ID,
        COMMON_PREFIXES,
        HARD_CAP_USD,
        NOMINAL_FPR,
        PRC_AUDITS,
        prc_audits,
    )

    mode = str(mode).strip().lower()
    if mode not in {"plan", "benchmark", "full"}:
        raise ValueError("mode must be plan, benchmark, or full")
    if mode != "plan" and approval_token != APPROVAL_TOKEN:
        raise PermissionError("paid proxy replay requires the approved $20 token")
    if gpu != "A10G":
        raise ValueError("the approved proxy preflight is frozen to A10G")
    if max_containers <= 0 or detection_max_containers <= 0:
        raise ValueError("container limits must be positive")

    all_indices = list(range(CANONICAL_NUM_PROMPTS))
    full_audits = prc_audits(require_full_entropy=True)
    full_plan = plan_cross_model_entropy_audits.remote(
        full_audits, all_indices, "0.6B", 13088
    )
    textseal_plan = plan_textseal_proxy_entropy.remote(all_indices)
    full_workload = summarize_cross_model_entropy_workload(full_plan)
    total_positions = (
        int(full_workload["teacher_forced_token_positions"])
        + int(textseal_plan["teacher_forced_token_positions"])
    )
    preflight = {
        "mode": mode,
        "generation_mode": "cache_only_no_text_generation",
        "generation_attempts": 0,
        "prc": full_plan,
        "textseal": textseal_plan,
        "remaining_teacher_forced_token_positions": total_positions,
        "expected_full_positions_from_empty_cache": 16_863_500,
        "hard_cap_usd": HARD_CAP_USD,
    }
    print(json.dumps({
        "mode": mode,
        "generation_attempts": 0,
        "prc_generation_records_verified": full_plan[
            "generation_records_verified"
        ],
        "prc_missing_trace_records": (
            full_workload["watermarked_trace_records_missing"]
            + full_workload["null_trace_records_missing"]
        ),
        "textseal_generation_records_verified": textseal_plan[
            "generation_records_verified"
        ],
        "textseal_missing_trace_records": textseal_plan[
            "missing_trace_records"
        ],
        "remaining_teacher_forced_token_positions": total_positions,
        "hard_cap_usd": HARD_CAP_USD,
    }, indent=2, sort_keys=True), flush=True)
    if mode == "plan":
        return

    estimator = CrossModelEntropyModel.with_options(
        **model_cls_options("0.6B", gpu, 1 if mode == "benchmark" else max_containers)
    )(
        entropy_model_size="0.6B",
        generation_model_size="8B",
        trace_kv_cache_implementation="static",
    )
    print(f"[proxy-8b] model ready: {estimator.ready.remote()}", flush=True)

    if mode == "benchmark":
        benchmark_indices = list(range(10))
        benchmark_plan = plan_cross_model_entropy_audits.remote(
            [full_audits[-1]], benchmark_indices, "0.6B", 13088
        )
        requests = cross_model_entropy_estimation_requests(benchmark_plan, 10)
        if not requests:
            raise RuntimeError(
                "benchmark traces already exist; use their saved benchmark manifest "
                "or choose uncached indices before projecting"
            )
        started = time.time()
        summaries = list(estimator.estimate.map(requests))
        wall_seconds = time.time() - started
        positions = sum(
            int(item["teacher_forced_token_positions"]) for item in summaries
        )
        method_seconds = sum(float(item["seconds"]) for item in summaries)
        if positions <= 0 or method_seconds <= 0:
            raise RuntimeError("benchmark did not perform measurable replay work")
        # Empirical all-in rate from the settled 14B->0.6B replay, with a 25%
        # guard for startup/long-context variation plus $0.25 CPU allowance.
        prior_all_in_usd_per_method_second = 1.22189737 / 3270.6197905540466
        guarded_rate = prior_all_in_usd_per_method_second * 1.25
        projected_total_usd = (
            method_seconds / positions * 16_863_500 * guarded_rate + 0.25
        )
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "campaign": "qwen3_8b_cache_only_proxy_benchmark",
            "approval_token_validated": True,
            "generation_attempts": 0,
            "gpu": gpu,
            "batch": 10,
            "max_containers": 1,
            "prompt_indices": benchmark_indices,
            "trace_T": 13088,
            "summaries": summaries,
            "teacher_forced_token_positions": positions,
            "measured_gpu_method_seconds": method_seconds,
            "wall_seconds": wall_seconds,
            "peak_cuda_allocated_bytes": max(
                int(item.get("peak_cuda_allocated_bytes", 0)) for item in summaries
            ),
            "peak_cuda_reserved_bytes": max(
                int(item.get("peak_cuda_reserved_bytes", 0)) for item in summaries
            ),
            "projection_basis_positions": 16_863_500,
            "prior_settled_all_in_usd_per_method_second": (
                prior_all_in_usd_per_method_second
            ),
            "projection_guard_multiplier": 1.25,
            "projected_total_usd": projected_total_usd,
            "hard_cap_usd": HARD_CAP_USD,
            "cost_gate_passed": projected_total_usd <= HARD_CAP_USD,
            "billing_status": "pending_exact_modal_reconciliation",
        }
        os.makedirs("outputs", exist_ok=True)
        benchmark_path = "outputs/proxy_8b_replay_benchmark.json"
        with open(benchmark_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        if projected_total_usd > HARD_CAP_USD:
            raise RuntimeError("benchmark projection exceeds the approved $20 cap")
        return

    benchmark_path = "outputs/proxy_8b_replay_benchmark.json"
    if not os.path.isfile(benchmark_path):
        raise FileNotFoundError("run the batch-10 benchmark before the full replay")
    with open(benchmark_path) as handle:
        benchmark = json.load(handle)
    if (
        not benchmark.get("cost_gate_passed")
        or float(benchmark.get("projected_total_usd", HARD_CAP_USD + 1)) > HARD_CAP_USD
    ):
        raise RuntimeError("saved benchmark does not pass the approved cost gate")

    campaign_started = time.time()
    requests = _proxy_8b_variable_batch_requests(full_plan)
    replay_started = time.time()
    trace_summaries = list(estimator.estimate.map(requests)) if requests else []
    replay_wall_seconds = time.time() - replay_started

    verified_plan = plan_cross_model_entropy_audits.remote(
        full_audits, all_indices, "0.6B", 13088
    )
    remaining = summarize_cross_model_entropy_workload(verified_plan)
    if remaining["teacher_forced_token_positions"]:
        raise RuntimeError(f"PRC proxy traces remain incomplete: {remaining}")

    textseal_verified = plan_textseal_proxy_entropy.remote(all_indices)
    textseal_requests = []
    for shard_index, missing in sorted(
        textseal_verified["missing_by_shard"].items(), key=lambda item: int(item[0])
    ):
        for chunk in _chunks(missing, 50):
            textseal_requests.append({
                "run_id": BASELINE_RUN_ID,
                "shard_index": int(shard_index),
                "artifact_tag": PRC_AUDITS[-1]["source_tag"],
                "prompt_indices": chunk,
            })
    textseal_started = time.time()
    textseal_summaries = (
        list(estimator.estimate_textseal.map(textseal_requests))
        if textseal_requests else []
    )
    textseal_wall_seconds = time.time() - textseal_started
    final_textseal_plan = plan_textseal_proxy_entropy.remote(all_indices)
    if final_textseal_plan["missing_trace_records"]:
        raise RuntimeError("TextSeal proxy traces remain incomplete")

    code_fingerprint = _local_code_fingerprint()
    detection_requests = []
    labels = []
    for audit in verified_plan["audits"]:
        score_prefixes = list(COMMON_PREFIXES)
        if int(audit["prefix_T"]) not in score_prefixes:
            score_prefixes.append(int(audit["prefix_T"]))
        for prefix in score_prefixes:
            for shard in prompt_detection_shards(all_indices, 50):
                detection_requests.append({
                    "source_tag": audit["source_tag"],
                    "prefix_T": prefix,
                    "watermarked_trace_T": int(audit["trace_T"]),
                    "estimator_chunk_size": int(
                        audit.get("estimator_chunk_size", 1)
                    ),
                    "null_T": 13088,
                    "null_trace_T": 13088,
                    "fpr": NOMINAL_FPR,
                    "entropy_model_size": "0.6B",
                    "code_fingerprint_sha256": code_fingerprint,
                    "prompt_indices": shard,
                })
                labels.append((str(audit["label"]), prefix))
    detector = detect_cross_model_entropy_prompt_shard.with_options(
        cpu=1.0,
        max_containers=min(detection_max_containers, len(detection_requests)),
    )
    detection_started = time.time()
    detection_summaries = list(detector.map(detection_requests))
    detection_wall_seconds = time.time() - detection_started
    grouped = {}
    for label, summary in zip(labels, detection_summaries):
        grouped.setdefault(label, []).append(summary)

    results = []
    prompt_rows = []
    with open("prompts.jsonl") as handle:
        prompt_rows = [json.loads(line) for line in handle if line.strip()]
    prompt_jsonl_path = "outputs/proxy_8b_prc_prompt_level.jsonl"
    with open(prompt_jsonl_path, "w") as prompt_handle:
        for audit in verified_plan["audits"]:
            score_prefixes = list(COMMON_PREFIXES)
            if int(audit["prefix_T"]) not in score_prefixes:
                score_prefixes.append(int(audit["prefix_T"]))
            for prefix in score_prefixes:
                aggregate = aggregate_cross_model_entropy_audit_shards.remote(
                    {**audit, "prefix_T": prefix},
                    all_indices,
                    "0.6B",
                    13088,
                    13088,
                    NOMINAL_FPR,
                    code_fingerprint,
                    grouped[(str(audit["label"]), prefix)],
                    detection_wall_seconds,
                )["payload"]
                results.append({
                    "label": audit["label"],
                    "eta": audit["eta"],
                    "prefix_T": prefix,
                    "boundary_status": audit["boundary_status"],
                    "counts": aggregate["counts"],
                })
                for result in aggregate["results"]:
                    index = int(result["prompt_idx"])
                    prompt_handle.write(json.dumps({
                        "prompt_index": index,
                        "prompt_id": prompt_rows[index].get("id", f"prompt-{index}"),
                        "eta": float(audit["eta"]),
                        "prefix_length": prefix,
                        "boundary_status": audit["boundary_status"],
                        "sample_type": (
                            "watermarked" if result["watermark"] else "null"
                        ),
                        "generation_model": "Qwen3-8B-Base",
                        "detector_probability_model": "Qwen3-0.6B-Base",
                        "scores": result["scores"],
                        "source_tag": audit["source_tag"],
                        "generation_attempts": 0,
                    }, sort_keys=True, allow_nan=False) + "\n")

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": "qwen3_8b_cache_only_proxy_analysis",
        "approval_token_validated": True,
        "generation_mode": "cache_only_no_text_generation",
        "generation_attempts": 0,
        "benchmark": benchmark,
        "preflight": preflight,
        "verified_prc_plan": verified_plan,
        "verified_textseal_plan": final_textseal_plan,
        "replay": {
            "gpu": gpu,
            "max_containers": max_containers,
            "prc_trace_batches": trace_summaries,
            "textseal_trace_batches": textseal_summaries,
            "prc_replay_wall_seconds": replay_wall_seconds,
            "textseal_replay_wall_seconds": textseal_wall_seconds,
            "measured_gpu_method_seconds": sum(
                float(item.get("seconds", 0.0))
                for item in trace_summaries + textseal_summaries
            ),
            "teacher_forced_token_positions": sum(
                int(item.get("teacher_forced_token_positions", 0))
                for item in trace_summaries + textseal_summaries
            ),
            "peak_cuda_allocated_bytes": max(
                [int(item.get("peak_cuda_allocated_bytes", 0))
                 for item in trace_summaries + textseal_summaries] or [0]
            ),
            "peak_cuda_reserved_bytes": max(
                [int(item.get("peak_cuda_reserved_bytes", 0))
                 for item in trace_summaries + textseal_summaries] or [0]
            ),
        },
        "detection_wall_seconds": detection_wall_seconds,
        "local_end_to_end_wall_seconds": time.time() - campaign_started,
        "results": results,
        "prompt_level_jsonl": prompt_jsonl_path,
        "billing_status": "pending_exact_modal_reconciliation",
    }
    manifest_path = "outputs/proxy_8b_campaign_manifest.json"
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps({
        "passed": True,
        "manifest_path": manifest_path,
        "prompt_level_jsonl": prompt_jsonl_path,
        "prc_result_cells": len(results),
        "textseal_proxy_traces": final_textseal_plan["cached_trace_records"],
        "generation_attempts": 0,
        "next_command": (
            "modal run baseline_comparison/modal_app.py::app.proxy-textseal-score "
            f"--approval-token {APPROVAL_TOKEN}"
        ),
    }, indent=2, sort_keys=True), flush=True)


@app.local_entrypoint(name="proxy-8b-quality")
def proxy_8b_quality_entrypoint(approval_token: str):
    """CPU-only native-8B quality aggregation for the proxy report."""
    from proxy_8b_analysis import APPROVAL_TOKEN

    if approval_token != APPROVAL_TOKEN:
        raise PermissionError("proxy quality aggregation needs the approved token")
    indices = list(range(CANONICAL_NUM_PROMPTS))
    results = list(
        proxy_8b_native_quality_shard.map(
            list(prompt_detection_shards(indices, 50))
        )
    )
    rows = [row for result in results for row in result["rows"]]
    if len(rows) != 2_500:
        raise AssertionError(f"quality aggregation produced {len(rows)} rows")
    if sum(int(row["generation_attempts"]) for row in rows) != 0:
        raise AssertionError("quality aggregation attempted generation")
    os.makedirs("outputs", exist_ok=True)
    path = "outputs/proxy_8b_native_quality_prompt_level.jsonl"
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "passed": True,
        "path": path,
        "rows": len(rows),
        "generation_attempts": 0,
        "model_loads": 0,
    }, indent=2, sort_keys=True))
