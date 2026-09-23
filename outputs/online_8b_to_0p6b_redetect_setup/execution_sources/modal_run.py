"""Modal runtime for fixed/online PRC generation and completion-only redetection.

Commands: generate_fixed, generate_online, generate_replicate, redetect.
All implementation lives here; generation keeps its original key/cache rules.
Redetection receives raw completions only, with coordinate 1's score set to zero.
"""

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from decimal import Decimal

import modal

from online_prc import GENERATION_SAMPLER_VERSION


# Shared runtime: app, dependency profiles, model loading and batching
# ----------------------------------------------------------------------------

RETIRED_DETECTION_MESSAGE = (
    "Prompt-dependent detection has been retired. Use modal_run.py::redetect "
    "with a frozen raw-completion manifest; use modal_run.py::generate_fixed or ::generate_online for generation."
)
SOURCE_MODULES = (
    "modal_run",
    "prc", "online_prc", "qwen", "constants", "detectors", "watermark_expt", "proxy_8b_analysis", "benchmarks",
)
# Keep the original dependency requirements for each execution profile. Changing
# generation numerics is a separate task from consolidating the runner.
ONLINE_PACKAGES = (
    "torch==2.4.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.4.5",
    "huggingface_hub==0.30.2", "scipy==1.14.1", "galois==0.4.2", "numba==0.59.1",
    "numpy==1.26.0", "pytest==8.3.3",
)
FIXED_PACKAGES = (
    "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub",
    "scipy", "galois", "numpy", "datasets", "aiohttp",
)
REPLICATE_PACKAGES = (
    "torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "scipy", "galois", "numpy",
)


def _image(packages, offline=False):
    env = {
        "HF_HOME": "/cache/hf", "HF_HUB_CACHE": "/cache/hf",
        "PRC_MODEL_CACHE_DIR": "/cache/models", "PRC_MODEL_SIZE": "0.6B", "PRC_MODEL_VARIANT": "base",
        "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    if offline:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = (modal.Image.debian_slim(python_version="3.11").pip_install(*packages).env(env)
              .add_local_file("prompts.jsonl", "/root/prompts.jsonl", copy=offline))
    if offline:
        result = result.add_local_dir("tests", "/root/tests", copy=True)
    return result.add_local_python_source(*SOURCE_MODULES)


image = _image(ONLINE_PACKAGES, offline=True)
fixed_image = _image(FIXED_PACKAGES)
replicate_image = _image(REPLICATE_PACKAGES)
hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
app = modal.App("prc-watermark", image=image)


def load_watermark_model(model_size):
    """Use the original Qwen loader in each model-specific Modal container."""
    os.environ["PRC_MODEL_SIZE"] = model_size
    os.environ["PRC_MODEL_VARIANT"] = "base"
    import watermark_expt
    return watermark_expt


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

def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def _format_rate(successes: int, total: int) -> str:
    return f"{int(successes)}/{int(total)} ({successes / max(total, 1):.1%})"


# Fixed PRC: original keys, generation and cache/result helpers
# ----------------------------------------------------------------------------

# ---- experiment config ------------------------------------------------------
FIXED_DEFAULT_N = 400
FIXED_DEFAULT_T = 3
FIXED_DEFAULT_ETA = 0.05
FIXED_DEFAULT_FPR = 1e-3  # 0.1%
FIXED_DEFAULT_BLOCKS = 1
FIXED_SEED = 12345
FIXED_MODEL_SIZE = "0.6B"
FIXED_VOCAB = 151_936
FIXED_GPU = "A10G"
FIXED_DEFAULT_MAX_CONTAINERS = 5
FIXED_DEFAULT_BATCH = 64
FIXED_DEFAULT_ENTROPY_BATCH = 8
FIXED_REQUIRED_R_FRAC = 0.99
FIXED_REQUIRED_R_SETTING = "0.99n"
FIXED_CANONICAL_NUM_PROMPTS = 500
FIXED_SHARD_RESULT_SCHEMA_VERSION = 1
FIXED_DETECTION_CHECKPOINT_SCHEMA_VERSION = 1
FIXED_DETECTION_CHECKPOINT_COMMIT_INTERVAL = 10
FIXED_DERIVED_TRACE_SCHEMA_VERSION = 1

FIXED_CSV_COLUMNS = [
    "eta",
    "T",
    "n",
    "r value",
    "r setting",
    "t",
    "Target FPR",
    "Entropy Model",
    "Generation Model",
    "Map TPR",
    "Entropy Aware TPR",
    "Naive TPR",
    "Log Hoeffding TPR",
    "Map FPR",
    "Entropy FPR",
    "Naive FPR",
    "Log Hoeffding FPR",
    "Entropy Trace Source",
    "Notes",
]


def fixed_normalize_model_size(model_size):
    value = FIXED_MODEL_SIZE if model_size is None else str(model_size).strip()
    if not value:
        value = FIXED_MODEL_SIZE
    upper = value.upper()
    if upper.endswith("B"):
        return upper[:-1] + "B"
    if upper.replace(".", "", 1).isdigit():
        return f"{upper}B"
    return value


def fixed_model_display(model_size):
    return f"Qwen3-{fixed_normalize_model_size(model_size)}-Base"


def fixed_entropy_model_tag(model_size):
    size = fixed_normalize_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def fixed_uses_cached_generation_trace(entropy_model_size,
                                 generation_model_size=FIXED_MODEL_SIZE):
    """Whether detection can reuse probabilities recorded during generation."""
    return fixed_normalize_model_size(entropy_model_size) == fixed_normalize_model_size(
        generation_model_size
    )


def fixed_entropy_trace_source(entropy_model_size,
                         generation_model_size=FIXED_MODEL_SIZE):
    if fixed_uses_cached_generation_trace(
            entropy_model_size, generation_model_size):
        return "cached_generation_p_trace"
    return f"estimated_{fixed_normalize_model_size(entropy_model_size)}"


def fixed_resolve_r(n, r=0, r_frac=0.0):
    explicit_r = int(r) if r else 0
    explicit_frac = float(r_frac) if r_frac else 0.0
    if explicit_r and explicit_frac:
        raise ValueError("Pass either --r or --r-frac, not both.")
    if explicit_r:
        return explicit_r
    if explicit_frac:
        return int(round(explicit_frac * n))
    return None


def fixed_resolve_new_run_r(n, r=0, r_frac=FIXED_REQUIRED_R_FRAC):
    """Enforce the project-wide r=round(0.99n) policy for new runs."""
    expected_r = int(round(FIXED_REQUIRED_R_FRAC * n))
    if r and int(r) != expected_r:
        raise ValueError(
            f"new runs require r=round(0.99n)={expected_r} for n={n}; "
            f"got explicit r={r}"
        )
    if r_frac and abs(float(r_frac) - FIXED_REQUIRED_R_FRAC) > 1e-12:
        raise ValueError(
            f"new runs require --r-frac {FIXED_REQUIRED_R_FRAC}; got {r_frac}"
        )
    return expected_r


def fixed_validate_r_for_keygen(n, t, r):
    if r is None:
        return
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    if r > n:
        raise ValueError(f"r must be <= n, got r={r}, n={n}")
    if n - r < t - 1:
        raise ValueError(
            f"r={r} is too large for n={n}, t={t}; need n-r >= t-1"
        )


def fixed_experiment_T(n):
    """Generated-token length for new runs: one length-n PRC code block."""
    return FIXED_DEFAULT_BLOCKS * int(n)


def _fixed_generation_scoped_root(root, generation_model_size=FIXED_MODEL_SIZE):
    """Keep legacy 0.6B paths while isolating every other generation model."""
    model_size = fixed_normalize_model_size(generation_model_size)
    if model_size == fixed_normalize_model_size(FIXED_MODEL_SIZE):
        return root
    return f"{root}/{fixed_entropy_model_tag(model_size)}"


def fixed_config_tag(n, t, eta, r=None, T=None,
               generation_model_size=FIXED_MODEL_SIZE):
    """Per-config tag for key-dependent artifacts.

    FPR is excluded because it only affects detection. r is included only when
    explicitly requested so old default-r caches keep their original tags.
    T is included for new runs so T=n caches cannot collide with old T=2n caches.
    """
    base = f"n{n}_t{t}_eta{eta:.2f}"
    if T is not None:
        base = f"{base}_T{int(T)}"
    if (fixed_normalize_model_size(generation_model_size)
            != fixed_normalize_model_size(FIXED_MODEL_SIZE)):
        base = f"{base}__gen-{fixed_entropy_model_tag(generation_model_size)}"
    return f"{base}_r{int(r)}" if r is not None else base


def fixed_art_path(n, t, eta, r=None, T=None,
             generation_model_size=FIXED_MODEL_SIZE):
    tag = fixed_config_tag(n, t, eta, r, T, generation_model_size)
    return f"/data/{tag}/artifacts.pt"


def fixed_wm_dir(n, t, eta, r=None, T=None,
           generation_model_size=FIXED_MODEL_SIZE):
    tag = fixed_config_tag(n, t, eta, r, T, generation_model_size)
    return f"/data/{tag}/wm"


def fixed_null_root(generation_model_size=FIXED_MODEL_SIZE):
    return _fixed_generation_scoped_root("/data/_nulls", generation_model_size)


def fixed_null_dir(T, generation_model_size=FIXED_MODEL_SIZE):
    return f"{fixed_null_root(generation_model_size)}/T{T}"


def fixed_wm_entropy_dir(tag, entropy_model_size):
    return f"/data/{tag}/entropy/{fixed_entropy_model_tag(entropy_model_size)}/wm"


def fixed_null_entropy_dir(T, entropy_model_size,
                     generation_model_size=FIXED_MODEL_SIZE):
    root = _fixed_generation_scoped_root(
        "/data/_null_entropy", generation_model_size
    )
    return f"{root}/{fixed_entropy_model_tag(entropy_model_size)}/T{T}"


def fixed_wm_trace_dir(tag, entropy_model_size):
    return f"/data/{tag}/detect_traces/{fixed_entropy_model_tag(entropy_model_size)}/wm"


def fixed_null_trace_dir(T, entropy_model_size,
                   generation_model_size=FIXED_MODEL_SIZE):
    root = _fixed_generation_scoped_root(
        "/data/_null_detection_traces", generation_model_size
    )
    return f"{root}/{fixed_entropy_model_tag(entropy_model_size)}/T{T}"


def fixed_detection_checkpoint_dir(tag, entropy_model_size, fpr):
    """Config-local detector records, separated by model and target FPR."""
    fpr_tag = _fixed_slug(f"{float(fpr):.12g}")
    return (
        f"/data/{tag}/detection_checkpoints/"
        f"{fixed_entropy_model_tag(entropy_model_size)}/fpr-{fpr_tag}"
    )


def fixed_validate_generation_record(record, generation_model_size,
                               source="generation", idx="?"):
    """Reject cache records from another model or unlabelled non-legacy data."""
    expected_size = fixed_normalize_model_size(generation_model_size)
    stored_size = record.get("generation_model_size")
    stored_display = record.get("generation_model")
    if stored_size is None:
        if expected_size != fixed_normalize_model_size(FIXED_MODEL_SIZE):
            raise ValueError(
                f"{source} cache index {idx} lacks generation-model metadata; "
                f"refusing to treat it as {fixed_model_display(expected_size)}"
            )
    elif fixed_normalize_model_size(stored_size) != expected_size:
        raise ValueError(
            f"{source} cache index {idx} was generated by "
            f"{fixed_model_display(stored_size)}, expected "
            f"{fixed_model_display(expected_size)}"
        )
    if (stored_display is not None
            and str(stored_display).strip() != fixed_model_display(expected_size)):
        raise ValueError(
            f"{source} cache index {idx} has generation model label "
            f"{stored_display!r}, expected {fixed_model_display(expected_size)!r}"
        )

    # Historical records remain readable. New-schema records, however, must be
    # self-contained and internally length-consistent.
    schema_version = record.get("generation_trace_schema_version")
    if schema_version is not None:
        from detectors import GENERATION_TRACE_SCHEMA_VERSION

        if int(schema_version) != GENERATION_TRACE_SCHEMA_VERSION:
            raise ValueError(
                f"{source} cache index {idx} has unsupported generation "
                f"trace schema {schema_version!r}"
            )
        required = (
            "watermark",
            "prompt_token_ids",
            "tokens",
            "prc_n",
            "p_trace",
            "observed_bucket_bits",
            "entropy_trace",
            "signed_entropy_trace",
            "codeword_signed_entropy_trace",
            "map_soft_tokens",
            "folded_signed_entropy",
            "folded_map_soft_tokens",
            "prc_codeword_bits",
            "prc_block_boundaries",
            "base_lm_entropy",
            "base_token_logprob",
            "partition_sha256",
            "encoding_key_sha256",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(
                f"{source} cache index {idx} is missing trace fields: "
                f"{', '.join(missing)}"
            )
        token_count = len(record["tokens"])
        prc_n = int(record["prc_n"])
        if prc_n <= 0:
            raise ValueError(
                f"{source} cache index {idx} has invalid prc_n={prc_n}"
            )
        for field in (
            "p_trace",
            "observed_bucket_bits",
            "entropy_trace",
            "signed_entropy_trace",
            "map_soft_tokens",
            "base_lm_entropy",
            "base_token_logprob",
        ):
            if len(record[field]) != token_count:
                raise ValueError(
                    f"{source} cache index {idx} has {len(record[field])} "
                    f"{field} values for {token_count} tokens"
                )
        for field in ("folded_signed_entropy", "folded_map_soft_tokens"):
            if len(record[field]) != prc_n:
                raise ValueError(
                    f"{source} cache index {idx} has {len(record[field])} "
                    f"{field} values for prc_n={prc_n}"
                )
        observed_values = {
            int(value) for value in record["observed_bucket_bits"]
        }
        if not observed_values.issubset({0, 1}):
            raise ValueError(
                f"{source} cache index {idx} has non-binary observed buckets"
            )
        expected_boundaries = [
            (start, min(start + prc_n, token_count))
            for start in range(0, token_count, prc_n)
        ]
        stored_boundaries = [
            tuple(int(value) for value in row)
            for row in record["prc_block_boundaries"]
        ]
        if stored_boundaries != expected_boundaries:
            raise ValueError(
                f"{source} cache index {idx} has invalid PRC block boundaries"
            )
        for field in ("partition_sha256", "encoding_key_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record[field])):
                raise ValueError(
                    f"{source} cache index {idx} has invalid {field}"
                )
        if bool(record.get("watermark")):
            codeword = record["prc_codeword_bits"]
            if codeword is None or len(codeword) != token_count:
                raise ValueError(
                    f"{source} cache index {idx} lacks its exact PRC codeword"
                )
            if not {int(value) for value in codeword}.issubset({0, 1}):
                raise ValueError(
                    f"{source} cache index {idx} has a non-binary PRC codeword"
                )
            if len(record["codeword_signed_entropy_trace"]) != token_count:
                raise ValueError(
                    f"{source} cache index {idx} has a misaligned "
                    "codeword-signed entropy trace"
                )
        elif (record["prc_codeword_bits"] is not None
              or record["codeword_signed_entropy_trace"] is not None):
            raise ValueError(
                f"{source} cache index {idx} is null but contains PRC "
                "codeword data"
            )


def fixed_find_complete_cache_T(root, min_T, prompt_indices_or_count, prefix):
    """Return the smallest complete T' >= min_T cache, or None.

    Cache directories are named T{length} and contain one {prefix}_XXXX.pt
    record per prompt.  A longer causal generation/trace can be truncated to
    any requested prefix length, so exact-length caches are not required. The
    third argument may be the legacy prompt count or an exact index iterable.
    """
    if not os.path.isdir(root):
        return None

    prompt_indices = _fixed_coerce_prompt_indices(prompt_indices_or_count)

    candidates = []
    for name in os.listdir(root):
        if not name.startswith("T"):
            continue
        try:
            candidate_T = int(name[1:])
        except ValueError:
            continue
        if candidate_T < int(min_T):
            continue
        cache_dir = os.path.join(root, name)
        if all(os.path.exists(os.path.join(
                cache_dir, f"{prefix}_{i:04d}.pt"))
               for i in prompt_indices):
            candidates.append(candidate_T)

    return min(candidates) if candidates else None




def _fixed_format_rate(count, total):
    denom = max(total, 1)
    return f"{count}/{total} ({count / denom:.1%})"


def fixed_prompt_indices_for_shard(prompt_start, num_prompts,
                             total_prompts=FIXED_CANONICAL_NUM_PROMPTS):
    """Return a validated contiguous range of global prompt indices."""
    start = int(prompt_start)
    count = int(num_prompts)
    total = int(total_prompts)
    if start < 0:
        raise ValueError(f"prompt_start must be >= 0, got {start}")
    if count <= 0:
        raise ValueError(f"num_prompts must be > 0, got {count}")
    if start + count > total:
        raise ValueError(
            f"prompt shard [{start}, {start + count}) exceeds canonical "
            f"prompt count {total}"
        )
    return list(range(start, start + count))


def _fixed_coerce_prompt_indices(prompt_indices_or_count):
    """Accept the old count API as well as an exact iterable of indices."""
    if isinstance(prompt_indices_or_count, int):
        if prompt_indices_or_count < 0:
            raise ValueError("prompt count must be nonnegative")
        return list(range(prompt_indices_or_count))
    indices = [int(i) for i in prompt_indices_or_count]
    if len(indices) != len(set(indices)):
        raise ValueError("prompt indices contain duplicates")
    if any(i < 0 for i in indices):
        raise ValueError("prompt indices must be nonnegative")
    return indices


def _fixed_json_safe(value):
    """Convert numpy/torch scalar containers to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _fixed_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_fixed_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _fixed_json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _fixed_atomic_write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w") as f:
        json.dump(_fixed_json_safe(payload), f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(temporary, path)


def _fixed_slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _fixed_canonical_json_sha256(value):
    encoded = json.dumps(
        _fixed_json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fixed_detection_checkpoint_identity(config, artifact_fingerprint,
                                   code_fingerprint, source, prompt_idx,
                                   tokens_sha256, p_trace_sha256):
    """Compatibility manifest for one detector result.

    A checkpoint is reusable only when the experiment configuration, frozen
    PRC artifact, detector source code, prompt/source, and exact detector
    inputs all match.
    """
    if isinstance(code_fingerprint, dict):
        detector_sha256 = code_fingerprint.get("sha256", "")
    else:
        detector_sha256 = str(code_fingerprint or "")
    return {
        "config": _fixed_json_safe(config),
        "artifact_fingerprint": str(artifact_fingerprint),
        "detector_implementation_sha256": str(detector_sha256),
        "source": str(source),
        "prompt_idx": int(prompt_idx),
        "tokens_sha256": str(tokens_sha256),
        "p_trace_sha256": str(p_trace_sha256),
    }


def _fixed_save_detection_checkpoint(path, identity, record):
    safe_record = _fixed_json_safe(record)
    payload = {
        "schema_version": FIXED_DETECTION_CHECKPOINT_SCHEMA_VERSION,
        "identity": _fixed_json_safe(identity),
        "identity_sha256": _fixed_canonical_json_sha256(identity),
        "record_sha256": _fixed_canonical_json_sha256(safe_record),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record": safe_record,
    }
    _fixed_atomic_write_json(path, payload)


def _fixed_load_detection_checkpoint(path, expected_identity):
    """Return a verified detector record, or None for stale/corrupt data."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if payload.get("schema_version") != FIXED_DETECTION_CHECKPOINT_SCHEMA_VERSION:
        return None
    identity = payload.get("identity")
    if identity != _fixed_json_safe(expected_identity):
        return None
    identity_sha256 = _fixed_canonical_json_sha256(identity)
    if payload.get("identity_sha256") != identity_sha256:
        return None
    record = payload.get("record")
    if not isinstance(record, dict):
        return None
    if payload.get("record_sha256") != _fixed_canonical_json_sha256(record):
        return None
    if record.get("source") != identity.get("source"):
        return None
    if int(record.get("prompt_idx", -1)) != int(identity.get("prompt_idx", -2)):
        return None
    if record.get("tokens_sha256") != identity.get("tokens_sha256"):
        return None
    if record.get("p_trace_sha256") != identity.get("p_trace_sha256"):
        return None
    return record


def fixed_shard_result_filename(tag, entropy_model_size, fpr, prompt_indices,
                          workspace_label="workspace"):
    indices = _fixed_coerce_prompt_indices(prompt_indices)
    if not indices:
        raise ValueError("cannot name an empty shard")
    return (
        f"{_fixed_slug(tag)}__{fixed_entropy_model_tag(entropy_model_size)}__"
        f"fpr-{_fixed_slug(f'{float(fpr):.12g}')}__"
        f"p{min(indices):04d}-{max(indices):04d}__"
        f"{_fixed_slug(workspace_label)}.json"
    )


def _fixed_local_code_fingerprint():
    """Fingerprint the local sources that determine generation/detection."""
    digest = hashlib.sha256()
    source_root = os.path.dirname(os.path.abspath(__file__))
    source_files = [
        "modal_run.py",
        "detectors.py",
        "prc.py",
        "watermark_expt.py",
        "qwen.py",
        "prompts.jsonl",
    ]
    for relative_path in source_files:
        digest.update(relative_path.encode())
        with open(os.path.join(source_root, relative_path), "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    try:
        revision = subprocess.run(
            ["git", "-C", source_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {"sha256": digest.hexdigest(), "git_revision": revision}


def _fixed_semantic_fingerprint(value):
    """Stable hash for nested experiment artifacts, including tensors."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def update(item):
        if item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda k: str(k)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        elif hasattr(item, "detach") and hasattr(item, "shape"):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(item, "tocsr"):
            sparse = item.tocsr()
            digest.update(b"sparse-csr")
            update(np.asarray(sparse.shape))
            update(np.asarray(sparse.indptr))
            update(np.asarray(sparse.indices))
            update(np.asarray(sparse.data))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(np.asarray(item))
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def _fixed_ensure_csv_schema(csv_out):
    parent = os.path.dirname(csv_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(csv_out) or os.path.getsize(csv_out) == 0:
        with open(csv_out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIXED_CSV_COLUMNS).writeheader()
        return

    with open(csv_out, newline="") as f:
        reader = csv.DictReader(f)
        old_columns = reader.fieldnames or []
        rows = list(reader)
    if old_columns == FIXED_CSV_COLUMNS:
        return

    backup = f"{csv_out}.pre_schema_update.bak"
    if not os.path.exists(backup):
        shutil.copy2(csv_out, backup)

    migrated = []
    for row in rows:
        migrated.append({
            "Target FPR": row.get("Target FPR", ""),
            "n": row.get("n", ""),
            "t": row.get("t", ""),
            "eta": row.get("eta", ""),
            "T": row.get("T", ""),
            "r setting": row.get("r setting", ""),
            "r value": row.get("r value", ""),
            "Map TPR": row.get("Map TPR", ""),
            "Entropy Aware TPR": row.get("Entropy Aware TPR", ""),
            "Naive TPR": row.get("Naive TPR", ""),
            "Log Hoeffding TPR": row.get("Log Hoeffding TPR", "skipped"),
            "Map FPR": row.get("Map FPR", ""),
            "Entropy FPR": row.get("Entropy FPR", row.get("FPR", "")),
            "Naive FPR": row.get("Naive FPR", row.get("FPR", "")),
            "Log Hoeffding FPR": row.get("Log Hoeffding FPR", "skipped"),
            "Entropy Model": row.get("Entropy Model", fixed_model_display(FIXED_MODEL_SIZE)),
            "Generation Model": row.get(
                "Generation Model", fixed_model_display(FIXED_MODEL_SIZE)
            ),
            "Entropy Trace Source": row.get(
                "Entropy Trace Source", "cached_generation_p_trace"
            ),
            "Notes": row.get("Notes", ""),
        })
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIXED_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def _fixed_append_summary_row(csv_out, row):
    _fixed_ensure_csv_schema(csv_out)
    row = dict(row)
    row.setdefault("Generation Model", fixed_model_display(FIXED_MODEL_SIZE))
    with open(csv_out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIXED_CSV_COLUMNS)
        writer.writerow(row)


def _fixed_summary_row_identity(row):
    return (
        float(row["eta"]),
        int(row["T"]),
        int(row["n"]),
        int(row["r value"]),
        str(row["r setting"]).strip(),
        int(row["t"]),
        float(row["Target FPR"]),
        str(row.get("Generation Model", fixed_model_display(FIXED_MODEL_SIZE))).strip(),
        str(row["Entropy Model"]).strip(),
    )


def _fixed_summary_row_exists(csv_out, candidate):
    if not os.path.exists(csv_out) or os.path.getsize(csv_out) == 0:
        return False
    candidate_identity = _fixed_summary_row_identity(candidate)
    with open(csv_out, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if _fixed_summary_row_identity(row) == candidate_identity:
                    return True
            except (KeyError, TypeError, ValueError):
                continue
    return False


# ---- image ------------------------------------------------------------------



# ---- artifact build (CPU only; no model load) -------------------------------
@app.function(name="fixed_build_artifacts", image=fixed_image, volumes={"/data": data_vol}, timeout=600)
def fixed_build_artifacts(num_prompts: int, n: int, t: int, eta: float,
                    r: int = 0, fresh: bool = False,
                    generation_model_size: str = FIXED_MODEL_SIZE) -> int:
    import json
    import os
    import shutil

    import numpy as np
    import torch
    from prc import KeyGen, parity_check_rank_info

    requested_r = fixed_resolve_new_run_r(n, r, 0.0)
    generation_model_size = fixed_normalize_model_size(generation_model_size)
    fixed_validate_r_for_keygen(n, t, requested_r)
    max_new_tokens = fixed_experiment_T(n)
    ap = fixed_art_path(
        n, t, eta, requested_r, max_new_tokens, generation_model_size
    )
    wmd = fixed_wm_dir(
        n, t, eta, requested_r, max_new_tokens, generation_model_size
    )
    os.makedirs(os.path.dirname(ap), exist_ok=True)

    config_sig = {
        "n": n,
        "t": t,
        "eta": eta,
        "T": max_new_tokens,
        "blocks": FIXED_DEFAULT_BLOCKS,
        "num_prompts": num_prompts,
        "gen_scheme": "single_codeword_batched",
        "generation_model_size": generation_model_size,
        "generation_model": fixed_model_display(generation_model_size),
        "keygen_seed": FIXED_SEED,
        "keygen_rng_version": "explicit_seed_v1",
    }
    if requested_r is not None:
        config_sig["r"] = requested_r

    data_vol.reload()
    if not fresh and os.path.exists(ap):
        prev = torch.load(ap, weights_only=False, map_location="cpu")
        if prev.get("config_sig") == config_sig:
            if not prev.get("artifact_fingerprint"):
                fingerprint_payload = {
                    key: prev[key] for key in (
                        "encoding_key", "decoding_key", "partition",
                        "prompt_ids_list", "seed", "config_sig",
                    )
                }
                prev["artifact_fingerprint"] = _fixed_semantic_fingerprint(
                    fingerprint_payload
                )
                torch.save(prev, ap)
                data_vol.commit()
            print(f"[build] reusing frozen key from {ap} (config matches)",
                  flush=True)
            return num_prompts
        print("[build] config changed -> rebuilding key, INVALIDATING wm cache",
              flush=True)

    tag_root = os.path.dirname(ap)
    invalidated = []
    for stale_dir in (
        wmd,
        os.path.join(tag_root, "entropy"),
        os.path.join(tag_root, "detect_traces"),
        os.path.join(tag_root, "detection_checkpoints"),
        os.path.join(tag_root, "shard_results"),
    ):
        if os.path.isdir(stale_dir):
            shutil.rmtree(stale_dir)
            invalidated.append(stale_dir)
    if invalidated:
        print("[build] cleared stale key-dependent caches: "
              f"{', '.join(invalidated)}", flush=True)

    torch.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)

    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=t,
        noise_rate=eta,
        r=requested_r,
        seed=FIXED_SEED,
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t_key = decoding_key
    rank_info = parity_check_rank_info(parity_check_matrix)
    actual_r = parity_check_matrix.shape[0]
    print(f"[build] PRC params: n={n} t={t_key} g={g} r={actual_r} "
          f"noise_rate={noise_rate:.4f}", flush=True)
    print(f"[build] parity rank: {rank_info['rank']}/{rank_info['rows']} "
          f"full_rank={rank_info['full_rank']}", flush=True)

    perm = torch.randperm(FIXED_VOCAB, generator=torch.Generator().manual_seed(FIXED_SEED))
    v0 = torch.zeros(FIXED_VOCAB, dtype=torch.bfloat16)
    v0[perm[: FIXED_VOCAB // 2]] = 1.0
    v1 = 1 - v0
    partition = torch.stack([v0, v1], dim=0)

    rows = []
    with open("/root/prompts.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= num_prompts:
                break
    if len(rows) < num_prompts:
        raise RuntimeError(f"prompts.jsonl has {len(rows)} rows, need {num_prompts}")
    prompt_ids_list = [row["prompt_tokens"] for row in rows]

    artifact = {
        "encoding_key": encoding_key,
        "decoding_key": decoding_key,
        "partition": partition,
        "prompt_ids_list": prompt_ids_list,
        "num_prompts": num_prompts,
        "n": n,
        "T": max_new_tokens,
        "generation_model_size": generation_model_size,
        "generation_model": fixed_model_display(generation_model_size),
        "seed": FIXED_SEED,
        "config_sig": config_sig,
        "parity_check_rank_info": rank_info,
    }
    artifact["artifact_fingerprint"] = _fixed_semantic_fingerprint({
        key: artifact[key] for key in (
            "encoding_key", "decoding_key", "partition", "prompt_ids_list",
            "seed", "config_sig",
        )
    })
    torch.save(artifact, ap)
    data_vol.commit()
    print(f"[build] wrote artifacts ({num_prompts} prompts) -> {ap}", flush=True)
    return num_prompts


# ---- cache probes (CPU; decide whether GPU work is needed) ------------------
@app.function(name="fixed_plan_generation", image=fixed_image, volumes={"/data": data_vol}, timeout=300)
def fixed_plan_generation(n: int, t: int, eta: float, prompt_indices: list,
                    r: int = 0,
                    generation_model_size: str = FIXED_MODEL_SIZE) -> dict:
    requested_r = int(r) if r else None
    generation_model_size = fixed_normalize_model_size(generation_model_size)
    T = fixed_experiment_T(n)
    wmd = fixed_wm_dir(
        n, t, eta, requested_r, T, generation_model_size
    )
    prompt_indices = _fixed_coerce_prompt_indices(prompt_indices)
    data_vol.reload()

    wm_missing = [i for i in prompt_indices
                  if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]

    model_null_root = fixed_null_root(generation_model_size)
    null_T = fixed_find_complete_cache_T(
        model_null_root, T, prompt_indices, "null"
    )
    if null_T is None:
        null_T = T
        d = fixed_null_dir(T, generation_model_size)
        null_missing = [
            i for i in prompt_indices
            if not os.path.exists(os.path.join(d, f"null_{i:04d}.pt"))
        ]
    else:
        null_missing = []

    return {
        "wm_missing": wm_missing,
        "null_missing": null_missing,
        "null_T": null_T,
        "T": T,
        "generation_model": fixed_model_display(generation_model_size),
        "null_root": model_null_root,
    }


def fixed_plan_entropy(tag: str, entropy_model_size: str, T: int, null_T: int,
                 prompt_indices: list,
                 generation_model_size: str = FIXED_MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- batched generation (GPU) -----------------------------------------------
@app.cls(
    image=fixed_image,
    gpu=FIXED_GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=FIXED_DEFAULT_MAX_CONTAINERS,
)
class FixedGenerationModel:
    tag: str = modal.parameter()
    model_size: str = modal.parameter()
    code_fingerprint_sha256: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        import torch

        self.model_size = fixed_normalize_model_size(self.model_size)
        os.environ["PRC_MODEL_SIZE"] = self.model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"
        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")
        artifact_model_size = fixed_normalize_model_size(
            art.get("generation_model_size", FIXED_MODEL_SIZE)
        )
        if artifact_model_size != self.model_size:
            raise ValueError(
                f"artifact generation model {artifact_model_size} does not "
                f"match requested model {self.model_size}"
            )

        we = load_watermark_model(self.model_size)
        self.we = we
        self.encoding_key = art["encoding_key"]
        self.prompts = art["prompt_ids_list"]
        self.n = art["n"]
        self.T = int(art.get("T", art.get("config_sig", {}).get("T", fixed_experiment_T(art["n"]))))
        self.partition_cpu = art["partition"]
        self.partition_fingerprint = we.tensor_sha256(self.partition_cpu)
        self.encoding_key_fingerprint = we.semantic_sha256(self.encoding_key)
        self.artifact_fingerprint = art.get("artifact_fingerprint")
        self.artifact_seed = art.get("seed")
        we.partition = self.partition_cpu.to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch

        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

    @modal.method()
    def ready(self) -> dict:
        """Warm and commit the shared model cache before scaling the fleet."""
        return {
            "generation_model": fixed_model_display(self.model_size),
            "model_cache_dir": os.environ.get("PRC_MODEL_CACHE_DIR", ""),
        }

    @modal.method()
    def generate_wm(self, prompt_indices: list) -> dict:
        import os
        import time

        import torch

        wmd = f"/data/{self.tag}/wm"
        data_vol.reload()
        os.makedirs(wmd, exist_ok=True)
        todo = [i for i in prompt_indices
                if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices)}

        t0 = time.time()
        batch = self._prompt_batch(todo)
        tokens, p_traces, trace_details = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=True, return_trace_details=True,
        )
        for row, i in enumerate(todo):
            record = self.we.build_prc_generation_record(
                batch[row],
                tokens[row],
                p_traces[row],
                self.partition_cpu,
                self.n,
                True,
                encoding_key_fingerprint=self.encoding_key_fingerprint,
                prc_codeword_bits=trace_details["prc_codeword_bits"][row],
                base_lm_entropy=trace_details["base_lm_entropy"][row],
                base_token_logprob=trace_details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": i,
                "watermark": True,
                "generation_model": fixed_model_display(self.model_size),
                "generation_model_size": self.model_size,
                "generation_model_variant": "base",
                "artifact_seed": self.artifact_seed,
                "artifact_fingerprint": self.artifact_fingerprint,
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
            })
            torch.save(
                record,
                os.path.join(wmd, f"wm_{i:04d}.pt"),
            )
        data_vol.commit()
        return {"generated": len(todo), "cached": len(prompt_indices) - len(todo),
                "dt": time.time() - t0}

    @modal.method()
    def generate_null(self, prompt_indices: list) -> dict:
        import os
        import time

        import torch

        nd = fixed_null_dir(self.T, self.model_size)
        data_vol.reload()
        os.makedirs(nd, exist_ok=True)
        todo = [i for i in prompt_indices
                if not os.path.exists(os.path.join(nd, f"null_{i:04d}.pt"))]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices)}

        t0 = time.time()
        batch = self._prompt_batch(todo)
        tokens, p_traces, trace_details = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=False, return_trace_details=True,
        )
        for row, i in enumerate(todo):
            record = self.we.build_prc_generation_record(
                batch[row],
                tokens[row],
                p_traces[row],
                self.partition_cpu,
                self.n,
                False,
                encoding_key_fingerprint=self.encoding_key_fingerprint,
                prc_codeword_bits=None,
                base_lm_entropy=trace_details["base_lm_entropy"][row],
                base_token_logprob=trace_details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": i,
                "watermark": False,
                "generation_model": fixed_model_display(self.model_size),
                "generation_model_size": self.model_size,
                "generation_model_variant": "base",
                "artifact_seed": self.artifact_seed,
                "artifact_fingerprint": self.artifact_fingerprint,
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
            })
            torch.save(
                record,
                os.path.join(nd, f"null_{i:04d}.pt"),
            )
        data_vol.commit()
        return {"generated": len(todo), "cached": len(prompt_indices) - len(todo),
                "dt": time.time() - t0}


# ---- optional alternate entropy model (GPU) ---------------------------------
def fixed_EntropyModel(*args, **kwargs):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- detection over cached generations (CPU) --------------------------------
def fixed_detect_all(n: int, t: int, eta: float, fpr: float, null_T: int,
               prompt_indices: list, r: int = 0,
               entropy_model_size: str = FIXED_MODEL_SIZE,
               null_entropy_T: int = 0,
               run_metadata: dict = None,
               generation_model_size: str = FIXED_MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def fixed_detect_all_any(n: int, t: int, eta: float, fpr: float,
                   num_prompts: int = 500, r: int = 0,
                   entropy_model_size: str = FIXED_MODEL_SIZE,
                   generation_model_size: str = FIXED_MODEL_SIZE) -> list:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def fixed_detect_map_summary(n: int, t: int, eta: float, fpr: float,
                       num_prompts: int = 500, r: int = 0,
                       entropy_model_size: str = FIXED_MODEL_SIZE,
                       generation_model_size: str = FIXED_MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def fixed_detect_legacy_first_block_summary(n: int, t: int, eta: float, fpr: float,
                                      num_prompts: int = 500, r: int = 0,
                                      entropy_model_size: str = FIXED_MODEL_SIZE,
                                      legacy_token_length: int = 0) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- driver -----------------------------------------------------------------
def fixed_main(num_prompts: int = 10, max_containers: int = FIXED_DEFAULT_MAX_CONTAINERS,
         prompt_start: int = 0,
         n: int = FIXED_DEFAULT_N, t: int = FIXED_DEFAULT_T, eta: float = FIXED_DEFAULT_ETA,
         fpr: float = FIXED_DEFAULT_FPR, fresh: bool = False,
         batch: int = FIXED_DEFAULT_BATCH,
         entropy_batch: int = FIXED_DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = FIXED_REQUIRED_R_FRAC,
         generation_model_size: str = FIXED_MODEL_SIZE,
         entropy_model_size: str = "",
         gpu: str = FIXED_GPU,
         csv_out: str = "hoeffding_results_summary.csv",
         shard_out: str = "", workspace_label: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def _fixed_aggregate_shard_payloads(payloads,
                              expected_num_prompts=FIXED_CANONICAL_NUM_PROMPTS):
    """Validate and aggregate shard payloads into one summary CSV row."""
    if not payloads:
        raise ValueError("at least one shard payload is required")

    expected_indices = set(range(int(expected_num_prompts)))
    reference_config = payloads[0].get("config")
    reference_artifact = payloads[0].get("artifact_fingerprint")
    reference_code = payloads[0].get("code_fingerprint")
    reference_rank = payloads[0].get("parity_check_rank_info", {})
    if not reference_config or not reference_artifact or not reference_code:
        raise ValueError("first shard is missing configuration fingerprints")
    if int(reference_config.get("canonical_num_prompts", -1)) != int(
            expected_num_prompts):
        raise ValueError("shard canonical prompt count does not match aggregation")

    by_source = {"wm": {}, "null": {}}
    shard_descriptors = []
    for shard_number, payload in enumerate(payloads):
        if payload.get("schema_version") != FIXED_SHARD_RESULT_SCHEMA_VERSION:
            raise ValueError(f"shard {shard_number} has unsupported schema")
        if payload.get("config") != reference_config:
            raise ValueError(f"shard {shard_number} configuration mismatch")
        if payload.get("artifact_fingerprint") != reference_artifact:
            raise ValueError(f"shard {shard_number} artifact/key mismatch")
        if payload.get("code_fingerprint") != reference_code:
            raise ValueError(f"shard {shard_number} code mismatch")
        if payload.get("parity_check_rank_info", {}) != reference_rank:
            raise ValueError(f"shard {shard_number} parity-rank mismatch")

        indices = _fixed_coerce_prompt_indices(payload.get("prompt_indices", []))
        index_set = set(indices)
        if not indices:
            raise ValueError(f"shard {shard_number} has no prompt indices")
        records = payload.get("records", [])
        if len(records) != 2 * len(indices):
            raise ValueError(
                f"shard {shard_number} has {len(records)} records; expected "
                f"{2 * len(indices)}"
            )
        canonical_records = json.dumps(
            _fixed_json_safe(records), sort_keys=True, separators=(",", ":")
        ).encode()
        checksum = hashlib.sha256(canonical_records).hexdigest()
        if checksum != payload.get("records_sha256"):
            raise ValueError(f"shard {shard_number} record checksum mismatch")

        seen_within = set()
        for record in records:
            idx = int(record["prompt_idx"])
            source = record.get("source")
            if source not in by_source:
                source = "wm" if record.get("watermark") else "null"
            key = (source, idx)
            if idx not in index_set:
                raise ValueError(
                    f"shard {shard_number} record {key} is outside its manifest"
                )
            if key in seen_within:
                raise ValueError(f"shard {shard_number} duplicates record {key}")
            if idx in by_source[source]:
                raise ValueError(
                    f"prompt {idx} source {source} appears in multiple shards"
                )
            if bool(record.get("watermark")) != (source == "wm"):
                raise ValueError(f"shard {shard_number} source flag mismatch {key}")
            seen_within.add(key)
            by_source[source][idx] = record
        expected_pairs = {(source, idx) for source in by_source for idx in indices}
        if seen_within != expected_pairs:
            missing = sorted(expected_pairs - seen_within)
            raise ValueError(
                f"shard {shard_number} is missing source/index records: {missing[:5]}"
            )

        workspace = str(payload.get("workspace_label", "unknown"))
        shard_descriptors.append({
            "workspace_label": workspace,
            "prompt_indices": indices,
            "records_sha256": checksum,
            "created_at": payload.get("created_at"),
            "null_cache_T": payload.get("null_cache_T"),
            "null_entropy_cache_T": payload.get("null_entropy_cache_T"),
        })

    for source, records in by_source.items():
        actual = set(records)
        if actual != expected_indices:
            missing = sorted(expected_indices - actual)
            extra = sorted(actual - expected_indices)
            raise ValueError(
                f"{source} prompt coverage mismatch; missing={missing[:10]} "
                f"extra={extra[:10]}"
            )

    wm = [by_source["wm"][i] for i in sorted(expected_indices)]
    null = [by_source["null"][i] for i in sorted(expected_indices)]
    naive_values = [r.get("decision_naive") for r in wm + null]
    has_naive = all(value is not None for value in naive_values)
    if not has_naive and any(value is not None for value in naive_values):
        raise ValueError("naive decisions are inconsistently present across shards")

    def positives(records, key):
        return sum(bool(record[key]) for record in records)

    map_tp = positives(wm, "decision_map")
    map_fp = positives(null, "decision_map")
    entropy_tp = positives(wm, "decision_entropy")
    entropy_fp = positives(null, "decision_entropy")
    naive_tp = positives(wm, "decision_naive") if has_naive else None
    naive_fp = positives(null, "decision_naive") if has_naive else None
    nwm, nnw = len(wm), len(null)
    config = reference_config
    workspace_count = len({
        shard["workspace_label"] for shard in shard_descriptors
    })
    notes = "; ".join([
        f"{config['generation_model']} generation",
        f"entropy model {config['entropy_model']}",
        config["entropy_trace_source"],
        (f"{len(shard_descriptors)}-shard prompt aggregation across "
         f"{workspace_count} workspaces"),
        "validated exact global prompt coverage with no gaps or duplicates",
        f"artifact_fingerprint={reference_artifact}",
        f"code_fingerprint={reference_code.get('sha256', '')}",
    ])
    row = {
        "eta": config["eta"],
        "T": config["T"],
        "n": config["n"],
        "r value": config["r_value"],
        "r setting": config["r_setting"],
        "t": config["t"],
        "Target FPR": f"{float(config['target_fpr']):.0e}",
        "Entropy Model": config["entropy_model"],
        "Generation Model": config["generation_model"],
        "Map TPR": _fixed_format_rate(map_tp, nwm),
        "Entropy Aware TPR": _fixed_format_rate(entropy_tp, nwm),
        "Naive TPR": _fixed_format_rate(naive_tp, nwm) if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _fixed_format_rate(map_fp, nnw),
        "Entropy FPR": _fixed_format_rate(entropy_fp, nnw),
        "Naive FPR": _fixed_format_rate(naive_fp, nnw) if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Trace Source": config["entropy_trace_source"],
        "Notes": notes,
    }
    aggregation = {
        "schema_version": FIXED_SHARD_RESULT_SCHEMA_VERSION,
        "config": config,
        "artifact_fingerprint": reference_artifact,
        "code_fingerprint": reference_code,
        "parity_check_rank_info": reference_rank,
        "expected_num_prompts": int(expected_num_prompts),
        "shards": shard_descriptors,
        "counts": {
            "wm_total": nwm,
            "null_total": nnw,
            "map_tp": map_tp,
            "map_fp": map_fp,
            "entropy_tp": entropy_tp,
            "entropy_fp": entropy_fp,
            "naive_tp": naive_tp,
            "naive_fp": naive_fp,
        },
        "summary_row": row,
        "aggregated_at": datetime.now(timezone.utc).isoformat(),
    }
    return row, aggregation


def aggregate_shards(shard_files: str,
                     csv_out: str = "hoeffding_results_summary.csv",
                     aggregate_out: str = "",
                     expected_num_prompts: int = FIXED_CANONICAL_NUM_PROMPTS):
    """Validate local shard JSON files and append one authoritative CSV row."""
    paths = [path.strip() for path in shard_files.split(",") if path.strip()]
    if not paths:
        raise ValueError("--shard-files must contain comma-separated JSON paths")
    payloads = []
    for path in paths:
        with open(path) as f:
            payloads.append(json.load(f))
    row, aggregation = _fixed_aggregate_shard_payloads(
        payloads, expected_num_prompts=expected_num_prompts
    )
    aggregation["local_shard_files"] = paths
    if not aggregate_out:
        config = aggregation["config"]
        entropy_size = (
            config["entropy_model"].replace("Qwen3-", "").replace("-Base", "")
        )
        generation_size = (
            config["generation_model"].replace("Qwen3-", "").replace("-Base", "")
        )
        aggregate_name = (
            f"eta{_fixed_slug(config['eta'])}_n{config['n']}_T{config['T']}_"
            f"r{config['r_value']}_"
            f"gen-{fixed_entropy_model_tag(generation_size)}_"
            f"{fixed_entropy_model_tag(entropy_size)}_"
            f"fpr-{_fixed_slug(config['target_fpr'])}.json"
        )
        aggregate_out = os.path.join("outputs", "aggregates", aggregate_name)
    _fixed_atomic_write_json(aggregate_out, aggregation)
    if _fixed_summary_row_exists(csv_out, row):
        raise ValueError(
            "the authoritative CSV already contains this experiment identity; "
            "not appending a duplicate row"
        )
    _fixed_append_summary_row(csv_out, row)
    print(f"[aggregate] validated {len(paths)} shards with "
          f"{expected_num_prompts} watermarked + {expected_num_prompts} null",
          flush=True)
    print(f"[aggregate] wrote audit manifest -> {aggregate_out}", flush=True)
    print(f"[aggregate] appended one summary row -> {csv_out}", flush=True)


def fixed_legacy_first_block_redetect(n: int = FIXED_DEFAULT_N, t: int = FIXED_DEFAULT_T,
                                eta: float = FIXED_DEFAULT_ETA,
                                fpr: float = FIXED_DEFAULT_FPR,
                                num_prompts: int = 500,
                                r: int = 0, r_frac: float = 0.0,
                                entropy_model_size: str = FIXED_MODEL_SIZE,
                                legacy_token_length: int = 0,
                                csv_out: str = "hoeffding_results_summary.csv"):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- re-detection sweep over all weight kinds (CPU-only, no regeneration) ---
def fixed_redetect(n: int = FIXED_DEFAULT_N, t: int = FIXED_DEFAULT_T, eta: float = FIXED_DEFAULT_ETA,
             fpr: float = FIXED_DEFAULT_FPR, num_prompts: int = 500,
             r: int = 0, r_frac: float = 0.0,
             generation_model_size: str = FIXED_MODEL_SIZE,
             entropy_model_size: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def fixed_redetect_map(n: int = FIXED_DEFAULT_N, t: int = FIXED_DEFAULT_T, eta: float = FIXED_DEFAULT_ETA,
                 fpr: float = FIXED_DEFAULT_FPR, num_prompts: int = 500,
                 r: int = 0, r_frac: float = 0.0,
                 generation_model_size: str = FIXED_MODEL_SIZE,
                 entropy_model_size: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- re-detect a whole set of configs in one Modal session ------------------
# (n, t, eta) for every config we've generated so far. Both cache layouts.
FIXED_REDETECT_CONFIGS = [
    (256, 3, 0.05), (400, 3, 0.05), (512, 3, 0.05), (1024, 3, 0.05),
    (400, 3, 0.20),
    (400, 5, 0.05), (512, 5, 0.05), (1024, 5, 0.05), (2048, 5, 0.05),
]


def fixed_redetect_all(fpr: float = 1e-3, num_prompts: int = 500):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def generate_fixed(num_prompts: int = 10, max_containers: int = FIXED_DEFAULT_MAX_CONTAINERS,
         prompt_start: int = 0,
         n: int = FIXED_DEFAULT_N, t: int = FIXED_DEFAULT_T, eta: float = FIXED_DEFAULT_ETA,
         fpr: float = FIXED_DEFAULT_FPR, fresh: bool = False,
         batch: int = FIXED_DEFAULT_BATCH,
         entropy_batch: int = FIXED_DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = FIXED_REQUIRED_R_FRAC,
         generation_model_size: str = FIXED_MODEL_SIZE,
         entropy_model_size: str = "",
         gpu: str = FIXED_GPU,
         csv_out: str = "hoeffding_results_summary.csv",
         shard_out: str = "", workspace_label: str = ""):
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if entropy_batch <= 0:
        raise ValueError(
            f"entropy_batch must be positive, got {entropy_batch}"
        )
    if max_containers <= 0:
        raise ValueError(
            f"max_containers must be positive, got {max_containers}"
        )
    gpu = str(gpu).strip()
    if not gpu:
        raise ValueError("gpu must be non-empty")
    prompt_indices = fixed_prompt_indices_for_shard(prompt_start, num_prompts)
    resolved_r = fixed_resolve_new_run_r(n, r, r_frac)
    fixed_validate_r_for_keygen(n, t, resolved_r)
    generation_model_size = fixed_normalize_model_size(generation_model_size)
    entropy_model_size = fixed_normalize_model_size(
        entropy_model_size or generation_model_size
    )
    T = fixed_experiment_T(n)
    tag = fixed_config_tag(
        n, t, eta, resolved_r, T, generation_model_size
    )
    workspace_label = (
        workspace_label.strip() if workspace_label else
        os.environ.get("MODAL_PROFILE", "workspace")
    )
    code_fingerprint = _fixed_local_code_fingerprint()
    is_complete_run = prompt_indices == list(range(FIXED_CANONICAL_NUM_PROMPTS))
    r_text = f"r={resolved_r} ({FIXED_REQUIRED_R_SETTING})"
    print(f"[main] config {tag}  FPR_target={fpr:g}  ({num_prompts} prompts, "
          f"global range={prompt_indices[0]}..{prompt_indices[-1]}, "
          f"batch={batch}, entropy_batch={entropy_batch}, {r_text}, "
          f"generation_model={fixed_model_display(generation_model_size)}, "
          f"entropy_model={fixed_model_display(entropy_model_size)}, fresh={fresh}) ...",
          flush=True)
    print(f"[main] GPU={gpu} max_containers={max_containers}", flush=True)

    fixed_build_artifacts.remote(
        FIXED_CANONICAL_NUM_PROMPTS, n, t, eta, resolved_r or 0, fresh,
        generation_model_size,
    )

    plan = fixed_plan_generation.remote(
        n, t, eta, prompt_indices, resolved_r or 0,
        generation_model_size,
    )
    wm_missing, null_missing = plan["wm_missing"], plan["null_missing"]
    null_T = plan["null_T"]
    print(f"[main] to generate: {len(wm_missing)} watermarked, "
          f"{len(null_missing)} null  (selected null store T={null_T}; "
          f"root={plan['null_root']}; scoring prefix T={T})",
          flush=True)

    if wm_missing or null_missing:
        from concurrent.futures import ThreadPoolExecutor

        model = FixedGenerationModel.with_options(
            gpu=gpu, max_containers=max_containers
        )(
            tag=tag,
            model_size=generation_model_size,
            code_fingerprint_sha256=code_fingerprint["sha256"],
        )
        cache_status = model.ready.remote()
        print(
            f"[main] model cache ready: {cache_status['generation_model']} "
            f"at {cache_status['model_cache_dir']}",
            flush=True,
        )
        work = []
        if wm_missing:
            work.append((
                "wm", model.generate_wm, _chunks(wm_missing, batch)
            ))
        if null_missing:
            work.append((
                "null", model.generate_null, _chunks(null_missing, batch)
            ))

        def _run_generation_map(item):
            kind, method, chunks = item
            return kind, list(method.map(chunks))

        # Watermarked and null generation are independent. Dispatching both
        # maps together lets batch=100 use 5+5=10 GPUs for 500 prompts.
        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            calls = list(pool.map(_run_generation_map, work))
        for kind, metas in calls:
            gen = sum(m.get("generated", 0) for m in metas)
            print(f"[main] {kind}: generated {gen} in {len(metas)} batches",
                  flush=True)
    else:
        print("[main] all generations cached -> skipping generation GPU fleet",
              flush=True)

    return {"generation_only": True, "tag": tag, "plan": plan}


# Seeded fixed replicates: isolated keys and reusable null caches
# ----------------------------------------------------------------------------

REPLICATE_SCHEME = "fixed_prc_replicate_v1"
REPLICATE_SEED = 12345
REPLICATE_MODEL_SIZE = "0.6B"
REPLICATE_MODEL_DISPLAY = "Qwen3-0.6B-Base"
REPLICATE_VOCAB = 151_936
REPLICATE_GPU = "A10G"
REPLICATE_DEFAULT_BATCH = 64
REPLICATE_DEFAULT_MAX_CONTAINERS = 5
REPLICATE_CANONICAL_NUM_PROMPTS = 500
REPLICATE_RESULT_SCHEMA_VERSION = 1
REPLICATE_CSV_COLUMNS = (
    "timestamp_utc", "scheme", "eta", "T", "n", "r value", "r setting",
    "t", "Target FPR", "Generation Model", "num prompts", "batch",
    "experiment seed", "Map TPR", "Map FPR", "Entropy Aware TPR",
    "Entropy FPR", "Naive TPR", "Naive FPR", "null cache T",
    "artifact fingerprint",
)




def replicate_config_tag(n, t, eta, experiment_seed):
    return (
        f"{REPLICATE_SCHEME}/qwen3_0p6b_base/"
        f"n{int(n)}_T{int(n)}_t{int(t)}_eta{float(eta):.2f}_rr99of100_"
        f"seed{int(experiment_seed)}"
    )


def replicate_artifact_path(tag):
    return f"/data/{tag}/artifacts.pt"


def replicate_wm_dir(tag):
    return f"/data/{tag}/wm"


def replicate_shared_null_dir(length):
    return f"/data/_nulls/T{int(length)}"






def _replicate_append_csv(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLICATE_CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in REPLICATE_CSV_COLUMNS})


def _replicate_local_code_fingerprint():
    digest = hashlib.sha256()
    for path in (
        "modal_run.py", "prc.py", "watermark_expt.py",
        "detectors.py", "qwen.py", "constants.py",
    ):
        digest.update(path.encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def _replicate_find_compatible_null_T(prompt_indices, requested_T):
    if not os.path.isdir("/data/_nulls"):
        return None
    candidates = []
    for name in os.listdir("/data/_nulls"):
        match = re.fullmatch(r"T(\d+)", name)
        if match and int(match.group(1)) >= int(requested_T):
            candidates.append(int(match.group(1)))
    for length in sorted(candidates):
        directory = replicate_shared_null_dir(length)
        if all(
            os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
            for index in prompt_indices
        ):
            return length
    return None





@app.function(name="replicate_build_artifacts", image=replicate_image, volumes={"/data": data_vol}, timeout=600)
def replicate_build_artifacts(n, t, eta, experiment_seed):
    import numpy as np
    import torch
    from detectors import semantic_sha256
    from prc import KeyGen, parity_check_rank_info

    n = int(n)
    t = int(t)
    experiment_seed = int(experiment_seed)
    r = int(round(0.99 * n))
    if n <= 0 or t < 2 or experiment_seed < 0:
        raise ValueError("invalid fixed replicate configuration")
    if n - r < t - 1:
        raise ValueError(f"n-r={n-r} is too small for t={t}")
    tag = replicate_config_tag(n, t, eta, experiment_seed)
    path = replicate_artifact_path(tag)
    config = {
        "scheme": REPLICATE_SCHEME,
        "n": n,
        "T": n,
        "t": t,
        "eta": float(eta),
        "r": r,
        "experiment_seed": experiment_seed,
        "partition_seed": REPLICATE_SEED,
        "generation_model": REPLICATE_MODEL_DISPLAY,
        "stopping_policy": "forced_length_v1",
    }
    data_vol.reload()
    if os.path.exists(path):
        previous = torch.load(path, weights_only=False, map_location="cpu")
        if previous.get("config_sig") == config:
            return {
                "tag": tag,
                "artifact_fingerprint": previous["artifact_fingerprint"],
                "reused": True,
            }
        raise RuntimeError(f"incompatible artifact already exists at {path}")

    np.random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=t,
        noise_rate=float(eta),
        r=r,
        seed=experiment_seed,
    )
    rank_info = parity_check_rank_info(decoding_key[1])
    if not rank_info["full_rank"] or rank_info["rank"] != r:
        raise RuntimeError(f"fixed parity matrix is not full rank: {rank_info}")

    permutation = torch.randperm(
        REPLICATE_VOCAB, generator=torch.Generator().manual_seed(REPLICATE_SEED)
    )
    bucket_zero = torch.zeros(REPLICATE_VOCAB, dtype=torch.bfloat16)
    bucket_zero[permutation[:REPLICATE_VOCAB // 2]] = 1.0
    partition = torch.stack([bucket_zero, 1 - bucket_zero], dim=0)

    rows = []
    with open("/root/prompts.jsonl") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= REPLICATE_CANONICAL_NUM_PROMPTS:
                break
    if len(rows) < REPLICATE_CANONICAL_NUM_PROMPTS:
        raise RuntimeError("prompts.jsonl does not contain 500 prompts")

    artifact = {
        "encoding_key": encoding_key,
        "decoding_key": decoding_key,
        "partition": partition,
        "prompt_ids_list": [row["prompt_tokens"] for row in rows],
        "n": n,
        "T": n,
        "r": r,
        "rank_info": rank_info,
        "experiment_seed": experiment_seed,
        "config_sig": config,
    }
    artifact["artifact_fingerprint"] = semantic_sha256(artifact)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(artifact, path)
    data_vol.commit()
    print(
        f"[build] {tag}: T=n={n}, t={t}, r={r}, "
        f"rank={rank_info['rank']}", flush=True,
    )
    return {
        "tag": tag,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "reused": False,
    }


@app.function(name="replicate_plan_generation", image=replicate_image, volumes={"/data": data_vol}, timeout=300)
def replicate_plan_generation(tag, prompt_indices, T):
    data_vol.reload()
    missing = [
        index for index in prompt_indices
        if not os.path.exists(os.path.join(replicate_wm_dir(tag), f"wm_{index:04d}.pt"))
    ]
    null_T = _replicate_find_compatible_null_T(prompt_indices, T)
    if null_T is None:
        null_T = int(T)
        null_missing = [
            index for index in prompt_indices
            if not os.path.exists(
                os.path.join(replicate_shared_null_dir(T), f"null_{index:04d}.pt")
            )
        ]
    else:
        null_missing = []
    return {"wm_missing": missing, "null_missing": null_missing, "null_T": null_T}


@app.cls(
    image=replicate_image,
    gpu=REPLICATE_GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=REPLICATE_DEFAULT_MAX_CONTAINERS,
)
class ReplicateGenerationModel:
    tag: str = modal.parameter()
    code_fingerprint_sha256: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from detectors import semantic_sha256, tensor_sha256

        data_vol.reload()
        artifact = torch.load(
            replicate_artifact_path(self.tag), weights_only=False, map_location="cpu"
        )
        self.encoding_key = artifact["encoding_key"]
        self.partition_cpu = artifact["partition"]
        self.partition_fingerprint = tensor_sha256(self.partition_cpu)
        self.key_fingerprint = semantic_sha256(self.encoding_key)
        self.artifact_fingerprint = artifact["artifact_fingerprint"]
        self.experiment_seed = artifact["experiment_seed"]
        self.prompts = artifact["prompt_ids_list"]
        self.n = int(artifact["n"])
        self.T = int(artifact["T"])

        we = load_watermark_model(REPLICATE_MODEL_SIZE)
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
    def ready(self):
        return {"model": REPLICATE_MODEL_DISPLAY, "T": self.T, "n": self.n}

    @modal.method()
    def generate_wm(self, prompt_indices):
        import time
        import torch

        data_vol.reload()
        directory = replicate_wm_dir(self.tag)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"wm_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}
        started = time.time()
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect(
            self.we.model,
            prompt_batch,
            self.T,
            self.encoding_key,
            self.partition,
            watermark=True,
            return_trace_details=True,
        )
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row], tokens[row], p_traces[row],
                self.partition_cpu, self.n, True,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=details["prc_codeword_bits"][row],
                base_lm_entropy=details["base_lm_entropy"][row],
                base_token_logprob=details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": int(index),
                "scheme": REPLICATE_SCHEME,
                "generation_model_size": REPLICATE_MODEL_SIZE,
                "generation_model": REPLICATE_MODEL_DISPLAY,
                "artifact_seed": self.experiment_seed,
                "artifact_fingerprint": self.artifact_fingerprint,
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
            })
            torch.save(record, os.path.join(directory, f"wm_{index:04d}.pt"))
        data_vol.commit()
        return {
            "generated": len(todo), "cached": len(prompt_indices) - len(todo),
            "batch": len(todo), "seconds": time.time() - started,
        }

    @modal.method()
    def generate_null(self, prompt_indices):
        import torch

        data_vol.reload()
        directory = replicate_shared_null_dir(self.T)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect(
            self.we.model, prompt_batch, self.T, self.encoding_key,
            self.partition, watermark=False, return_trace_details=True,
        )
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row], tokens[row], p_traces[row],
                self.partition_cpu, self.n, False,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=None,
                base_lm_entropy=details["base_lm_entropy"][row],
                base_token_logprob=details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": int(index),
                "generation_model_size": REPLICATE_MODEL_SIZE,
                "generation_model": REPLICATE_MODEL_DISPLAY,
            })
            torch.save(record, os.path.join(directory, f"null_{index:04d}.pt"))
        data_vol.commit()
        return {"generated": len(todo), "cached": 0, "batch": len(todo)}


def replicate_detect_all(tag, prompt_indices, null_T, fpr, batch, code_fingerprint_sha256):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def replicate_main(num_prompts: int = REPLICATE_CANONICAL_NUM_PROMPTS,
         n: int = 256, t: int = 3, eta: float = 0.05,
         fpr: float = 1e-3, batch: int = REPLICATE_DEFAULT_BATCH,
         experiment_seed: int = 54321,
         max_containers: int = REPLICATE_DEFAULT_MAX_CONTAINERS,
         gpu: str = REPLICATE_GPU,
         csv_out: str = "fixed_replicate_results_summary.csv"):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def generate_replicate(num_prompts: int = REPLICATE_CANONICAL_NUM_PROMPTS,
         n: int = 256, t: int = 3, eta: float = 0.05,
         fpr: float = 1e-3, batch: int = REPLICATE_DEFAULT_BATCH,
         experiment_seed: int = 54321,
         max_containers: int = REPLICATE_DEFAULT_MAX_CONTAINERS,
         gpu: str = REPLICATE_GPU,
         csv_out: str = "fixed_replicate_results_summary.csv"):
    if not 0 < num_prompts <= REPLICATE_CANONICAL_NUM_PROMPTS:
        raise ValueError("num_prompts must be in [1,500]")
    if batch <= 0 or max_containers <= 0 or experiment_seed < 0:
        raise ValueError("batch, max_containers, and experiment_seed are invalid")
    prompt_indices = list(range(int(num_prompts)))
    tag = replicate_config_tag(n, t, eta, experiment_seed)
    code_fingerprint = _replicate_local_code_fingerprint()
    print(
        f"[main] {REPLICATE_SCHEME}: T=n={n}, t={t}, eta={eta}, fpr={fpr:g}, "
        f"prompts={num_prompts}, batch={batch}, seed={experiment_seed}, "
        f"GPU={gpu}, max_containers={max_containers}", flush=True,
    )
    build = replicate_build_artifacts.remote(n, t, eta, experiment_seed)
    print(
        f"[main] artifact {'reused' if build['reused'] else 'built'}: "
        f"{build['artifact_fingerprint']}", flush=True,
    )
    plan = replicate_plan_generation.remote(tag, prompt_indices, n)
    print(
        f"[main] generation plan: wm_missing={len(plan['wm_missing'])}, "
        f"null_missing={len(plan['null_missing'])}, null_T={plan['null_T']}",
        flush=True,
    )
    if plan["wm_missing"] or plan["null_missing"]:
        from concurrent.futures import ThreadPoolExecutor

        model = ReplicateGenerationModel.with_options(
            gpu=gpu, max_containers=max_containers
        )(tag=tag, code_fingerprint_sha256=code_fingerprint)
        print(f"[main] model ready: {model.ready.remote()}", flush=True)
        work = []
        if plan["wm_missing"]:
            work.append(("wm", model.generate_wm, _chunks(plan["wm_missing"], batch)))
        if plan["null_missing"]:
            work.append((
                "null", model.generate_null, _chunks(plan["null_missing"], batch)
            ))

        def run_map(item):
            name, method, chunks = item
            return name, list(method.map(chunks))

        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            mapped = list(pool.map(run_map, work))
        for name, records in mapped:
            print(
                f"[main] {name}: generated="
                f"{sum(record['generated'] for record in records)}, "
                f"batch_sizes={[record['batch'] for record in records if record['batch']]}",
                flush=True,
            )

    return {"generation_only": True, "tag": tag, "plan": plan}


# Online PRC: causal keys, generation, continuation and cache/result helpers
# ----------------------------------------------------------------------------

def online_legacy_config_tag(n: int, t: int, eta: float,
                      experiment_seed: int = SEED,
                      generation_model_size: str = MODEL_SIZE) -> str:
    tag = (
        f"{SCHEME}/{model_cache_name(generation_model_size)}/"
        f"n{int(n)}_T{int(n)}_t{int(t)}_eta{float(eta):.2f}_rr99of100"
    )
    if int(experiment_seed) != SEED:
        tag += f"_seed{int(experiment_seed)}"
    return tag


def online_config_tag(n: int, t: int, eta: float,
               experiment_seed: int = SEED,
               generation_model_size: str = MODEL_SIZE,
               kv_cache_implementation: str = DEFAULT_KV_CACHE_IMPLEMENTATION,
               ) -> str:
    """Sampler-v2 namespace; legacy caches remain readable as sources."""
    implementation = normalize_kv_cache_implementation(
        kv_cache_implementation
    )
    tag = (
        f"{online_legacy_config_tag(n, t, eta, experiment_seed, generation_model_size)}"
        f"_sampler-{SAMPLER_CACHE_TAG}"
    )
    if implementation != DEFAULT_KV_CACHE_IMPLEMENTATION:
        tag += f"_kvcache-{kv_cache_version(implementation)}"
    return tag


def online_artifact_path(tag: str) -> str:
    return f"/data/{tag}/artifacts.pt"


def online_wm_dir(tag: str) -> str:
    return f"/data/{tag}/wm"


def online_shared_null_dir(length: int,
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
        online_shared_null_dir(length, generation_model_size),
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
        possible_tags = [online_config_tag(
            T_value, t, eta, experiment_seed, model_size,
            kv_cache_implementation,
        )]
        if kv_cache_implementation == DEFAULT_KV_CACHE_IMPLEMENTATION:
            possible_tags.append(online_legacy_config_tag(
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
        "modal_run.py", "online_prc.py", "watermark_expt.py",
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





@app.function(name="online_proxy_unit_tests_remote", cpu=2.0, timeout=900)
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


@app.function(name="online_proxy_8b_native_quality_shard", cpu=2.0, volumes={"/data": data_vol}, timeout=1800)
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
            online_artifact_path(audit["source_tag"]),
            weights_only=False,
            map_location="cpu",
        )
        for index in indices:
            record = torch.load(
                os.path.join(online_wm_dir(audit["source_tag"]), f"wm_{index:04d}.pt"),
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
        online_artifact_path(PRC_AUDITS[-1]["source_tag"]),
        weights_only=False,
        map_location="cpu",
    )
    null_manifest = load_null_cache_manifest(NULL_TRACE_T, "8B")
    if null_manifest is None:
        raise FileNotFoundError("the canonical 8B null cache manifest is missing")
    for index in indices:
        record = torch.load(
            os.path.join(
                online_shared_null_dir(NULL_TRACE_T, "8B"),
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


@app.function(name="online_build_artifacts", volumes={"/data": data_vol}, timeout=600)
def online_build_artifacts(num_prompts: int, n: int, t: int, eta: float,
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
    tag = online_config_tag(
        n, t, eta, experiment_seed, generation_model_size,
        kv_cache_implementation,
    )
    path = online_artifact_path(tag)
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
    if os.path.isdir(online_wm_dir(tag)):
        shutil.rmtree(online_wm_dir(tag))
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
    null_root = os.path.dirname(online_shared_null_dir(0, model_size))
    if not os.path.isdir(null_root):
        return None
    candidates = []
    for name in os.listdir(null_root):
        match = re.fullmatch(r"T(\d+)", name)
        if match and int(match.group(1)) >= int(requested_T):
            candidates.append(int(match.group(1)))
    for length in sorted(candidates):
        directory = online_shared_null_dir(length, model_size)
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


@app.function(name="online_plan_generation", volumes={"/data": data_vol}, timeout=300)
def online_plan_generation(tag: str, prompt_indices: list[int], T: int,
                    allow_wm_reuse: bool = True,
                    null_kv_cache_implementation: str = "",
                    include_null: bool = True) -> dict:
    import torch

    data_vol.reload()
    requested_artifact = torch.load(
        online_artifact_path(tag), weights_only=False, map_location="cpu"
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
        if not os.path.exists(os.path.join(online_wm_dir(tag), f"wm_{index:04d}.pt"))
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
            and os.path.isdir(online_shared_null_dir(null_T, generation_model_size))
            and any(
                name.startswith("null_") and name.endswith(".pt")
                for name in os.listdir(
                    online_shared_null_dir(null_T, generation_model_size)
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
                    online_shared_null_dir(T, generation_model_size),
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


@app.function(name="online_plan_null_cache_generation", volumes={"/data": data_vol}, timeout=1800)
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
        online_artifact_path(tag), weights_only=False, map_location="cpu"
    )
    model_size = artifact_generation_model_size(artifact)
    requested_T = int(requested_T)
    implementation = normalize_kv_cache_implementation(
        null_kv_cache_implementation
    )
    null_root = os.path.dirname(online_shared_null_dir(0, model_size))
    candidates = []
    if os.path.isdir(null_root):
        for name in os.listdir(null_root):
            match = re.fullmatch(r"T(\d+)", name)
            if match and int(match.group(1)) >= requested_T:
                candidates.append(int(match.group(1)))

    rejected_candidates = []
    for length in sorted(candidates):
        directory = online_shared_null_dir(length, model_size)
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

    directory = online_shared_null_dir(requested_T, model_size)
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


@app.function(name="online_verify_shared_null_cache", volumes={"/data": data_vol}, timeout=1800)
def verify_shared_null_cache(
    tag: str,
    prompt_indices: list[int],
    null_T: int,
) -> dict:
    """Perform the cache-only acceptance audit after null generation."""
    import torch

    data_vol.reload()
    artifact = torch.load(
        online_artifact_path(tag), weights_only=False, map_location="cpu"
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
            online_shared_null_dir(null_T, model_size),
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


@app.function(name="online_plan_cross_model_entropy_audits", cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
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
            online_artifact_path(source_tag), weights_only=False, map_location="cpu"
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
                online_wm_dir(source_tag), f"wm_{index:04d}.pt"
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
            online_shared_null_dir(null_T, generation_model_size),
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


@app.function(name="online_plan_textseal_proxy_entropy", cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
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
        online_artifact_path(PRC_AUDITS[-1]["source_tag"]),
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
class OnlineGenerationModel:
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
            online_artifact_path(self.tag), weights_only=False, map_location="cpu"
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

        we = load_watermark_model(self.model_size)
        _numpy_pickle_compat()
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
        directory = online_wm_dir(self.tag)
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
                online_artifact_path(resume_source_tag),
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
                    online_wm_dir(resume_source_tag), f"wm_{index:04d}.pt"
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
        directory = online_shared_null_dir(self.T, self.model_size)
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


@app.function(name="online_audit_continuation", volumes={"/data": data_vol}, timeout=600)
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
        online_artifact_path(target_tag), weights_only=False, map_location="cpu"
    )
    source_artifact = torch.load(
        online_artifact_path(source_tag), weights_only=False, map_location="cpu"
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
            os.path.join(online_wm_dir(source_tag), f"wm_{index:04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        target_record = torch.load(
            os.path.join(online_wm_dir(target_tag), f"wm_{index:04d}.pt"),
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


@app.function(name="online_prepare_sweep_ceiling", volumes={"/data": data_vol}, timeout=600)
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
        online_artifact_path(target_tag), weights_only=False, map_location="cpu"
    )
    reference_artifact = torch.load(
        online_artifact_path(reference_tag), weights_only=False, map_location="cpu"
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
        os.path.dirname(online_wm_dir(target_tag)),
        "quarantine",
        f"prefix-mismatch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )
    for index in prompt_indices:
        reference_path = os.path.join(
            online_wm_dir(reference_tag), f"wm_{index:04d}.pt"
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

        target_path = os.path.join(online_wm_dir(target_tag), f"wm_{index:04d}.pt")
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


def detect_full_audit_prompt_shard(request: dict) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(name="online_aggregate_full_audit_shards", cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
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

    _numpy_pickle_compat()
    started = time.time()
    data_vol.reload()
    artifact = torch.load(
        online_artifact_path(tag), weights_only=False, map_location="cpu"
    )
    source_artifact = torch.load(
        online_artifact_path(watermarked_source_tag),
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


def detect_cross_model_entropy_prompt_shard(request: dict) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(name="online_aggregate_cross_model_entropy_audit_shards", cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
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
        online_artifact_path(source_tag), weights_only=False, map_location="cpu"
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


def detect_all(tag: str, prompt_indices: list[int], null_T: int, fpr: float,
               batch: int, code_fingerprint_sha256: str,
               wm_source_tag: str = "",
               wm_reuse_mode: str = "exact_cache",
               wm_resume_source_tag: str = "",
               wm_resume_source_T: int = 0) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def detect_saved_prefix(source_tag: str, prefix_T: int,
                        prompt_indices: list[int], null_T: int,
                        fpr: float, code_fingerprint_sha256: str) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def prepare_map_prefix_shard(request: dict) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def detect_map_prefix_grid_serial(source_tag: str,
                                  prefix_lengths: list[int],
                                  prompt_indices: list[int], fpr: float,
                                  code_fingerprint_sha256: str,
                                  target_map_tpr: float,
                                  stop_after_first_below: bool = True,
                                  persist_results: bool = True) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(name="online_aggregate_map_prefix_shards", cpu=1.0, volumes={"/data": data_vol}, timeout=1800)
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
        online_artifact_path(source_tag), weights_only=False, map_location="cpu"
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

    model = OnlineGenerationModel.with_options(**model_cls_options(
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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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


@app.function(name="online_compare_kv_cache_records", volumes={"/data": data_vol}, timeout=600)
def compare_kv_cache_records(n: int, t: int, eta: float,
                             experiment_seed: int,
                             prompt_indices: list[int],
                             generation_model_size: str = MODEL_SIZE) -> dict:
    """Compare isolated concat/static online records field-for-field."""
    import numpy as np
    import torch

    generation_model_size = normalize_model_size(generation_model_size)
    concat_tag = online_config_tag(
        n, t, eta, experiment_seed, generation_model_size, "concat"
    )
    static_tag = online_config_tag(
        n, t, eta, experiment_seed, generation_model_size, "static"
    )
    data_vol.reload()
    concat_artifact = torch.load(
        online_artifact_path(concat_tag), weights_only=False, map_location="cpu"
    )
    static_artifact = torch.load(
        online_artifact_path(static_tag), weights_only=False, map_location="cpu"
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
            os.path.join(online_wm_dir(concat_tag), f"wm_{int(index):04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        static_record = torch.load(
            os.path.join(online_wm_dir(static_tag), f"wm_{int(index):04d}.pt"),
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
    tag = online_config_tag(
        n, t, eta, experiment_seed, generation_model_size, "static"
    )
    model = OnlineGenerationModel.with_options(**model_cls_options(
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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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
    tag = online_config_tag(
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
    build = online_build_artifacts.remote(
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
        model = OnlineGenerationModel.with_options(**model_cls_options(
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


def online_main(num_prompts: int = CANONICAL_NUM_PROMPTS,
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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def redetect_prefix(source_n: int = 400, prefix_n: int = 256,
                    num_prompts: int = CANONICAL_NUM_PROMPTS,
                    t: int = 3, eta: float = 0.05,
                    fpr: float = 1e-3, experiment_seed: int = SEED,
                    generation_model_size: str = MODEL_SIZE,
                    kv_cache_implementation: str = (
                        DEFAULT_KV_CACHE_IMPLEMENTATION
                    )):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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


def proxy_8b_entrypoint(
    mode: str = "plan",
    approval_token: str = "",
    gpu: str = "A10G",
    max_containers: int = 5,
    detection_max_containers: int = 10,
):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


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


def generate_online(num_prompts: int = CANONICAL_NUM_PROMPTS,
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
    tag = online_config_tag(
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
    build = online_build_artifacts.remote(
        num_prompts, n, t, eta, experiment_seed, fresh,
        generation_model_size, kv_cache_implementation,
    )
    print(
        f"[main] artifact {'reused' if build['reused'] else 'built'}: "
        f"{build['artifact_fingerprint']}", flush=True,
    )
    plan = online_plan_generation.remote(
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

        model = OnlineGenerationModel.with_options(**model_cls_options(
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

    return {"generation_only": True, "tag": tag, "plan": plan}


# Redetection: one raw-completion trace pipeline for both constructions
# ----------------------------------------------------------------------------

EXECUTION_FILES = ("qwen.py", "detectors.py", "prc.py", "online_prc.py",
                   "modal_run.py",
                   "watermark_expt.py", "constants.py")

# Completion-only redetection shares this app's image, model loader and batching.
REDETECT_PROTOCOL = "completion_only_raw_abstain_v1"
REDETECT_CSV = "outputs/redetection/redetection_results_summary.csv"
REDETECT_CSV_COLUMNS = FIXED_CSV_COLUMNS[:5] + ["PRC Construction"] + FIXED_CSV_COLUMNS[5:9] + [
    "Old Posterior TPR", "Posterior TPR", "Old Entropy Aware TPR", "Entropy Aware TPR",
    "Naive TPR", "Posterior FPR", "Entropy FPR", "Naive FPR", "Entropy Trace Source", "Notes",
]
redetect_results = modal.Volume.from_name("prc-completion-only", create_if_missing=False)
redetect_archive = modal.Volume.from_name("prc-research-archive", create_if_missing=False)


def _redetect_sha(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _redetect_model_spec(spec):
    """Validate pinned single-file or sharded BF16 detector checkpoints."""
    size = spec["size"]
    if (size not in ("0.6B", "4B", "8B") or spec["id"] != f"Qwen/Qwen3-{size}-Base"
            or spec["dtype"] != "bfloat16" or spec["cache_directory"] != f"models/Qwen3-{size}-Base"
            or not re.fullmatch(r"[0-9a-f]{40}", spec["revision"])):
        raise ValueError("expected a pinned BF16 Qwen3-0.6B-Base, Qwen3-4B-Base or Qwen3-8B-Base detector")
    weights = ({"model.safetensors": spec["weights_sha256"]} if size == "0.6B" else spec["weight_files"])
    hashes = [spec["tokenizer_sha256"], *weights.values()]
    if size in ("4B", "8B"):
        hashes.append(spec["index_sha256"])
        if not weights or any(not re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", p) for p in weights):
            raise ValueError("invalid detector weight-shard names")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes):
        raise ValueError("detector files require SHA-256 checksums")
    return weights


def _verify_redetection_checkpoint(spec, cache_root="/cache"):
    from pathlib import Path
    weights = _redetect_model_spec(spec)
    cache = Path(cache_root)/spec["cache_directory"]
    files = {**weights, "tokenizer.json": spec["tokenizer_sha256"]}
    if spec["size"] in ("4B", "8B"):
        index = cache/"model.safetensors.index.json"
        files[index.name] = spec["index_sha256"]
        if set(json.loads(index.read_text())["weight_map"].values()) != set(weights):
            raise ValueError("detector index does not match the frozen weight shards")
    for name, digest in files.items():
        if _redetect_sha(cache/name) != digest:
            raise ValueError(f"checkpoint differs from frozen detector manifest: {name}")
    for name in weights:
        metadata = cache/".cache/huggingface/download"/(name+".metadata")
        if metadata.read_text().splitlines()[0] != spec["revision"]:
            raise ValueError("checkpoint revision differs from frozen detector manifest")


def _redetect_write(path, value):
    """Atomic per-batch writes; a failed worker cannot publish half a trace."""
    from pathlib import Path
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    if path.suffix == ".pt":
        torch.save(value, temporary)
    else:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def _redetect_load(path):
    import torch
    from galois._fields import _factory
    _numpy_pickle_compat()
    # Newer fixed-key caches pickle their field class through this factory.
    # Reconstruct the same field in the pinned galois 0.4.2 runtime.
    if not hasattr(_factory, "_reconstruct_field_class"):
        _factory._reconstruct_field_class = lambda args, kwargs: _factory.GF(*args, **kwargs)
    return torch.load(path, weights_only=False, map_location="cpu")


def _redetect_source(ref, roots):
    from pathlib import Path, PurePosixPath
    name = PurePosixPath(ref["path"])
    if name.is_absolute() or any(p in ("", ".", "..") for p in ref["path"].split("/")):
        raise ValueError("source path must stay within its volume")
    path = Path(roots[ref["volume"]])/name
    if path.stat().st_size != ref["bytes"] or _redetect_sha(path) != ref["sha256"]:
        raise ValueError(f"cached source changed: {path}")
    return _redetect_load(path)


def _prepare_redetection(case, model, execution, roots, destination):
    """Verify frozen sources on CPU and export only tokens and partition to GPU."""
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import torch
    from detectors import semantic_sha256, tensor_sha256
    if (case["construction"], case["fpr_policy"]) not in {
        ("fixed", "block_or_bonferroni"), ("online", "one_shot"), ("online", "alpha_spending_v1")
    }:
        raise ValueError("PRC construction and original FPR policy must agree")
    size, lengths = case["batch_size"], case["lengths"]
    if type(size) is not int or size < 1 or not lengths or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("batch size and prefix lengths must be positive integers")
    ids = [(r["source"], r["prompt_idx"]) for r in case["records"]]
    null_policy = case.get("null_policy", "evaluate")
    if null_policy not in ("evaluate", "not_evaluated"):
        raise ValueError("invalid null evaluation policy")
    sources = {"wm"} if null_policy == "not_evaluated" else {"wm", "null"}
    if len(set(ids)) != len(ids) or {s for s, _ in ids} != sources:
        raise ValueError("require unique candidates matching the null evaluation policy")
    if not 0 < case["fpr"] < 1 or not case["weights"] or any(w not in ("map", "entropy") for w in case["weights"]):
        raise ValueError("invalid detector weights or FPR")
    raw = _redetect_source(case["artifact"], roots)
    key = "online_key" if case["construction"] == "online" else "decoding_key"
    artifact = {"partition": raw["partition"], key: raw[key]}
    partition = artifact["partition"].to(torch.bfloat16)
    if (partition.ndim != 2 or partition.shape[0] != 2
            or not torch.all((partition == 0) | (partition == 1)) or not torch.all(partition.sum(0) == 1)):
        raise ValueError("invalid original partition")
    maximum = max(lengths)
    if maximum > 40960 or case["cache"] not in ("concat", "static"):
        raise ValueError("unsupported completion length or KV cache")
    def extract(ref):
        record = _redetect_source(ref["file"], roots)
        tokens = record["tokens"]
        if (record["prompt_idx"] != ref["prompt_idx"] or record["watermark"] != (ref["source"] == "wm")
                or tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64) or len(tokens) < maximum):
            raise ValueError("candidate label, tokens or length changed")
        tokens = tokens[:maximum].to(torch.int64).clone()
        digest = hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest()
        if digest != ref["tokens_sha256"] or torch.any(tokens < 0) or torch.any(tokens >= partition.shape[1]):
            raise ValueError("candidate tokens differ from frozen inputs")
        return tokens  # Original prompt and generation probabilities are never exported.
    with ThreadPoolExecutor(max_workers=8) as pool:
        tokens = list(pool.map(extract, case["records"]))
    run = {"protocol": REDETECT_PROTOCOL, "schema_version": 2, "case": case, "model": model, "execution": execution}
    root = Path(destination)/REDETECT_PROTOCOL/"integrated"/semantic_sha256(run)[:24]
    _redetect_write(root/"manifest.json", run)
    _redetect_write(root/"artifact.pt", artifact)
    batches = []
    for start in range(0, len(tokens), size):
        directory = root/"batches"/f"{start:06d}"
        inputs = {"tokens": torch.stack(tokens[start:start+size]), "partition": partition}
        identity = {"protocol": REDETECT_PROTOCOL, "run": root.name, "start": start,
                    "count": len(inputs["tokens"]), "length": maximum, "cache": case["cache"],
                    "input_sha256": semantic_sha256(inputs)}
        _redetect_write(directory/"inputs.pt", inputs)
        batches.append({"root": str(directory.relative_to(destination)), "identity": identity})
    return {"root": str(root.relative_to(destination)), "run": run, "batches": batches,
            "artifact_sha256": _redetect_sha(root/"artifact.pt"), "partition_sha256": tensor_sha256(partition)}


def _redetect_inputs(batch, destination):
    from pathlib import Path
    import torch
    from detectors import semantic_sha256
    value = _redetect_load(Path(destination)/batch["root"]/"inputs.pt")
    identity = batch["identity"]
    if (identity["protocol"] != REDETECT_PROTOCOL
            or set(value) != {"tokens", "partition"} or semantic_sha256(value) != identity["input_sha256"]
            or value["tokens"].shape != (identity["count"], identity["length"])
            or value["tokens"].dtype != torch.int64):
        raise ValueError("GPU inputs changed; only frozen completion tokens and partition are accepted")
    return value


def _redetect_trace(path, identity):
    import torch
    from detectors import tensor_sha256
    payload = _redetect_load(path)
    if "probabilities_2_to_T" not in payload or payload.get("identity") != identity:
        raise ValueError("cached trace is not a compatible completion-only trace")
    trace = payload["probabilities_2_to_T"]
    if (payload["identity"] != identity or identity["protocol"] != REDETECT_PROTOCOL
            or trace.shape != (identity["count"], identity["length"]-1) or trace.dtype != torch.float32
            or not torch.isfinite(trace).all() or torch.any((trace < 0) | (trace > 1))
            or tensor_sha256(trace) != payload["probabilities_sha256"]):
        raise ValueError("cached trace has incompatible identity, shape or probabilities")
    return payload


def _recover_redetection_batch(model, batch, destination, validate=False):
    from pathlib import Path
    import time
    import torch
    from detectors import tensor_sha256
    from qwen import completion_only_partition_trace_batch, make_kv_cache
    inputs = _redetect_inputs(batch, destination)
    path = Path(destination)/batch["root"]/"trace.pt"
    identity = batch["identity"]
    if path.exists():
        saved = _redetect_trace(path, identity)
        if validate and not saved["full_validation"]:
            raise ValueError("representative cached batch lacks full validation")
        return {"root": batch["root"], "cached": True}
    device = next(model.parameters()).device
    tokens, part = inputs["tokens"].to(device), inputs["partition"][1].to(device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trace = completion_only_partition_trace_batch(model, tokens, part, identity["cache"])
    if validate and tokens.shape[1] > 1:
        # One independent token-step replay per batch shape, before fanout.
        cache = make_kv_cache(identity["cache"], max_length=tokens.shape[1]-1)
        with torch.no_grad():
            reference = torch.stack([
                (model(tokens[:, i:i+1], cache=cache)[:, -1].softmax(-1)*part).sum(-1).cpu()
                for i in range(tokens.shape[1]-1)
            ], dim=1).float()
        if not torch.equal(trace, reference):
            raise ValueError("completion-only replay differs from independent reference")
        length = min(65, tokens.shape[1])
        reverse = completion_only_partition_trace_batch(model, tokens.flip(0)[:, :length], part, identity["cache"])
        if not torch.equal(reverse.flip(0), trace[:, :length-1]):
            raise ValueError("batch-order/prefix consistency failed")
    allocated = torch.cuda.max_memory_allocated() if cuda else 0
    reserved = torch.cuda.max_memory_reserved() if cuda else 0
    if cuda and allocated >= .85*torch.cuda.get_device_properties(device).total_memory:
        raise ValueError("batch exceeds the live GPU memory margin")
    payload = {"identity": identity, "probabilities_2_to_T": trace,
               "probabilities_sha256": tensor_sha256(trace), "full_validation": validate,
               "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved,
               "seconds": time.monotonic()-started}
    _redetect_write(path, payload)
    _redetect_trace(path, identity)
    return {"root": batch["root"], "cached": False, "seconds": payload["seconds"]}


def _score_redetection(prepared, destination):
    from pathlib import Path
    import math
    from detectors import detect_hoeffding, detect_online_hoeffding
    root = Path(destination)/prepared["root"]
    if _redetect_sha(root/"artifact.pt") != prepared["artifact_sha256"]:
        raise ValueError("original scoring artifact changed")
    artifact = _redetect_load(root/"artifact.pt")
    case = prepared["run"]["case"]
    records, hashes = [], {}
    for batch in prepared["batches"]:
        inputs = _redetect_inputs(batch, destination)
        path = Path(destination)/batch["root"]/"trace.pt"
        traces = _redetect_trace(path, batch["identity"])["probabilities_2_to_T"].numpy()
        hashes[batch["root"]] = _redetect_sha(path)
        for row, p in enumerate(traces):
            ref = case["records"][batch["identity"]["start"]+row]
            scores = {}
            for length in case["lengths"]:
                scores[str(length)] = {}
                for weight in case["weights"]:
                    common = dict(fpr=case["fpr"], weight=weight, return_info=True)
                    args = (inputs["tokens"][row, :length], p[:length-1], artifact["partition"])
                    if case["construction"] == "fixed":
                        decision, info = detect_hoeffding(artifact["decoding_key"], *args, **common)
                    else:
                        decision, info = detect_online_hoeffding(artifact["online_key"], *args, fpr_policy=case["fpr_policy"], **common)
                    info = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in info.items()}
                    scores[str(length)][weight] = {"decision": bool(decision), **info}
            records.append({k: ref[k] for k in ("source", "prompt_idx", "tokens_sha256")} | {"scores": scores})
    expected = [(r["source"], r["prompt_idx"]) for r in case["records"]]
    if [(r["source"], r["prompt_idx"]) for r in records] != expected:
        raise ValueError("incomplete or reordered candidate coverage")
    counts = {str(n): {w: {s: {"detected": sum(r["scores"][str(n)][w]["decision"] for r in records if r["source"] == s),
                              "count": sum(r["source"] == s for r in records)}
                          for s in ("wm", "null")} for w in case["weights"]} for n in case["lengths"]}
    key = artifact["online_key"] if case["construction"] == "online" else artifact["decoding_key"]
    settings = ({"eta": key["noise_rate"], "t": key["check_weight"],
                 "r_setting": f"causal round({key['row_rate_numerator']/key['row_rate_denominator']:g}L), startup-clamped"}
                if case["construction"] == "online" else
                {"eta": key[4], "t": key[-1], "n": key[1].shape[1], "r": key[1].shape[0],
                 "r_setting": f"{key[1].shape[0]}/{key[1].shape[1]}"})
    report = {"passed": True, "protocol": REDETECT_PROTOCOL, "counts": counts, "settings": settings,
              "trace_shard_sha256": hashes, "records": records}
    _redetect_write(root/"full.json", report)
    _redetect_write(root/"summary.json", {k: v for k, v in report.items() if k != "records"})
    return report


def _append_redetection_csv(prepared, report, csv_out):
    """Append completed results in the Hoeffding CSV format; reruns are idempotent."""
    import fcntl
    from pathlib import Path
    if not report.get("passed") or report.get("protocol") != REDETECT_PROTOCOL:
        raise ValueError("only completed raw-completion results can enter the CSV")
    run = prepared["run"]
    case, settings = run["case"], report["settings"]
    rows = []
    for length in case["lengths"]:
        count = report["counts"][str(length)]
        info = report["records"][0]["scores"][str(length)][case["weights"][0]]
        row = {"eta": settings["eta"], "T": length, "n": settings.get("n", length),
               "r value": settings.get("r", info.get("r")), "t": settings["t"],
               "r setting": settings["r_setting"],
               "PRC Construction": "online_causal_prc_v1" if case["construction"] == "online" else "fixed_prc",
               "Target FPR": f"{case['fpr']:g}", "Entropy Model": run["model"]["id"].split("/")[-1],
               "Generation Model": case["generation_model"], "Entropy Trace Source": REDETECT_PROTOCOL,
               "Notes": f"BF16; coordinate 1=0; FPR policy={case['fpr_policy']}; batch={case['batch_size']}; "
                        f"GPU={run['execution']['gpu']}; commit={run['execution']['git_commit']}; "
                        f"run={prepared['root']}; report=full.json"}
        if case.get("null_policy") == "not_evaluated":
            row["Notes"] += f"; N={sum(r['source'] == 'wm' for r in case['records'])}; null N=0; empirical FPR not evaluated"
        if run["execution"].get("source_hashes_authoritative"):
            row["Notes"] += "; working-tree execution; source hashes in manifest"
        for weight, tpr, fpr in [("map", "Posterior TPR", "Posterior FPR"), ("entropy", "Entropy Aware TPR", "Entropy FPR"),
                                  ("naive", "Naive TPR", "Naive FPR")]:
            for source, column in [("wm", tpr), ("null", fpr)]:
                row[column] = (_format_rate(count[weight][source]["detected"], count[weight][source]["count"])
                               if weight in count and count[weight][source]["count"] else "skipped")
        old = case.get("old_tpr")
        if old and old["detector_model"] != run["model"]["id"]:
            raise ValueError("old TPR belongs to a different detector model")
        for weight, column in [("map", "Old Posterior TPR"), ("entropy", "Old Entropy Aware TPR")]:
            previous = old["counts"].get(str(length), {}).get(weight) if old else None
            if previous and (previous["count"] != sum(ref["source"] == "wm" for ref in case["records"])
                             or not 0 <= previous["detected"] <= previous["count"]):
                raise ValueError("old and redetected TPR candidate counts differ")
            row[column] = _format_rate(previous["detected"], previous["count"]) if previous else "unavailable"
        if old:
            row["Notes"] += f"; old TPR source={old['source']}; old evidence SHA256={old['evidence_sha256']}"
        rows.append({k: str(row[k]) for k in REDETECT_CSV_COLUMNS})
    path = Path(csv_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        reader = csv.DictReader(handle)
        existing = list(reader)
        if reader.fieldnames and reader.fieldnames != REDETECT_CSV_COLUMNS:
            raise ValueError("redetection CSV columns differ from the Hoeffding results schema")
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=REDETECT_CSV_COLUMNS, lineterminator="\n")
        if not reader.fieldnames:
            writer.writeheader()
        writer.writerows(row for row in rows if row not in existing)
        handle.flush()
        os.fsync(handle.fileno())


@app.function(cpu=4, memory=8192, timeout=3600,
              volumes={"/data": data_vol, "/archive": redetect_archive, "/results": redetect_results})
def prepare_redetection(case, model, execution):
    redetect_results.reload()
    prepared = _prepare_redetection(case, model, execution, {"data": "/data", "archive": "/archive"}, "/results")
    redetect_results.commit()
    return prepared


@app.function(cpu=4, memory=8192, timeout=3600, volumes={"/results": redetect_results})
def finish_redetection(prepared):
    redetect_results.reload()
    result = _score_redetection(prepared, "/results")
    redetect_results.commit()
    return {"root": prepared["root"], "counts": result["counts"]}


@app.local_entrypoint()
def redetect(manifest: str, stage: str = "preflight", gpu: str = "A100-80GB", max_containers: int = 10,
             csv_out: str = REDETECT_CSV):
    """Redetect frozen completions; no text generation or original prompt input."""
    from pathlib import Path
    import subprocess
    from detectors import semantic_sha256
    if stage not in ("preflight", "smoke", "full") or not 1 <= max_containers <= 10:
        raise ValueError("choose preflight/smoke/full and 1..10 workers")
    # Modal validates GPU names; batching is validated on the selected hardware.
    content = json.loads(Path(manifest).read_text())
    if content["protocol"] != REDETECT_PROTOCOL or content["schema_version"] != 1 or not content["cases"]:
        raise ValueError("expected a frozen raw-completion manifest")
    spec = content["model"]
    _redetect_model_spec(spec)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    files = EXECUTION_FILES
    for name in files:
        if subprocess.check_output(["git", "show", f"{commit}:{name}"]) != Path(name).read_bytes():
            raise ValueError(f"commit execution code before running: {name}")
    execution = {"git_commit": commit, "files": {p: _redetect_sha(p) for p in files},
                 "gpu": gpu, "allocator": "expandable_segments:True"}
    local = Path("outputs/redetection/.archive/runs")/semantic_sha256({"manifest": content, "execution": execution})[:24]
    for case in content["cases"]:
        prepared = prepare_redetection.remote(case, spec, execution)
        _redetect_write(local/case["id"]/"prepared.json", prepared)
        print(f"Verified {len(case['records'])} candidates, batch {case['batch_size']}, lengths {case['lengths']}")
        if stage == "preflight":
            continue
        worker = RedetectionModel.with_options(
            **{**model_cls_options(spec["size"], gpu, max_containers),
               "memory": 65536 if spec["size"] == "8B" else 8192, "scaledown_window": 2},
        )(entropy_model_size=spec["size"], generation_model_size=case["generation_model"].removeprefix("Qwen3-").removesuffix("-Base"),
          trace_kv_cache_implementation=case["cache"], completion_model=json.dumps(spec, sort_keys=True))
        # Validate one representative for each actual shape, including a short tail.
        representatives = {}
        for batch in prepared["batches"]:
            representatives.setdefault(batch["identity"]["count"], batch)
        for batch in representatives.values():
            worker.redetect_batch.remote(batch, validate=True)
        if stage == "smoke":
            continue
        for done in worker.redetect_batch.map(prepared["batches"]):
            print(json.dumps(done), flush=True)
        result = finish_redetection.remote(prepared)
        for name in ("full.json", "summary.json"):
            (local/case["id"]/name).write_bytes(b"".join(redetect_results.read_file(result["root"]+"/"+name)))
        _append_redetection_csv(prepared, json.loads((local/case["id"]/"full.json").read_text()), csv_out)
        print(json.dumps(result), flush=True)


@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache, "/results": redetect_results},
    timeout=7200,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class RedetectionModel:
    """Teacher-force cached tokens with an alternate, smaller LM."""
    entropy_model_size: str = modal.parameter()
    generation_model_size: str = modal.parameter()
    trace_kv_cache_implementation: str = modal.parameter()
    completion_model: str = modal.parameter(default="")

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
        if not self.completion_model and self.trace_kv_cache_implementation != "static":
            raise ValueError(
                "cross-model entropy replay requires the optimized static "
                "KV cache"
            )
        os.environ["PRC_MODEL_SIZE"] = self.entropy_model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"
        if self.completion_model:
            spec = json.loads(self.completion_model)
            if spec["size"] != self.entropy_model_size:
                raise ValueError("detector model and frozen manifest disagree")
            _verify_redetection_checkpoint(spec)
        we = load_watermark_model(self.entropy_model_size)
        self.we = we
        if self.completion_model:
            import torch
            torch.set_num_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            if next(we.model.parameters()).dtype != torch.bfloat16:
                raise ValueError("completion-only replay requires BF16")
            we.model.eval().requires_grad_(False)
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
    def redetect_batch(self, batch: dict, validate: bool = False) -> dict:
        if not self.completion_model:
            raise ValueError("use a pinned completion-only model configuration")
        redetect_results.reload()
        result = _recover_redetection_batch(self.we.model, batch, "/results", validate)
        redetect_results.commit()
        return result

# Public commands; generation and redetection remain explicitly separate.
generate_fixed = app.local_entrypoint()(generate_fixed)
generate_online = app.local_entrypoint()(generate_online)
generate_replicate = app.local_entrypoint()(generate_replicate)
aggregate_shards = app.local_entrypoint()(aggregate_shards)
build_null_cache = app.local_entrypoint()(build_null_cache)
compare_kv_caches = app.local_entrypoint()(compare_kv_caches)
validate_kv_cache_runtime_smoke = app.local_entrypoint()(validate_kv_cache_runtime_smoke)
proxy_8b_quality = app.local_entrypoint(name="proxy-8b-quality")(proxy_8b_quality_entrypoint)
