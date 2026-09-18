"""
Run the PRC watermark experiment on Modal with the Hoeffding detector.

Cost-optimized pipeline:
  - build_artifacts (CPU): freeze the PRC key, seeded partition map, and
    tokenized RealNews prompts. Written to a Volume per config.
  - Model.generate_wm / generate_null (GPU): batched generation, one batch per
    .map() call.
  - Optional EntropyModel pass (GPU): teacher-force cached generations with a
    different Qwen3 model to estimate P[partition 1] for detection.
  - detect_all (CPU): Hoeffding detection over cached generations/traces.

Null generations depend only on the model, prompt, and sampling setup, not on
the key (n, t, eta, r). They live in shared stores keyed by generated length;
any complete longer store can serve a shorter requested prefix.
Current eta=0.1 runs use T=n: one generated length-n code block per prompt.

Usage:
    modal run modal_run.py
    modal run modal_run.py --num-prompts 500 --max-containers 10
    modal run modal_run.py --num-prompts 500 --n 512 --t 3 --eta 0.1 \
      --r-frac 0.99 --fpr 1e-3 --entropy-model-size 4B
    modal run modal_run.py --num-prompts 500 --max-containers 10 \
      --n 768 --t 3 --eta 0.1 --r-frac 0.99 --fpr 1e-3 \
      --generation-model-size 8B --gpu H100 --batch 100

Generation-model caches are isolated. The historical Qwen3-0.6B-Base paths
remain unchanged, while other models use explicit model-qualified directories.
"""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone

import modal

RETIRED_DETECTION_MESSAGE = (
    "Prompt-dependent detection has been retired. Use modal_online_run.py::redetect "
    "with a frozen raw-completion manifest. Generation-only commands remain available."
)

# ---- experiment config ------------------------------------------------------
DEFAULT_N = 400
DEFAULT_T = 3
DEFAULT_ETA = 0.05
DEFAULT_FPR = 1e-3  # 0.1%
DEFAULT_BLOCKS = 1
SEED = 12345
MODEL_SIZE = "0.6B"
VOCAB = 151_936
GPU = "A10G"
DEFAULT_MAX_CONTAINERS = 5
DEFAULT_BATCH = 64
DEFAULT_ENTROPY_BATCH = 8
REQUIRED_R_FRAC = 0.99
REQUIRED_R_SETTING = "0.99n"
CANONICAL_NUM_PROMPTS = 500
SHARD_RESULT_SCHEMA_VERSION = 1
DETECTION_CHECKPOINT_SCHEMA_VERSION = 1
DETECTION_CHECKPOINT_COMMIT_INTERVAL = 10
DERIVED_TRACE_SCHEMA_VERSION = 1

CSV_COLUMNS = [
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


def normalize_model_size(model_size):
    value = MODEL_SIZE if model_size is None else str(model_size).strip()
    if not value:
        value = MODEL_SIZE
    upper = value.upper()
    if upper.endswith("B"):
        return upper[:-1] + "B"
    if upper.replace(".", "", 1).isdigit():
        return f"{upper}B"
    return value


def model_display(model_size):
    return f"Qwen3-{normalize_model_size(model_size)}-Base"


def entropy_model_tag(model_size):
    size = normalize_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def uses_cached_generation_trace(entropy_model_size,
                                 generation_model_size=MODEL_SIZE):
    """Whether detection can reuse probabilities recorded during generation."""
    return normalize_model_size(entropy_model_size) == normalize_model_size(
        generation_model_size
    )


def entropy_trace_source(entropy_model_size,
                         generation_model_size=MODEL_SIZE):
    if uses_cached_generation_trace(
            entropy_model_size, generation_model_size):
        return "cached_generation_p_trace"
    return f"estimated_{normalize_model_size(entropy_model_size)}"


def resolve_r(n, r=0, r_frac=0.0):
    explicit_r = int(r) if r else 0
    explicit_frac = float(r_frac) if r_frac else 0.0
    if explicit_r and explicit_frac:
        raise ValueError("Pass either --r or --r-frac, not both.")
    if explicit_r:
        return explicit_r
    if explicit_frac:
        return int(round(explicit_frac * n))
    return None


def resolve_new_run_r(n, r=0, r_frac=REQUIRED_R_FRAC):
    """Enforce the project-wide r=round(0.99n) policy for new runs."""
    expected_r = int(round(REQUIRED_R_FRAC * n))
    if r and int(r) != expected_r:
        raise ValueError(
            f"new runs require r=round(0.99n)={expected_r} for n={n}; "
            f"got explicit r={r}"
        )
    if r_frac and abs(float(r_frac) - REQUIRED_R_FRAC) > 1e-12:
        raise ValueError(
            f"new runs require --r-frac {REQUIRED_R_FRAC}; got {r_frac}"
        )
    return expected_r


def validate_r_for_keygen(n, t, r):
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


def experiment_T(n):
    """Generated-token length for new runs: one length-n PRC code block."""
    return DEFAULT_BLOCKS * int(n)


def _generation_scoped_root(root, generation_model_size=MODEL_SIZE):
    """Keep legacy 0.6B paths while isolating every other generation model."""
    model_size = normalize_model_size(generation_model_size)
    if model_size == normalize_model_size(MODEL_SIZE):
        return root
    return f"{root}/{entropy_model_tag(model_size)}"


def config_tag(n, t, eta, r=None, T=None,
               generation_model_size=MODEL_SIZE):
    """Per-config tag for key-dependent artifacts.

    FPR is excluded because it only affects detection. r is included only when
    explicitly requested so old default-r caches keep their original tags.
    T is included for new runs so T=n caches cannot collide with old T=2n caches.
    """
    base = f"n{n}_t{t}_eta{eta:.2f}"
    if T is not None:
        base = f"{base}_T{int(T)}"
    if (normalize_model_size(generation_model_size)
            != normalize_model_size(MODEL_SIZE)):
        base = f"{base}__gen-{entropy_model_tag(generation_model_size)}"
    return f"{base}_r{int(r)}" if r is not None else base


def art_path(n, t, eta, r=None, T=None,
             generation_model_size=MODEL_SIZE):
    tag = config_tag(n, t, eta, r, T, generation_model_size)
    return f"/data/{tag}/artifacts.pt"


def wm_dir(n, t, eta, r=None, T=None,
           generation_model_size=MODEL_SIZE):
    tag = config_tag(n, t, eta, r, T, generation_model_size)
    return f"/data/{tag}/wm"


def null_root(generation_model_size=MODEL_SIZE):
    return _generation_scoped_root("/data/_nulls", generation_model_size)


def null_dir(T, generation_model_size=MODEL_SIZE):
    return f"{null_root(generation_model_size)}/T{T}"


def wm_entropy_dir(tag, entropy_model_size):
    return f"/data/{tag}/entropy/{entropy_model_tag(entropy_model_size)}/wm"


def null_entropy_dir(T, entropy_model_size,
                     generation_model_size=MODEL_SIZE):
    root = _generation_scoped_root(
        "/data/_null_entropy", generation_model_size
    )
    return f"{root}/{entropy_model_tag(entropy_model_size)}/T{T}"


def wm_trace_dir(tag, entropy_model_size):
    return f"/data/{tag}/detect_traces/{entropy_model_tag(entropy_model_size)}/wm"


def null_trace_dir(T, entropy_model_size,
                   generation_model_size=MODEL_SIZE):
    root = _generation_scoped_root(
        "/data/_null_detection_traces", generation_model_size
    )
    return f"{root}/{entropy_model_tag(entropy_model_size)}/T{T}"


def detection_checkpoint_dir(tag, entropy_model_size, fpr):
    """Config-local detector records, separated by model and target FPR."""
    fpr_tag = _slug(f"{float(fpr):.12g}")
    return (
        f"/data/{tag}/detection_checkpoints/"
        f"{entropy_model_tag(entropy_model_size)}/fpr-{fpr_tag}"
    )


def validate_generation_record(record, generation_model_size,
                               source="generation", idx="?"):
    """Reject cache records from another model or unlabelled non-legacy data."""
    expected_size = normalize_model_size(generation_model_size)
    stored_size = record.get("generation_model_size")
    stored_display = record.get("generation_model")
    if stored_size is None:
        if expected_size != normalize_model_size(MODEL_SIZE):
            raise ValueError(
                f"{source} cache index {idx} lacks generation-model metadata; "
                f"refusing to treat it as {model_display(expected_size)}"
            )
    elif normalize_model_size(stored_size) != expected_size:
        raise ValueError(
            f"{source} cache index {idx} was generated by "
            f"{model_display(stored_size)}, expected "
            f"{model_display(expected_size)}"
        )
    if (stored_display is not None
            and str(stored_display).strip() != model_display(expected_size)):
        raise ValueError(
            f"{source} cache index {idx} has generation model label "
            f"{stored_display!r}, expected {model_display(expected_size)!r}"
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


def find_complete_cache_T(root, min_T, prompt_indices_or_count, prefix):
    """Return the smallest complete T' >= min_T cache, or None.

    Cache directories are named T{length} and contain one {prefix}_XXXX.pt
    record per prompt.  A longer causal generation/trace can be truncated to
    any requested prefix length, so exact-length caches are not required. The
    third argument may be the legacy prompt count or an exact index iterable.
    """
    if not os.path.isdir(root):
        return None

    prompt_indices = _coerce_prompt_indices(prompt_indices_or_count)

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


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _format_rate(count, total):
    denom = max(total, 1)
    return f"{count}/{total} ({count / denom:.1%})"


def prompt_indices_for_shard(prompt_start, num_prompts,
                             total_prompts=CANONICAL_NUM_PROMPTS):
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


def _coerce_prompt_indices(prompt_indices_or_count):
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


def _json_safe(value):
    """Convert numpy/torch scalar containers to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _atomic_write_json(path, payload):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "w") as f:
        json.dump(_json_safe(payload), f, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(temporary, path)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _canonical_json_sha256(value):
    encoded = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _detection_checkpoint_identity(config, artifact_fingerprint,
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
        "config": _json_safe(config),
        "artifact_fingerprint": str(artifact_fingerprint),
        "detector_implementation_sha256": str(detector_sha256),
        "source": str(source),
        "prompt_idx": int(prompt_idx),
        "tokens_sha256": str(tokens_sha256),
        "p_trace_sha256": str(p_trace_sha256),
    }


def _save_detection_checkpoint(path, identity, record):
    safe_record = _json_safe(record)
    payload = {
        "schema_version": DETECTION_CHECKPOINT_SCHEMA_VERSION,
        "identity": _json_safe(identity),
        "identity_sha256": _canonical_json_sha256(identity),
        "record_sha256": _canonical_json_sha256(safe_record),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "record": safe_record,
    }
    _atomic_write_json(path, payload)


def _load_detection_checkpoint(path, expected_identity):
    """Return a verified detector record, or None for stale/corrupt data."""
    try:
        with open(path) as f:
            payload = json.load(f)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if payload.get("schema_version") != DETECTION_CHECKPOINT_SCHEMA_VERSION:
        return None
    identity = payload.get("identity")
    if identity != _json_safe(expected_identity):
        return None
    identity_sha256 = _canonical_json_sha256(identity)
    if payload.get("identity_sha256") != identity_sha256:
        return None
    record = payload.get("record")
    if not isinstance(record, dict):
        return None
    if payload.get("record_sha256") != _canonical_json_sha256(record):
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


def shard_result_filename(tag, entropy_model_size, fpr, prompt_indices,
                          workspace_label="workspace"):
    indices = _coerce_prompt_indices(prompt_indices)
    if not indices:
        raise ValueError("cannot name an empty shard")
    return (
        f"{_slug(tag)}__{entropy_model_tag(entropy_model_size)}__"
        f"fpr-{_slug(f'{float(fpr):.12g}')}__"
        f"p{min(indices):04d}-{max(indices):04d}__"
        f"{_slug(workspace_label)}.json"
    )


def _local_code_fingerprint():
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


def _semantic_fingerprint(value):
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


def _ensure_csv_schema(csv_out):
    parent = os.path.dirname(csv_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(csv_out) or os.path.getsize(csv_out) == 0:
        with open(csv_out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        return

    with open(csv_out, newline="") as f:
        reader = csv.DictReader(f)
        old_columns = reader.fieldnames or []
        rows = list(reader)
    if old_columns == CSV_COLUMNS:
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
            "Entropy Model": row.get("Entropy Model", model_display(MODEL_SIZE)),
            "Generation Model": row.get(
                "Generation Model", model_display(MODEL_SIZE)
            ),
            "Entropy Trace Source": row.get(
                "Entropy Trace Source", "cached_generation_p_trace"
            ),
            "Notes": row.get("Notes", ""),
        })
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def _append_summary_row(csv_out, row):
    _ensure_csv_schema(csv_out)
    row = dict(row)
    row.setdefault("Generation Model", model_display(MODEL_SIZE))
    with open(csv_out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def _summary_row_identity(row):
    return (
        float(row["eta"]),
        int(row["T"]),
        int(row["n"]),
        int(row["r value"]),
        str(row["r setting"]).strip(),
        int(row["t"]),
        float(row["Target FPR"]),
        str(row.get("Generation Model", model_display(MODEL_SIZE))).strip(),
        str(row["Entropy Model"]).strip(),
    )


def _summary_row_exists(csv_out, candidate):
    if not os.path.exists(csv_out) or os.path.getsize(csv_out) == 0:
        return False
    candidate_identity = _summary_row_identity(candidate)
    with open(csv_out, newline="") as f:
        for row in csv.DictReader(f):
            try:
                if _summary_row_identity(row) == candidate_identity:
                    return True
            except (KeyError, TypeError, ValueError):
                continue
    return False


# ---- image ------------------------------------------------------------------
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
        "datasets",
        "aiohttp",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "PRC_MODEL_CACHE_DIR": "/cache/models",
            "PRC_MODEL_SIZE": MODEL_SIZE,
            "PRC_MODEL_VARIANT": "base",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "qwen", "constants", "detectors", "watermark_expt", "benchmarks"
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)

app = modal.App("prc-hoeffding", image=image)


# ---- artifact build (CPU only; no model load) -------------------------------
@app.function(volumes={"/data": data_vol}, timeout=600)
def build_artifacts(num_prompts: int, n: int, t: int, eta: float,
                    r: int = 0, fresh: bool = False,
                    generation_model_size: str = MODEL_SIZE) -> int:
    import json
    import os
    import shutil

    import numpy as np
    import torch
    from prc import KeyGen, parity_check_rank_info

    requested_r = resolve_new_run_r(n, r, 0.0)
    generation_model_size = normalize_model_size(generation_model_size)
    validate_r_for_keygen(n, t, requested_r)
    max_new_tokens = experiment_T(n)
    ap = art_path(
        n, t, eta, requested_r, max_new_tokens, generation_model_size
    )
    wmd = wm_dir(
        n, t, eta, requested_r, max_new_tokens, generation_model_size
    )
    os.makedirs(os.path.dirname(ap), exist_ok=True)

    config_sig = {
        "n": n,
        "t": t,
        "eta": eta,
        "T": max_new_tokens,
        "blocks": DEFAULT_BLOCKS,
        "num_prompts": num_prompts,
        "gen_scheme": "single_codeword_batched",
        "generation_model_size": generation_model_size,
        "generation_model": model_display(generation_model_size),
        "keygen_seed": SEED,
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
                prev["artifact_fingerprint"] = _semantic_fingerprint(
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

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=t,
        noise_rate=eta,
        r=requested_r,
        seed=SEED,
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t_key = decoding_key
    rank_info = parity_check_rank_info(parity_check_matrix)
    actual_r = parity_check_matrix.shape[0]
    print(f"[build] PRC params: n={n} t={t_key} g={g} r={actual_r} "
          f"noise_rate={noise_rate:.4f}", flush=True)
    print(f"[build] parity rank: {rank_info['rank']}/{rank_info['rows']} "
          f"full_rank={rank_info['full_rank']}", flush=True)

    perm = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(SEED))
    v0 = torch.zeros(VOCAB, dtype=torch.bfloat16)
    v0[perm[: VOCAB // 2]] = 1.0
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
        "generation_model": model_display(generation_model_size),
        "seed": SEED,
        "config_sig": config_sig,
        "parity_check_rank_info": rank_info,
    }
    artifact["artifact_fingerprint"] = _semantic_fingerprint({
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
@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(n: int, t: int, eta: float, prompt_indices: list,
                    r: int = 0,
                    generation_model_size: str = MODEL_SIZE) -> dict:
    requested_r = int(r) if r else None
    generation_model_size = normalize_model_size(generation_model_size)
    T = experiment_T(n)
    wmd = wm_dir(
        n, t, eta, requested_r, T, generation_model_size
    )
    prompt_indices = _coerce_prompt_indices(prompt_indices)
    data_vol.reload()

    wm_missing = [i for i in prompt_indices
                  if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]

    model_null_root = null_root(generation_model_size)
    null_T = find_complete_cache_T(
        model_null_root, T, prompt_indices, "null"
    )
    if null_T is None:
        null_T = T
        d = null_dir(T, generation_model_size)
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
        "generation_model": model_display(generation_model_size),
        "null_root": model_null_root,
    }


@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_entropy(tag: str, entropy_model_size: str, T: int, null_T: int,
                 prompt_indices: list,
                 generation_model_size: str = MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- batched generation (GPU) -----------------------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class Model:
    tag: str = modal.parameter()
    model_size: str = modal.parameter()
    code_fingerprint_sha256: str = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        import torch

        self.model_size = normalize_model_size(self.model_size)
        os.environ["PRC_MODEL_SIZE"] = self.model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"
        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")
        artifact_model_size = normalize_model_size(
            art.get("generation_model_size", MODEL_SIZE)
        )
        if artifact_model_size != self.model_size:
            raise ValueError(
                f"artifact generation model {artifact_model_size} does not "
                f"match requested model {self.model_size}"
            )

        import watermark_expt as we
        self.we = we
        self.encoding_key = art["encoding_key"]
        self.prompts = art["prompt_ids_list"]
        self.n = art["n"]
        self.T = int(art.get("T", art.get("config_sig", {}).get("T", experiment_T(art["n"]))))
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
            "generation_model": model_display(self.model_size),
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
                "generation_model": model_display(self.model_size),
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

        nd = null_dir(self.T, self.model_size)
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
                "generation_model": model_display(self.model_size),
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
def EntropyModel(*args, **kwargs):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- detection over cached generations (CPU) --------------------------------
@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(n: int, t: int, eta: float, fpr: float, null_T: int,
               prompt_indices: list, r: int = 0,
               entropy_model_size: str = MODEL_SIZE,
               null_entropy_T: int = 0,
               run_metadata: dict = None,
               generation_model_size: str = MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all_any(n: int, t: int, eta: float, fpr: float,
                   num_prompts: int = 500, r: int = 0,
                   entropy_model_size: str = MODEL_SIZE,
                   generation_model_size: str = MODEL_SIZE) -> list:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_map_summary(n: int, t: int, eta: float, fpr: float,
                       num_prompts: int = 500, r: int = 0,
                       entropy_model_size: str = MODEL_SIZE,
                       generation_model_size: str = MODEL_SIZE) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_legacy_first_block_summary(n: int, t: int, eta: float, fpr: float,
                                      num_prompts: int = 500, r: int = 0,
                                      entropy_model_size: str = MODEL_SIZE,
                                      legacy_token_length: int = 0) -> dict:
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- driver -----------------------------------------------------------------
@app.local_entrypoint()
def main(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         prompt_start: int = 0,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False,
         batch: int = DEFAULT_BATCH,
         entropy_batch: int = DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = REQUIRED_R_FRAC,
         generation_model_size: str = MODEL_SIZE,
         entropy_model_size: str = "",
         gpu: str = GPU,
         csv_out: str = "hoeffding_results_summary.csv",
         shard_out: str = "", workspace_label: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


def _aggregate_shard_payloads(payloads,
                              expected_num_prompts=CANONICAL_NUM_PROMPTS):
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
        if payload.get("schema_version") != SHARD_RESULT_SCHEMA_VERSION:
            raise ValueError(f"shard {shard_number} has unsupported schema")
        if payload.get("config") != reference_config:
            raise ValueError(f"shard {shard_number} configuration mismatch")
        if payload.get("artifact_fingerprint") != reference_artifact:
            raise ValueError(f"shard {shard_number} artifact/key mismatch")
        if payload.get("code_fingerprint") != reference_code:
            raise ValueError(f"shard {shard_number} code mismatch")
        if payload.get("parity_check_rank_info", {}) != reference_rank:
            raise ValueError(f"shard {shard_number} parity-rank mismatch")

        indices = _coerce_prompt_indices(payload.get("prompt_indices", []))
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
            _json_safe(records), sort_keys=True, separators=(",", ":")
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
        "Map TPR": _format_rate(map_tp, nwm),
        "Entropy Aware TPR": _format_rate(entropy_tp, nwm),
        "Naive TPR": _format_rate(naive_tp, nwm) if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _format_rate(map_fp, nnw),
        "Entropy FPR": _format_rate(entropy_fp, nnw),
        "Naive FPR": _format_rate(naive_fp, nnw) if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Trace Source": config["entropy_trace_source"],
        "Notes": notes,
    }
    aggregation = {
        "schema_version": SHARD_RESULT_SCHEMA_VERSION,
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


@app.local_entrypoint()
def aggregate_shards(shard_files: str,
                     csv_out: str = "hoeffding_results_summary.csv",
                     aggregate_out: str = "",
                     expected_num_prompts: int = CANONICAL_NUM_PROMPTS):
    """Validate local shard JSON files and append one authoritative CSV row."""
    paths = [path.strip() for path in shard_files.split(",") if path.strip()]
    if not paths:
        raise ValueError("--shard-files must contain comma-separated JSON paths")
    payloads = []
    for path in paths:
        with open(path) as f:
            payloads.append(json.load(f))
    row, aggregation = _aggregate_shard_payloads(
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
            f"eta{_slug(config['eta'])}_n{config['n']}_T{config['T']}_"
            f"r{config['r_value']}_"
            f"gen-{entropy_model_tag(generation_size)}_"
            f"{entropy_model_tag(entropy_size)}_"
            f"fpr-{_slug(config['target_fpr'])}.json"
        )
        aggregate_out = os.path.join("outputs", "aggregates", aggregate_name)
    _atomic_write_json(aggregate_out, aggregation)
    if _summary_row_exists(csv_out, row):
        raise ValueError(
            "the authoritative CSV already contains this experiment identity; "
            "not appending a duplicate row"
        )
    _append_summary_row(csv_out, row)
    print(f"[aggregate] validated {len(paths)} shards with "
          f"{expected_num_prompts} watermarked + {expected_num_prompts} null",
          flush=True)
    print(f"[aggregate] wrote audit manifest -> {aggregate_out}", flush=True)
    print(f"[aggregate] appended one summary row -> {csv_out}", flush=True)


@app.local_entrypoint()
def legacy_first_block_redetect(n: int = DEFAULT_N, t: int = DEFAULT_T,
                                eta: float = DEFAULT_ETA,
                                fpr: float = DEFAULT_FPR,
                                num_prompts: int = 500,
                                r: int = 0, r_frac: float = 0.0,
                                entropy_model_size: str = MODEL_SIZE,
                                legacy_token_length: int = 0,
                                csv_out: str = "hoeffding_results_summary.csv"):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- re-detection sweep over all weight kinds (CPU-only, no regeneration) ---
@app.local_entrypoint()
def redetect(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
             fpr: float = DEFAULT_FPR, num_prompts: int = 500,
             r: int = 0, r_frac: float = 0.0,
             generation_model_size: str = MODEL_SIZE,
             entropy_model_size: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.local_entrypoint()
def redetect_map(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
                 fpr: float = DEFAULT_FPR, num_prompts: int = 500,
                 r: int = 0, r_frac: float = 0.0,
                 generation_model_size: str = MODEL_SIZE,
                 entropy_model_size: str = ""):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


# ---- re-detect a whole set of configs in one Modal session ------------------
# (n, t, eta) for every config we've generated so far. Both cache layouts.
REDETECT_CONFIGS = [
    (256, 3, 0.05), (400, 3, 0.05), (512, 3, 0.05), (1024, 3, 0.05),
    (400, 3, 0.20),
    (400, 5, 0.05), (512, 5, 0.05), (1024, 5, 0.05), (2048, 5, 0.05),
]


@app.local_entrypoint()
def redetect_all(fpr: float = 1e-3, num_prompts: int = 500):
    raise RuntimeError(RETIRED_DETECTION_MESSAGE)


@app.local_entrypoint()
def generate_fixed(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         prompt_start: int = 0,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False,
         batch: int = DEFAULT_BATCH,
         entropy_batch: int = DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = REQUIRED_R_FRAC,
         generation_model_size: str = MODEL_SIZE,
         entropy_model_size: str = "",
         gpu: str = GPU,
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
    prompt_indices = prompt_indices_for_shard(prompt_start, num_prompts)
    resolved_r = resolve_new_run_r(n, r, r_frac)
    validate_r_for_keygen(n, t, resolved_r)
    generation_model_size = normalize_model_size(generation_model_size)
    entropy_model_size = normalize_model_size(
        entropy_model_size or generation_model_size
    )
    T = experiment_T(n)
    tag = config_tag(
        n, t, eta, resolved_r, T, generation_model_size
    )
    workspace_label = (
        workspace_label.strip() if workspace_label else
        os.environ.get("MODAL_PROFILE", "workspace")
    )
    code_fingerprint = _local_code_fingerprint()
    is_complete_run = prompt_indices == list(range(CANONICAL_NUM_PROMPTS))
    r_text = f"r={resolved_r} ({REQUIRED_R_SETTING})"
    print(f"[main] config {tag}  FPR_target={fpr:g}  ({num_prompts} prompts, "
          f"global range={prompt_indices[0]}..{prompt_indices[-1]}, "
          f"batch={batch}, entropy_batch={entropy_batch}, {r_text}, "
          f"generation_model={model_display(generation_model_size)}, "
          f"entropy_model={model_display(entropy_model_size)}, fresh={fresh}) ...",
          flush=True)
    print(f"[main] GPU={gpu} max_containers={max_containers}", flush=True)

    build_artifacts.remote(
        CANONICAL_NUM_PROMPTS, n, t, eta, resolved_r or 0, fresh,
        generation_model_size,
    )

    plan = plan_generation.remote(
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

        model = Model.with_options(
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
