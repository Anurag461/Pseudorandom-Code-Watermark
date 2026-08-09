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
      --n 768 --t 3 --eta 0.1 --r 760 --fpr 1e-3 \
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
CANONICAL_NUM_PROMPTS = 500
SHARD_RESULT_SCHEMA_VERSION = 1
DETECTION_CHECKPOINT_SCHEMA_VERSION = 1
DETECTION_CHECKPOINT_COMMIT_INTERVAL = 10

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
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "PRC_MODEL_SIZE": MODEL_SIZE,
            "PRC_MODEL_VARIANT": "base",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "qwen", "constants", "detectors", "watermark_expt"
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

    requested_r = int(r) if r else None
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
    prompt_indices = _coerce_prompt_indices(prompt_indices)
    generation_model_size = normalize_model_size(generation_model_size)
    if uses_cached_generation_trace(
            entropy_model_size, generation_model_size):
        return {"wm_missing": [], "null_missing": [],
                "T": T, "null_T": null_T, "null_entropy_T": null_T}

    data_vol.reload()
    wdir = wm_entropy_dir(tag, entropy_model_size)
    wm_missing = [i for i in prompt_indices
                  if not os.path.exists(os.path.join(wdir, f"wm_{i:04d}.pt"))]

    null_entropy_root = os.path.dirname(null_entropy_dir(
        T, entropy_model_size, generation_model_size
    ))
    null_entropy_T = find_complete_cache_T(
        null_entropy_root, T, prompt_indices, "null"
    )
    if null_entropy_T is None:
        null_entropy_T = T
        ndir = null_entropy_dir(
            T, entropy_model_size, generation_model_size
        )
        null_missing = [
            i for i in prompt_indices
            if not os.path.exists(os.path.join(ndir, f"null_{i:04d}.pt"))
        ]
    else:
        null_missing = []

    return {"wm_missing": wm_missing, "null_missing": null_missing,
            "T": T, "null_T": null_T,
            "null_entropy_T": null_entropy_T}


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
        we.partition = art["partition"].to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch

        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

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
        tokens, p_traces = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=True,
        )
        for row, i in enumerate(todo):
            torch.save(
                {"prompt_idx": i, "watermark": True,
                 "generation_model": model_display(self.model_size),
                 "generation_model_size": self.model_size,
                 "tokens": tokens[row].cpu(), "p_trace": p_traces[row]},
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
        tokens, p_traces = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=False,
        )
        for row, i in enumerate(todo):
            torch.save(
                {"prompt_idx": i, "watermark": False,
                 "generation_model": model_display(self.model_size),
                 "generation_model_size": self.model_size,
                 "tokens": tokens[row].cpu(), "p_trace": p_traces[row]},
                os.path.join(nd, f"null_{i:04d}.pt"),
            )
        data_vol.commit()
        return {"generated": len(todo), "cached": len(prompt_indices) - len(todo),
                "dt": time.time() - t0}


# ---- optional alternate entropy model (GPU) ---------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=7200,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class EntropyModel:
    tag: str = modal.parameter()
    model_size: str = modal.parameter()
    generation_model_size: str = modal.parameter()
    T: int = modal.parameter()
    null_T: int = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        import torch

        self.model_size = normalize_model_size(self.model_size)
        self.generation_model_size = normalize_model_size(
            self.generation_model_size
        )
        os.environ["PRC_MODEL_SIZE"] = self.model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"

        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")

        import watermark_expt as we
        self.we = we
        self.prompts = art["prompt_ids_list"]
        self.partition_cpu = art["partition"]
        self.partition = art["partition"].to(we.device)
        self.n = art["n"]
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch

        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

    def _save_trace_payload(self, path, idx, source, tokens, p_trace):
        import numpy as np
        import torch
        from detectors import binary_entropy, fold_soft_token, tokens_to_bits

        bits = tokens_to_bits(tokens, self.partition_cpu)
        entropy = binary_entropy(p_trace) / np.log(2)
        signs = (1 - 2 * bits.astype(np.int64)).astype(np.float64)
        signed_entropy = signs * entropy
        folded = fold_soft_token(bits, p_trace, self.n)
        torch.save(
            {
                "prompt_idx": idx,
                "source": source,
                "generation_model": model_display(
                    self.generation_model_size
                ),
                "generation_model_size": self.generation_model_size,
                "entropy_model": model_display(self.model_size),
                "entropy_trace_source": entropy_trace_source(
                    self.model_size, self.generation_model_size
                ),
                "tokens_len": int(tokens.numel()),
                "p_trace": p_trace,
                "entropy_trace": entropy.astype(np.float32),
                "signed_entropy_trace": signed_entropy.astype(np.float32),
                "folded_signed_entropy": folded.astype(np.float32),
            },
            path,
        )

    def _estimate(self, prompt_indices, source):
        import os
        import time

        import torch

        if source == "wm":
            src_dir = f"/data/{self.tag}/wm"
            out_dir = wm_entropy_dir(self.tag, self.model_size)
            prefix = "wm"
        elif source == "null":
            src_dir = null_dir(self.null_T, self.generation_model_size)
            out_dir = null_entropy_dir(
                self.T, self.model_size, self.generation_model_size
            )
            prefix = "null"
        else:
            raise ValueError(f"unknown entropy source {source}")

        data_vol.reload()
        os.makedirs(out_dir, exist_ok=True)
        todo = [
            i for i in prompt_indices
            if not os.path.exists(os.path.join(out_dir, f"{prefix}_{i:04d}.pt"))
        ]
        if not todo:
            return {"estimated": 0, "cached": len(prompt_indices), "source": source}

        t0 = time.time()
        records = [
            torch.load(os.path.join(src_dir, f"{prefix}_{i:04d}.pt"),
                       weights_only=False, map_location="cpu")
            for i in todo
        ]
        for record, i in zip(records, todo):
            validate_generation_record(
                record, self.generation_model_size, source, i
            )
            if int(record.get("prompt_idx", i)) != i:
                raise ValueError(
                    f"{source} cache prompt mismatch for index {i}: "
                    f"record has {record.get('prompt_idx')}"
                )
            if len(record["tokens"]) < self.T:
                raise ValueError(
                    f"{source} cache index {i} has {len(record['tokens'])} "
                    f"tokens, need at least {self.T}"
                )
        token_batch = torch.stack([rec["tokens"][:self.T].long() for rec in records])
        prompt_batch = self._prompt_batch(todo)
        p_traces = self.we.estimate_partition_trace_batch(
            self.we.model, prompt_batch, token_batch, self.partition
        )
        for row, i in enumerate(todo):
            tokens = token_batch[row].cpu()
            self._save_trace_payload(
                os.path.join(out_dir, f"{prefix}_{i:04d}.pt"),
                i,
                source,
                tokens,
                p_traces[row],
            )

        data_vol.commit()
        return {"estimated": len(todo), "cached": len(prompt_indices) - len(todo),
                "source": source, "dt": time.time() - t0}

    @modal.method()
    def estimate_wm(self, prompt_indices: list) -> dict:
        return self._estimate(prompt_indices, "wm")

    @modal.method()
    def estimate_null(self, prompt_indices: list) -> dict:
        return self._estimate(prompt_indices, "null")


# ---- detection over cached generations (CPU) --------------------------------
@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(n: int, t: int, eta: float, fpr: float, null_T: int,
               prompt_indices: list, r: int = 0,
               entropy_model_size: str = MODEL_SIZE,
               null_entropy_T: int = 0,
               run_metadata: dict = None,
               generation_model_size: str = MODEL_SIZE) -> dict:
    import os

    import numpy as np
    import torch
    from detectors import (
        binary_entropy,
        detect_hoeffding,
        fold_soft_token,
        tokens_to_bits,
    )
    from prc import parity_check_rank_info

    requested_r = int(r) if r else None
    T = experiment_T(n)
    generation_model_size = normalize_model_size(generation_model_size)
    tag = config_tag(
        n, t, eta, requested_r, T, generation_model_size
    )
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(
        model_size, generation_model_size
    )
    null_entropy_T = int(null_entropy_T) if null_entropy_T else T
    source_label = entropy_trace_source(model_size, generation_model_size)
    prompt_indices = _coerce_prompt_indices(prompt_indices)
    if not prompt_indices:
        raise ValueError("detection requires at least one prompt index")
    run_metadata = dict(run_metadata or {})
    ap = art_path(
        n, t, eta, requested_r, T, generation_model_size
    )
    wmd = wm_dir(
        n, t, eta, requested_r, T, generation_model_size
    )
    nd = null_dir(null_T, generation_model_size)
    data_vol.reload()
    art = torch.load(ap, weights_only=False, map_location="cpu")
    artifact_model_size = normalize_model_size(
        art.get("generation_model_size", MODEL_SIZE)
    )
    if artifact_model_size != generation_model_size:
        raise ValueError(
            f"artifact generation model {artifact_model_size} does not "
            f"match requested model {generation_model_size}"
        )
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    artifact_fingerprint = art.get("artifact_fingerprint")
    if not artifact_fingerprint:
        artifact_fingerprint = _semantic_fingerprint({
            key: art[key] for key in (
                "encoding_key", "decoding_key", "partition",
                "prompt_ids_list", "seed", "config_sig",
            )
        })
    rank_info = art.get("parity_check_rank_info")
    if rank_info is None:
        rank_info = parity_check_rank_info(decoding_key[1])

    workspace_label = run_metadata.get("workspace_label", "workspace")
    code_fingerprint = run_metadata.get("code_fingerprint", {})
    config = {
        "n": n,
        "T": T,
        "t": t,
        "eta": eta,
        "r_value": rank_info.get("rows", requested_r),
        "r_setting": run_metadata.get(
            "r_setting", "explicit" if requested_r is not None else "default"
        ),
        "target_fpr": fpr,
        "generation_model": model_display(generation_model_size),
        "entropy_model": model_display(model_size),
        "entropy_trace_source": source_label,
        "seed": art.get("seed", SEED),
        "canonical_num_prompts": run_metadata.get(
            "canonical_num_prompts", CANONICAL_NUM_PROMPTS
        ),
    }
    checkpoint_root = detection_checkpoint_dir(tag, model_size, fpr)
    checkpoint_stats = {"reused": 0, "computed": 0, "stale_or_corrupt": 0}

    trace_saved = 0

    def _prefix(values, source, idx, field):
        if len(values) < T:
            raise ValueError(
                f"{source} cache index {idx} has {len(values)} {field} values, "
                f"need at least {T}"
            )
        return values[:T]

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(
                _prefix(record["p_trace"], source, idx, "p_trace"),
                dtype=np.float64,
            )
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(
                null_entropy_dir(
                    null_entropy_T, model_size, generation_model_size
                ),
                f"null_{idx:04d}.pt",
            )
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(
            _prefix(est["p_trace"], source, idx, "p_trace"),
            dtype=np.float64,
        )

    def _save_detection_trace(source, idx, tokens, p_trace):
        nonlocal trace_saved
        if source == "wm":
            out_dir = wm_trace_dir(tag, model_size)
            prefix = "wm"
        else:
            out_dir = null_trace_dir(
                T, model_size, generation_model_size
            )
            prefix = "null"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}_{idx:04d}.pt")
        if os.path.exists(path):
            return
        bits = tokens_to_bits(tokens, partition)
        entropy = binary_entropy(p_trace) / np.log(2)
        signs = (1 - 2 * bits.astype(np.int64)).astype(np.float64)
        signed_entropy = signs * entropy
        folded = fold_soft_token(bits, p_trace, n)
        torch.save(
            {
                "prompt_idx": idx,
                "source": source,
                "generation_model": model_display(generation_model_size),
                "generation_model_size": normalize_model_size(
                    generation_model_size
                ),
                "entropy_model": model_display(model_size),
                "entropy_trace_source": source_label,
                "tokens_len": int(tokens.numel()),
                "entropy_trace": entropy.astype(np.float32),
                "signed_entropy_trace": signed_entropy.astype(np.float32),
                "folded_signed_entropy": folded.astype(np.float32),
            },
            path,
        )
        trace_saved += 1

    def _input_hashes(tokens, p_trace):
        tokens_sha256 = hashlib.sha256(
            tokens.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        p_trace_sha256 = hashlib.sha256(
            np.ascontiguousarray(p_trace).tobytes()
        ).hexdigest()
        return tokens_sha256, p_trace_sha256

    def _run(tokens, p_trace, wm_flag, idx,
             tokens_sha256, p_trace_sha256):
        dm, im = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="map", return_info=True)
        de, ie = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="entropy", return_info=True)
        out = {
            "prompt_idx": idx,
            "source": "wm" if wm_flag else "null",
            "watermark": wm_flag,
            "decision_map": bool(dm),
            "stat_map": float(im["statistic"]),
            "thr_map": float(im["threshold"]),
            "decision_entropy": bool(de),
            "stat_entropy": float(ie["statistic"]),
            "thr_entropy": float(ie["threshold"]),
            "decision_naive": None,
            "stat_naive": None,
            "thr_naive": None,
            "decision_log": None,
            "stat_log": None,
            "thr_log": None,
            "n_tokens": int(tokens.numel()),
            "generation_model": model_display(generation_model_size),
            "entropy_model": model_display(model_size),
            "entropy_trace_source": source_label,
            "parity_check_rank_info": rank_info,
            "tokens_sha256": tokens_sha256,
            "p_trace_sha256": p_trace_sha256,
        }
        if use_generation_trace:
            dn, ino = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                       fpr=fpr, weight="naive", return_info=True)
            out.update({
                "decision_naive": bool(dn),
                "stat_naive": float(ino["statistic"]),
                "thr_naive": float(ino["threshold"]),
            })
        return out

    def _checkpointed_run(tokens, p_trace, wm_flag, idx):
        source = "wm" if wm_flag else "null"
        tokens_sha256, p_trace_sha256 = _input_hashes(tokens, p_trace)
        identity = _detection_checkpoint_identity(
            config,
            artifact_fingerprint,
            code_fingerprint,
            source,
            idx,
            tokens_sha256,
            p_trace_sha256,
        )
        checkpoint_path = os.path.join(
            checkpoint_root, source, f"{source}_{idx:04d}.json"
        )
        existed = os.path.exists(checkpoint_path)
        cached = _load_detection_checkpoint(checkpoint_path, identity)
        if cached is not None:
            checkpoint_stats["reused"] += 1
            return cached
        if existed:
            checkpoint_stats["stale_or_corrupt"] += 1
        record = _run(
            tokens, p_trace, wm_flag, idx, tokens_sha256, p_trace_sha256
        )
        _save_detection_checkpoint(checkpoint_path, identity, record)
        checkpoint_stats["computed"] += 1
        return record

    out = []
    committed_checkpoint_count = 0

    for position, i in enumerate(prompt_indices, start=1):
        gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        validate_generation_record(gw, generation_model_size, "wm", i)
        wm_tokens = _prefix(gw["tokens"], "wm", i, "token")
        wm_p = _p_trace("wm", i, gw)
        _save_detection_trace("wm", i, wm_tokens, wm_p)
        out.append(_checkpointed_run(wm_tokens, wm_p, True, i))

        gn = torch.load(os.path.join(nd, f"null_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        validate_generation_record(gn, generation_model_size, "null", i)
        null_tokens = _prefix(gn["tokens"], "null", i, "token")
        null_p = _p_trace("null", i, gn)
        _save_detection_trace("null", i, null_tokens, null_p)
        out.append(_checkpointed_run(null_tokens, null_p, False, i))

        if (position % DETECTION_CHECKPOINT_COMMIT_INTERVAL == 0
                and checkpoint_stats["computed"] > committed_checkpoint_count):
            data_vol.commit()
            committed_checkpoint_count = checkpoint_stats["computed"]

    filename = shard_result_filename(
        tag, model_size, fpr, prompt_indices, workspace_label
    )
    remote_path = os.path.join(
        f"/data/{tag}/shard_results/{entropy_model_tag(model_size)}",
        filename,
    )
    safe_records = _json_safe(out)
    records_json = json.dumps(
        safe_records, sort_keys=True, separators=(",", ":")
    ).encode()
    payload = {
        "schema_version": SHARD_RESULT_SCHEMA_VERSION,
        "config": config,
        "artifact_fingerprint": artifact_fingerprint,
        "code_fingerprint": run_metadata.get("code_fingerprint", {}),
        "workspace_label": workspace_label,
        "prompt_indices": prompt_indices,
        "prompt_start": min(prompt_indices),
        "prompt_stop": max(prompt_indices) + 1,
        "record_count": len(safe_records),
        "records_sha256": hashlib.sha256(records_json).hexdigest(),
        "null_cache_T": null_T,
        "null_cache_root": null_root(generation_model_size),
        "null_entropy_cache_T": null_entropy_T,
        "null_detection_trace_root": null_trace_dir(
            T, model_size, generation_model_size
        ),
        "parity_check_rank_info": _json_safe(rank_info),
        "detection_checkpointing": {
            "schema_version": DETECTION_CHECKPOINT_SCHEMA_VERSION,
            "commit_interval_prompts": DETECTION_CHECKPOINT_COMMIT_INTERVAL,
            **checkpoint_stats,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": safe_records,
    }
    _atomic_write_json(remote_path, payload)
    data_vol.commit()
    return {"results": out, "shard_payload": payload,
            "checkpoint_stats": checkpoint_stats,
            "remote_shard_path": remote_path}


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all_any(n: int, t: int, eta: float, fpr: float,
                   num_prompts: int = 500, r: int = 0,
                   entropy_model_size: str = MODEL_SIZE,
                   generation_model_size: str = MODEL_SIZE) -> list:
    """Re-detect a config with ALL weight kinds, auto-detecting the cache layout:
      - NEW layout (config_tag/wm/wm_XXXX.pt + shared _nulls/T*) -> batched pipeline
      - OLD layout (config_tag/gens/gen_XXXX.pt, watermark flag inside)
    Free: model-free CPU detection over already-cached generations, no regen."""
    import glob
    import os
    import numpy as np
    import torch
    from detectors import detect_hoeffding, WEIGHT_KINDS

    T = experiment_T(n)
    requested_r = int(r) if r else None
    generation_model_size = normalize_model_size(generation_model_size)
    tag = config_tag(
        n, t, eta, requested_r, T, generation_model_size
    )
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(
        model_size, generation_model_size
    )
    ap = art_path(
        n, t, eta, requested_r, T, generation_model_size
    )
    data_vol.reload()
    art = torch.load(ap, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    null_cache_T = T
    null_entropy_cache_T = T

    def _prefix(values, source, idx, field):
        if len(values) < T:
            raise ValueError(
                f"{source} cache index {idx} has {len(values)} {field} values, "
                f"need at least {T}"
            )
        return values[:T]

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(
                _prefix(record["p_trace"], source, idx, "p_trace"),
                dtype=np.float64,
            )
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(
                null_entropy_dir(
                    null_entropy_cache_T, model_size,
                    generation_model_size,
                ),
                f"null_{idx:04d}.pt",
            )
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(
            _prefix(est["p_trace"], source, idx, "p_trace"),
            dtype=np.float64,
        )

    def decisions(tokens, p_trace, wm_flag, idx):
        row = {"prompt_idx": idx, "watermark": bool(wm_flag)}
        weight_kinds = WEIGHT_KINDS if use_generation_trace else tuple(
            wname for wname in WEIGHT_KINDS if wname != "naive"
        )
        for wname in weight_kinds:
            row[f"decision_{wname}"] = bool(detect_hoeffding(
                decoding_key, tokens, p_trace, partition, fpr=fpr, weight=wname))
        return row

    wmd = f"/data/{tag}/wm"
    new_layout = os.path.isdir(wmd) and glob.glob(os.path.join(wmd, "wm_*.pt"))

    out = []
    if new_layout:
        root = null_root(generation_model_size)
        null_cache_T = find_complete_cache_T(root, T, num_prompts, "null")
        if null_cache_T is None:
            raise FileNotFoundError(
                f"No complete shared null store found with T >= {T}"
            )
        if not use_generation_trace:
            entropy_root = os.path.dirname(null_entropy_dir(
                T, model_size, generation_model_size
            ))
            null_entropy_cache_T = find_complete_cache_T(
                entropy_root, T, num_prompts, "null"
            )
            if null_entropy_cache_T is None:
                raise FileNotFoundError(
                    f"No complete {model_display(model_size)} null entropy "
                    f"store found with T >= {T}"
                )
        for i in range(num_prompts):
            gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            validate_generation_record(
                gw, generation_model_size, "wm", i
            )
            wm_tokens = _prefix(gw["tokens"], "wm", i, "token")
            out.append(decisions(wm_tokens, _p_trace("wm", i, gw), True, i))
            gn = torch.load(os.path.join(
                root, f"T{null_cache_T}", f"null_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            validate_generation_record(
                gn, generation_model_size, "null", i
            )
            null_tokens = _prefix(gn["tokens"], "null", i, "token")
            out.append(decisions(null_tokens, _p_trace("null", i, gn), False, i))
    else:
        for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
            g = torch.load(path, weights_only=False, map_location="cpu")
            idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
            tokens = g["tokens"][:T]
            out.append(decisions(tokens, _p_trace("wm" if g["watermark"] else "null", idx, g),
                                 bool(g["watermark"]), idx))
    return out


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_map_summary(n: int, t: int, eta: float, fpr: float,
                       num_prompts: int = 500, r: int = 0,
                       entropy_model_size: str = MODEL_SIZE,
                       generation_model_size: str = MODEL_SIZE) -> dict:
    """CPU-only cache redetect for the CSV columns we need: map, entropy, naive."""
    import glob
    import os
    import numpy as np
    import torch
    from detectors import detect_hoeffding

    T = experiment_T(n)
    requested_r = int(r) if r else None
    generation_model_size = normalize_model_size(generation_model_size)
    tag = config_tag(
        n, t, eta, requested_r, T, generation_model_size
    )
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(
        model_size, generation_model_size
    )
    data_vol.reload()
    art = torch.load(
        art_path(n, t, eta, requested_r, T, generation_model_size),
        weights_only=False, map_location="cpu"
    )
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    null_cache_T = T
    null_entropy_cache_T = T

    def _prefix(values, source, idx, field):
        if len(values) < T:
            raise ValueError(
                f"{source} cache index {idx} has {len(values)} {field} values, "
                f"need at least {T}"
            )
        return values[:T]

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(
                _prefix(record["p_trace"], source, idx, "p_trace"),
                dtype=np.float64,
            )
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(
                null_entropy_dir(
                    null_entropy_cache_T, model_size,
                    generation_model_size,
                ),
                f"null_{idx:04d}.pt",
            )
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(
            _prefix(est["p_trace"], source, idx, "p_trace"),
            dtype=np.float64,
        )

    counts = {
        "wm_total": 0,
        "null_total": 0,
        "map_tp": 0,
        "map_fp": 0,
        "entropy_tp": 0,
        "entropy_fp": 0,
        "naive_tp": None if not use_generation_trace else 0,
        "naive_fp": None if not use_generation_trace else 0,
    }

    def _score(tokens, p_trace, watermark):
        dm = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="map")
        de = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="entropy")
        dn = None
        if use_generation_trace:
            dn = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="naive")
        if watermark:
            counts["wm_total"] += 1
            counts["map_tp"] += int(dm)
            counts["entropy_tp"] += int(de)
            if dn is not None:
                counts["naive_tp"] += int(dn)
        else:
            counts["null_total"] += 1
            counts["map_fp"] += int(dm)
            counts["entropy_fp"] += int(de)
            if dn is not None:
                counts["naive_fp"] += int(dn)

    wmd = f"/data/{tag}/wm"
    new_layout = os.path.isdir(wmd) and glob.glob(os.path.join(wmd, "wm_*.pt"))
    if new_layout:
        root = null_root(generation_model_size)
        null_cache_T = find_complete_cache_T(root, T, num_prompts, "null")
        if null_cache_T is None:
            raise FileNotFoundError(
                f"No complete shared null store found with T >= {T}"
            )
        if not use_generation_trace:
            entropy_root = os.path.dirname(null_entropy_dir(
                T, model_size, generation_model_size
            ))
            null_entropy_cache_T = find_complete_cache_T(
                entropy_root, T, num_prompts, "null"
            )
            if null_entropy_cache_T is None:
                raise FileNotFoundError(
                    f"No complete {model_display(model_size)} null entropy "
                    f"store found with T >= {T}"
                )
        for i in range(num_prompts):
            gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            validate_generation_record(
                gw, generation_model_size, "wm", i
            )
            wm_tokens = _prefix(gw["tokens"], "wm", i, "token")
            _score(wm_tokens, _p_trace("wm", i, gw), True)
            gn = torch.load(os.path.join(
                root, f"T{null_cache_T}", f"null_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            validate_generation_record(
                gn, generation_model_size, "null", i
            )
            null_tokens = _prefix(gn["tokens"], "null", i, "token")
            _score(null_tokens, _p_trace("null", i, gn), False)
    else:
        for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
            g = torch.load(path, weights_only=False, map_location="cpu")
            idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
            tokens = g["tokens"][:T]
            watermark = bool(g["watermark"])
            source = "wm" if watermark else "null"
            _score(tokens, _p_trace(source, idx, g), watermark)

    counts.update({
        "n": n,
        "t": t,
        "eta": eta,
        "T": T,
        "r": requested_r,
        "generation_model": model_display(generation_model_size),
        "entropy_model": model_display(model_size),
        "entropy_trace_source": entropy_trace_source(
            model_size, generation_model_size
        ),
    })
    return counts


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_legacy_first_block_summary(n: int, t: int, eta: float, fpr: float,
                                      num_prompts: int = 500, r: int = 0,
                                      entropy_model_size: str = MODEL_SIZE,
                                      legacy_token_length: int = 0) -> dict:
    """Redetect old T=2n cache rows using only the first length-n block."""
    import glob
    import os

    import numpy as np
    import torch
    from detectors import detect_hoeffding
    from prc import parity_check_rank_info

    score_T = experiment_T(n)
    cache_T = (
        int(legacy_token_length) if legacy_token_length else 2 * int(n)
    )
    requested_r = int(r) if r else None
    tag = config_tag(n, t, eta, requested_r, None)
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(model_size)

    data_vol.reload()
    art = torch.load(f"/data/{tag}/artifacts.pt", weights_only=False,
                     map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    rank_info = art.get("parity_check_rank_info")
    if rank_info is None:
        rank_info = parity_check_rank_info(decoding_key[1])

    def _p_trace(idx, record):
        if use_generation_trace:
            p_trace = np.asarray(record["p_trace"], dtype=np.float64)
        else:
            path = os.path.join(
                f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
                f"gen_{idx:04d}.pt",
            )
            est = torch.load(path, weights_only=False, map_location="cpu")
            p_trace = np.asarray(est["p_trace"], dtype=np.float64)
        if p_trace.shape[0] < score_T:
            raise ValueError(
                f"{tag} gen_{idx:04d} has p_trace length {p_trace.shape[0]}, "
                f"need at least {score_T}"
            )
        return p_trace[:score_T]

    counts = {
        "wm_total": 0,
        "null_total": 0,
        "map_tp": 0,
        "map_fp": 0,
        "entropy_tp": 0,
        "entropy_fp": 0,
        "naive_tp": None if not use_generation_trace else 0,
        "naive_fp": None if not use_generation_trace else 0,
    }

    def _score(tokens, p_trace, watermark):
        dm = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="map")
        de = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="entropy")
        dn = None
        if use_generation_trace:
            dn = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="naive")
        if watermark:
            counts["wm_total"] += 1
            counts["map_tp"] += int(dm)
            counts["entropy_tp"] += int(de)
            if dn is not None:
                counts["naive_tp"] += int(dn)
        else:
            counts["null_total"] += 1
            counts["map_fp"] += int(dm)
            counts["entropy_fp"] += int(de)
            if dn is not None:
                counts["naive_fp"] += int(dn)

    for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
        g = torch.load(path, weights_only=False, map_location="cpu")
        idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
        watermark = bool(g["watermark"])
        if watermark and counts["wm_total"] >= num_prompts:
            continue
        if not watermark and counts["null_total"] >= num_prompts:
            continue
        tokens = g["tokens"][:score_T]
        if tokens.numel() < score_T:
            raise ValueError(
                f"{tag} gen_{idx:04d} has token length {tokens.numel()}, "
                f"need at least {score_T}"
            )
        _score(tokens, _p_trace(idx, g), watermark)

    counts.update({
        "n": n,
        "t": t,
        "eta": eta,
        "T": score_T,
        "legacy_T": cache_T,
        "legacy_tag": tag,
        "r": requested_r,
        "entropy_model": model_display(model_size),
        "entropy_trace_source": entropy_trace_source(model_size),
        "parity_check_rank_info": rank_info,
    })
    return counts


# ---- driver -----------------------------------------------------------------
@app.local_entrypoint()
def main(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         prompt_start: int = 0,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False,
         batch: int = DEFAULT_BATCH,
         entropy_batch: int = DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = 0.0,
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
    resolved_r = resolve_r(n, r, r_frac)
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
    r_text = f"r={resolved_r}" if resolved_r is not None else "r=default"
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
        )(tag=tag, model_size=generation_model_size)
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

    null_entropy_T = null_T
    if uses_cached_generation_trace(
            entropy_model_size, generation_model_size):
        print("[main] entropy trace: using cached generation p_trace", flush=True)
    else:
        eplan = plan_entropy.remote(
            tag, entropy_model_size, T, null_T, prompt_indices,
            generation_model_size,
        )
        wm_e_missing = eplan["wm_missing"]
        null_e_missing = eplan["null_missing"]
        null_entropy_T = eplan["null_entropy_T"]
        print(f"[main] entropy traces to estimate with "
              f"{model_display(entropy_model_size)}: "
              f"{len(wm_e_missing)} watermarked, {len(null_e_missing)} null "
              f"(selected null entropy store T={null_entropy_T}; "
              f"scoring prefix T={T})",
              flush=True)
        if wm_e_missing or null_e_missing:
            estimator = EntropyModel.with_options(
                gpu=gpu, max_containers=max_containers
            )(
                tag=tag,
                model_size=entropy_model_size,
                generation_model_size=generation_model_size,
                T=T,
                null_T=null_T,
            )
            ecalls = []
            if wm_e_missing:
                ecalls.append(("wm entropy", list(estimator.estimate_wm.map(
                    _chunks(wm_e_missing, entropy_batch)))))
            if null_e_missing:
                ecalls.append(("null entropy", list(estimator.estimate_null.map(
                    _chunks(null_e_missing, entropy_batch)))))
            for kind, metas in ecalls:
                est = sum(m.get("estimated", 0) for m in metas)
                print(f"[main] {kind}: estimated {est} in {len(metas)} batches",
                      flush=True)
        else:
            print("[main] all alternate entropy traces cached", flush=True)

    detect_label = "map + entropy-aware + naive"
    if not uses_cached_generation_trace(
            entropy_model_size, generation_model_size):
        detect_label = "map + entropy-aware (naive/log skipped for alternate entropy)"
    print(f"[main] detecting ({detect_label}) ...", flush=True)
    r_setting = f"{r_frac:g}n" if r_frac else (
        "explicit" if r else "default"
    )
    detection = detect_all.remote(
        n, t, eta, fpr, null_T, prompt_indices,
        resolved_r or 0, entropy_model_size, null_entropy_T,
        {
            "workspace_label": workspace_label,
            "canonical_num_prompts": CANONICAL_NUM_PROMPTS,
            "r_setting": r_setting,
            "code_fingerprint": code_fingerprint,
        },
        generation_model_size,
    )
    results = detection["results"]
    shard_payload = detection["shard_payload"]
    checkpoint_stats = detection.get("checkpoint_stats", {})
    print(
        "[main] detection checkpoints: "
        f"reused={checkpoint_stats.get('reused', 0)}, "
        f"computed={checkpoint_stats.get('computed', 0)}, "
        f"stale_or_corrupt={checkpoint_stats.get('stale_or_corrupt', 0)}",
        flush=True,
    )
    if not shard_out:
        shard_out = os.path.join(
            "outputs", "shards",
            shard_result_filename(
                tag, entropy_model_size, fpr, prompt_indices, workspace_label
            ),
        )
    _atomic_write_json(shard_out, shard_payload)
    print(f"[main] saved local shard result -> {shard_out}", flush=True)
    print(f"[main] saved remote shard result -> "
          f"{detection['remote_shard_path']}", flush=True)

    wm = sorted([r0 for r0 in results if r0["watermark"]],
                key=lambda r0: r0["prompt_idx"])
    nw = sorted([r0 for r0 in results if not r0["watermark"]],
                key=lambda r0: r0["prompt_idx"])
    tp_m = sum(r0["decision_map"] for r0 in wm)
    fp_m = sum(r0["decision_map"] for r0 in nw)
    tp_e = sum(r0["decision_entropy"] for r0 in wm)
    fp_e = sum(r0["decision_entropy"] for r0 in nw)
    has_naive = any(r0.get("decision_naive") is not None for r0 in results)
    tp_n = sum(r0["decision_naive"] for r0 in wm) if has_naive else 0
    fp_n = sum(r0["decision_naive"] for r0 in nw) if has_naive else 0
    nwm, nnw = max(len(wm), 1), max(len(nw), 1)
    rank_info = results[0].get("parity_check_rank_info", {}) if results else {}

    print("\n=== Summary (Hoeffding detector, proven FPR) ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g}  T={T}  {r_text}",
          flush=True)
    if rank_info:
        print(f"  parity rank: {rank_info.get('rank')}/{rank_info.get('rows')} "
              f"full_rank={rank_info.get('full_rank')}", flush=True)
    print(f"  map (default):  TPR {tp_m}/{len(wm)} ({tp_m/nwm:.1%})   "
          f"FPR {fp_m}/{len(nw)} ({fp_m/nnw:.1%})", flush=True)
    print(f"  entropy-aware:  TPR {tp_e}/{len(wm)} ({tp_e/nwm:.1%})   "
          f"FPR {fp_e}/{len(nw)} ({fp_e/nnw:.1%})", flush=True)
    if has_naive:
        print(f"  naive        :  TPR {tp_n}/{len(wm)} ({tp_n/nwm:.1%})   "
              f"FPR {fp_n}/{len(nw)} ({fp_n/nnw:.1%})", flush=True)
    else:
        print("  naive        :  skipped", flush=True)
    print("  log_hoeffding:  skipped", flush=True)

    rank_note = ""
    if rank_info:
        rank_note = (
            f"r={rank_info.get('rows')} rank={rank_info.get('rank')}/"
            f"{rank_info.get('rows')} full_rank={rank_info.get('full_rank')}"
        )
        if not rank_info.get("full_rank", False):
            rank_note += "; WARNING parity matrix was not full rank"

    notes = "; ".join(part for part in [
        f"{model_display(generation_model_size)} generation",
        f"entropy model {model_display(entropy_model_size)}",
        entropy_trace_source(
            entropy_model_size, generation_model_size
        ),
        "map=Bayes-optimal soft-token S_j=E[c|observed bit,p]",
        f"batched Modal pipeline with null cache T={null_T} truncated to T={T}",
        (f"alternate null entropy cache T={null_entropy_T} truncated to T={T}"
         if not uses_cached_generation_trace(
             entropy_model_size, generation_model_size
         ) else ""),
        "single length-n code block (T=n)",
        "Hoeffding threshold tau=sqrt(2V*log(1/F)); one block so block FPR equals target F",
        rank_note,
    ] if part)

    row = {
        "Target FPR": f"{fpr:.0e}",
        "n": n,
        "t": t,
        "eta": eta,
        "T": T,
        "r setting": r_setting,
        "r value": rank_info.get("rows", resolved_r or ""),
        "Map TPR": _format_rate(tp_m, len(wm)),
        "Entropy Aware TPR": _format_rate(tp_e, len(wm)),
        "Naive TPR": _format_rate(tp_n, len(wm)) if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _format_rate(fp_m, len(nw)),
        "Entropy FPR": _format_rate(fp_e, len(nw)),
        "Naive FPR": _format_rate(fp_n, len(nw)) if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Model": model_display(entropy_model_size),
        "Generation Model": model_display(generation_model_size),
        "Entropy Trace Source": entropy_trace_source(
            entropy_model_size, generation_model_size
        ),
        "Notes": notes,
    }
    if is_complete_run:
        _append_summary_row(csv_out, row)
        print(f"[main] appended 500/500 summary row to {csv_out}", flush=True)
    else:
        print("[main] partial shard: summary CSV not modified; use "
              "aggregate_shards after all prompt shards finish", flush=True)


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
    """Append a T=n redetect row from legacy T=2n cached generations."""
    resolved_r = resolve_r(n, r, r_frac)
    entropy_model_size = normalize_model_size(entropy_model_size)
    score_T = experiment_T(n)
    cache_T = (
        int(legacy_token_length) if legacy_token_length else 2 * int(n)
    )
    legacy_tag = config_tag(n, t, eta, resolved_r, None)
    print(f"[legacy_first_block_redetect] cache={legacy_tag} cache_T={cache_T} "
          f"score_T={score_T} FPR_target={fpr:g} "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)

    s = detect_legacy_first_block_summary.remote(
        n, t, eta, fpr, num_prompts, resolved_r or 0, entropy_model_size,
        cache_T,
    )

    wm_total = max(s["wm_total"], 1)
    null_total = max(s["null_total"], 1)
    has_naive = s["naive_tp"] is not None
    rank_info = s.get("parity_check_rank_info", {})

    print("\n=== Legacy First-Block Redetect Summary ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g} "
          f"T={score_T} legacy_T={cache_T} r={resolved_r}", flush=True)
    if rank_info:
        print(f"  parity rank: {rank_info.get('rank')}/{rank_info.get('rows')} "
              f"full_rank={rank_info.get('full_rank')}", flush=True)
    print(f"  entropy model: {s['entropy_model']} ({s['entropy_trace_source']})",
          flush=True)
    print(f"  map    : TPR {s['map_tp']}/{s['wm_total']} "
          f"({s['map_tp']/wm_total:.1%})   FPR {s['map_fp']}/"
          f"{s['null_total']} ({s['map_fp']/null_total:.1%})", flush=True)
    print(f"  entropy: TPR {s['entropy_tp']}/{s['wm_total']} "
          f"({s['entropy_tp']/wm_total:.1%})   FPR {s['entropy_fp']}/"
          f"{s['null_total']} ({s['entropy_fp']/null_total:.1%})", flush=True)
    if has_naive:
        print(f"  naive  : TPR {s['naive_tp']}/{s['wm_total']} "
              f"({s['naive_tp']/wm_total:.1%})   FPR {s['naive_fp']}/"
              f"{s['null_total']} ({s['naive_fp']/null_total:.1%})", flush=True)
    else:
        print("  naive  : skipped", flush=True)

    rank_note = ""
    if rank_info:
        rank_note = (
            f"r={rank_info.get('rows')} rank={rank_info.get('rank')}/"
            f"{rank_info.get('rows')} full_rank={rank_info.get('full_rank')}"
        )
        if not rank_info.get("full_rank", False):
            rank_note += "; WARNING parity matrix was not full rank"

    notes = "; ".join(part for part in [
        "Qwen3-0.6B-Base generation",
        f"entropy model {s['entropy_model']}",
        s["entropy_trace_source"],
        "map=Bayes-optimal soft-token S_j=E[c|observed bit,p]",
        f"legacy redetect from {legacy_tag}",
        f"used first n tokens from legacy T={cache_T} cached generations",
        "appended as new T=n row; old T=2n row retained",
        "Hoeffding threshold tau=sqrt(2V*log(1/F)); one block so block FPR equals target F",
        rank_note,
    ] if part)

    row = {
        "Target FPR": f"{fpr:.0e}",
        "n": n,
        "t": t,
        "eta": eta,
        "T": score_T,
        "r setting": f"{r_frac:g}n" if r_frac else (
            "explicit" if r else "default"
        ),
        "r value": rank_info.get("rows", resolved_r or ""),
        "Map TPR": _format_rate(s["map_tp"], s["wm_total"]),
        "Entropy Aware TPR": _format_rate(s["entropy_tp"], s["wm_total"]),
        "Naive TPR": _format_rate(s["naive_tp"], s["wm_total"])
        if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _format_rate(s["map_fp"], s["null_total"]),
        "Entropy FPR": _format_rate(s["entropy_fp"], s["null_total"]),
        "Naive FPR": _format_rate(s["naive_fp"], s["null_total"])
        if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Model": s["entropy_model"],
        "Entropy Trace Source": s["entropy_trace_source"],
        "Notes": notes,
    }
    _append_summary_row(csv_out, row)
    print(f"[legacy_first_block_redetect] appended summary row to {csv_out}",
          flush=True)


# ---- re-detection sweep over all weight kinds (CPU-only, no regeneration) ---
@app.local_entrypoint()
def redetect(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
             fpr: float = DEFAULT_FPR, num_prompts: int = 500,
             r: int = 0, r_frac: float = 0.0,
             generation_model_size: str = MODEL_SIZE,
             entropy_model_size: str = ""):
    """Re-detect a config (either cache layout) with ALL weight kinds, ranked.
    Free: model-free CPU detection over already-cached generations."""
    from detectors import WEIGHT_KINDS

    resolved_r = resolve_r(n, r, r_frac)
    generation_model_size = normalize_model_size(generation_model_size)
    entropy_model_size = normalize_model_size(
        entropy_model_size or generation_model_size
    )
    T = experiment_T(n)
    tag = config_tag(
        n, t, eta, resolved_r, T, generation_model_size
    )
    print(f"[redetect] {tag}  FPR_target={fpr:g}  "
          f"generation_model={model_display(generation_model_size)} "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)
    results = detect_all_any.remote(n, t, eta, fpr, num_prompts,
                                    resolved_r or 0, entropy_model_size,
                                    generation_model_size)
    wm = [r for r in results if r["watermark"]]
    nw = [r for r in results if not r["watermark"]]
    nwm, nnw = max(len(wm), 1), max(len(nw), 1)
    print(f"\n=== Re-detect (Hoeffding, proven FPR) n={n} t={t} eta={eta} "
          f"F={fpr:g}  T={T} ===", flush=True)
    rows = []
    for name in WEIGHT_KINDS:
        key = f"decision_{name}"
        if not any(key in row for row in results):
            continue
        tp = sum(r.get(key, False) for r in wm)
        fp = sum(r.get(key, False) for r in nw)
        rows.append((tp, fp, name))
    for tp, fp, name in sorted(rows, reverse=True):     # best TPR first
        flag = "  <- baseline" if name == "entropy" else ""
        print(f"  {name:8s}: TPR {tp}/{len(wm)} ({tp/nwm:.1%})   "
              f"FPR {fp}/{len(nw)} ({fp/nnw:.1%}){flag}", flush=True)


@app.local_entrypoint()
def redetect_map(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
                 fpr: float = DEFAULT_FPR, num_prompts: int = 500,
                 r: int = 0, r_frac: float = 0.0,
                 generation_model_size: str = MODEL_SIZE,
                 entropy_model_size: str = ""):
    """CPU-only redetect for map/entropy/naive CSV summary columns."""
    resolved_r = resolve_r(n, r, r_frac)
    generation_model_size = normalize_model_size(generation_model_size)
    entropy_model_size = normalize_model_size(
        entropy_model_size or generation_model_size
    )
    T = experiment_T(n)
    tag = config_tag(
        n, t, eta, resolved_r, T, generation_model_size
    )
    print(f"[redetect_map] {tag} FPR_target={fpr:g} "
          f"generation_model={model_display(generation_model_size)} "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)
    s = detect_map_summary.remote(n, t, eta, fpr, num_prompts,
                                  resolved_r or 0, entropy_model_size,
                                  generation_model_size)
    wm_total = max(s["wm_total"], 1)
    null_total = max(s["null_total"], 1)
    print("\n=== Map Redetect Summary ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g} T={T} r={resolved_r}",
          flush=True)
    print(f"  entropy model: {s['entropy_model']} ({s['entropy_trace_source']})",
          flush=True)
    print(f"  map    : TPR {s['map_tp']}/{s['wm_total']} "
          f"({s['map_tp']/wm_total:.1%})   FPR {s['map_fp']}/{s['null_total']} "
          f"({s['map_fp']/null_total:.1%})", flush=True)
    print(f"  entropy: TPR {s['entropy_tp']}/{s['wm_total']} "
          f"({s['entropy_tp']/wm_total:.1%})   FPR {s['entropy_fp']}/"
          f"{s['null_total']} ({s['entropy_fp']/null_total:.1%})", flush=True)
    if s["naive_tp"] is None:
        print("  naive  : skipped", flush=True)
    else:
        print(f"  naive  : TPR {s['naive_tp']}/{s['wm_total']} "
              f"({s['naive_tp']/wm_total:.1%})   FPR {s['naive_fp']}/"
              f"{s['null_total']} ({s['naive_fp']/null_total:.1%})", flush=True)


# ---- re-detect a whole set of configs in one Modal session ------------------
# (n, t, eta) for every config we've generated so far. Both cache layouts.
REDETECT_CONFIGS = [
    (256, 3, 0.05), (400, 3, 0.05), (512, 3, 0.05), (1024, 3, 0.05),
    (400, 3, 0.20),
    (400, 5, 0.05), (512, 5, 0.05), (1024, 5, 0.05), (2048, 5, 0.05),
]


@app.local_entrypoint()
def redetect_all(fpr: float = 1e-3, num_prompts: int = 500):
    """Re-detect EVERY cached config with all weight kinds, one Modal session.
    Prints a compact map-vs-entropy line per config plus each full leaderboard."""
    from detectors import WEIGHT_KINDS

    summary = []
    for (n, t, eta) in REDETECT_CONFIGS:
        print(f"\n########## n={n} t={t} eta={eta} ##########", flush=True)
        results = detect_all_any.remote(n, t, eta, fpr, num_prompts)
        wm = [r for r in results if r["watermark"]]
        nw = [r for r in results if not r["watermark"]]
        nwm, nnw = max(len(wm), 1), max(len(nw), 1)
        rows = []
        for name in WEIGHT_KINDS:
            tp = sum(r[f"decision_{name}"] for r in wm)
            fp = sum(r[f"decision_{name}"] for r in nw)
            rows.append((tp, fp, name))
        for tp, fp, name in sorted(rows, reverse=True):
            flag = "  <- baseline" if name == "entropy" else ""
            print(f"  {name:8s}: TPR {tp}/{len(wm)} ({tp/nwm:.1%})   "
                  f"FPR {fp}/{len(nw)} ({fp/nnw:.1%}){flag}", flush=True)
        by = {name: (tp, fp) for tp, fp, name in rows}
        summary.append((n, t, eta, by["map"], by["entropy"], nwm, nnw))

    print("\n================ map vs entropy (all configs) ================",
          flush=True)
    for n, t, eta, (mtp, mfp), (etp, efp), nwm, nnw in summary:
        print(f"  n={n:>4} t={t} eta={eta:<4}: map {mtp/nwm:5.1%} vs entropy "
              f"{etp/nwm:5.1%}  (+{(mtp-etp)/nwm:.1%})   FPR map {mfp}/{nnw} "
              f"ent {efp}/{nnw}", flush=True)
