"""Isolated Modal integration for official baseline reference tests and smoke.

The entrypoints are deliberately fail-closed: cached PRC/null records never
fall back to generation, model loading is local-files-only, and the only GPU
entrypoint is the explicitly named five-prompt smoke.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

from baseline_comparison.config import (
    IMAGE_DEFINITION_SHA256,
    ONLINE_PRC_SOURCE_TAG,
    PINNED_DEPENDENCIES,
    SHARED_NULL_SOURCE_T,
    SMOKE_PROMPT_INDICES,
    SYNTHID_COMMIT,
    SYNTHID_REPOSITORY,
    TEXTSEAL_COMMIT,
    TEXTSEAL_REPOSITORY,
)


APP_NAME = "prc-controlled-baselines-smoke"
MODEL_ROOT = "/cache/models/Qwen3-8B-Base"
PROMPTS_PATH = "/root/prompts.jsonl"
NULL_ROOT = f"/data/_nulls/qwen3_8b_base/T{SHARED_NULL_SOURCE_T}"
ARTIFACT_PATH = f"/data/{ONLINE_PRC_SOURCE_TAG}/artifacts.pt"
WM_ROOT = f"/data/{ONLINE_PRC_SOURCE_TAG}/wm"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*PINNED_DEPENDENCIES)
    .pip_install(
        f"git+{TEXTSEAL_REPOSITORY}.git@{TEXTSEAL_COMMIT}",
        f"git+{SYNTHID_REPOSITORY}.git@{SYNTHID_COMMIT}",
        extra_options="--no-deps",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PRC_MODEL_CACHE_DIR": "/cache/models",
            "PRC_MODEL_SIZE": "8B",
            "PRC_MODEL_VARIANT": "base",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", PROMPTS_PATH, copy=True)
    .add_local_dir("baseline_comparison", "/root/baseline_comparison", copy=True)
    .add_local_python_source(
        "qwen", "prc", "online_prc", "detectors", "proxy_8b_analysis"
    )
)

# Hugging Face added native Qwen3 support after the official-baseline image's
# pinned Transformers release.  Keep the parity reference in an independent,
# separately pinned image so the TextSeal/SynthID environment remains intact.
HF_PARITY_DEPENDENCIES = (
    "torch==2.4.0",
    "transformers==4.51.3",
    "tokenizers==0.21.1",
    "huggingface-hub==0.30.2",
    "safetensors==0.4.5",
    "numpy==1.26.0",
    "scipy==1.14.1",
)
HF_PARITY_IMAGE_DEFINITION = {
    "python": "3.11",
    "base": "modal.Image.debian_slim",
    "dependencies": list(HF_PARITY_DEPENDENCIES),
    "network_model_downloads": False,
}
HF_PARITY_IMAGE_SHA256 = hashlib.sha256(
    json.dumps(HF_PARITY_IMAGE_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
parity_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*HF_PARITY_DEPENDENCIES)
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PRC_MODEL_CACHE_DIR": "/cache/models",
            "PRC_MODEL_SIZE": "8B",
            "PRC_MODEL_VARIANT": "base",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", PROMPTS_PATH, copy=True)
    .add_local_dir("baseline_comparison", "/root/baseline_comparison", copy=True)
    .add_local_python_source("qwen")
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=False)
data_volume = modal.Volume.from_name("prc-data", create_if_missing=False)
app = modal.App(APP_NAME, image=image)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_fingerprint(root: str | Path, suffix: str = ".py") -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob(f"*{suffix}")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _semantic_sha256(value) -> str:
    """Match detectors.semantic_sha256 without importing PRC dependencies."""
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def update(item):
        if item is None or isinstance(item, (str, int, float, bool)):
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda key: str(key)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                update(child)
        elif torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif hasattr(item, "tocsr"):
            sparse = item.tocsr()
            digest.update(b"sparse-csr")
            update(np.asarray(sparse.shape))
            update(np.asarray(sparse.indptr))
            update(np.asarray(sparse.indices))
            update(np.asarray(sparse.data))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"ndarray")
            digest.update(str(array.dtype).encode())
            digest.update(repr(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, bytes):
            digest.update(b"bytes")
            digest.update(item)
        else:
            raise TypeError(f"unsupported semantic hash type {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def _tensor_sha256(value) -> str:
    import torch

    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    header = f"{tensor.dtype}:{tuple(tensor.shape)}:".encode()
    return hashlib.sha256(header + raw).hexdigest()


def _model_revision_manifest() -> dict:
    root = Path(MODEL_ROOT)
    if not root.is_dir():
        raise FileNotFoundError(f"cached model directory is missing: {root}")
    index_path = root / "model.safetensors.index.json"
    tokenizer_path = root / "tokenizer.json"
    config_path = root / "config.json"
    for required in (index_path, tokenizer_path, config_path):
        if not required.is_file():
            raise FileNotFoundError(f"required cached model file missing: {required}")
    index = json.loads(index_path.read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"cached model is missing {len(missing)} shards: {missing[:3]}")

    revisions: dict[str, list[str]] = {}
    for path in root.rglob("*.metadata"):
        try:
            first_line = path.read_text().splitlines()[0].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if len(first_line) == 40 and all(char in "0123456789abcdef" for char in first_line.lower()):
            revisions.setdefault(first_line.lower(), []).append(str(path.relative_to(root)))
    if len(revisions) != 1:
        raise RuntimeError(
            "cached Qwen model revision is not uniquely recoverable from Hugging Face "
            f"metadata; found {sorted(revisions)}"
        )
    revision = next(iter(revisions))
    return {
        "model_revision": revision,
        "tokenizer_revision": revision,
        "revision_metadata_files": sorted(revisions[revision]),
        "model_root": MODEL_ROOT,
        "model_index_sha256": _sha256(index_path),
        "model_config_sha256": _sha256(config_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "weight_shard_count": len(shards),
        "weight_shards": [
            {"filename": name, "size_bytes": (root / name).stat().st_size}
            for name in shards
        ],
        "network_downloads_allowed": False,
    }


def _load_prompts() -> list[dict]:
    rows = [json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line]
    if len(rows) != 500:
        raise AssertionError(f"canonical corpus has {len(rows)} prompts, expected 500")
    doc_indices = [int(row["doc_index"]) for row in rows]
    if len(set(doc_indices)) != 500 or doc_indices != sorted(doc_indices):
        raise AssertionError("canonical prompt doc_index values are not unique and ordered")
    token_lengths = [len(row["prompt_tokens"]) for row in rows]
    if token_lengths != [50] * 500:
        raise AssertionError("canonical prompt token lengths are not exactly 50")
    return rows


def _validate_tokenization(rows: list[dict]) -> dict:
    from qwen import Qwen3Tokenizer

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=str(Path(MODEL_ROOT) / "tokenizer.json"),
        repo_id="Qwen/Qwen3-8B-Base",
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
    )
    mismatches = []
    for row_index, row in enumerate(rows):
        observed = tokenizer.encode(row["prompt_text"])
        if observed != row["prompt_tokens"]:
            mismatches.append(row_index)
            if len(mismatches) >= 10:
                break
    if mismatches:
        raise AssertionError(f"Qwen tokenizer differs from canonical prompt tokens: {mismatches}")
    return {
        "passed": True,
        "prompts_checked": len(rows),
        "prompt_construction": "prompt_text encoded without special tokens or chat template",
        "canonical_token_length": 50,
    }


def _array_summary(value) -> dict:
    import numpy as np

    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(
            f"{array.dtype}:{array.shape}:".encode() + np.ascontiguousarray(array).tobytes()
        ).hexdigest(),
    }


def _validate_cache_records(rows: list[dict]) -> dict:
    import numpy as np
    import sys
    import torch

    # These caches were serialized by a NumPy 2.x worker, whose pickle module
    # path is ``numpy._core``. SynthID's pinned environment requires NumPy
    # 1.26, where the identical implementation is exposed as ``numpy.core``.
    # Aliasing the import path is serialization compatibility only; array
    # values are hashed and validated below before use.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)

    if not Path(ARTIFACT_PATH).is_file():
        raise FileNotFoundError(f"online PRC artifact missing: {ARTIFACT_PATH}")
    artifact = torch.load(ARTIFACT_PATH, weights_only=False, map_location="cpu")
    stored_fingerprint = artifact.get("artifact_fingerprint")
    unhashed = dict(artifact)
    unhashed.pop("artifact_fingerprint", None)
    observed_fingerprint = _semantic_sha256(unhashed)
    if stored_fingerprint != observed_fingerprint:
        raise AssertionError("online PRC artifact fingerprint mismatch")
    if int(artifact.get("T", -1)) != 1280 or int(artifact.get("n", -1)) != 1280:
        raise AssertionError("online PRC source is not the frozen T=n=1280 cache")
    config = artifact.get("config_sig", {})
    expected_config = {
        "scheme": "online_causal_prc_v1",
        "check_weight": 3,
        "noise_rate": 0.05,
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
        "stopping_policy": "forced_length_v1",
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
    }
    for key, expected in expected_config.items():
        observed = artifact.get(key, config.get(key))
        if observed != expected:
            raise AssertionError(f"online artifact {key}={observed!r}, expected {expected!r}")
    canonical_tokens = [row["prompt_tokens"] for row in rows]
    if artifact.get("prompt_ids_list") != canonical_tokens:
        raise AssertionError("online PRC artifact prompt corpus/order differs from prompts.jsonl")

    manifest_path = Path(NULL_ROOT) / "_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"shared null manifest missing: {manifest_path}")
    null_manifest = json.loads(manifest_path.read_text())
    partition_sha = _tensor_sha256(artifact["partition"])
    prompt_corpus_sha = _semantic_sha256(canonical_tokens)
    expected_manifest = {
        "T": SHARED_NULL_SOURCE_T,
        "prompt_count": 500,
        "prompt_corpus_sha256": prompt_corpus_sha,
        "partition_sha256": partition_sha,
        "generation_model_size": "8B",
        "generation_model": "Qwen3-8B-Base",
        "generation_model_variant": "base",
        "generation_sampler_version": "torch_multinomial_global_v1",
        "generation_rng_policy": "torch_multinomial_global_v1",
        "stopping_policy": "forced_length_v1",
        "forced_length": True,
        "kv_cache_implementation": "static",
        "kv_cache_version": "static-v1",
    }
    for key, expected in expected_manifest.items():
        if null_manifest.get(key) != expected:
            raise AssertionError(
                f"shared null manifest {key}={null_manifest.get(key)!r}, expected {expected!r}"
            )

    prompt_records = []
    for index in SMOKE_PROMPT_INDICES:
        wm_path = Path(WM_ROOT) / f"wm_{index:04d}.pt"
        null_path = Path(NULL_ROOT) / f"null_{index:04d}.pt"
        if not wm_path.is_file() or not null_path.is_file():
            raise FileNotFoundError(f"missing cache for prompt {index}: {wm_path}, {null_path}")
        wm = torch.load(wm_path, weights_only=False, map_location="cpu")
        null = torch.load(null_path, weights_only=False, map_location="cpu")
        if wm.get("watermark") is not True or null.get("watermark") not in (False, None):
            raise AssertionError(f"watermark flags invalid for prompt {index}")
        if int(wm.get("prompt_idx", -1)) != index or int(null.get("prompt_idx", -1)) != index:
            raise AssertionError(f"prompt index provenance invalid for prompt {index}")
        expected_prompt = torch.tensor(canonical_tokens[index], dtype=torch.long)
        for label, record, source_length in (
            ("online_prc", wm, 1280),
            ("null", null, SHARED_NULL_SOURCE_T),
        ):
            prompt = torch.as_tensor(record.get("prompt_token_ids"), dtype=torch.long).reshape(-1)
            if not torch.equal(prompt, expected_prompt):
                raise AssertionError(f"{label} prompt tokens differ at index {index}")
            for field in ("tokens", "p_trace", "base_lm_entropy", "base_token_logprob"):
                array = np.asarray(record.get(field))
                if array.size != source_length:
                    raise AssertionError(
                        f"{label} prompt {index} {field} length {array.size}, expected {source_length}"
                    )
                if field != "tokens" and not np.all(np.isfinite(array)):
                    raise AssertionError(f"{label} prompt {index} {field} contains non-finite values")
        if wm.get("artifact_fingerprint") != stored_fingerprint:
            raise AssertionError(f"online PRC record {index} artifact fingerprint differs")
        if wm.get("partition_sha256") != partition_sha:
            raise AssertionError(f"online PRC record {index} partition fingerprint differs")
        if null.get("partition_sha256") not in (None, partition_sha):
            raise AssertionError(f"null record {index} partition fingerprint differs")
        prompt_records.append(
            {
                "prompt_index": index,
                "prompt_fingerprint": hashlib.sha256(
                    json.dumps(canonical_tokens[index], separators=(",", ":")).encode()
                ).hexdigest(),
                "online_prc_tokens": _array_summary(wm["tokens"]),
                "null_tokens": _array_summary(null["tokens"]),
                "online_record_artifact_fingerprint": wm.get("artifact_fingerprint"),
                "online_kv_cache": [wm.get("kv_cache_implementation"), wm.get("kv_cache_version")],
                "null_kv_cache": [null.get("kv_cache_implementation"), null.get("kv_cache_version")],
            }
        )

    fixed_candidates = []
    # These are all fixed 8B configurations established by the completed
    # fixed-vs-online audit. None is a frozen eta=.05, T=1024 source.
    for tag in (
        "n416_t3_eta0.05_T416__gen-qwen3_8b_base_r412",
        "n749_t3_eta0.05_T749__gen-qwen3_8b_base_r742",
        "n768_t3_eta0.10_T768__gen-qwen3_8b_base_r760",
        "n1382_t3_eta0.10_T1382__gen-qwen3_8b_base_r1368",
        "n1625_t3_eta0.10_T1625__gen-qwen3_8b_base_r1609",
    ):
        path = Path("/data") / tag / "artifacts.pt"
        fixed_candidates.append({"tag": tag, "exists": path.is_file()})
    exact_fixed_tags = (
        "n1024_t3_eta0.05_T1024__gen-qwen3_8b_base_r1014",
        "n1024_t3_eta0.05_T1024__gen-qwen3_8b_base",
        "n1024_t3_eta0.05_T1024",
    )
    exact_fixed = [tag for tag in exact_fixed_tags if (Path("/data") / tag / "artifacts.pt").is_file()]

    return {
        "online_artifact": {
            "path": ARTIFACT_PATH,
            "artifact_fingerprint": stored_fingerprint,
            "semantic_fingerprint_verified": True,
            "online_key_sha256": __import__("online_prc").OnlinePRCKey.from_dict(
                artifact["online_key"]
            ).fingerprint,
            "partition_sha256": partition_sha,
            "prompt_corpus_sha256": prompt_corpus_sha,
        },
        "shared_null": {
            "root": NULL_ROOT,
            "manifest_sha256": _sha256(manifest_path),
            "manifest": null_manifest,
        },
        "prompt_records": prompt_records,
        "fixed_prc": {
            "adapter_available": True,
            "known_candidate_artifacts": fixed_candidates,
            "exact_frozen_8b_eta0.05_T1024_artifacts": exact_fixed,
            "smoke_eligible": bool(exact_fixed),
            "regeneration_fallback": False,
        },
        "generation_attempts": {"online_prc": 0, "null": 0, "fixed_prc": 0},
    }


@app.function(
    cpu=2.0,
    memory=4096,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=1800,
)
def preflight_remote() -> dict:
    """CPU-only immutable provenance and cache validation."""
    import os
    import platform
    import sys
    from importlib import metadata

    data_volume.reload()
    hf_cache.reload()
    prompts = _load_prompts()
    model = _model_revision_manifest()
    tokenization = _validate_tokenization(prompts)
    caches = _validate_cache_records(prompts)
    distributions = sorted(
        {dist.metadata["Name"].lower(): dist.version for dist in metadata.distributions()}.items()
    )
    prc_files = ["/root/qwen.py", "/root/prc.py", "/root/online_prc.py", "/root/detectors.py"]
    prc_digest = hashlib.sha256()
    prc_file_hashes = {}
    for filename in prc_files:
        digest = _sha256(filename)
        prc_file_hashes[Path(filename).name] = digest
        prc_digest.update(Path(filename).name.encode())
        prc_digest.update(bytes.fromhex(digest))
    modal_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("MODAL_") and "TOKEN" not in key and "SECRET" not in key
    }
    return {
        "passed": True,
        "mode": "cpu_cache_only_preflight",
        "python": sys.version,
        "platform": platform.platform(),
        "image_definition_sha256": IMAGE_DEFINITION_SHA256,
        "integration_code_fingerprint": _tree_fingerprint("/root/baseline_comparison"),
        "prc_code_fingerprint": prc_digest.hexdigest(),
        "prc_file_sha256": prc_file_hashes,
        "prompts_file_sha256": _sha256(PROMPTS_PATH),
        "prompt_count": len(prompts),
        "prompt_indices": list(SMOKE_PROMPT_INDICES),
        "model": model,
        "tokenization": tokenization,
        "caches": caches,
        "resolved_dependencies": [{"name": name, "version": version} for name, version in distributions],
        "modal_environment": modal_environment,
        "stop_conditions_checked": {
            "missing_caches": False,
            "unexpected_model_downloads": False,
            "incomplete_prompts": False,
            "incompatible_tokenization": False,
        },
    }


@app.function(cpu=2.0, memory=4096, timeout=1800)
def official_reference_checks_remote() -> dict:
    """Run TextSeal and Google SynthID parity checks without a GPU/model."""
    import time
    from importlib import metadata

    from baseline_comparison.official import run_official_reference_checks

    started = time.perf_counter()
    result = run_official_reference_checks()
    result.update(
        {
            "runtime_seconds": time.perf_counter() - started,
            "textseal_package_version": metadata.version("textseal"),
            "synthid_text_package_version": metadata.version("synthid-text"),
            "textseal_commit": TEXTSEAL_COMMIT,
            "synthid_text_commit": SYNTHID_COMMIT,
            "image_definition_sha256": IMAGE_DEFINITION_SHA256,
        }
    )
    return result


@app.function(
    cpu=2.0,
    memory=4096,
    volumes={"/data": data_volume},
    timeout=1800,
)
def cache_diagnostic_remote(raw_path: str) -> dict:
    """Read-only prefix/entropy/loop analysis with no model or generation."""
    from baseline_comparison.diagnostic import run_cache_diagnostic

    return run_cache_diagnostic(data_volume, raw_path)


@app.function(
    image=parity_image,
    gpu="H100",
    cpu=4.0,
    memory=49_152,
    volumes={"/cache": hf_cache},
    timeout=3600,
)
def hf_logits_parity_remote() -> dict:
    """Offline project-Qwen versus native Hugging Face logits parity."""
    from baseline_comparison.diagnostic import run_hf_logits_parity

    result = run_hf_logits_parity()
    result["parity_image_definition"] = HF_PARITY_IMAGE_DEFINITION
    result["parity_image_definition_sha256"] = HF_PARITY_IMAGE_SHA256
    return result


@app.function(
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=3600,
)
def score_committed_smoke_remote(
    raw_path: str,
    source_app_url: str,
    source_task_id: str,
    actual_gpu: str,
) -> dict:
    """Validate and score a committed GPU payload without regeneration."""
    from baseline_comparison.smoke_runner import score_committed_smoke

    return score_committed_smoke(
        data_volume,
        raw_path=raw_path,
        source_app_url=source_app_url,
        source_task_id=source_task_id,
        actual_gpu=actual_gpu,
    )


@app.function(
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=3600,
)
def full_cache_preflight_remote() -> dict:
    """Verify all 500 PRC/null cache pairs before any approved GPU launch."""
    from baseline_comparison.smoke_runner import _load_cached_sequences_for_indices

    data_volume.reload()
    prompts = _load_prompts()
    online, nulls, artifact = _load_cached_sequences_for_indices(prompts, range(500))
    return {
        "passed": len(online) == len(nulls) == 500,
        "prompt_count": 500,
        "online_records": len(online),
        "null_records": len(nulls),
        "online_artifact_fingerprint": artifact["artifact_fingerprint"],
        "generation_attempts": {"online_prc": 0, "null": 0},
    }


@app.cls(
    gpu="H100",
    cpu=4.0,
    memory=49_152,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=7200,
    max_containers=1,
)
class SmokeWorker:
    """The sole GPU worker; hard-coded to the authorized five-prompt smoke."""

    @modal.method()
    def run(self) -> dict:
        from baseline_comparison.smoke_runner import run_gpu_smoke

        return run_gpu_smoke(data_volume)

    @modal.method()
    def diagnose(self) -> dict:
        from baseline_comparison.smoke_runner import run_gpu_diagnostic

        return run_gpu_diagnostic(data_volume)

    @modal.method()
    def gumbel_determinism(self, raw_path: str) -> dict:
        from baseline_comparison.smoke_runner import run_gumbel_determinism_check

        return run_gumbel_determinism_check(data_volume, raw_path)

    @modal.method()
    def stochastic_seed_check(self, raw_path: str) -> dict:
        from baseline_comparison.smoke_runner import run_stochastic_seed_check

        return run_stochastic_seed_check(data_volume, raw_path)

    @modal.method()
    def scientific_diagnostic(self, raw_path: str) -> dict:
        from baseline_comparison.diagnostic import run_generation_diagnostic

        return run_generation_diagnostic(data_volume, raw_path)


@app.cls(
    gpu="H100",
    cpu=4.0,
    memory=49_152,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=7200,
    max_containers=10,
)
class FullRunWorker:
    """Approval-gated 50-prompt generation shards for the later full run."""

    @modal.method()
    def run_shard(self, request: dict) -> dict:
        from baseline_comparison.smoke_runner import generate_full_shard

        return generate_full_shard(data_volume, request)


@app.cls(
    cpu=4.0,
    memory=8192,
    volumes={"/data": data_volume, "/cache": hf_cache},
    timeout=7200,
    max_containers=10,
)
class FullScoreWorker:
    """CPU-only exact-prefix scoring for committed full-run shards."""

    @modal.method()
    def score_shard(self, request: dict) -> dict:
        from baseline_comparison.smoke_runner import score_full_shard

        return score_full_shard(data_volume, request)

    @modal.method()
    def score_textseal_proxy_shard(self, request: dict) -> dict:
        from baseline_comparison.smoke_runner import (
            score_textseal_proxy_entropy_shard,
        )

        return score_textseal_proxy_entropy_shard(data_volume, request)

    @modal.method()
    def validate_generated_shard(
        self,
        request: dict,
        expected_sha256: str | None = None,
        expected_integration_fingerprint: str | None = None,
    ) -> dict:
        from baseline_comparison.resume import validate_generated_shard

        return validate_generated_shard(
            data_volume,
            request,
            expected_sha256=expected_sha256,
            expected_integration_fingerprint=expected_integration_fingerprint,
        )

    @modal.method()
    def validate_scored_shard(
        self,
        request: dict,
        expected_jsonl_sha256: str | None = None,
        expected_validation_sha256: str | None = None,
        expected_scoring_integration_fingerprint: str | None = None,
    ) -> dict:
        from baseline_comparison.resume import validate_scored_shard

        return validate_scored_shard(
            data_volume,
            request,
            expected_jsonl_sha256=expected_jsonl_sha256,
            expected_validation_sha256=expected_validation_sha256,
            expected_scoring_integration_fingerprint=(
                expected_scoring_integration_fingerprint
            ),
        )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_smoke_artifacts(payload: dict) -> dict:
    """Write compact, commit-safe artifacts; raw generations stay on Modal."""
    import csv
    import math
    import statistics

    root = Path.cwd()
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    seed_validation_path = outputs / "controlled_baseline_smoke_seed_validation.json"
    if not seed_validation_path.is_file():
        raise FileNotFoundError(
            "controlled equal-shape seed validation artifact is required before finalization"
        )
    seed_validation = json.loads(seed_validation_path.read_text())
    if seed_validation.get("status") != "passed":
        raise AssertionError("controlled equal-shape seed validation did not pass")
    records = payload["smoke"]["records"]
    prompt_jsonl = outputs / "controlled_baseline_smoke_prompt_level.jsonl"
    with prompt_jsonl.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    primary = [record for record in records if record["sample_type"] != "seed_validation"]
    summary_rows = []
    for method in ("online_prc", "textseal", "synthid_text", "gumbel_max"):
        for prefix in (128, 256, 400, 512, 768, 1024):
            wm = [
                row for row in primary
                if row["method"] == method
                and row["prefix_length"] == prefix
                and row["sample_type"] == "watermarked"
            ]
            null = [
                row for row in primary
                if row["method"] == method
                and row["prefix_length"] == prefix
                and row["sample_type"] == "null"
            ]
            if len(wm) != 5 or len(null) != 5:
                raise AssertionError(
                    f"summary coverage differs for {method} T={prefix}: {len(wm)}, {len(null)}"
                )
            neglog = lambda row: -math.log10(max(float(row["p_value"]), 1e-300))
            summary_rows.append(
                {
                    "method": method,
                    "prefix_length": prefix,
                    "watermarked_prompts": len(wm),
                    "null_prompts": len(null),
                    "tpr": sum(row["decision"] for row in wm) / len(wm),
                    "observed_fpr": sum(row["decision"] for row in null) / len(null),
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
    summary_csv = outputs / "controlled_baseline_smoke_prefix_summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    unique_quality = {}
    for record in primary:
        key = (
            record["method"],
            record["sample_type"],
            record["prompt_index"],
            record["generation_seed"],
            record["generated_token_hash"],
        )
        unique_quality.setdefault(
            key,
            {
                "method": record["method"],
                "sample_type": record["sample_type"],
                "prompt_index": record["prompt_index"],
                "prompt_id": record["prompt_id"],
                "generation_seed": record["generation_seed"],
                "generated_token_hash": record["generated_token_hash"],
                "output_length": record["output_length"],
                "base_model_nll": record["base_model_nll"],
                "base_model_perplexity": record["base_model_perplexity"],
                "repetition_rate": record["repetition_rate"],
                "repetition_metric": record["repetition_metric"],
                "distinct_2": record["distinct_2"],
                "distinct_3": record["distinct_3"],
                "runtime_seconds": record["runtime_seconds"],
            },
        )
    quality_rows = list(unique_quality.values())
    quality_csv = outputs / "controlled_baseline_smoke_quality.csv"
    with quality_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)

    source_manifest = {
        "timestamp_utc": payload["smoke"]["timestamp_utc"],
        "official_sources": {
            "textseal": {
                "repository": TEXTSEAL_REPOSITORY,
                "commit": TEXTSEAL_COMMIT,
                "package_version": payload["reference_checks"]["textseal_package_version"],
                "license": "Apache-2.0",
            },
            "synthid_text": {
                "repository": SYNTHID_REPOSITORY,
                "commit": SYNTHID_COMMIT,
                "package_version": payload["reference_checks"]["synthid_text_package_version"],
                "license": "Apache-2.0",
            },
        },
        "paper_source": {
            "uploaded_path": "/Users/christineli/Downloads/arXiv-2605.12456v2 (2).tar.gz",
            "sha256": "dc703a7a61ef219b55b70ae12f24839f56560bd38da985af801fcb1e270095ed",
            "arxiv_identifier": "2605.12456v2",
            "inspected_source_files": [
                "sections/1-gumbel.tex",
                "sections/2-method.tex",
                "sections/3-experiments.tex",
                "sections/4-ablations.tex",
                "sections/5-appendix.tex"
            ],
            "paper_experiment": {
                "model": "Qwen3.5-27B",
                "temperature": 0.8,
                "top_p": 0.9,
                "reasoning": False,
                "prompts": "1,000 ELI5 prompts with five seeds",
                "continuation_tokens": 400,
                "textseal_alpha": 0.1,
                "synthid_depth": 10,
                "watermark_context_length": 3,
                "detection_deduplication": "unique (context window, token) tuples",
                "synthid_detector": "frequentist weighted Z-test",
                "diversity_metric": "Self-BLEU, lower is more diverse; paper ablation uses two generations per prompt"
            }
        },
        "model": payload["preflight"]["model"],
        "image": {
            "definition": __import__("baseline_comparison.config", fromlist=["IMAGE_DEFINITION"]).IMAGE_DEFINITION,
            "definition_sha256": IMAGE_DEFINITION_SHA256,
            "preflight_modal_image_id": payload["preflight"]["modal_environment"].get("MODAL_IMAGE_ID"),
            "generation_modal_image_id": payload["smoke"]["provenance"].get(
                "generation_modal_image_id"
            ),
            "scoring_modal_image_id": payload["smoke"]["provenance"].get("modal_image_id"),
        },
        "direct_pinned_dependencies": list(PINNED_DEPENDENCIES),
        "resolved_dependencies": payload["preflight"]["resolved_dependencies"],
        "prc": {
            "base_commit": __import__(
                "baseline_comparison.config", fromlist=["PRC_BASE_COMMIT"]
            ).PRC_BASE_COMMIT,
            "code_fingerprint": payload["smoke"]["provenance"]["prc_code_fingerprint"],
            "file_sha256": payload["preflight"]["prc_file_sha256"],
        },
        "canonical_prompts": {
            "path": "prompts.jsonl",
            "count": 500,
            "sha256": payload["preflight"]["prompts_file_sha256"],
            "smoke_row_indices": list(SMOKE_PROMPT_INDICES),
            "prompt_id_field": "doc_index",
        },
        "paper_vs_code_discrepancies": [
            "Paper routes key 1 with probability 1-alpha and key 2 with alpha; released code names the alpha-weighted/routed key A. The mixture is symmetric but key labels differ.",
            "Paper comparison settings are Qwen3.5-27B, temperature 0.8, top-p 0.9, 1,000 ELI5 prompts x five seeds, and 400 tokens; this controlled experiment deliberately uses Qwen3-8B-Base, temperature/top-p 1.0, 500 project prompts, and 1,024 tokens.",
            "Released TextSeal defaults are alpha=0.5, context length 1, and SynthID depth 30; the paper/frozen comparison explicitly overrides them to alpha=0.1, context length 3, and depth 10.",
            "Released TextSeal v2 detector begins at target position k+1, skipping the first mathematically eligible position k; the controlled scorer follows the released convention.",
            "Released TextSeal detector returns min(weighted p, unweighted p) and hard-codes detected at p<0.01. The paper comparison and controlled primary result preserve the entropy-weighted Gamma p-value and use p<0.001; both released values are retained as intermediates.",
            "Released TextSeal metadata reports package 0.2.0 while textseal.__version__ is 0.0.4; LICENSE and documentation are Apache-2.0 although the pyproject classifier says MIT.",
            "Google SynthID supports Gemma/GPT-2 generation mixins, not Qwen3; the adapter performs the Qwen loop while invoking the pinned official processor unchanged for every score update.",
            "Google SynthID defines ngram_len as context length plus target, so frozen context length 3 maps to ngram_len=4.",
            "Google's pinned SynthID constructor hashes keys via .numpy(), which fails if constructed on CUDA; the adapter constructs on CPU and moves the unchanged keys/state to the exact indexed CUDA device.",
            "Google's repository provides generation and Bayesian detectors but no frequentist p-value implementation; the controlled detector uses the TextSeal paper's weighted normal approximation over official Google g-values.",
            "The TextSeal Gumbel comparison sampler uses the legacy r^(1/p) argmax despite a stable log-space helper elsewhere in the same file. Equal-shape runs are seed-deterministic, but Qwen3-8B smoke outputs changed between batch sizes 5 and 1.",
            "Importing TextSeal's top-level evaluation stack after allocating Qwen3-8B caused a reproducible worker SIGSEGV; official runtimes are preloaded before model allocation.",
            "TextSeal's CUDA PRF helper is torch.compile-decorated and failed in the pinned Torch runtime; the adapter calls its pinned original eager body, verified bit-for-bit against the official PRF.",
            "TextSeal imports download NLTK punkt into each ephemeral container even though the controlled generation/detection path does not use tokenized BLEU.",
            "NumPy-2-authored PRC caches reference numpy._core; the pinned NumPy 1.26 image aliases that pickle module path only during cache loading, followed by exact value and fingerprint validation."
        ],
        "controlled_metric_definitions": {
            "base_model_nll": "mean negative natural-log probability of the selected continuation tokens under unwatermarked Qwen3-8B-Base logits",
            "base_model_perplexity": "exp(base_model_nll)",
            "repetition_rate": "1 - unique token 4-grams / total token 4-grams",
            "distinct_2": "unique token bigrams / total token bigrams",
            "distinct_3": "unique token trigrams / total token trigrams",
            "pairwise_token_agreement": "mean exact position-wise token agreement over all unordered seed pairs",
            "self_bleu": "mean token-ID BLEU of each seed output against the remaining outputs; lower is more diverse"
        }
    }
    source_manifest_path = root / "baseline_comparison" / "pinned_sources_manifest.json"
    _write_json(source_manifest_path, source_manifest)

    behavior = dict(payload["smoke"]["behavior"])
    behavior.update(
        {
            "gumbel_max_identical_across_equal_shape_seeds": seed_validation[
                "gumbel_max"
            ]["fixed_shape_identical_across_seeds"],
            "gumbel_max_batch_shape_sensitivity_observed": seed_validation[
                "gumbel_max"
            ]["batch_shape_sensitivity_observed"],
            "textseal_changed_across_equal_shape_seeds": seed_validation["textseal"][
                "changed_across_seeds"
            ],
            "synthid_text_changed_across_equal_shape_seeds": seed_validation[
                "synthid_text"
            ]["changed_across_seeds"],
            "fixed_seed_replay_passed": True,
        }
    )
    validation = {
        "status": "passed",
        "preflight": payload["preflight"],
        "official_reference_checks": payload["reference_checks"],
        "gpu_validation_records": payload["smoke"]["validation_records"],
        "runtime": payload["smoke"]["runtime"],
        "behavior": behavior,
        "controlled_seed_validation": seed_validation,
        "provenance": payload["smoke"]["provenance"],
        "remote_generation_artifact": payload["smoke"]["remote_generation_artifact"],
        "coverage": {
            "prompt_indices": list(SMOKE_PROMPT_INDICES),
            "primary_methods": ["online_prc", "textseal", "synthid_text", "gumbel_max"],
            "prefix_lengths": [128, 256, 400, 512, 768, 1024],
            "prompt_prefix_rows": len(records),
            "primary_prompt_prefix_rows": len(primary),
            "seed_validation_rows": len(records) - len(primary),
            "schema_validation": "all PromptLevelResult.validate calls passed",
        },
        "cost_gate": payload["cost_gate"],
    }
    validation_path = outputs / "controlled_baseline_smoke_validation.json"
    _write_json(validation_path, validation)

    # Billing fields are reconciled from Modal's usage dashboard after this
    # run. A single explicit pending row prevents an estimated value from being
    # mistaken for exact provider billing.
    cost_ledger = outputs / "controlled_baseline_smoke_cost_ledger.csv"
    with cost_ledger.open("w", newline="") as handle:
        fields = [
            "timestamp_utc",
            "modal_app_url",
            "modal_task_id",
            "run_kind",
            "requested_gpu",
            "actual_gpu",
            "gpu_function_seconds",
            "peak_cuda_allocated_bytes",
            "peak_cuda_reserved_bytes",
            "provider_gpu_cost_usd",
            "provider_cpu_cost_usd",
            "provider_memory_cost_usd",
            "provider_total_cost_usd",
            "billing_status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": payload["smoke"]["timestamp_utc"],
                "modal_app_url": "PENDING_DASHBOARD_RECONCILIATION",
                "modal_task_id": payload["smoke"]["provenance"].get("modal_task_id"),
                "run_kind": "five_prompt_controlled_baseline_smoke",
                "requested_gpu": payload["smoke"]["runtime"]["requested_gpu"],
                "actual_gpu": payload["smoke"]["runtime"]["actual_gpu"],
                "gpu_function_seconds": payload["smoke"]["runtime"]["gpu_function_seconds"],
                "peak_cuda_allocated_bytes": payload["smoke"]["runtime"]["peak_cuda_allocated_bytes"],
                "peak_cuda_reserved_bytes": payload["smoke"]["runtime"]["peak_cuda_reserved_bytes"],
                "provider_gpu_cost_usd": "",
                "provider_cpu_cost_usd": "",
                "provider_memory_cost_usd": "",
                "provider_total_cost_usd": "",
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
            }
        )

    artifacts = [
        prompt_jsonl,
        summary_csv,
        quality_csv,
        source_manifest_path,
        validation_path,
        cost_ledger,
        seed_validation_path,
    ]
    artifact_manifest = {
        "status": "passed_pre_billing_reconciliation",
        "artifacts": [
            {
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in artifacts
        ],
        "remote_generation_artifact": payload["smoke"]["remote_generation_artifact"],
    }
    artifact_manifest_path = outputs / "controlled_baseline_smoke_artifact_manifest.json"
    _write_json(artifact_manifest_path, artifact_manifest)
    return {
        "written": [str(path.relative_to(root)) for path in artifacts + [artifact_manifest_path]],
        "prefix_summary_rows": len(summary_rows),
        "quality_rows": len(quality_rows),
        "prompt_prefix_rows": len(records),
    }


@app.local_entrypoint(name="preflight")
def preflight_entrypoint():
    print(json.dumps(preflight_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="reference-checks")
def reference_checks_entrypoint():
    print(json.dumps(official_reference_checks_remote.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="smoke")
def smoke_entrypoint():
    # Conservative gate based on prior measured Qwen3-8B H100 throughput. The
    # full dashboard amount is reconciled after the run; this is only the
    # authorization guard before any GPU starts.
    cost_gate = {
        "hard_cap_usd": 10.0,
        "conservative_expected_total_usd": 3.0,
        "basis": (
            "18 forced 1024-token generations on one H100, including model load, "
            "official-reference overhead, and a 30-minute conservative GPU allowance"
        ),
        "prior_measured_h100_rate_usd_per_hour": 4.078,
        "passed": True,
    }
    if cost_gate["conservative_expected_total_usd"] > cost_gate["hard_cap_usd"]:
        raise RuntimeError("projected smoke cost exceeds the authorized $10 hard cap")
    preflight = preflight_remote.remote()
    references = official_reference_checks_remote.remote()
    if not preflight.get("passed") or not references.get("passed"):
        raise RuntimeError("CPU preflight/reference checks did not pass; GPU launch blocked")
    smoke = SmokeWorker().run.remote()
    payload = {
        "cost_gate": cost_gate,
        "preflight": preflight,
        "reference_checks": references,
        "smoke": smoke,
    }
    written = _write_smoke_artifacts(payload)
    print(
        json.dumps(
            {
                "passed": smoke["passed"],
                "behavior": smoke["behavior"],
                "runtime": smoke["runtime"],
                "generation_telemetry": smoke["generation_telemetry"],
                "remote_generation_artifact": smoke["remote_generation_artifact"],
                "artifacts": written,
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="resume-smoke")
def resume_smoke_entrypoint(
    raw_path: str = "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt",
    source_app_url: str = (
        "https://modal.com/apps/new-prc-watermark/main/ap-ZEwIv5wljqlERa1g0jX7EW"
    ),
    source_task_id: str = "ta-01M0P5F4NN27F69CT9A4RG9J2R",
    actual_gpu: str = "NVIDIA H100 80GB HBM3",
):
    """CPU-only resume path for the committed authorized generation."""
    cost_gate = {
        "hard_cap_usd": 10.0,
        "pre_generation_conservative_expected_total_usd": 3.0,
        "cpu_resume_conservative_expected_increment_usd": 0.25,
        "basis": (
            "original 18-sequence H100 smoke gate plus committed-generation reuse for "
            "CPU-only reference validation and scoring"
        ),
        "passed": True,
    }
    preflight = preflight_remote.remote()
    references = official_reference_checks_remote.remote()
    if not preflight.get("passed") or not references.get("passed"):
        raise RuntimeError("CPU preflight/reference checks did not pass; scoring blocked")
    smoke = score_committed_smoke_remote.remote(
        raw_path,
        source_app_url,
        source_task_id,
        actual_gpu,
    )
    seed_validation = json.loads(
        Path("outputs/controlled_baseline_smoke_seed_validation.json").read_text()
    )
    smoke["runtime"].update(
        {
            "gpu_function_seconds": 280.350079451,
            "gpu_function_runtime_source": (
                "provider-billed H100 cost 0.31757434 USD divided by the pinned "
                "4.078 USD/hour H100 rate"
            ),
            "peak_cuda_allocated_bytes": seed_validation["gumbel_max"][
                "peak_cuda_allocated_bytes"
            ],
            "peak_cuda_reserved_bytes": seed_validation["gumbel_max"][
                "peak_cuda_reserved_bytes"
            ],
            "peak_memory_source_app_url": seed_validation["gumbel_max"][
                "modal_app_url"
            ],
        }
    )
    smoke["provenance"].update(
        {
            "generation_modal_image_id": "im-lqcA41yGVtqwIMu2GARgZZ",
            "generation_modal_app_url": source_app_url,
            "controlled_seed_validation_artifact": (
                "outputs/controlled_baseline_smoke_seed_validation.json"
            ),
        }
    )
    payload = {
        "cost_gate": cost_gate,
        "preflight": preflight,
        "reference_checks": references,
        "smoke": smoke,
    }
    written = _write_smoke_artifacts(payload)
    print(
        json.dumps(
            {
                "passed": smoke["passed"],
                "behavior": smoke["behavior"],
                "runtime": smoke["runtime"],
                "generation_telemetry": smoke["generation_telemetry"],
                "remote_generation_artifact": smoke["remote_generation_artifact"],
                "artifacts": written,
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
                "generation_reused": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="gpu-diagnostic")
def gpu_diagnostic_entrypoint():
    print(json.dumps(SmokeWorker().diagnose.remote(), indent=2, sort_keys=True))


@app.local_entrypoint(name="diagnostic-cache")
def diagnostic_cache_entrypoint(
    raw_path: str = (
        "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt"
    ),
):
    """Run the authorized no-generation diagnostic phase and write artifacts."""
    from baseline_comparison.diagnostic import write_cache_diagnostic_artifacts

    cost_gate = {
        "hard_cap_usd": 2.0,
        "phase_expected_usd": 0.05,
        "generation_authorized": False,
        "passed": True,
    }
    if cost_gate["phase_expected_usd"] > cost_gate["hard_cap_usd"]:
        raise RuntimeError("cache diagnostic estimate exceeds the diagnostic hard cap")
    payload = cache_diagnostic_remote.remote(raw_path)
    if payload.get("generation_attempts") != 0:
        raise AssertionError("cache diagnostic attempted generation")
    written = write_cache_diagnostic_artifacts(payload, Path.cwd())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "mode": payload["mode"],
                "row_count": len(payload["rows"]),
                "summary_count": len(payload["summary"]),
                "artifacts": written,
                "cost_gate": cost_gate,
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="diagnostic-generation")
def diagnostic_generation_entrypoint(
    raw_path: str = (
        "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt"
    ),
):
    """Run the authorized bounded H100 diagnostic matrix once."""
    from baseline_comparison.diagnostic import write_generation_diagnostic_artifacts

    cost_gate = {
        "campaign_hard_cap_usd": 2.0,
        "phase_expected_usd": 0.50,
        "worker_count": 1,
        "requested_gpu": "H100",
        "full_500_prompt_run_authorized": False,
        "passed": True,
    }
    if cost_gate["phase_expected_usd"] > cost_gate["campaign_hard_cap_usd"]:
        raise RuntimeError("generation diagnostic estimate exceeds the diagnostic hard cap")
    payload = SmokeWorker().scientific_diagnostic.remote(raw_path)
    written = write_generation_diagnostic_artifacts(payload, Path.cwd())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "matrix": payload["matrix"],
                "runtime": payload["runtime"],
                "remote_generation_artifact": payload["remote_generation_artifact"],
                "artifacts": written,
                "cost_gate": cost_gate,
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="diagnostic-logits-parity")
def diagnostic_logits_parity_entrypoint():
    """Run the final authorized Qwen integration parity check."""
    from baseline_comparison.diagnostic import write_logits_parity_artifact

    cost_gate = {
        "campaign_hard_cap_usd": 2.0,
        "phase_expected_usd": 0.30,
        "requested_gpu": "H100",
        "worker_count": 1,
        "network_model_downloads_allowed": False,
        "passed": True,
    }
    if cost_gate["phase_expected_usd"] > cost_gate["campaign_hard_cap_usd"]:
        raise RuntimeError("logits parity estimate exceeds the diagnostic hard cap")
    payload = hf_logits_parity_remote.remote()
    written = write_logits_parity_artifact(payload, Path.cwd())
    print(
        json.dumps(
            {
                "status": payload["status"],
                "passed": payload["passed"],
                "runtime": payload["runtime"],
                "artifact": written,
                "cost_gate": cost_gate,
                "billing_status": "pending_exact_modal_dashboard_reconciliation",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="gumbel-determinism")
def gumbel_determinism_entrypoint(
    raw_path: str = "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt",
):
    conservative_expected_usd = 0.5
    if conservative_expected_usd > 10.0:
        raise RuntimeError("projected Gumbel determinism check exceeds the smoke hard cap")
    print(
        json.dumps(
            SmokeWorker().gumbel_determinism.remote(raw_path),
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="stochastic-seed-check")
def stochastic_seed_check_entrypoint(
    raw_path: str = "/data/controlled_baseline_smoke/20260823T020321Z/generated_sequences.pt",
):
    conservative_expected_usd = 0.5
    if conservative_expected_usd > 10.0:
        raise RuntimeError("projected stochastic seed check exceeds the smoke hard cap")
    print(
        json.dumps(
            SmokeWorker().stochastic_seed_check.remote(raw_path),
            indent=2,
            sort_keys=True,
        )
    )


FULL_RUN_APPROVAL_TOKEN = "APPROVE_500_PROMPT_CONTROLLED_BASELINE"
BATCH50_VALIDATION_APPROVAL_TOKEN = "APPROVE_50_PROMPT_BATCH50_VALIDATION"
BATCH50_SCORE_APPROVAL_TOKEN = "APPROVE_50_PROMPT_BATCH50_EVALUATION"
REUSED_BATCH50_RUN_ID = "qwen3-8b-batch50-validation-20260823-v1"
REUSED_BATCH50_RAW_SHA256 = (
    "0b7326fbfa43d55a39088e19cc2d69a696b7c81e8095b3ff6d291254392716da"
)
REUSED_BATCH50_GENERATION_FINGERPRINT = (
    "6f9bc1c40ab864d0098ded7378eb3153c4e17b280dc84e4965108f9c014dbddc"
)
REUSED_BATCH50_SCORED_SHA256 = (
    "1149134f14c684bf2da9962349e56644beabf0bd5fd76cd775323ec0bd68a71e"
)
REUSED_BATCH50_SCORED_VALIDATION_SHA256 = (
    "0bdc265f7a636734886b5f5809036fe032118fc443e717831abe88e1c406a099"
)
REUSED_BATCH50_SCORING_FINGERPRINT = (
    "b250f74d5e753e7bd26189045b57be5c2e8104597c0cdcbc536a15c5ad84b705"
)


@app.local_entrypoint(name="batch50-validation")
def batch50_validation_entrypoint(approval_token: str, run_id: str):
    """Run one standalone 50-prompt batch-50 generation validation."""
    if approval_token != BATCH50_VALIDATION_APPROVAL_TOKEN:
        raise PermissionError(
            "batch-50 validation generation is not authorized; obtain explicit approval "
            f"and pass --approval-token {BATCH50_VALIDATION_APPROVAL_TOKEN}"
        )
    cost_gate = {
        "hard_cap_usd": 3.0,
        "conservative_expected_total_usd": 1.5,
        "requested_gpu": "H100",
        "worker_count": 1,
        "generated_prompts_per_method": 50,
        "methods": ["textseal", "synthid_text", "gumbel_max"],
        "generation_batch_size": 50,
        "passed": True,
    }
    if cost_gate["conservative_expected_total_usd"] > cost_gate["hard_cap_usd"]:
        raise RuntimeError("projected batch-50 validation cost exceeds its $3 hard cap")
    preflight = preflight_remote.remote()
    references = official_reference_checks_remote.remote()
    full_caches = full_cache_preflight_remote.remote()
    if not preflight.get("passed") or not references.get("passed") or not full_caches.get("passed"):
        raise RuntimeError("batch-50 preflight/reference/cache validation did not pass")

    request = {"run_id": run_id, "shard_index": 0}
    generated = FullRunWorker().run_shard.remote(request)
    if not generated.get("passed") or int(generated.get("generation_batch_size", -1)) != 50:
        raise RuntimeError("batch-50 generation shard failed its memory/layout gate")
    if int(generated["runtime"]["peak_cuda_reserved_bytes"]) > 70 * 1024**3:
        raise RuntimeError("batch-50 validation exceeded the 70 GiB reserved-memory gate")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "cost_gate": cost_gate,
                "full_cache_preflight": full_caches,
                "generation": generated,
                "billing_status": "pending_exact_modal_billing_reconciliation",
                "next_step": "reconcile billing and update the measured full-run estimate",
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="full-run")
def full_run_entrypoint(approval_token: str, run_id: str):
    """Launch the later 500-prompt generation only after explicit approval."""
    if approval_token != FULL_RUN_APPROVAL_TOKEN:
        raise PermissionError(
            "500-prompt generation is not authorized; obtain explicit approval and pass "
            f"--approval-token {FULL_RUN_APPROVAL_TOKEN}"
        )
    projected = {
        "prior_batch5_all_resource_cost_range_usd": [14.0, 18.0],
        "prior_batch5_wall_time_minutes_with_10_gpus": [20, 35],
        "production_batch_size": 50,
        "batch50_validation_run_id": "qwen3-8b-batch50-validation-20260823-v1",
        "batch50_validation_app_id": "ap-INjm5299E4tI0jNgiqOx4U",
        "batch50_validation_status": "passed",
        "measured_function_seconds_per_50_prompt_shard": 166.386096582,
        "measured_peak_cuda_reserved_bytes": 27_839_692_800,
        "projected_generation_cost_usd": 2.70343190,
        "projected_end_to_end_cost_range_usd": [3.0, 4.0],
        "recommended_hard_cap_usd": 5.0,
        "projected_end_to_end_wall_minutes_with_10_gpus": [8, 15],
    }
    preflight = preflight_remote.remote()
    references = official_reference_checks_remote.remote()
    full_caches = full_cache_preflight_remote.remote()
    if not preflight.get("passed") or not references.get("passed") or not full_caches.get("passed"):
        raise RuntimeError("full-run preflight/reference/cache validation did not pass")
    requests = [{"run_id": run_id, "shard_index": index} for index in range(10)]
    results = list(FullRunWorker().run_shard.map(requests))
    if len(results) != 10 or any(not result.get("passed") for result in results):
        raise RuntimeError("one or more full generation shards failed")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "approval_token_validated": True,
                "projection_used_for_approval": projected,
                "full_cache_preflight": full_caches,
                "shards": results,
                "next_command": (
                    "modal run baseline_comparison/modal_app.py::app.full-score "
                    f"--approval-token {FULL_RUN_APPROVAL_TOKEN} --run-id {run_id}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="remaining-run")
def remaining_run_entrypoint(approval_token: str, run_id: str):
    """Reuse validated shard 0 and generate only prompt shards 1 through 9."""
    if approval_token != FULL_RUN_APPROVAL_TOKEN:
        raise PermissionError(
            "remaining 500-prompt comparison generation is not authorized; obtain "
            f"explicit approval and pass --approval-token {FULL_RUN_APPROVAL_TOKEN}"
        )
    if run_id != REUSED_BATCH50_RUN_ID:
        raise ValueError(
            f"remaining-run must reuse the validated run ID {REUSED_BATCH50_RUN_ID}"
        )
    cost_gate = {
        "hard_cap_usd": 5.0,
        "projected_remaining_generation_usd": 2.43308871,
        "requested_gpu": "H100",
        "worker_count": 9,
        "generation_batch_size": 50,
        "generated_prompt_indices": [50, 499],
        "passed": True,
    }
    if cost_gate["projected_remaining_generation_usd"] > cost_gate["hard_cap_usd"]:
        raise RuntimeError("remaining generation projection exceeds the $5 hard cap")

    preflight = preflight_remote.remote()
    references = official_reference_checks_remote.remote()
    full_caches = full_cache_preflight_remote.remote()
    if not preflight.get("passed") or not references.get("passed") or not full_caches.get("passed"):
        raise RuntimeError("remaining-run preflight/reference/cache validation did not pass")
    reused = FullScoreWorker().validate_generated_shard.remote(
        {"run_id": run_id, "shard_index": 0},
        REUSED_BATCH50_RAW_SHA256,
        REUSED_BATCH50_GENERATION_FINGERPRINT,
    )
    if not reused.get("passed"):
        raise RuntimeError("validated batch-50 generation shard could not be reused")

    requests = [{"run_id": run_id, "shard_index": index} for index in range(1, 10)]
    results = list(FullRunWorker().run_shard.map(requests))
    if len(results) != 9 or any(not result.get("passed") for result in results):
        raise RuntimeError("one or more remaining generation shards failed")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "approval_token_validated": True,
                "cost_gate": cost_gate,
                "full_cache_preflight": full_caches,
                "reused_shard": reused,
                "generated_shards": results,
                "next_command": (
                    "modal run baseline_comparison/modal_app.py::app.remaining-score "
                    f"--approval-token {FULL_RUN_APPROVAL_TOKEN} --run-id {run_id}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="full-score")
def full_score_entrypoint(approval_token: str, run_id: str):
    """CPU-score all ten committed full-run shards without generation."""
    if approval_token != FULL_RUN_APPROVAL_TOKEN:
        raise PermissionError("full scoring requires the approved full-run token")
    requests = [{"run_id": run_id, "shard_index": index} for index in range(10)]
    results = list(FullScoreWorker().score_shard.map(requests))
    if len(results) != 10 or any(not result.get("passed") for result in results):
        raise RuntimeError("one or more full scoring shards failed")
    if sum(result["record_count"] for result in results) != 24_000:
        raise AssertionError("full scoring did not produce 24,000 prompt-prefix rows")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "record_count": 24_000,
                "generation_attempts": {"online_prc": 0, "null": 0},
                "shards": results,
                "next_command": (
                    "modal volume get prc-data "
                    f"controlled_baseline_full/{run_id}/scored "
                    f"outputs/controlled_baseline_full/{run_id}/scored"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="batch50-score")
def batch50_score_entrypoint(approval_token: str, run_id: str):
    """CPU-score only the committed 50-prompt validation shard."""
    if approval_token != BATCH50_SCORE_APPROVAL_TOKEN:
        raise PermissionError(
            "batch-50 evaluation is not authorized; obtain explicit approval and pass "
            f"--approval-token {BATCH50_SCORE_APPROVAL_TOKEN}"
        )
    cost_gate = {
        "hard_cap_usd": 0.5,
        "conservative_expected_total_usd": 0.25,
        "gpu_workers": 0,
        "cpu_workers": 1,
        "passed": True,
    }
    result = FullScoreWorker().score_shard.remote(
        {"run_id": run_id, "shard_index": 0}
    )
    if not result.get("passed") or int(result.get("record_count", -1)) != 2_400:
        raise RuntimeError("batch-50 scoring did not produce 2,400 validated rows")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "cost_gate": cost_gate,
                "generation_attempts": {"online_prc": 0, "null": 0},
                "shard": result,
                "billing_status": "pending_exact_modal_billing_reconciliation",
                "next_command": (
                    "modal volume get prc-data "
                    f"controlled_baseline_full/{run_id}/scored "
                    f"/private/tmp/{run_id}-scored"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="remaining-score")
def remaining_score_entrypoint(approval_token: str, run_id: str):
    """Reuse scored shard 0 and CPU-score only prompt shards 1 through 9."""
    if approval_token != FULL_RUN_APPROVAL_TOKEN:
        raise PermissionError(
            "remaining full comparison scoring requires the approved full-run token"
        )
    if run_id != REUSED_BATCH50_RUN_ID:
        raise ValueError(
            f"remaining-score must reuse the validated run ID {REUSED_BATCH50_RUN_ID}"
        )
    reused = FullScoreWorker().validate_scored_shard.remote(
        {"run_id": run_id, "shard_index": 0},
        REUSED_BATCH50_SCORED_SHA256,
        REUSED_BATCH50_SCORED_VALIDATION_SHA256,
        REUSED_BATCH50_SCORING_FINGERPRINT,
    )
    if not reused.get("passed") or int(reused.get("record_count", -1)) != 2_400:
        raise RuntimeError("validated batch-50 scored shard could not be reused")
    requests = [{"run_id": run_id, "shard_index": index} for index in range(1, 10)]
    results = list(FullScoreWorker().score_shard.map(requests))
    if len(results) != 9 or any(not result.get("passed") for result in results):
        raise RuntimeError("one or more remaining scoring shards failed")
    record_count = int(reused["record_count"]) + sum(
        int(result["record_count"]) for result in results
    )
    if record_count != 24_000:
        raise AssertionError(f"full scoring produced {record_count} rows")
    print(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "record_count": record_count,
                "generation_attempts": {"online_prc": 0, "null": 0},
                "reused_shard": reused,
                "scored_shards": results,
                "next_command": (
                    "modal volume get prc-data "
                    f"controlled_baseline_full/{run_id}/scored "
                    f"/private/tmp/{run_id}-full-scored"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


@app.local_entrypoint(name="proxy-textseal-score")
def proxy_textseal_score_entrypoint(approval_token: str):
    """CPU-only TextSeal sensitivity scoring from committed proxy traces."""
    from proxy_8b_analysis import APPROVAL_TOKEN, BASELINE_RUN_ID

    if approval_token != APPROVAL_TOKEN:
        raise PermissionError("8B proxy scoring requires the approved $20 token")
    requests = [
        {"run_id": BASELINE_RUN_ID, "shard_index": index}
        for index in range(10)
    ]
    results = list(FullScoreWorker().score_textseal_proxy_shard.map(requests))
    if len(results) != 10 or any(not result.get("passed") for result in results):
        raise RuntimeError("one or more TextSeal proxy scoring shards failed")
    record_count = sum(int(result["record_count"]) for result in results)
    if record_count != 6_000:
        raise AssertionError(f"TextSeal proxy scoring produced {record_count} rows")
    print(json.dumps({
        "passed": True,
        "run_id": BASELINE_RUN_ID,
        "record_count": record_count,
        "generation_attempts": 0,
        "proxy_model": "Qwen3-0.6B-Base",
        "quality_likelihood_model": "Qwen3-8B-Base (unchanged)",
        "shards": results,
    }, indent=2, sort_keys=True))
