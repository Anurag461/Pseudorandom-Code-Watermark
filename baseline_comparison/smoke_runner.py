"""Remote-only Qwen3-8B five-prompt smoke implementation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np
import torch

from .config import (
    CONTEXT_LENGTH,
    GENERATION_SETTINGS,
    GUMBEL_KEY,
    IMAGE_DEFINITION_SHA256,
    MAX_NEW_TOKENS,
    MODEL_ID,
    NOMINAL_FPR,
    ONLINE_PRC_ETA,
    ONLINE_PRC_SOURCE_TAG,
    ONLINE_PRC_T,
    PREFIX_LENGTHS,
    PRIMARY_SEED,
    PRC_BASE_COMMIT,
    PRC_REPOSITORY,
    SECONDARY_SEED,
    SHARED_NULL_SOURCE_T,
    SMOKE_PROMPT_INDICES,
    SYNTHID_COMMIT,
    SYNTHID_CONTEXT_HISTORY_SIZE,
    SYNTHID_DEPTH,
    SYNTHID_KEYS,
    SYNTHID_REPOSITORY,
    TEMPERATURE,
    TEXTSEAL_ALPHA,
    TEXTSEAL_COMMIT,
    TEXTSEAL_KEY_A,
    TEXTSEAL_KEY_B,
    TEXTSEAL_REPOSITORY,
    TOKENIZER_ID,
    TOP_P,
)
from .official import (
    gumbel_generator,
    official_gumbel_scores,
    official_synthid_g_values,
    official_textseal_fused_scores,
    synthid_processor,
    textseal_config,
    textseal_generator,
)
from .schema import PromptLevelResult
from .scoring import (
    deduplicated_positions,
    gumbel_gamma_test,
    prc_hoeffding_test,
    quality_metrics,
    synthid_normal_test,
    textseal_gamma_test,
)


MODEL_ROOT = "/cache/models/Qwen3-8B-Base"
PROMPTS_PATH = "/root/prompts.jsonl"
NULL_ROOT = f"/data/_nulls/qwen3_8b_base/T{SHARED_NULL_SOURCE_T}"
ARTIFACT_PATH = f"/data/{ONLINE_PRC_SOURCE_TAG}/artifacts.pt"
WM_ROOT = f"/data/{ONLINE_PRC_SOURCE_TAG}/wm"
VOCAB_SIZE = 151_936


def _numpy_pickle_compat() -> None:
    # See modal_app._validate_cache_records. Keep this alias scoped to cache I/O.
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


def _token_sha256(token_ids: Sequence[int]) -> str:
    array = np.asarray(token_ids, dtype=np.int64)
    return hashlib.sha256(
        f"int64:{array.shape}:".encode() + np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def _semantic_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _prompt_fingerprint(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(map(int, tokens)), separators=(",", ":")).encode()
    ).hexdigest()


def _integration_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path("/root/baseline_comparison")
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _prc_fingerprint() -> str:
    digest = hashlib.sha256()
    for name in ("qwen.py", "prc.py", "online_prc.py", "detectors.py"):
        path = Path("/root") / name
        digest.update(name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def preload_official_runtimes() -> dict:
    """Import official packages before the 8B CUDA allocation.

    TextSeal's package initializer imports its unrelated evaluation stack. In
    the pinned environment that late import segfaults if the 8B model is
    already resident. Preloading is an ordering workaround only.
    """
    print("[smoke] preloading pinned TextSeal/SynthID runtimes", flush=True)
    textseal = textseal_generator()
    gumbel = gumbel_generator()
    synthid = synthid_processor("cpu")
    print("[smoke] official runtime preload complete", flush=True)
    return {
        "textseal": textseal,
        "gumbel_max": gumbel,
        "synthid_hash_iv": int(synthid.hash_iv),
    }


def _model_revision() -> str:
    revisions = set()
    for path in Path(MODEL_ROOT).rglob("*.metadata"):
        try:
            revision = path.read_text().splitlines()[0].strip().lower()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if len(revision) == 40 and all(char in "0123456789abcdef" for char in revision):
            revisions.add(revision)
    if len(revisions) != 1:
        raise RuntimeError(f"model revision is ambiguous: {sorted(revisions)}")
    return next(iter(revisions))


def load_qwen3_8b():
    """Load the exact project Qwen implementation from the offline cache."""
    from qwen import Qwen3Model, return_qwen_config
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("the smoke model must run on CUDA")
    root = Path(MODEL_ROOT)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"offline model index missing: {index_path}")
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    missing = [name for name in shard_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"offline model shards missing: {missing}")

    print("[smoke] constructing Qwen3-8B directly on CUDA", flush=True)
    config = return_qwen_config("8B")
    with torch.device("cuda"):
        model = Qwen3Model(config)

    tied_weights = "lm_head.weight" not in index["weight_map"]
    if tied_weights:
        model.out_head.weight = model.tok_emb.weight

    targets = {
        "model.embed_tokens.weight": model.tok_emb.weight,
        "model.norm.weight": model.final_norm.scale,
    }
    if not tied_weights:
        targets["lm_head.weight"] = model.out_head.weight
    for layer, block in enumerate(model.trf_blocks):
        targets.update(
            {
                f"model.layers.{layer}.self_attn.q_proj.weight": block.att.W_query.weight,
                f"model.layers.{layer}.self_attn.k_proj.weight": block.att.W_key.weight,
                f"model.layers.{layer}.self_attn.v_proj.weight": block.att.W_value.weight,
                f"model.layers.{layer}.self_attn.o_proj.weight": block.att.out_proj.weight,
                f"model.layers.{layer}.self_attn.q_norm.weight": block.att.q_norm.scale,
                f"model.layers.{layer}.self_attn.k_norm.weight": block.att.k_norm.scale,
                f"model.layers.{layer}.input_layernorm.weight": block.norm1.scale,
                f"model.layers.{layer}.mlp.gate_proj.weight": block.ff.fc1.weight,
                f"model.layers.{layer}.mlp.up_proj.weight": block.ff.fc2.weight,
                f"model.layers.{layer}.mlp.down_proj.weight": block.ff.fc3.weight,
                f"model.layers.{layer}.post_attention_layernorm.weight": block.norm2.scale,
            }
        )
    missing_targets = sorted(set(targets) - set(index["weight_map"]))
    if missing_targets:
        raise KeyError(f"cached Qwen index lacks integration weights: {missing_targets[:3]}")

    loaded_targets = set()
    for name in shard_names:
        print(f"[smoke] loading offline shard {name}", flush=True)
        shard = load_file(str(root / name), device="cpu")
        for hf_name, tensor in shard.items():
            target = targets.get(hf_name)
            if target is None:
                continue
            if tuple(target.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Qwen weight shape differs for {hf_name}: {target.shape} != {tensor.shape}"
                )
            with torch.no_grad():
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded_targets.add(hf_name)
        del shard
        gc.collect()
    if loaded_targets != set(targets):
        missing_loaded = sorted(set(targets) - loaded_targets)
        raise KeyError(f"Qwen loader did not assign weights: {missing_loaded[:3]}")
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    print("[smoke] Qwen3-8B offline model load complete", flush=True)
    return model


@torch.no_grad()
def generate_method(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    method: str,
    seed: int,
) -> tuple[list[dict], dict]:
    """Generate a forced 1,024-token batch using the pinned official sampler."""
    from qwen import StaticKVCache

    if method not in {"textseal", "synthid_text", "gumbel_max"}:
        raise ValueError(f"unsupported generated method {method}")
    if not prompts or any(len(prompt) != 50 for prompt in prompts):
        raise ValueError("the smoke requires nonempty, exactly 50-token prompts")
    device = torch.device("cuda")
    batch_size = len(prompts)
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    torch.cuda.synchronize()
    started = time.perf_counter()

    all_tokens = torch.empty(
        (batch_size, 50 + MAX_NEW_TOKENS), dtype=torch.long, device=device
    )
    all_tokens[:, :50] = torch.tensor(prompts, dtype=torch.long, device=device)
    generated = torch.empty((batch_size, MAX_NEW_TOKENS), dtype=torch.long)
    logprobs = torch.empty((batch_size, MAX_NEW_TOKENS), dtype=torch.float32)
    entropies = torch.empty((batch_size, MAX_NEW_TOKENS), dtype=torch.float32)
    cache = StaticKVCache(max_length=50 + MAX_NEW_TOKENS)
    print(
        f"[smoke] {method} seed={seed} prefill start batch={batch_size}",
        flush=True,
    )
    logits = model(all_tokens[:, :50], cache=cache)[:, -1]
    torch.cuda.synchronize()
    print(f"[smoke] {method} seed={seed} prefill complete", flush=True)

    sampler = None
    processor = None
    reference_processor = None
    synthid_reference_max_abs_difference = 0.0
    synthid_reference_indices_equal = True
    if method == "textseal":
        sampler = textseal_generator()
    elif method == "gumbel_max":
        sampler = gumbel_generator()
    else:
        processor = synthid_processor(device)
        # One independent official processor follows prompt 0 for every smoke
        # step. This checks the adapter's exact generation-time score update on
        # real model logits while adding only 1/5 of a second full-batch pass.
        reference_processor = synthid_processor(device)

    for position in range(MAX_NEW_TOKENS):
        base_log_probs = torch.log_softmax(logits.float(), dim=-1)
        base_probs = torch.exp(base_log_probs)
        base_entropy = -(base_probs * base_log_probs).sum(dim=-1)

        if method in {"textseal", "gumbel_max"}:
            context = all_tokens[:, 50 + position - CONTEXT_LENGTH : 50 + position]
            next_token = sampler.sample_next(
                logits, context, temperature=float(TEMPERATURE), top_p=float(TOP_P)
            )
        else:
            updated, indices, _ = processor.watermarked_call(
                all_tokens[:, : 50 + position], logits
            )
            reference_updated, reference_indices, _ = reference_processor.watermarked_call(
                all_tokens[:1, : 50 + position], logits[:1]
            )
            synthid_reference_indices_equal &= bool(
                torch.equal(indices[:1], reference_indices)
            )
            difference = float(
                torch.max(torch.abs(updated[:1].float() - reference_updated.float())).item()
            )
            synthid_reference_max_abs_difference = max(
                synthid_reference_max_abs_difference, difference
            )
            probs = torch.softmax(updated.float(), dim=-1)
            selected = torch.multinomial(probs, num_samples=1)
            next_token = torch.gather(indices, 1, selected).reshape(-1)

        selected_logprob = base_log_probs.gather(1, next_token[:, None]).squeeze(1)
        generated[:, position] = next_token.detach().cpu()
        logprobs[:, position] = selected_logprob.detach().cpu()
        entropies[:, position] = base_entropy.detach().cpu()
        all_tokens[:, 50 + position] = next_token
        if position + 1 < MAX_NEW_TOKENS:
            logits = model(next_token[:, None], cache=cache)[:, -1]

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs = []
    for row in range(batch_size):
        outputs.append(
            {
                "token_ids": generated[row].tolist(),
                "base_token_logprobs": logprobs[row].double().tolist(),
                "base_entropies": entropies[row].double().tolist(),
            }
        )
    telemetry = {
        "method": method,
        "seed": int(seed),
        "batch_size": batch_size,
        "generated_sequences": batch_size,
        "generated_tokens": batch_size * MAX_NEW_TOKENS,
        "method_seconds": elapsed,
        "seconds_per_prompt": elapsed / batch_size,
        "tokens_per_second": batch_size * MAX_NEW_TOKENS / elapsed,
        "synthid_official_smoke_reference": {
            "prompt_index": 0 if method == "synthid_text" else None,
            "indices_equal": synthid_reference_indices_equal if method == "synthid_text" else None,
            "max_abs_score_difference": (
                synthid_reference_max_abs_difference if method == "synthid_text" else None
            ),
        },
    }
    return outputs, telemetry


def run_gpu_diagnostic(data_volume) -> dict:
    """No-generation diagnostic for the pinned model/KV-cache runtime."""
    from qwen import StaticKVCache

    print(
        f"[diagnostic] torch={torch.__version__} cuda={torch.version.cuda} "
        f"gpu={torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}",
        flush=True,
    )
    preloaded = preload_official_runtimes()
    data_volume.reload()
    rows = [json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line]
    _load_cached_sequences(rows)
    model = load_qwen3_8b()
    prompt_one = torch.tensor([rows[0]["prompt_tokens"]], dtype=torch.long, device="cuda")
    prompt_five = torch.tensor(
        [rows[index]["prompt_tokens"] for index in SMOKE_PROMPT_INDICES],
        dtype=torch.long,
        device="cuda",
    )
    checks = []
    with torch.no_grad():
        print("[diagnostic] batch1 no-cache forward start", flush=True)
        output = model(prompt_one)
        torch.cuda.synchronize()
        checks.append(
            {
                "name": "batch1_no_cache",
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "last_logit_sum": float(output[:, -1].float().sum().item()),
            }
        )
        del output
        print("[diagnostic] batch1 no-cache forward complete", flush=True)

        print("[diagnostic] batch1 static-cache prefill start", flush=True)
        cache_one = StaticKVCache(max_length=50 + MAX_NEW_TOKENS)
        output = model(prompt_one, cache=cache_one)
        torch.cuda.synchronize()
        checks.append(
            {
                "name": "batch1_static_cache_prefill",
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "cache_length": cache_one.get_seq_len(),
                "last_logit_sum": float(output[:, -1].float().sum().item()),
            }
        )
        del output, cache_one
        print("[diagnostic] batch1 static-cache prefill complete", flush=True)

        print("[diagnostic] batch5 static-cache prefill start", flush=True)
        cache_five = StaticKVCache(max_length=50 + MAX_NEW_TOKENS)
        output = model(prompt_five, cache=cache_five)
        torch.cuda.synchronize()
        checks.append(
            {
                "name": "batch5_static_cache_prefill",
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "cache_length": cache_five.get_seq_len(),
                "last_logit_sum": float(output[:, -1].float().sum().item()),
            }
        )
        print("[diagnostic] batch5 static-cache prefill complete", flush=True)

        print("[diagnostic] batch5 base log-probability/entropy start", flush=True)
        last_logits = output[:, -1]
        base_log_probs = torch.log_softmax(last_logits.float(), dim=-1)
        entropy = -(base_log_probs.exp() * base_log_probs).sum(dim=-1)
        torch.cuda.synchronize()
        checks.append(
            {
                "name": "batch5_base_distribution",
                "shape": list(base_log_probs.shape),
                "finite": bool(
                    torch.isfinite(base_log_probs).all().item()
                    and torch.isfinite(entropy).all().item()
                ),
                "entropy_sum": float(entropy.sum().item()),
            }
        )
        print("[diagnostic] batch5 base log-probability/entropy complete", flush=True)

        print("[diagnostic] TextSeal preloaded sampler retrieval start", flush=True)
        sampler = preloaded["textseal"]
        print("[diagnostic] TextSeal preloaded sampler retrieval complete", flush=True)
        context = prompt_five[:, -CONTEXT_LENGTH:]
        print("[diagnostic] TextSeal one-step sample start", flush=True)
        next_token = sampler.sample_next(
            last_logits,
            context,
            temperature=float(TEMPERATURE),
            top_p=float(TOP_P),
        )
        torch.cuda.synchronize()
        checks.append(
            {
                "name": "batch5_textseal_one_step",
                "shape": list(next_token.shape),
                "finite": True,
                "token_ids": next_token.detach().cpu().tolist(),
            }
        )
        print("[diagnostic] TextSeal one-step sample complete", flush=True)
    return {
        "passed": all(check["finite"] for check in checks),
        "checks": checks,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "actual_gpu": torch.cuda.get_device_name(0),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }


def run_gumbel_determinism_check(data_volume, raw_path: str) -> dict:
    """Control seed while holding prompt, batch shape, and model fixed."""
    print("[gumbel-check] validating caches and committed smoke", flush=True)
    preload_official_runtimes()
    data_volume.reload()
    prompt_rows = [
        json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    _load_cached_sequences(prompt_rows)
    _numpy_pickle_compat()
    committed = torch.load(raw_path, weights_only=False, map_location="cpu")
    committed_secondary = committed["sequences"][f"gumbel_max/seed{SECONDARY_SEED}"][0]

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_qwen3_8b()
    model_load_seconds = time.perf_counter() - started
    outputs_by_seed = {}
    telemetry = []
    for seed in (PRIMARY_SEED, SECONDARY_SEED):
        outputs, item = generate_method(
            model,
            [prompt_rows[0]["prompt_tokens"]],
            method="gumbel_max",
            seed=seed,
        )
        outputs_by_seed[seed] = outputs[0]
        telemetry.append(item)

    primary_tokens = outputs_by_seed[PRIMARY_SEED]["token_ids"]
    secondary_tokens = outputs_by_seed[SECONDARY_SEED]["token_ids"]
    equal_shape_identical = primary_tokens == secondary_tokens
    committed_secondary_identical = (
        secondary_tokens == committed_secondary["token_ids"]
    )
    if not equal_shape_identical:
        raise AssertionError(
            "official Gumbel comparison path changed across RNG seeds at equal batch shape"
        )
    if not committed_secondary_identical:
        raise AssertionError(
            "equal-shape Gumbel rerun differs from the committed batch-1 continuation"
        )
    torch.cuda.synchronize()
    return {
        "passed": True,
        "prompt_index": 0,
        "batch_size": 1,
        "seeds": [PRIMARY_SEED, SECONDARY_SEED],
        "equal_shape_identical_across_seeds": equal_shape_identical,
        "committed_secondary_reproduced": committed_secondary_identical,
        "token_hashes": {
            str(seed): _token_sha256(outputs_by_seed[seed]["token_ids"])
            for seed in (PRIMARY_SEED, SECONDARY_SEED)
        },
        "committed_primary_batch5_hash": _token_sha256(
            committed["sequences"][f"gumbel_max/seed{PRIMARY_SEED}"][0]["token_ids"]
        ),
        "committed_secondary_batch1_hash": _token_sha256(
            committed_secondary["token_ids"]
        ),
        "batch_shape_sensitivity_observed": (
            committed["sequences"][f"gumbel_max/seed{PRIMARY_SEED}"][0]["token_ids"]
            != secondary_tokens
        ),
        "generation_telemetry": telemetry,
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "function_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "requested_gpu": "H100",
            "actual_gpu": torch.cuda.get_device_name(0),
        },
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "remote_generation_artifact": raw_path,
    }


def run_stochastic_seed_check(data_volume, raw_path: str) -> dict:
    """Check fixed-seed replay and different-seed behavior at equal shape."""
    print("[seed-check] validating TextSeal/SynthID equal-shape behavior", flush=True)
    preload_official_runtimes()
    data_volume.reload()
    prompt_rows = [
        json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    _load_cached_sequences(prompt_rows)
    _numpy_pickle_compat()
    committed = torch.load(raw_path, weights_only=False, map_location="cpu")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_qwen3_8b()
    model_load_seconds = time.perf_counter() - started
    checks = []
    all_telemetry = []
    for method in ("textseal", "synthid_text"):
        outputs_by_seed = {}
        for seed in (PRIMARY_SEED, SECONDARY_SEED):
            outputs, telemetry = generate_method(
                model,
                [prompt_rows[0]["prompt_tokens"]],
                method=method,
                seed=seed,
            )
            outputs_by_seed[seed] = outputs[0]
            all_telemetry.append(telemetry)
        primary_tokens = outputs_by_seed[PRIMARY_SEED]["token_ids"]
        secondary_tokens = outputs_by_seed[SECONDARY_SEED]["token_ids"]
        committed_secondary = committed["sequences"][f"{method}/seed{SECONDARY_SEED}"][0]
        fixed_seed_reproduced = secondary_tokens == committed_secondary["token_ids"]
        changed_across_seeds = primary_tokens != secondary_tokens
        if not fixed_seed_reproduced:
            raise AssertionError(f"{method} did not reproduce its committed fixed-seed output")
        if not changed_across_seeds:
            raise AssertionError(f"{method} did not change across controlled RNG seeds")
        checks.append(
            {
                "method": method,
                "fixed_seed_reproduced": fixed_seed_reproduced,
                "changed_across_seeds": changed_across_seeds,
                "primary_secondary_token_agreement": sum(
                    left == right for left, right in zip(primary_tokens, secondary_tokens)
                )
                / MAX_NEW_TOKENS,
                "token_hashes": {
                    str(seed): _token_sha256(outputs_by_seed[seed]["token_ids"])
                    for seed in (PRIMARY_SEED, SECONDARY_SEED)
                },
            }
        )
    torch.cuda.synchronize()
    return {
        "passed": True,
        "prompt_index": 0,
        "batch_size": 1,
        "seeds": [PRIMARY_SEED, SECONDARY_SEED],
        "checks": checks,
        "generation_telemetry": all_telemetry,
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "function_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "requested_gpu": "H100",
            "actual_gpu": torch.cuda.get_device_name(0),
        },
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "remote_generation_artifact": raw_path,
    }


def _load_cached_sequences_for_indices(
    prompt_rows: Sequence[dict], prompt_indices: Sequence[int]
) -> tuple[list[dict], list[dict], dict]:
    _numpy_pickle_compat()
    artifact = torch.load(ARTIFACT_PATH, weights_only=False, map_location="cpu")
    online = []
    nulls = []
    for index in prompt_indices:
        wm = torch.load(
            str(Path(WM_ROOT) / f"wm_{index:04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        null = torch.load(
            str(Path(NULL_ROOT) / f"null_{index:04d}.pt"),
            weights_only=False,
            map_location="cpu",
        )
        for label, record, source_length in (
            ("online_prc", wm, 1280),
            ("null", null, SHARED_NULL_SOURCE_T),
        ):
            if int(np.asarray(record["tokens"]).size) != source_length:
                raise ValueError(f"{label} prompt {index} cache length changed")
            if list(map(int, record["prompt_token_ids"])) != prompt_rows[index]["prompt_tokens"]:
                raise ValueError(f"{label} prompt {index} cache prompt changed")
        online.append(wm)
        nulls.append(null)
    return online, nulls, artifact


def _load_cached_sequences(prompt_rows: Sequence[dict]) -> tuple[list[dict], list[dict], dict]:
    return _load_cached_sequences_for_indices(prompt_rows, SMOKE_PROMPT_INDICES)


def _method_configuration(method: str) -> tuple[dict, int | None, str, str, str]:
    if method == "textseal":
        return (
            {
                "alpha": TEXTSEAL_ALPHA,
                "context_length": CONTEXT_LENGTH,
                "deduplication": "unique (context_3, token), released v2 path starts at position k+1",
                "primary_test": "entropy-weighted moment-matched Gamma",
                "official_min_weighted_unweighted_p_preserved_as_intermediate_only": True,
            },
            TEXTSEAL_KEY_A,
            f"dual keys A={TEXTSEAL_KEY_A}, B={TEXTSEAL_KEY_B}; released alpha weights/routes A",
            TEXTSEAL_REPOSITORY,
            TEXTSEAL_COMMIT,
        )
    if method == "synthid_text":
        return (
            {
                "depth": SYNTHID_DEPTH,
                "keys": list(SYNTHID_KEYS),
                "ngram_len": CONTEXT_LENGTH + 1,
                "context_length": CONTEXT_LENGTH,
                "context_history_size": SYNTHID_CONTEXT_HISTORY_SIZE,
                "num_leaves": 2,
                "apply_top_k": False,
                "detector": "frequentist weighted normal approximation from TextSeal comparison",
                "deduplication": "unique (context_3, token), released TextSeal v2 position convention",
                "generation_repeated_context_mask": "Google official context-only mask enabled",
            },
            SYNTHID_KEYS[0],
            "Google official 10-key domain; all keys recorded in method configuration",
            SYNTHID_REPOSITORY,
            SYNTHID_COMMIT,
        )
    if method == "gumbel_max":
        return (
            {
                "context_length": CONTEXT_LENGTH,
                "deduplication": "unique (context_3, token), released v2 position convention",
                "generation": "TextSeal GumbelmaxGenerator comparison path",
                "detector": "exact Gamma(sum -log(1-r); shape=N, scale=1)",
            },
            GUMBEL_KEY,
            "TextSeal uniform PRF",
            TEXTSEAL_REPOSITORY,
            TEXTSEAL_COMMIT,
        )
    if method == "online_prc":
        return (
            {
                "t": ONLINE_PRC_T,
                "eta": ONLINE_PRC_ETA,
                "weight": "map",
                "fpr_policy": "one_shot",
                "source_T": 1280,
                "deduplication": "not applicable",
            },
            PRIMARY_SEED,
            "online PRC compact HMAC support/OTP key artifact",
            PRC_REPOSITORY,
            PRC_BASE_COMMIT,
        )
    raise ValueError(method)


def _result(
    *,
    prompt_row: dict,
    prompt_index: int,
    sample_type: str,
    method: str,
    seed: int,
    token_ids: Sequence[int],
    base_logprobs: Sequence[float],
    prefix: int,
    dedup_count: int,
    test: dict,
    model_revision: str,
    integration_fingerprint: str,
    prc_fingerprint: str,
    artifact_fingerprint: str,
    provenance: dict,
    runtime_seconds: float,
    diversity_fields: dict | None = None,
) -> dict:
    config, key_seed, key_domain, repo, commit = _method_configuration(method)
    tokens = list(map(int, token_ids))
    quality = quality_metrics(tokens, base_logprobs)
    result = PromptLevelResult(
        prompt_index=int(prompt_index),
        prompt_id=f"doc-{int(prompt_row['doc_index'])}",
        prompt_fingerprint=_prompt_fingerprint(prompt_row["prompt_tokens"]),
        sample_type=sample_type,
        method=method,
        method_configuration=config,
        model_id=MODEL_ID,
        model_revision=model_revision,
        tokenizer_id=TOKENIZER_ID,
        tokenizer_revision=model_revision,
        generation_seed=int(seed),
        key_seed=key_seed,
        key_domain=key_domain,
        generation_settings=GENERATION_SETTINGS,
        generated_token_count=len(tokens),
        generated_token_hash=_token_sha256(tokens),
        prefix_length=int(prefix),
        deduplicated_sample_count=int(dedup_count),
        statistic=float(test["statistic"]),
        p_value=float(test["p_value"]),
        calibration_type=str(test["calibration_type"]),
        threshold=float(test["threshold"]),
        decision=bool(test["decision"]),
        base_model_nll=quality["base_model_nll"],
        base_model_perplexity=quality["base_model_perplexity"],
        output_length=quality["output_length"],
        repetition_rate=quality["repetition_rate"],
        repetition_metric=quality["repetition_metric"],
        distinct_2=quality["distinct_2"],
        distinct_3=quality["distinct_3"],
        source_repository_url=repo,
        source_repository_commit=commit,
        prc_code_fingerprint=prc_fingerprint,
        integration_code_fingerprint=integration_fingerprint,
        image_fingerprint=IMAGE_DEFINITION_SHA256,
        artifact_fingerprint=artifact_fingerprint,
        cache_or_generation_provenance=provenance,
        runtime_seconds=float(runtime_seconds),
        intermediate_values=dict(test.get("intermediate", {})),
        diversity_fields=diversity_fields or {},
    )
    return result.to_dict()


def _score_baseline_sequence(
    *,
    method: str,
    sequence: dict,
    prompt_row: dict,
    prompt_index: int,
    sample_type: str,
    seed: int,
    model_revision: str,
    integration_fingerprint: str,
    prc_fingerprint: str,
    provenance: dict,
    runtime_seconds: float,
    diversity_fields: dict | None = None,
) -> tuple[list[dict], dict]:
    from textseal.watermarking.detector import TextSealDetector

    tokens = list(map(int, sequence["token_ids"][:MAX_NEW_TOKENS]))
    logprobs = list(map(float, sequence["base_token_logprobs"][:MAX_NEW_TOKENS]))
    entropies = list(map(float, sequence["base_entropies"][:MAX_NEW_TOKENS]))
    if not (len(tokens) == len(logprobs) == len(entropies) == MAX_NEW_TOKENS):
        raise ValueError(f"{method} prompt {prompt_index} is not exactly 1024 tokens")
    token_hash = _token_sha256(tokens)
    artifact_fingerprint = _semantic_fingerprint(
        {
            "method": method,
            "prompt_index": prompt_index,
            "seed": seed,
            "token_hash": token_hash,
            "provenance": provenance,
        }
    )
    results = []
    exact_prefix_checks = []
    textseal_reference_deltas = []
    synthid_g_hashes = []
    full_positions = deduplicated_positions(tokens, CONTEXT_LENGTH)
    if method == "textseal":
        full_evidence = official_textseal_fused_scores(tokens, full_positions)
    elif method == "gumbel_max":
        full_evidence = official_gumbel_scores(tokens, full_positions)
    else:
        full_evidence = official_synthid_g_values(tokens, full_positions)

    for prefix in PREFIX_LENGTHS:
        prefix_tokens = tokens[:prefix]
        positions = deduplicated_positions(prefix_tokens, CONTEXT_LENGTH)
        if method == "textseal":
            direct = official_textseal_fused_scores(prefix_tokens, positions)
            selected = full_evidence[: len(positions)]
            exact_delta = float(np.max(np.abs(direct - selected))) if len(direct) else 0.0
            selected_entropies = [entropies[position] for position in positions]
            test = textseal_gamma_test(
                direct, selected_entropies, alpha=TEXTSEAL_ALPHA, nominal_fpr=NOMINAL_FPR
            )
            # Shift generation-step entropies into the official teacher-forced
            # convention: detector entropy index p-1 predicts target p.
            official_entropies = entropies[1:prefix]
            official = TextSealDetector(
                None, textseal_config(), scoring_method="v2"
            )._score_text(prefix_tokens, official_entropies, scoring_method="v2")
            official_weighted_p = float(official["p_value_weighted"])
            reference_delta = abs(official_weighted_p - test["p_value"])
            # The pinned detector accumulates its scalar float32 PRF tensors
            # through NumPy, whereas the common scorer casts those identical
            # PRF values to float64 before the weighted reduction.  Validate
            # numerical parity and, separately, exact decision parity; this is
            # not a change to either statistic or calibration formula.
            if not math.isclose(
                official_weighted_p,
                test["p_value"],
                rel_tol=2e-6,
                abs_tol=2e-7,
            ):
                raise AssertionError(
                    f"TextSeal official p mismatch on {sample_type} prompt {prompt_index} "
                    f"T={prefix}: {reference_delta}"
                )
            if bool(official_weighted_p < NOMINAL_FPR) != bool(test["decision"]):
                raise AssertionError(
                    f"TextSeal official/common decisions differ on {sample_type} "
                    f"prompt {prompt_index} T={prefix}"
                )
            textseal_reference_deltas.append(reference_delta)
            test["intermediate"].update(
                {
                    "official_p_value_weighted": float(official["p_value_weighted"]),
                    "official_p_value_unweighted": float(official["p_value_unweighted"]),
                    "official_min_p_value": float(official["p_value"]),
                    "official_hardcoded_decision_at_0.01": bool(official["detected"]),
                    "official_common_abs_p_delta": reference_delta,
                    "official_common_p_tolerance": {"relative": 2e-6, "absolute": 2e-7},
                    "primary_decision_rule": "weighted p < 0.001",
                }
            )
        elif method == "gumbel_max":
            direct = official_gumbel_scores(prefix_tokens, positions)
            selected = full_evidence[: len(positions)]
            exact_delta = float(np.max(np.abs(direct - selected))) if len(direct) else 0.0
            test = gumbel_gamma_test(direct, nominal_fpr=NOMINAL_FPR)
        else:
            direct = official_synthid_g_values(prefix_tokens, positions)
            selected = full_evidence[: len(positions)]
            exact_delta = float(np.max(np.abs(direct - selected))) if direct.size else 0.0
            if not np.array_equal(direct, selected):
                raise AssertionError(
                    f"SynthID prefix g-values differ at prompt {prompt_index}, T={prefix}"
                )
            test = synthid_normal_test(direct, nominal_fpr=NOMINAL_FPR)
            synthid_g_hashes.append(
                hashlib.sha256(np.ascontiguousarray(direct).tobytes()).hexdigest()
            )
            test["intermediate"]["official_google_g_values_sha256"] = synthid_g_hashes[-1]
        if exact_delta > 0.0:
            raise AssertionError(
                f"{method} exact-prefix evidence differs at prompt {prompt_index}, T={prefix}: "
                f"{exact_delta}"
            )
        exact_prefix_checks.append(
            {"prefix_length": prefix, "deduplicated_count": len(positions), "max_abs_delta": exact_delta}
        )
        results.append(
            _result(
                prompt_row=prompt_row,
                prompt_index=prompt_index,
                sample_type=sample_type,
                method=method,
                seed=seed,
                token_ids=tokens,
                base_logprobs=logprobs,
                prefix=prefix,
                dedup_count=len(positions),
                test=test,
                model_revision=model_revision,
                integration_fingerprint=integration_fingerprint,
                prc_fingerprint=prc_fingerprint,
                artifact_fingerprint=artifact_fingerprint,
                provenance=provenance,
                runtime_seconds=runtime_seconds,
                diversity_fields=diversity_fields,
            )
        )
    return results, {
        "method": method,
        "prompt_index": prompt_index,
        "sample_type": sample_type,
        "seed": seed,
        "token_hash": token_hash,
        "exact_prefix_checks": exact_prefix_checks,
        "max_textseal_official_p_difference": (
            max(textseal_reference_deltas) if textseal_reference_deltas else None
        ),
        "synthid_g_value_hashes": synthid_g_hashes,
    }


def _score_prc_sequences(
    *,
    online_records: list[dict],
    null_records: list[dict],
    artifact: dict,
    prompt_rows: list[dict],
    model_revision: str,
    integration_fingerprint: str,
    prc_fingerprint: str,
    prompt_indices: Sequence[int] = SMOKE_PROMPT_INDICES,
) -> tuple[list[dict], list[dict]]:
    from detectors import detect_online_map_prefix_grid
    from online_prc import OnlinePRCKey

    key = OnlinePRCKey.from_dict(artifact["online_key"])
    partition = artifact["partition"]
    records = []
    validation = []
    for sample_type, cached, source_root in (
        ("watermarked", online_records, WM_ROOT),
        ("null", null_records, NULL_ROOT),
    ):
        for row, prompt_index in enumerate(prompt_indices):
            record = cached[row]
            tokens = list(map(int, np.asarray(record["tokens"])[:MAX_NEW_TOKENS]))
            p_trace = np.asarray(record["p_trace"], dtype=np.float64)[:MAX_NEW_TOKENS]
            logprobs = list(
                map(float, np.asarray(record["base_token_logprob"])[:MAX_NEW_TOKENS])
            )
            if len(tokens) != MAX_NEW_TOKENS or len(p_trace) != MAX_NEW_TOKENS:
                raise ValueError(f"cached PRC/{sample_type} prompt {prompt_index} is incomplete")
            scored = detect_online_map_prefix_grid(
                key,
                tokens,
                p_trace,
                partition,
                list(PREFIX_LENGTHS),
                fpr=NOMINAL_FPR,
                fpr_policy="one_shot",
            )
            for prefix, info in zip(PREFIX_LENGTHS, scored):
                test = prc_hoeffding_test(
                    info["statistic"], info["V"], nominal_fpr=NOMINAL_FPR
                )
                if bool(info["decision"]) != bool(test["decision"]):
                    raise AssertionError(
                        f"PRC common p-bound decision differs at {sample_type} "
                        f"prompt {prompt_index}, T={prefix}"
                    )
                if not math.isclose(
                    float(info["threshold"]), float(test["threshold"]), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise AssertionError("PRC common threshold differs from project detector")
                test["intermediate"].update(
                    {
                        "official_project_detector": dict(info),
                        "deduplication": "not applicable to online PRC",
                    }
                )
                provenance = {
                    "mode": "cache_only",
                    "source_root": source_root,
                    "source_T": 1280 if sample_type == "watermarked" else SHARED_NULL_SOURCE_T,
                    "source_artifact_fingerprint": artifact["artifact_fingerprint"],
                    "generation_attempts": 0,
                    "record_prompt_index": prompt_index,
                    "kv_cache_implementation": record.get("kv_cache_implementation"),
                    "kv_cache_version": record.get("kv_cache_version"),
                }
                records.append(
                    _result(
                        prompt_row=prompt_rows[prompt_index],
                        prompt_index=prompt_index,
                        sample_type=sample_type,
                        method="online_prc",
                        seed=int(record.get("experiment_seed", PRIMARY_SEED)),
                        token_ids=tokens,
                        base_logprobs=logprobs,
                        prefix=prefix,
                        dedup_count=prefix,
                        test=test,
                        model_revision=model_revision,
                        integration_fingerprint=integration_fingerprint,
                        prc_fingerprint=prc_fingerprint,
                        artifact_fingerprint=artifact["artifact_fingerprint"],
                        provenance=provenance,
                        runtime_seconds=0.0,
                    )
                )
            validation.append(
                {
                    "method": "online_prc",
                    "sample_type": sample_type,
                    "prompt_index": prompt_index,
                    "token_hash": _token_sha256(tokens),
                    "prefix_grid_equivalent_to_direct_detector": True,
                    "generation_attempts": 0,
                }
            )
    return records, validation


def _score_generated_payload(
    *,
    generated: dict[tuple[str, int], list[dict]],
    generation_telemetry: list[dict],
    raw_path: str,
    prompt_rows: list[dict],
    online_records: list[dict],
    null_records: list[dict],
    prc_artifact: dict,
    model_revision: str,
    integration_fingerprint: str,
    prc_fingerprint: str,
    actual_gpu: str,
) -> dict:
    """Score a committed generation payload without model access.

    Keeping scoring independent of generation lets a validation-only failure
    resume on Modal CPU from the immutable raw artifact, rather than spending
    another H100 pass on identical continuations.
    """
    gumbel_deterministic = (
        generated[("gumbel_max", PRIMARY_SEED)][0]["token_ids"]
        == generated[("gumbel_max", SECONDARY_SEED)][0]["token_ids"]
    )
    textseal_changed = (
        generated[("textseal", PRIMARY_SEED)][0]["token_ids"]
        != generated[("textseal", SECONDARY_SEED)][0]["token_ids"]
    )
    synthid_changed = (
        generated[("synthid_text", PRIMARY_SEED)][0]["token_ids"]
        != generated[("synthid_text", SECONDARY_SEED)][0]["token_ids"]
    )

    records, validation = _score_prc_sequences(
        online_records=online_records,
        null_records=null_records,
        artifact=prc_artifact,
        prompt_rows=prompt_rows,
        model_revision=model_revision,
        integration_fingerprint=integration_fingerprint,
        prc_fingerprint=prc_fingerprint,
    )

    telemetry_lookup = {
        (item["method"], item["seed"]): item for item in generation_telemetry
    }
    for method in ("textseal", "synthid_text", "gumbel_max"):
        for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
            sequence = generated[(method, PRIMARY_SEED)][row]
            runtime = telemetry_lookup[(method, PRIMARY_SEED)]["seconds_per_prompt"]
            diversity = {}
            if prompt_index == 0:
                other = generated[(method, SECONDARY_SEED)][0]["token_ids"]
                primary = sequence["token_ids"]
                diversity["secondary_seed"] = SECONDARY_SEED
                diversity["primary_secondary_token_agreement"] = sum(
                    left == right for left, right in zip(primary, other)
                ) / MAX_NEW_TOKENS
                diversity["primary_secondary_identical"] = primary == other
                diversity["primary_batch_size"] = 5
                diversity["secondary_batch_size"] = 1
                diversity["seed_effect_interpretable"] = False
            scored, checked = _score_baseline_sequence(
                method=method,
                sequence=sequence,
                prompt_row=prompt_rows[prompt_index],
                prompt_index=prompt_index,
                sample_type="watermarked",
                seed=PRIMARY_SEED,
                model_revision=model_revision,
                integration_fingerprint=integration_fingerprint,
                prc_fingerprint=prc_fingerprint,
                provenance={
                    "mode": "generated",
                    "remote_generation_artifact": raw_path,
                    "generation_artifact_key": f"{method}/seed{PRIMARY_SEED}/row{row}",
                    "requested_gpu": "H100",
                    "actual_gpu": actual_gpu,
                },
                runtime_seconds=runtime,
                diversity_fields=diversity,
            )
            records.extend(scored)
            validation.append(checked)

        # All three detectors score the exact same five cached null texts.
        for row, prompt_index in enumerate(SMOKE_PROMPT_INDICES):
            null = null_records[row]
            sequence = {
                "token_ids": list(map(int, np.asarray(null["tokens"])[:MAX_NEW_TOKENS])),
                "base_token_logprobs": list(
                    map(float, np.asarray(null["base_token_logprob"])[:MAX_NEW_TOKENS])
                ),
                # Cached PRC entropies are bits; TextSeal uses natural logs.
                "base_entropies": list(
                    map(float, np.asarray(null["base_lm_entropy"])[:MAX_NEW_TOKENS] * math.log(2.0))
                ),
            }
            scored, checked = _score_baseline_sequence(
                method=method,
                sequence=sequence,
                prompt_row=prompt_rows[prompt_index],
                prompt_index=prompt_index,
                sample_type="null",
                seed=int(null.get("experiment_seed", PRIMARY_SEED)),
                model_revision=model_revision,
                integration_fingerprint=integration_fingerprint,
                prc_fingerprint=prc_fingerprint,
                provenance={
                    "mode": "cache_only",
                    "source_root": NULL_ROOT,
                    "source_T": SHARED_NULL_SOURCE_T,
                    "generation_attempts": 0,
                    "record_prompt_index": prompt_index,
                    "kv_cache_implementation": null.get("kv_cache_implementation"),
                    "kv_cache_version": null.get("kv_cache_version"),
                    "method_specific_key_handling": (
                        "same null text, scored under this detector's fixed key domain"
                    ),
                },
                runtime_seconds=0.0,
            )
            records.extend(scored)
            validation.append(checked)

        secondary_sequence = generated[(method, SECONDARY_SEED)][0]
        secondary_scored, secondary_checked = _score_baseline_sequence(
            method=method,
            sequence=secondary_sequence,
            prompt_row=prompt_rows[0],
            prompt_index=0,
            sample_type="seed_validation",
            seed=SECONDARY_SEED,
            model_revision=model_revision,
            integration_fingerprint=integration_fingerprint,
            prc_fingerprint=prc_fingerprint,
            provenance={
                "mode": "generated_seed_validation",
                "remote_generation_artifact": raw_path,
                "generation_artifact_key": f"{method}/seed{SECONDARY_SEED}/row0",
                "requested_gpu": "H100",
                "actual_gpu": actual_gpu,
            },
            runtime_seconds=telemetry_lookup[(method, SECONDARY_SEED)]["seconds_per_prompt"],
            diversity_fields={
                "primary_seed": PRIMARY_SEED,
                "primary_secondary_identical": (
                    generated[(method, PRIMARY_SEED)][0]["token_ids"]
                    == secondary_sequence["token_ids"]
                ),
                "primary_batch_size": 5,
                "secondary_batch_size": 1,
                "seed_effect_interpretable": False,
            },
        )
        records.extend(secondary_scored)
        validation.append(secondary_checked)

    if len(records) != 258:
        raise AssertionError(f"smoke produced {len(records)} prompt-prefix records, expected 258")
    if any(not np.isfinite(float(record["p_value"])) for record in records):
        raise AssertionError("smoke produced a non-finite p-value")
    if any(record["output_length"] != MAX_NEW_TOKENS for record in records):
        raise AssertionError("smoke schema contains a non-1024 output")

    return {
        "passed": True,
        "records": records,
        "validation_records": validation,
        "generation_telemetry": generation_telemetry,
        "behavior": {
            "gumbel_max_primary_batch5_vs_secondary_batch1_identical": gumbel_deterministic,
            "gumbel_max_seed_determinism_requires_equal_shape_control": True,
            "textseal_changed_across_seeds": textseal_changed,
            "synthid_text_changed_across_seeds": synthid_changed,
        },
        "remote_generation_artifact": {
            "path": raw_path,
            "size_bytes": Path(raw_path).stat().st_size,
            "semantic_fingerprint": _semantic_fingerprint(
                {
                    "model_revision": model_revision,
                    "prompt_indices": list(SMOKE_PROMPT_INDICES),
                    "token_hashes": {
                        f"{method}/seed{seed}": [
                            _token_sha256(sequence["token_ids"]) for sequence in outputs
                        ]
                        for (method, seed), outputs in generated.items()
                    },
                }
            ),
        },
    }


def run_gpu_smoke(data_volume) -> dict:
    """Run only the authorized five canonical prompts and two seed checks."""
    print("[smoke] GPU worker entered; validating caches before model load", flush=True)
    preload_official_runtimes()
    data_volume.reload()
    prompt_rows = [json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    prompt_batch = [prompt_rows[index]["prompt_tokens"] for index in SMOKE_PROMPT_INDICES]
    online_records, null_records, prc_artifact = _load_cached_sequences(prompt_rows)
    model_revision = _model_revision()
    integration_fingerprint = _integration_fingerprint()
    prc_fingerprint = _prc_fingerprint()

    torch.cuda.reset_peak_memory_stats()
    overall_started = time.perf_counter()
    model_load_started = time.perf_counter()
    model = load_qwen3_8b()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - model_load_started
    generated: dict[tuple[str, int], list[dict]] = {}
    generation_telemetry = []
    for method in ("textseal", "synthid_text", "gumbel_max"):
        print(f"[smoke] primary generation start: {method}", flush=True)
        outputs, telemetry = generate_method(
            model, prompt_batch, method=method, seed=PRIMARY_SEED
        )
        print(
            f"[smoke] primary generation complete: {method} "
            f"seconds={telemetry['method_seconds']:.3f}",
            flush=True,
        )
        generated[(method, PRIMARY_SEED)] = outputs
        generation_telemetry.append(telemetry)
    for method in ("textseal", "synthid_text", "gumbel_max"):
        print(f"[smoke] secondary seed generation start: {method}", flush=True)
        outputs, telemetry = generate_method(
            model, prompt_batch[:1], method=method, seed=SECONDARY_SEED
        )
        print(
            f"[smoke] secondary seed generation complete: {method} "
            f"seconds={telemetry['method_seconds']:.3f}",
            flush=True,
        )
        generated[(method, SECONDARY_SEED)] = outputs
        generation_telemetry.append(telemetry)

    for (method, seed), outputs in generated.items():
        for row, output in enumerate(outputs):
            if len(output["token_ids"]) != MAX_NEW_TOKENS:
                raise AssertionError(
                    f"{method} seed {seed} row {row} produced {len(output['token_ids'])} tokens"
                )
            for field in ("base_token_logprobs", "base_entropies"):
                values = np.asarray(output[field], dtype=np.float64)
                if values.size != MAX_NEW_TOKENS or not np.all(np.isfinite(values)):
                    raise AssertionError(f"{method} seed {seed} {field} is incomplete/non-finite")

    gumbel_deterministic = (
        generated[("gumbel_max", PRIMARY_SEED)][0]["token_ids"]
        == generated[("gumbel_max", SECONDARY_SEED)][0]["token_ids"]
    )
    textseal_changed = (
        generated[("textseal", PRIMARY_SEED)][0]["token_ids"]
        != generated[("textseal", SECONDARY_SEED)][0]["token_ids"]
    )
    synthid_changed = (
        generated[("synthid_text", PRIMARY_SEED)][0]["token_ids"]
        != generated[("synthid_text", SECONDARY_SEED)][0]["token_ids"]
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    remote_root = f"/data/controlled_baseline_smoke/{timestamp}"
    Path(remote_root).mkdir(parents=True, exist_ok=False)
    raw_path = str(Path(remote_root) / "generated_sequences.pt")
    raw_payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_indices": list(SMOKE_PROMPT_INDICES),
        "model_revision": model_revision,
        "image_definition_sha256": IMAGE_DEFINITION_SHA256,
        "integration_code_fingerprint": integration_fingerprint,
        "generation_settings": GENERATION_SETTINGS,
        "sequences": {
            f"{method}/seed{seed}": outputs
            for (method, seed), outputs in generated.items()
        },
        "generation_telemetry": generation_telemetry,
    }
    torch.save(raw_payload, raw_path)
    data_volume.commit()

    actual_gpu = torch.cuda.get_device_name(0)
    scored = _score_generated_payload(
        generated=generated,
        generation_telemetry=generation_telemetry,
        raw_path=raw_path,
        prompt_rows=prompt_rows,
        online_records=online_records,
        null_records=null_records,
        prc_artifact=prc_artifact,
        model_revision=model_revision,
        integration_fingerprint=integration_fingerprint,
        prc_fingerprint=prc_fingerprint,
        actual_gpu=actual_gpu,
    )
    torch.cuda.synchronize()
    overall_seconds = time.perf_counter() - overall_started
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    scored.update({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "gpu_function_seconds": overall_seconds,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "requested_gpu": "H100",
            "actual_gpu": actual_gpu,
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
        },
        "provenance": {
            "model_revision": model_revision,
            "tokenizer_revision": model_revision,
            "image_definition_sha256": IMAGE_DEFINITION_SHA256,
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "integration_code_fingerprint": integration_fingerprint,
            "prc_code_fingerprint": prc_fingerprint,
            "online_artifact_fingerprint": prc_artifact["artifact_fingerprint"],
            "online_key_sha256": __import__("online_prc").OnlinePRCKey.from_dict(
                prc_artifact["online_key"]
            ).fingerprint,
            "generation_attempts": {"online_prc": 0, "null": 0, "fixed_prc": 0},
            "fixed_prc_smoke_omitted": True,
            "fixed_prc_reason": "no exact frozen 8B eta=0.05 T=1024 cache; regeneration forbidden",
        },
    })
    return scored


def score_committed_smoke(
    data_volume,
    *,
    raw_path: str,
    source_app_url: str,
    source_task_id: str,
    actual_gpu: str,
) -> dict:
    """Resume validation/scoring from a committed raw smoke on Modal CPU."""
    print(f"[resume] loading committed generation artifact {raw_path}", flush=True)
    preload_official_runtimes()
    data_volume.reload()
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"committed smoke artifact is missing: {raw_path}")
    _numpy_pickle_compat()
    raw = torch.load(path, weights_only=False, map_location="cpu")
    if raw.get("prompt_indices") != list(SMOKE_PROMPT_INDICES):
        raise AssertionError("committed smoke prompt indices differ from the frozen five")
    if raw.get("generation_settings") != GENERATION_SETTINGS:
        raise AssertionError("committed smoke generation settings differ from the frozen settings")
    model_revision = _model_revision()
    if raw.get("model_revision") != model_revision:
        raise AssertionError("committed smoke model revision differs from the cached model")

    expected_keys = {
        f"{method}/seed{seed}"
        for method in ("textseal", "synthid_text", "gumbel_max")
        for seed in (PRIMARY_SEED, SECONDARY_SEED)
    }
    if set(raw.get("sequences", {})) != expected_keys:
        raise AssertionError("committed smoke method/seed coverage is incomplete")
    generated = {}
    for key, outputs in raw["sequences"].items():
        method, seed_label = key.split("/seed", 1)
        generated[(method, int(seed_label))] = outputs
        expected_rows = 5 if int(seed_label) == PRIMARY_SEED else 1
        if len(outputs) != expected_rows:
            raise AssertionError(f"{key} has {len(outputs)} rows, expected {expected_rows}")
        for output in outputs:
            if len(output.get("token_ids", [])) != MAX_NEW_TOKENS:
                raise AssertionError(f"{key} has a non-1024 continuation")

    prompt_rows = [
        json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    online_records, null_records, prc_artifact = _load_cached_sequences(prompt_rows)
    integration_fingerprint = _integration_fingerprint()
    prc_fingerprint = _prc_fingerprint()
    scored = _score_generated_payload(
        generated=generated,
        generation_telemetry=raw["generation_telemetry"],
        raw_path=raw_path,
        prompt_rows=prompt_rows,
        online_records=online_records,
        null_records=null_records,
        prc_artifact=prc_artifact,
        model_revision=model_revision,
        integration_fingerprint=integration_fingerprint,
        prc_fingerprint=prc_fingerprint,
        actual_gpu=actual_gpu,
    )
    timed_generation_seconds = float(
        sum(item["method_seconds"] for item in raw["generation_telemetry"])
    )
    scored.update(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "model_load_seconds": None,
                "gpu_function_seconds": None,
                "timed_generation_loops_seconds": timed_generation_seconds,
                "gpu_function_runtime_source": (
                    "provider task billing; reconcile in cost ledger because the generation "
                    "worker stopped after committing output but before returning telemetry"
                ),
                "peak_cuda_allocated_bytes": None,
                "peak_cuda_reserved_bytes": None,
                "requested_gpu": "H100",
                "actual_gpu": actual_gpu,
                "cuda_capability": None,
            },
            "provenance": {
                "model_revision": model_revision,
                "tokenizer_revision": model_revision,
                "image_definition_sha256": IMAGE_DEFINITION_SHA256,
                "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
                "modal_task_id": source_task_id,
                "source_generation_app_url": source_app_url,
                "scoring_modal_task_id": os.environ.get("MODAL_TASK_ID"),
                "integration_code_fingerprint": integration_fingerprint,
                "generation_integration_code_fingerprint": raw.get(
                    "integration_code_fingerprint"
                ),
                "prc_code_fingerprint": prc_fingerprint,
                "online_artifact_fingerprint": prc_artifact["artifact_fingerprint"],
                "online_key_sha256": __import__("online_prc").OnlinePRCKey.from_dict(
                    prc_artifact["online_key"]
                ).fingerprint,
                "generation_attempts": {"online_prc": 0, "null": 0, "fixed_prc": 0},
                "generated_text_reused_after_validation_only_stop": True,
                "fixed_prc_smoke_omitted": True,
                "fixed_prc_reason": (
                    "no exact frozen 8B eta=0.05 T=1024 cache; regeneration forbidden"
                ),
            },
        }
    )
    print("[resume] committed smoke scoring and validation passed", flush=True)
    return scored


FULL_SHARD_COUNT = 10
FULL_SHARD_SIZE = 50
FULL_GENERATION_BATCH_SIZE = 50


def _validated_full_request(request: dict) -> tuple[str, int, list[int]]:
    run_id = str(request.get("run_id", ""))
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("run_id must contain only letters, digits, hyphens, and underscores")
    shard_index = int(request["shard_index"])
    if not 0 <= shard_index < FULL_SHARD_COUNT:
        raise ValueError(f"shard_index must be in [0, {FULL_SHARD_COUNT})")
    start = shard_index * FULL_SHARD_SIZE
    return run_id, shard_index, list(range(start, start + FULL_SHARD_SIZE))


def generate_full_shard(data_volume, request: dict) -> dict:
    """Generate one approved 50-prompt shard and commit before scoring."""
    run_id, shard_index, prompt_indices = _validated_full_request(request)
    print(f"[full] generation shard {shard_index} cache/model validation", flush=True)
    preload_official_runtimes()
    data_volume.reload()
    prompt_rows = [
        json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line
    ]
    if len(prompt_rows) != 500:
        raise AssertionError("canonical prompt corpus is incomplete")
    _load_cached_sequences_for_indices(prompt_rows, prompt_indices)
    model_revision = _model_revision()
    integration_fingerprint = _integration_fingerprint()

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = load_qwen3_8b()
    model_load_seconds = time.perf_counter() - started
    generated: dict[str, list[dict]] = {}
    telemetry = []
    for method in ("textseal", "synthid_text", "gumbel_max"):
        method_outputs = []
        for batch_start in range(0, FULL_SHARD_SIZE, FULL_GENERATION_BATCH_SIZE):
            batch_indices = prompt_indices[
                batch_start : batch_start + FULL_GENERATION_BATCH_SIZE
            ]
            outputs, item = generate_method(
                model,
                [prompt_rows[index]["prompt_tokens"] for index in batch_indices],
                method=method,
                seed=PRIMARY_SEED,
            )
            item["prompt_indices"] = batch_indices
            telemetry.append(item)
            for output in outputs:
                output["runtime_seconds"] = item["seconds_per_prompt"]
            method_outputs.extend(outputs)
        if len(method_outputs) != FULL_SHARD_SIZE:
            raise AssertionError(f"{method} full shard coverage is incomplete")
        if any(len(output["token_ids"]) != MAX_NEW_TOKENS for output in method_outputs):
            raise AssertionError(f"{method} full shard contains a non-1024 continuation")
        generated[method] = method_outputs

    raw_root = Path("/data/controlled_baseline_full") / run_id / "generated"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_path = raw_root / f"shard_{shard_index:02d}.pt"
    if raw_path.exists():
        raise FileExistsError(f"full-run shard already exists: {raw_path}")
    raw = {
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "model_revision": model_revision,
        "image_definition_sha256": IMAGE_DEFINITION_SHA256,
        "integration_code_fingerprint": integration_fingerprint,
        "generation_settings": GENERATION_SETTINGS,
        "generation_batch_size": FULL_GENERATION_BATCH_SIZE,
        "generation_attempts": {"online_prc": 0, "null": 0},
        "seed": PRIMARY_SEED,
        "sequences": generated,
        "generation_telemetry": telemetry,
        "runtime": {
            "model_load_seconds": model_load_seconds,
            "function_seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "requested_gpu": "H100",
            "actual_gpu": torch.cuda.get_device_name(0),
        },
        "modal_image_id": os.environ.get("MODAL_IMAGE_ID"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
    }
    torch.save(raw, raw_path)
    data_volume.commit()
    return {
        "passed": True,
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "generated_sequences": sum(len(value) for value in generated.values()),
        "path": str(raw_path),
        "size_bytes": raw_path.stat().st_size,
        "runtime": raw["runtime"],
        "generation_batch_size": FULL_GENERATION_BATCH_SIZE,
        "modal_task_id": raw["modal_task_id"],
    }


def score_full_shard(data_volume, request: dict) -> dict:
    """CPU-score one committed full-run shard into shared-schema JSONL."""
    run_id, shard_index, prompt_indices = _validated_full_request(request)
    preload_official_runtimes()
    data_volume.reload()
    prompt_rows = [
        json.loads(line) for line in Path(PROMPTS_PATH).read_text().splitlines() if line
    ]
    online_records, null_records, prc_artifact = _load_cached_sequences_for_indices(
        prompt_rows, prompt_indices
    )
    raw_path = (
        Path("/data/controlled_baseline_full")
        / run_id
        / "generated"
        / f"shard_{shard_index:02d}.pt"
    )
    if not raw_path.is_file():
        raise FileNotFoundError(f"full-run generated shard is missing: {raw_path}")
    _numpy_pickle_compat()
    raw = torch.load(raw_path, weights_only=False, map_location="cpu")
    if raw.get("prompt_indices") != prompt_indices:
        raise AssertionError("full-run shard prompt ordering differs")
    if raw.get("generation_settings") != GENERATION_SETTINGS:
        raise AssertionError("full-run shard settings differ")
    if raw.get("model_revision") != _model_revision():
        raise AssertionError("full-run shard model revision differs")
    if int(raw.get("generation_batch_size", -1)) != FULL_GENERATION_BATCH_SIZE:
        raise AssertionError("full-run shard generation batch size differs")

    integration_fingerprint = _integration_fingerprint()
    prc_fingerprint = _prc_fingerprint()
    records, validation = _score_prc_sequences(
        online_records=online_records,
        null_records=null_records,
        artifact=prc_artifact,
        prompt_rows=prompt_rows,
        model_revision=raw["model_revision"],
        integration_fingerprint=integration_fingerprint,
        prc_fingerprint=prc_fingerprint,
        prompt_indices=prompt_indices,
    )
    for method in ("textseal", "synthid_text", "gumbel_max"):
        outputs = raw["sequences"].get(method)
        if outputs is None or len(outputs) != FULL_SHARD_SIZE:
            raise AssertionError(f"{method} full-run shard output coverage differs")
        for row, prompt_index in enumerate(prompt_indices):
            sequence = outputs[row]
            scored, checked = _score_baseline_sequence(
                method=method,
                sequence=sequence,
                prompt_row=prompt_rows[prompt_index],
                prompt_index=prompt_index,
                sample_type="watermarked",
                seed=PRIMARY_SEED,
                model_revision=raw["model_revision"],
                integration_fingerprint=integration_fingerprint,
                prc_fingerprint=prc_fingerprint,
                provenance={
                    "mode": "generated_full_shard",
                    "remote_generation_artifact": str(raw_path),
                    "generation_artifact_key": f"{method}/row{row}",
                    "requested_gpu": "H100",
                    "actual_gpu": raw["runtime"]["actual_gpu"],
                    "shard_index": shard_index,
                },
                runtime_seconds=float(sequence["runtime_seconds"]),
            )
            records.extend(scored)
            validation.append(checked)

        for row, prompt_index in enumerate(prompt_indices):
            null = null_records[row]
            sequence = {
                "token_ids": list(map(int, np.asarray(null["tokens"])[:MAX_NEW_TOKENS])),
                "base_token_logprobs": list(
                    map(float, np.asarray(null["base_token_logprob"])[:MAX_NEW_TOKENS])
                ),
                "base_entropies": list(
                    map(float, np.asarray(null["base_lm_entropy"])[:MAX_NEW_TOKENS] * math.log(2.0))
                ),
            }
            scored, checked = _score_baseline_sequence(
                method=method,
                sequence=sequence,
                prompt_row=prompt_rows[prompt_index],
                prompt_index=prompt_index,
                sample_type="null",
                seed=int(null.get("experiment_seed", PRIMARY_SEED)),
                model_revision=raw["model_revision"],
                integration_fingerprint=integration_fingerprint,
                prc_fingerprint=prc_fingerprint,
                provenance={
                    "mode": "cache_only",
                    "source_root": NULL_ROOT,
                    "source_T": SHARED_NULL_SOURCE_T,
                    "generation_attempts": 0,
                    "record_prompt_index": prompt_index,
                    "shard_index": shard_index,
                    "method_specific_key_handling": (
                        "same null text, scored under this detector's fixed key domain"
                    ),
                },
                runtime_seconds=0.0,
            )
            records.extend(scored)
            validation.append(checked)

    if len(records) != FULL_SHARD_SIZE * 4 * 2 * len(PREFIX_LENGTHS):
        raise AssertionError(f"full scoring shard produced {len(records)} records")
    scored_root = Path("/data/controlled_baseline_full") / run_id / "scored"
    scored_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = scored_root / f"shard_{shard_index:02d}.jsonl"
    validation_path = scored_root / f"shard_{shard_index:02d}_validation.json"
    if jsonl_path.exists() or validation_path.exists():
        raise FileExistsError(f"full scored shard already exists for {shard_index}")
    with jsonl_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    validation_path.write_text(
        json.dumps(
            {
                "passed": True,
                "run_id": run_id,
                "shard_index": shard_index,
                "prompt_indices": prompt_indices,
                "record_count": len(records),
                "generation_attempts": {"online_prc": 0, "null": 0},
                "validation_records": validation,
                "generation_artifact": str(raw_path),
                "generation_integration_code_fingerprint": raw[
                    "integration_code_fingerprint"
                ],
                "scoring_integration_code_fingerprint": integration_fingerprint,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    data_volume.commit()
    return {
        "passed": True,
        "run_id": run_id,
        "shard_index": shard_index,
        "prompt_indices": prompt_indices,
        "record_count": len(records),
        "jsonl_path": str(jsonl_path),
        "validation_path": str(validation_path),
    }
