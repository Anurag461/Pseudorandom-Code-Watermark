from __future__ import annotations
import gc
import json
import time
from pathlib import Path
from typing import Sequence
import torch
from .config import (
    CONTEXT_LENGTH,
    TEMPERATURE,
    TOP_P,
    TEXTSEAL_ALPHA,
    SYNTHID_KEYS,
    MAX_NEW_TOKENS,
)
from .official import textseal_generator, gumbel_generator, synthid_processor

MODEL_ROOT = "/cache/models/Qwen3-8B-Base"


def load_qwen3_8b():
    from prc_watermark.qwen import Qwen3Model, return_qwen_config
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("the model must run on CUDA")
    root = Path(MODEL_ROOT)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"offline model index missing: {index_path}")
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    missing = [name for name in shard_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"offline model shards missing: {missing}")
    print("constructing Qwen3-8B directly on CUDA", flush=True)
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
        raise KeyError(
            f"cached Qwen index lacks integration weights: {missing_targets[:3]}"
        )
    loaded_targets = set()
    for name in shard_names:
        print(f"loading offline shard {name}", flush=True)
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
    print("Qwen3-8B offline model load complete", flush=True)
    return model


@torch.no_grad()
def generate_method(
    model,
    prompts: Sequence[Sequence[int]],
    *,
    method: str,
    seed: int,
    textseal_alpha: float = TEXTSEAL_ALPHA,
    synthid_keys: Sequence[int] = SYNTHID_KEYS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    device: str = "cuda",
) -> tuple[list[dict], dict]:
    from prc_watermark.qwen import StaticKVCache

    if method not in {"textseal", "synthid_text", "gumbel_max", "null"}:
        raise ValueError(f"unsupported generated method {method}")
    if not prompts or any((len(prompt) != 50 for prompt in prompts)):
        raise ValueError("generation requires nonempty, exactly 50-token prompts")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    device = torch.device(device)
    model.eval()
    batch_size = len(prompts)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.synchronize()
    started = time.perf_counter()
    all_tokens = torch.empty(
        (batch_size, 50 + max_new_tokens), dtype=torch.long, device=device
    )
    all_tokens[:, :50] = torch.tensor(prompts, dtype=torch.long, device=device)
    generated = torch.empty((batch_size, max_new_tokens), dtype=torch.long)
    logprobs = torch.empty((batch_size, max_new_tokens), dtype=torch.float32)
    entropies = torch.empty((batch_size, max_new_tokens), dtype=torch.float32)
    cache = StaticKVCache(max_length=50 + max_new_tokens)
    print(f"{method} seed={seed} prefill start batch={batch_size}", flush=True)
    logits = model(all_tokens[:, :50], cache=cache)[:, -1]
    if device.type == "cuda":
        torch.cuda.synchronize()
    print(f"{method} seed={seed} prefill complete", flush=True)
    sampler = None
    processor = None
    reference_processor = None
    synthid_reference_max_abs_difference = 0.0
    synthid_reference_indices_equal = True
    if method == "textseal":
        sampler = textseal_generator(alpha=textseal_alpha)
    elif method == "gumbel_max":
        sampler = gumbel_generator()
    elif method == "synthid_text":
        processor = synthid_processor(device, keys=synthid_keys)
        reference_processor = synthid_processor(device, keys=synthid_keys)
    for position in range(max_new_tokens):
        base_log_probs = torch.log_softmax(logits.float(), dim=-1)
        base_probs = torch.exp(base_log_probs)
        base_entropy = -(base_probs * base_log_probs).sum(dim=-1)
        if method in {"textseal", "gumbel_max"}:
            context = all_tokens[:, 50 + position - CONTEXT_LENGTH : 50 + position]
            next_token = sampler.sample_next(
                logits, context, temperature=float(TEMPERATURE), top_p=float(TOP_P)
            )
        elif method == "null":
            next_token = torch.multinomial(
                torch.softmax(logits.float(), dim=-1), 1
            ).reshape(-1)
        else:
            (updated, indices, _) = processor.watermarked_call(
                all_tokens[:, : 50 + position], logits
            )
            (reference_updated, reference_indices, _) = (
                reference_processor.watermarked_call(
                    all_tokens[:1, : 50 + position], logits[:1]
                )
            )
            synthid_reference_indices_equal &= bool(
                torch.equal(indices[:1], reference_indices)
            )
            difference = float(
                torch.max(
                    torch.abs(updated[:1].float() - reference_updated.float())
                ).item()
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
        if position + 1 < max_new_tokens:
            logits = model(next_token[:, None], cache=cache)[:, -1]
    if device.type == "cuda":
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
        "generated_tokens": batch_size * max_new_tokens,
        "method_seconds": elapsed,
        "seconds_per_prompt": elapsed / batch_size,
        "tokens_per_second": batch_size * max_new_tokens / elapsed,
        "synthid_reference": {
            "prompt_index": 0 if method == "synthid_text" else None,
            "indices_equal": (
                synthid_reference_indices_equal if method == "synthid_text" else None
            ),
            "max_abs_score_difference": (
                synthid_reference_max_abs_difference
                if method == "synthid_text"
                else None
            ),
        },
    }
    return (outputs, telemetry)
