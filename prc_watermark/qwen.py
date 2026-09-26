import torch
import torch.nn as nn
import json
from pathlib import Path


class FeedForward(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class RMSNorm(nn.Module):

    def __init__(self, emb_dim, eps=1e-06, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    def forward(self, x):
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        norm_x = norm_x * self.scale
        if self.shift is not None:
            norm_x = norm_x + self.shift
        return norm_x.to(input_dtype)


def compute_rope_params(
    head_dim, theta_base=10000, context_length=4096, dtype=torch.float32
):
    assert head_dim % 2 == 0, "Embedding dimension must be even"
    inv_freq = 1.0 / theta_base ** (
        torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float() / head_dim
    )
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (cos, sin)


def apply_rope(x, cos, sin, pos_offset=0):
    (batch_size, num_heads, seq_len, head_dim) = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[pos_offset : pos_offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[pos_offset : pos_offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = x * cos + rotated * sin
    return x_rotated.to(dtype=x.dtype)


class KVCache:

    def __init__(self):
        self.cache = {}

    def update(self, layer_idx, k, v):
        if layer_idx in self.cache:
            (past_k, past_v) = self.cache[layer_idx]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        self.cache[layer_idx] = (k, v)
        return (k, v)

    def get_seq_len(self):
        if not self.cache:
            return 0
        (any_k, _) = next(iter(self.cache.values()))
        return any_k.shape[2]

    def reset(self):
        self.cache = {}


CONCAT_KV_CACHE_VERSION = "concat-v1"
STATIC_KV_CACHE_VERSION = "static-v1"
KV_CACHE_IMPLEMENTATIONS = ("concat", "static")


def normalize_kv_cache_implementation(implementation="concat"):
    value = str(implementation or "concat").strip().lower()
    aliases = {"dynamic": "concat", "legacy": "concat", "preallocated": "static"}
    value = aliases.get(value, value)
    if value not in KV_CACHE_IMPLEMENTATIONS:
        raise ValueError(
            f"kv cache implementation must be one of {KV_CACHE_IMPLEMENTATIONS}; got {implementation!r}"
        )
    return value


def kv_cache_version(implementation="concat"):
    implementation = normalize_kv_cache_implementation(implementation)
    return (
        STATIC_KV_CACHE_VERSION
        if implementation == "static"
        else CONCAT_KV_CACHE_VERSION
    )


class StaticKVCache:

    def __init__(self, max_length):
        self.max_length = int(max_length)
        if self.max_length <= 0:
            raise ValueError("static KV cache max_length must be positive")
        self.cache = {}
        self._lengths = {}

    @staticmethod
    def _shape_without_sequence(tensor):
        if tensor.ndim != 4:
            raise ValueError(
                "KV tensors must have shape (batch, num_kv_groups, sequence, head_dim)"
            )
        return tensor.shape[:2] + tensor.shape[3:]

    def _allocate_layer(self, layer_idx, k, v):
        if self._shape_without_sequence(k) != self._shape_without_sequence(v):
            raise ValueError("key and value cache shapes are incompatible")
        shape = (int(k.shape[0]), int(k.shape[1]), self.max_length, int(k.shape[3]))
        self.cache[layer_idx] = (
            torch.empty(shape, dtype=k.dtype, device=k.device),
            torch.empty(shape, dtype=v.dtype, device=v.device),
        )
        self._lengths[layer_idx] = 0

    def update(self, layer_idx, k, v):
        self._shape_without_sequence(k)
        self._shape_without_sequence(v)
        if int(k.shape[2]) != int(v.shape[2]):
            raise ValueError("key and value updates must have equal lengths")
        if int(k.shape[2]) <= 0:
            raise ValueError("KV cache updates must be nonempty")
        if layer_idx not in self.cache:
            self._allocate_layer(layer_idx, k, v)
        (key_cache, value_cache) = self.cache[layer_idx]
        expected_shape = (
            int(key_cache.shape[0]),
            int(key_cache.shape[1]),
            int(key_cache.shape[3]),
        )
        if self._shape_without_sequence(k) != expected_shape:
            raise ValueError("key update shape changed after cache allocation")
        if self._shape_without_sequence(v) != expected_shape:
            raise ValueError("value update shape changed after cache allocation")
        if k.dtype != key_cache.dtype or k.device != key_cache.device:
            raise ValueError("key update dtype/device changed after cache allocation")
        if v.dtype != value_cache.dtype or v.device != value_cache.device:
            raise ValueError("value update dtype/device changed after cache allocation")
        start = int(self._lengths[layer_idx])
        end = start + int(k.shape[2])
        if end > self.max_length:
            raise ValueError(
                f"static KV cache capacity {self.max_length} exceeded by update ending at {end}"
            )
        with torch.no_grad():
            key_cache[:, :, start:end, :].copy_(k)
            value_cache[:, :, start:end, :].copy_(v)
        self._lengths[layer_idx] = end
        return (key_cache[:, :, :end, :], value_cache[:, :, :end, :])

    def get_seq_len(self):
        if not self._lengths:
            return 0
        lengths = set(self._lengths.values())
        if len(lengths) != 1:
            raise RuntimeError(
                "static KV cache layers have inconsistent sequence lengths"
            )
        return next(iter(lengths))

    def reset(self):
        for layer_idx in self._lengths:
            self._lengths[layer_idx] = 0

    def allocated_bytes(self):
        return sum(
            (
                tensor.numel() * tensor.element_size()
                for pair in self.cache.values()
                for tensor in pair
            )
        )


def make_kv_cache(implementation="concat", max_length=None):
    implementation = normalize_kv_cache_implementation(implementation)
    if implementation == "concat":
        return KVCache()
    if max_length is None:
        raise ValueError("static KV cache requires max_length")
    return StaticKVCache(max_length)


def teacher_force_partition_entropy_trace_batch(
    model,
    prompt_ids_batch,
    generated_tokens_batch,
    partition_one_mask,
    kv_cache_implementation="concat",
    chunk_size=1,
    return_full_entropy=True,
):
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("teacher-forcing chunk_size must be positive")
    decode_length = int(generated_tokens_batch.shape[1])
    if decode_length == 0:
        empty = torch.empty(
            (int(generated_tokens_batch.shape[0]), 0), dtype=torch.float32
        )
        return (empty, empty.clone() if return_full_entropy else None)
    cache = make_kv_cache(
        kv_cache_implementation,
        max_length=int(prompt_ids_batch.shape[1]) + max(decode_length - 1, 0),
    )
    part1 = partition_one_mask.to(prompt_ids_batch.device)
    partition_chunks = []
    entropy_chunks = []

    def append_evidence(step_logits):
        probabilities = torch.softmax(step_logits, dim=-1)
        partition_chunks.append(
            (probabilities * part1.to(step_logits.device)).sum(dim=-1).detach().cpu()
        )
        if return_full_entropy:
            log_probabilities = torch.log_softmax(step_logits.float(), dim=-1)
            full_probabilities = torch.exp(log_probabilities)
            entropy_chunks.append(
                -(full_probabilities * log_probabilities).sum(dim=-1).detach().cpu()
            )

    model.eval()
    with torch.no_grad():
        logits = model(prompt_ids_batch, cache=cache)[:, -1]
        append_evidence(logits[:, None, :])
        predicted = 1
        while predicted < decode_length:
            count = min(chunk_size, decode_length - predicted)
            logits = model(
                generated_tokens_batch[:, predicted - 1 : predicted - 1 + count],
                cache=cache,
            )
            append_evidence(logits)
            predicted += count
    return (
        torch.cat(partition_chunks, dim=1).float(),
        torch.cat(entropy_chunks, dim=1).float() if return_full_entropy else None,
    )


def completion_only_partition_trace_batch(
    model, completion_tokens, partition_one_mask, kv_cache_implementation="static"
):
    if (
        not isinstance(completion_tokens, torch.Tensor)
        or completion_tokens.dtype != torch.int64
        or completion_tokens.ndim != 2
        or (min(completion_tokens.shape) == 0)
    ):
        raise ValueError("expected nonempty int64 B x T completion tokens")
    if (
        partition_one_mask.ndim != 1
        or not torch.all((partition_one_mask == 0) | (partition_one_mask == 1))
        or torch.any(completion_tokens < 0)
        or torch.any(completion_tokens >= partition_one_mask.numel())
    ):
        raise ValueError("invalid partition or completion token IDs")
    trace = teacher_force_partition_trace_batch(
        model,
        completion_tokens[:, :1],
        completion_tokens[:, 1:],
        partition_one_mask,
        kv_cache_implementation=kv_cache_implementation,
        chunk_size=1,
    )
    if not torch.isfinite(trace).all() or torch.any((trace < 0) | (trace > 1)):
        raise ValueError("invalid response-only partition probabilities")
    return trace


def teacher_force_partition_trace_batch(
    model,
    prompt_ids_batch,
    generated_tokens_batch,
    partition_one_mask,
    kv_cache_implementation="concat",
    chunk_size=1,
):
    (partition_trace, _) = teacher_force_partition_entropy_trace_batch(
        model,
        prompt_ids_batch,
        generated_tokens_batch,
        partition_one_mask,
        kv_cache_implementation=kv_cache_implementation,
        chunk_size=chunk_size,
        return_full_entropy=False,
    )
    return partition_trace


class GroupedQueryAttention(nn.Module):

    def __init__(
        self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None
    ):
        super().__init__()
        assert (
            num_heads % num_kv_groups == 0
        ), "num_heads must be divisible by num_kv_groups"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        if head_dim is None:
            assert (
                d_in % num_heads == 0
            ), "`d_in` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(
            d_in, num_kv_groups * head_dim, bias=False, dtype=dtype
        )
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-06)
            self.k_norm = RMSNorm(head_dim, eps=1e-06)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin, cache=None, layer_idx=None, pos_offset=0):
        (b, num_tokens, _) = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        queries = apply_rope(queries, cos, sin, pos_offset=pos_offset)
        keys = apply_rope(keys, cos, sin, pos_offset=pos_offset)
        if cache is not None:
            (keys, values) = cache.update(layer_idx, keys, values)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        context = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )
        return self.out_proj(context)


class TransformerBlock(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-06)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-06)

    def forward(self, x, mask, cos, sin, cache=None, layer_idx=None, pos_offset=0):
        shortcut = x
        x = self.norm1(x)
        x = self.att(
            x, mask, cos, sin, cache=cache, layer_idx=layer_idx, pos_offset=pos_offset
        )
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


class Qwen3Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        (cos, sin) = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx, cache=None, key_padding_mask=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x.shape[1]
        pos_offset = cache.get_seq_len() if cache is not None else 0
        total_len = pos_offset + num_tokens
        q_pos = torch.arange(pos_offset, total_len, device=x.device)
        k_pos = torch.arange(total_len, device=x.device)
        mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
        if key_padding_mask is not None:
            kpm = key_padding_mask.to(device=x.device, dtype=torch.bool)
            if kpm.shape[1] < total_len:
                extra = torch.zeros(
                    kpm.shape[0],
                    total_len - kpm.shape[1],
                    dtype=torch.bool,
                    device=x.device,
                )
                kpm = torch.cat([kpm, extra], dim=1)
            mask = mask.unsqueeze(0).unsqueeze(0) | kpm[:, None, None, :]
            diag = q_pos.unsqueeze(1) == k_pos.unsqueeze(0)
            mask = mask & ~diag.unsqueeze(0).unsqueeze(0)
        for layer_idx, block in enumerate(self.trf_blocks):
            x = block(
                x,
                mask,
                self.cos,
                self.sin,
                cache=cache,
                layer_idx=layer_idx,
                pos_offset=pos_offset,
            )
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits


def calc_model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size
    total_buffers = sum((buf.numel() for buf in model.buffers()))
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    total_memory_gb = total_memory_bytes / 1024**3
    return total_memory_gb


def load_weights_into_qwen(model, param_config, params):

    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}"
            )
        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
        return left

    model.tok_emb.weight = assign(
        model.tok_emb.weight,
        params["model.embed_tokens.weight"],
        "model.embed_tokens.weight",
    )
    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight",
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight",
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight",
        )
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight",
        )
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight",
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight",
            )
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight",
        )
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight",
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight",
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight",
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight",
        )
    model.final_norm.scale = assign(
        model.final_norm.scale, params["model.norm.weight"], "model.norm.weight"
    )
    if "lm_head.weight" in params:
        model.out_head.weight = assign(
            model.out_head.weight, params["lm_head.weight"], "lm_head.weight"
        )
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


import re


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|object_ref_start|>",
        "<|object_ref_end|>",
        "<|box_start|>",
        "<|box_end|>",
        "<|quad_start|>",
        "<|quad_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|vision_pad|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<think>",
        "</think>",
    ]
    _SPLIT_RE = re.compile("(<\\|[^>]+?\\|>|<think>|</think>)")

    def __init__(
        self,
        tokenizer_file_path="tokenizer.json",
        repo_id=None,
        apply_chat_template=True,
        add_generation_prompt=False,
        add_thinking=False,
    ):
        from tokenizers import Tokenizer

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking
        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(str(tok_file))
        self._special_to_id = {}
        for t in self._SPECIALS:
            tid = self._tok.token_to_id(t)
            if tid is not None:
                self._special_to_id[t] = tid
        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id
        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "<|endoftext|>"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = text.strip()
        if stripped in self._special_to_id and "\n" not in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            text = self._wrap_chat(text)
        ids = []
        for part in filter(None, self._SPLIT_RE.split(text)):
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.extend(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def return_qwen_config(CHOOSE_MODEL: str):
    if CHOOSE_MODEL == "0.6B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 1024,
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 3072,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    elif CHOOSE_MODEL == "1.7B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 2048,
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    elif CHOOSE_MODEL == "4B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 2560,
            "n_heads": 32,
            "n_layers": 36,
            "hidden_dim": 9728,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    elif CHOOSE_MODEL == "8B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 4096,
            "n_heads": 32,
            "n_layers": 36,
            "hidden_dim": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    elif CHOOSE_MODEL == "14B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 5120,
            "n_heads": 40,
            "n_layers": 40,
            "hidden_dim": 17408,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    elif CHOOSE_MODEL == "32B":
        QWEN3_CONFIG = {
            "vocab_size": 151936,
            "context_length": 40960,
            "emb_dim": 5120,
            "n_heads": 64,
            "n_layers": 64,
            "hidden_dim": 25600,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1000000.0,
            "dtype": torch.bfloat16,
        }
    else:
        raise ValueError(f"{CHOOSE_MODEL} is not supported.")
    return QWEN3_CONFIG


def load_model(directory, size="0.6B", variant="base", device="cuda"):
    from safetensors.torch import load_file

    directory = Path(directory)
    config = return_qwen_config(size)
    model = Qwen3Model(config)
    model.to(device)
    index = directory / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values()))
        if index.exists()
        else ["model.safetensors"]
    )
    weights = {}
    for filename in files:
        weights.update(load_file(str(directory / filename)))
    load_weights_into_qwen(model, config, weights)
    del weights
    model.to(device).eval()
    repo_id = f"Qwen/Qwen3-{size}" + ("-Base" if variant == "base" else "")
    tokenizer = Qwen3Tokenizer(
        str(directory / "tokenizer.json"),
        repo_id=repo_id,
        apply_chat_template=variant != "base",
        add_generation_prompt=variant != "base",
        add_thinking=variant == "reasoning",
    )
    return (model, tokenizer)
