from __future__ import annotations
import numpy as np
import torch
from prc_watermark.prc import Encode
from prc_watermark.qwen import (
    KVCache,
    normalize_kv_cache_implementation,
    make_kv_cache,
    kv_cache_version,
)


def signed_to_bits(signed: torch.Tensor) -> torch.Tensor:
    return ((1 - signed) / 2).long()


def generate_text_watermark_prc(
    model,
    token_ids,
    max_new_tokens,
    encoding_key,
    partition_map,
    eos_token_id=None,
    watermark=True,
    collect_trace_details=False,
):
    device = next(model.parameters()).device
    model.eval()
    n = encoding_key[0].shape[0]

    def _fresh_codeword():
        if watermark:
            signed = Encode(encoding_key)
            return signed_to_bits(signed).to(device).float()
        return torch.bernoulli(torch.full((n,), 0.5)).to(device)

    codeword = _fresh_codeword()
    partition_map = partition_map.to(device)
    with torch.no_grad():
        cache = KVCache()
        logits = model(token_ids, cache=cache)[:, -1]
        for pos in range(max_new_tokens):
            if pos > 0 and pos % n == 0:
                codeword = _fresh_codeword()
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * partition_map[1].to(logits.device)).sum(dim=-1)
            if collect_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                base_lm_entropy = -(base_probs * base_log_probs).sum(dim=-1)
                base_lm_entropy = base_lm_entropy / np.log(2)
            if watermark:
                xi = codeword[pos % n]
                bern_p = torch.where(
                    p1 <= 0.5, 2 * xi * p1, 1 - 2 * (1 - xi) * (1 - p1)
                ).clamp(0.0, 1.0)
                b = torch.bernoulli(bern_p).long()
                mask = partition_map[b].to(logits.device)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                sample_logits = logits
            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if collect_trace_details:
                base_token_logprob = base_log_probs.gather(1, next_token).squeeze(1)
                codeword_bit = None
                if watermark:
                    codeword_bit = xi.expand(token_ids.shape[0]).detach().cpu()
                yield (
                    next_token,
                    p1.detach().cpu(),
                    {
                        "prc_codeword_bit": codeword_bit,
                        "base_lm_entropy": base_lm_entropy.detach().cpu(),
                        "base_token_logprob": base_token_logprob.detach().cpu(),
                    },
                )
            else:
                yield (next_token, p1.detach().cpu())
            logits = model(next_token, cache=cache)[:, -1]


def generate_and_collect(generator):
    tok_chunks, p_chunks = ([], [])
    for item in generator:
        next_token, p1 = item[:2]
        tok_chunks.append(next_token)
        p_chunks.append(p1)
    if not tok_chunks:
        return (torch.zeros(0, dtype=torch.long), np.zeros(0, dtype=np.float64))
    tokens = torch.cat(tok_chunks, dim=1).flatten()
    p_trace = torch.stack(p_chunks).flatten().float().numpy().astype(np.float64)
    return (tokens, p_trace)


def generate_and_collect_detailed(generator):
    tok_chunks, p_chunks = ([], [])
    codeword_chunks, entropy_chunks, logprob_chunks = ([], [], [])
    saw_null_codeword = False
    for item in generator:
        if len(item) != 3:
            raise ValueError("detailed collection requires collect_trace_details=True")
        next_token, p1, details = item
        tok_chunks.append(next_token)
        p_chunks.append(p1)
        entropy_chunks.append(details["base_lm_entropy"])
        logprob_chunks.append(details["base_token_logprob"])
        codeword_bit = details["prc_codeword_bit"]
        if codeword_bit is None:
            saw_null_codeword = True
        else:
            codeword_chunks.append(codeword_bit)
    if not tok_chunks:
        empty = np.zeros(0, dtype=np.float32)
        return (
            torch.zeros(0, dtype=torch.long),
            np.zeros(0, dtype=np.float64),
            {
                "prc_codeword_bits": None,
                "base_lm_entropy": empty,
                "base_token_logprob": empty.copy(),
            },
        )
    if saw_null_codeword and codeword_chunks:
        raise ValueError("generation mixed null and PRC codeword trace steps")
    tokens = torch.cat(tok_chunks, dim=1).flatten()
    p_trace = torch.stack(p_chunks, dim=1).flatten().float().numpy().astype(np.float64)
    codeword_bits = None
    if codeword_chunks:
        codeword_bits = (
            torch.stack(codeword_chunks, dim=1).flatten().to(dtype=torch.uint8).numpy()
        )
    details = {
        "prc_codeword_bits": codeword_bits,
        "base_lm_entropy": torch.stack(entropy_chunks, dim=1).flatten().float().numpy(),
        "base_token_logprob": torch.stack(logprob_chunks, dim=1)
        .flatten()
        .float()
        .numpy(),
    }
    return (tokens, p_trace, details)


def generate_batch_and_collect(
    model,
    prompt_ids_batch,
    max_new_tokens,
    encoding_key,
    partition_map,
    watermark=True,
    return_trace_details=False,
):
    device = next(model.parameters()).device
    model.eval()
    B = prompt_ids_batch.shape[0]
    n = encoding_key[0].shape[0]
    pm = partition_map.to(device)
    part1 = pm[1]

    def _fresh_codewords():
        rows = [
            signed_to_bits(Encode(encoding_key)).to(device).float() for _ in range(B)
        ]
        return torch.stack(rows, dim=0)

    codeword = _fresh_codewords() if watermark else None
    tok_steps, p_steps = ([], [])
    codeword_steps, lm_entropy_steps, token_logprob_steps = ([], [], [])
    with torch.no_grad():
        cache = KVCache()
        logits = model(prompt_ids_batch, cache=cache)[:, -1]
        for pos in range(max_new_tokens):
            if watermark and pos > 0 and (pos % n == 0):
                codeword = _fresh_codewords()
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * part1.to(logits.device)).sum(dim=-1)
            if return_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                lm_entropy_steps.append(
                    -(base_probs * base_log_probs)
                    .sum(dim=-1)
                    .div(np.log(2))
                    .detach()
                    .cpu()
                )
            if watermark:
                xi = codeword[:, pos % n]
                bern_p = torch.where(
                    p1 <= 0.5, 2 * xi * p1, 1 - 2 * (1 - xi) * (1 - p1)
                ).clamp(0.0, 1.0)
                b = torch.bernoulli(bern_p).long()
                mask = pm[b].to(logits.device)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                sample_logits = logits
            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            tok_steps.append(next_token)
            p_steps.append(p1.detach().cpu())
            if return_trace_details:
                token_logprob_steps.append(
                    base_log_probs.gather(1, next_token).squeeze(1).detach().cpu()
                )
                if watermark:
                    codeword_steps.append(xi.detach().cpu())
            logits = model(next_token, cache=cache)[:, -1]
    tokens = torch.cat(tok_steps, dim=1).cpu()
    p_traces = torch.stack(p_steps, dim=1).float().numpy().astype(np.float64)
    if not return_trace_details:
        return (tokens, p_traces)
    details = {
        "prc_codeword_bits": (
            torch.stack(codeword_steps, dim=1).to(dtype=torch.uint8).numpy()
            if watermark
            else None
        ),
        "base_lm_entropy": torch.stack(lm_entropy_steps, dim=1).float().numpy(),
        "base_token_logprob": torch.stack(token_logprob_steps, dim=1).float().numpy(),
    }
    return (tokens, p_traces, details)


def generate_batch_and_collect_online(
    model,
    prompt_ids_batch,
    max_new_tokens,
    online_key,
    partition_map,
    watermark=True,
    return_trace_details=False,
    document_seeds=None,
    prefix_tokens_batch=None,
    prefix_codeword_bits_batch=None,
    kv_cache_implementation="concat",
):
    device = next(model.parameters()).device
    from prc_watermark.prc import (
        GENERATION_SAMPLER_VERSION,
        OnlinePRCEncoder,
        OnlinePRCKey,
        document_uniform,
    )

    if isinstance(online_key, dict):
        online_key = OnlinePRCKey.from_dict(online_key)
    model.eval()
    batch_size = int(prompt_ids_batch.shape[0])
    target_length = int(max_new_tokens)
    kv_cache_implementation = normalize_kv_cache_implementation(kv_cache_implementation)
    if target_length < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if watermark:
        if document_seeds is None:
            raise ValueError("online watermark generation requires document_seeds")
        if len(document_seeds) != batch_size:
            raise ValueError(
                f"got {len(document_seeds)} document seeds for batch {batch_size}"
            )
        encoder = OnlinePRCEncoder(online_key, document_seeds)
    else:
        encoder = None
    if prefix_tokens_batch is None:
        prefix_tokens = torch.empty(
            (batch_size, 0), dtype=torch.long, device=prompt_ids_batch.device
        )
    else:
        if not watermark:
            raise ValueError("saved-prefix resume is only supported for watermark=True")
        prefix_tokens = torch.as_tensor(
            prefix_tokens_batch, dtype=torch.long, device=prompt_ids_batch.device
        )
        if prefix_tokens.ndim != 2 or prefix_tokens.shape[0] != batch_size:
            raise ValueError(
                "prefix_tokens_batch must have shape (batch_size, prefix_length)"
            )
    prefix_length = int(prefix_tokens.shape[1])
    if prefix_length > target_length:
        raise ValueError(
            f"saved prefix length {prefix_length} exceeds target {target_length}"
        )
    if prefix_length:
        if prefix_codeword_bits_batch is None:
            raise ValueError(
                "resumed watermark generation requires prefix_codeword_bits_batch"
            )
        prefix_codeword = np.asarray(prefix_codeword_bits_batch, dtype=np.uint8)
        if prefix_codeword.shape != (batch_size, prefix_length):
            raise ValueError("prefix_codeword_bits_batch must match prefix token shape")
        expected_prefix = encoder.encode_to_length(prefix_length)
        if not np.array_equal(expected_prefix, prefix_codeword):
            raise ValueError(
                "saved prefix PRC bits do not match the requested online key, document seeds, and positions"
            )
    elif prefix_codeword_bits_batch is not None:
        supplied = np.asarray(prefix_codeword_bits_batch)
        if supplied.size:
            raise ValueError("nonempty prefix codeword supplied without prefix tokens")
    pm = partition_map.to(device)
    part1 = pm[1]
    print(
        (
            f"Watermark Enabled ({online_key.scheme}), batch={batch_size}, resume_prefix={prefix_length}"
            if watermark
            else f"Watermark Disabled, batch={batch_size}"
        ),
        flush=True,
    )
    token_steps, p_steps = ([], [])
    codeword_steps, entropy_steps, logprob_steps = ([], [], [])
    with torch.no_grad():
        cache = make_kv_cache(
            kv_cache_implementation,
            max_length=int(prompt_ids_batch.shape[1]) + target_length,
        )
        logits = model(prompt_ids_batch, cache=cache)[:, -1]
        for position in range(prefix_length):
            logits = model(prefix_tokens[:, position : position + 1], cache=cache)[
                :, -1
            ]
        for position in range(prefix_length, target_length):
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * part1.to(logits.device)).sum(dim=-1)
            if return_trace_details:
                base_log_probs = torch.log_softmax(logits.float(), dim=-1)
                base_probs = torch.exp(base_log_probs)
                entropy_steps.append(
                    -(base_probs * base_log_probs)
                    .sum(dim=-1)
                    .div(np.log(2))
                    .detach()
                    .cpu()
                )
            if watermark:
                xi = torch.as_tensor(
                    encoder.next_bits(), dtype=torch.float32, device=logits.device
                )
                bern_p = torch.where(
                    p1 <= 0.5, 2 * xi * p1, 1 - 2 * (1 - xi) * (1 - p1)
                ).clamp(0.0, 1.0)
                bucket_uniform = torch.tensor(
                    [
                        document_uniform(seed, "lm-bucket/v1", position)
                        for seed in document_seeds
                    ],
                    dtype=torch.float64,
                    device=logits.device,
                )
                bucket = (bucket_uniform < bern_p.double()).long()
                mask = pm[bucket].to(logits.device)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                xi = None
                sample_logits = logits
            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            if watermark:
                token_uniform = torch.tensor(
                    [
                        document_uniform(seed, "lm-token/v1", position)
                        for seed in document_seeds
                    ],
                    dtype=sample_probs.dtype,
                    device=sample_probs.device,
                ).unsqueeze(1)
                cumulative = torch.cumsum(sample_probs, dim=-1)
                cumulative[:, -1] = 1.0
                next_token = torch.searchsorted(
                    cumulative, token_uniform, right=False
                ).clamp_max(sample_probs.shape[1] - 1)
            else:
                next_token = torch.multinomial(sample_probs, num_samples=1)
            token_steps.append(next_token)
            p_steps.append(p1.detach().cpu())
            if return_trace_details:
                logprob_steps.append(
                    base_log_probs.gather(1, next_token).squeeze(1).detach().cpu()
                )
                if watermark:
                    codeword_steps.append(xi.detach().cpu())
            logits = model(next_token, cache=cache)[:, -1]
    suffix_length = target_length - prefix_length
    if suffix_length:
        tokens = torch.cat(token_steps, dim=1).cpu()
        p_traces = torch.stack(p_steps, dim=1).float().numpy().astype(np.float64)
    else:
        tokens = torch.empty((batch_size, 0), dtype=torch.long)
        p_traces = np.empty((batch_size, 0), dtype=np.float64)
    if not return_trace_details:
        return (tokens, p_traces)
    details = {
        "prc_codeword_bits": (
            torch.stack(codeword_steps, dim=1).to(dtype=torch.uint8).numpy()
            if watermark and suffix_length
            else np.empty((batch_size, 0), dtype=np.uint8) if watermark else None
        ),
        "base_lm_entropy": (
            torch.stack(entropy_steps, dim=1).float().numpy()
            if suffix_length
            else np.empty((batch_size, 0), dtype=np.float32)
        ),
        "base_token_logprob": (
            torch.stack(logprob_steps, dim=1).float().numpy()
            if suffix_length
            else np.empty((batch_size, 0), dtype=np.float32)
        ),
        "online_sampler_version": GENERATION_SAMPLER_VERSION if watermark else None,
        "kv_cache_implementation": kv_cache_implementation,
        "kv_cache_version": kv_cache_version(kv_cache_implementation),
        "resume_prefix_length": prefix_length,
    }
    return (tokens, p_traces, details)


def left_pad_batch(prompt_id_lists, pad_id):
    B = len(prompt_id_lists)
    L = max((len(p) for p in prompt_id_lists))
    ids = torch.full((B, L), pad_id, dtype=torch.long)
    key_pad = torch.ones(B, L, dtype=torch.bool)
    for i, p in enumerate(prompt_id_lists):
        ids[i, L - len(p) :] = torch.tensor(p, dtype=torch.long)
        key_pad[i, L - len(p) :] = False
    return (ids, key_pad)


def generate_batch_padded_and_collect(
    model,
    prompt_ids_batch,
    key_padding_mask,
    max_new_tokens,
    encoding_key,
    partition_map,
    eos_token_id=None,
    watermark=True,
):
    device = next(model.parameters()).device
    model.eval()
    B = prompt_ids_batch.shape[0]
    n = encoding_key[0].shape[0]
    pm = partition_map.to(device)
    part1 = pm[1]
    prompt_ids_batch = prompt_ids_batch.to(device)
    kpm = key_padding_mask.to(device)

    def _fresh_codewords():
        rows = [
            signed_to_bits(Encode(encoding_key)).to(device).float() for _ in range(B)
        ]
        return torch.stack(rows, dim=0)

    codeword = _fresh_codewords() if watermark else None
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    tok_steps, p_steps = ([], [])
    with torch.no_grad():
        cache = KVCache()
        logits = model(prompt_ids_batch, cache=cache, key_padding_mask=kpm)[:, -1]
        for pos in range(max_new_tokens):
            if watermark and pos > 0 and (pos % n == 0):
                codeword = _fresh_codewords()
            probs = torch.softmax(logits, dim=-1)
            p1 = (probs * part1.to(logits.device)).sum(dim=-1)
            if watermark:
                xi = codeword[:, pos % n]
                bern_p = torch.where(
                    p1 <= 0.5, 2 * xi * p1, 1 - 2 * (1 - xi) * (1 - p1)
                ).clamp(0.0, 1.0)
                b = torch.bernoulli(bern_p).long()
                mask = pm[b].to(logits.device)
                sample_logits = logits.masked_fill(mask == 0, float("-inf"))
            else:
                sample_logits = logits
            sample_probs = torch.softmax(sample_logits.float(), dim=-1)
            next_token = torch.multinomial(sample_probs, num_samples=1)
            if eos_token_id is not None:
                next_token = next_token.masked_fill(finished.unsqueeze(1), eos_token_id)
            tok_steps.append(next_token)
            p_steps.append(p1.detach().cpu())
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(1) == eos_token_id)
                if bool(finished.all()):
                    break
            logits = model(next_token, cache=cache, key_padding_mask=kpm)[:, -1]
    tokens = torch.cat(tok_steps, dim=1).cpu()
    p_traces = torch.stack(p_steps, dim=1).float().numpy().astype(np.float64)
    return (tokens, p_traces)
