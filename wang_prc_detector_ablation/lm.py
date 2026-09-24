"""Cached Base model generation and strictly completion-only replay.

Imported by paid GPU workers only. No model is instantiated at import time.
"""
import gc
import json
import time
from pathlib import Path
import numpy as np
from .config import DESIGN, MODEL, PROMPTS, sha256, fingerprint
from .storage import load_arrays, save_arrays, exists, write_json
from . import hierarchy


def load_model(cache_root):
    import torch
    from safetensors.torch import load_file
    from qwen import Qwen3Model, Qwen3Tokenizer, return_qwen_config, load_weights_into_qwen
    directory = Path(cache_root) / MODEL['cache_directory']
    expected = {**MODEL['weight_files'], 'model.safetensors.index.json': MODEL['index_sha256'],
                'tokenizer.json': MODEL['tokenizer_sha256']}
    verified = {}
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Cached model file does not match pinned checkpoint: {name}')
        verified[name] = digest
    for name in MODEL['weight_files']:
        metadata = directory / '.cache/huggingface/download' / (name + '.metadata')
        if not metadata.exists() or metadata.read_text().splitlines()[0] != MODEL['revision']:
            raise ValueError(f'Cached checkpoint revision mismatch: {name}')
    cfg = return_qwen_config('8B')
    if cfg['vocab_size'] != DESIGN['vocab']:
        raise ValueError('Unexpected model vocabulary')
    if (cfg['vocab_size'] - 1).bit_length() != DESIGN['bits']:
        raise ValueError('Derived token width differs from the frozen design')
    cfg['context_length'] = 2048  # Same RoPE positions; only the unused table tail is omitted.
    cfg['dtype'] = torch.bfloat16
    model = Qwen3Model(cfg)
    params = {}
    for name in MODEL['weight_files']:
        params.update(load_file(str(directory / name), device='cpu'))
    load_weights_into_qwen(model, cfg, params)
    del params
    gc.collect()
    model = model.to('cuda').eval().requires_grad_(False)
    tokenizer = Qwen3Tokenizer(str(directory / 'tokenizer.json'), repo_id=MODEL['id'],
                              apply_chat_template=False, add_generation_prompt=False,
                              add_thinking=False)
    prompt_ids = [tokenizer.encode(p, chat_wrapped=False) for p in PROMPTS]
    if any(not p for p in prompt_ids) or max(map(len, prompt_ids)) + 1024 > 2048:
        raise ValueError('Unexpected Base prompt length')
    return model, tokenizer, prompt_ids, dict(files=verified, revision=MODEL['revision'],
                                              torch=torch.__version__, device=torch.cuda.get_device_name())


def generate(model, prompt_ids, pad_id, specs, codewords):
    import torch
    from qwen import make_kv_cache
    batch, length = len(specs), specs[0]['tokens']
    temp = specs[0]['temperature']
    assert all(s['temperature'] == temp and s['tokens'] == length for s in specs)
    assert all((s['source'] == 'wm') == (codewords is not None) for s in specs)
    plen = max(map(len, prompt_ids))
    inputs = torch.full((batch, plen), pad_id, dtype=torch.long, device='cuda')
    padding = torch.ones_like(inputs, dtype=torch.bool)
    for row, ids in enumerate(prompt_ids):
        inputs[row, -len(ids):] = torch.tensor(ids, device='cuda')
        padding[row, -len(ids):] = False
    cache = make_kv_cache('static', max_length=plen + length)
    uniforms = torch.tensor(np.stack([np.random.default_rng(s['seed']).random(
        (length, DESIGN['bits'])) for s in specs]), device='cuda', dtype=torch.float64)
    codes = None if codewords is None else torch.tensor(
        codewords[:, :length * DESIGN['bits']].reshape(batch, length, DESIGN['bits']),
        device='cuda', dtype=torch.float64)
    tokens, paths, entropies = [], [], []
    with torch.inference_mode():
        logits = model(inputs, cache=cache, key_padding_mask=padding)[:, -1]
        for i in range(length):
            p = hierarchy.probabilities(logits, temp)
            token, path = hierarchy.walk(p, uniforms=uniforms[:, i],
                                         codeword=None if codes is None else codes[:, i])
            tokens.append(token)
            paths.append(path)
            entropies.append(hierarchy.entropy(p))
            if i + 1 < length:
                logits = model(token[:, None], cache=cache, key_padding_mask=padding)[:, -1]
    result = dict(tokens=torch.stack(tokens, 1).cpu().numpy(),
                  generation_p1=torch.stack(paths, 1).cpu().numpy(),
                  generation_entropy=torch.stack(entropies, 1).cpu().numpy())
    from .wang import token_bits
    recovered = token_bits(result['tokens'], DESIGN['bits'])
    u = uniforms.cpu().numpy()
    p = result['generation_p1']
    x = None if codes is None else codes.cpu().numpy()
    q = p if x is None else np.where(p <= .5, 2*p*x, 1-2*(1-p)*(1-x))
    if not np.array_equal(recovered, (u < q).astype(np.uint8)):
        raise ValueError('Saved probabilities/uniforms do not reproduce observed sampler branches')
    result['observed_bits'] = recovered
    validate_trace(result, length)
    return result


def replay(model, completion_ids, temperature, *, capture=None, cache_factory=None):
    """No prompt parameter, no tokenizer call and no synthetic initial token.

    capture is used only by the bounded smoke check. A fresh cache starts with
    completion token 0; its output predicts token 1. All coordinates for token
    0 are stored NaN (no probability exists) and later mapped to zero evidence.
    """
    import torch
    if cache_factory is None:
        from qwen import make_kv_cache
        cache_factory = lambda n: make_kv_cache('static', max_length=n)
    tokens = torch.as_tensor(completion_ids, device=next(model.parameters()).device,
                             dtype=torch.long)
    batch, length = tokens.shape
    if length < 2:
        raise ValueError('Replay needs at least two completion tokens')
    cache = cache_factory(length - 1)
    paths, entropies = [], []
    with torch.inference_mode():
        for i in range(1, length):
            model_input = tokens[:, i - 1:i]
            if capture is not None:
                capture.append(model_input.cpu().numpy().copy())
            logits = model(model_input, cache=cache)[:, -1]
            p = hierarchy.probabilities(logits, temperature)
            _, path = hierarchy.walk(p, observed=tokens[:, i])
            paths.append(path)
            entropies.append(hierarchy.entropy(p))
    return dict(replay_p1=np.concatenate((np.full((batch, 1, DESIGN['bits']), np.nan),
                    torch.stack(paths, 1).cpu().numpy()), axis=1),
                replay_entropy=np.concatenate((np.full((batch, 1), np.nan),
                    torch.stack(entropies, 1).cpu().numpy()), axis=1))


def validate_trace(trace, length):
    ids = trace['tokens']
    if ids.ndim != 2 or ids.shape[1] != length or np.any(ids < 0) or np.any(ids >= DESIGN['vocab']):
        raise ValueError('Invalid generated token IDs/length')
    for name in ('generation_p1', 'replay_p1'):
        if name not in trace:
            continue
        p = trace[name][:, 1:] if name == 'replay_p1' else trace[name]
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError(f'Invalid {name}')
        if trace[name].shape != (*ids.shape, DESIGN['bits']):
            raise ValueError(f'Unexpected {name} shape')
        if name == 'replay_p1' and not np.isnan(trace[name][:, 0]).all():
            raise ValueError('Replay supplied forbidden first-token probabilities')


def run_temperature(root, temperature, cache_root, provenance, smoke=False, commit=None, loaded=None):
    import torch
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['fingerprint'] != fingerprint() or manifest['smoke'] != smoke:
        raise ValueError('Preparation/config mismatch')
    specs = [s for s in manifest['inventory'] if s['temperature'] == temperature]
    begun = time.monotonic()
    pending = []
    for s in specs:
        path = root / 'traces' / (s['id'] + '.npz')
        expected = dict(sample=s, fingerprint=fingerprint(), context='completion-only')
        if exists(path):
            arrays, _ = load_arrays(path, expected)
            validate_trace({k: v[None] for k, v in arrays.items() if k != 'prompt_ids'}, s['tokens'])
        else:
            pending.append(s)
    if not pending:
        if smoke:
            prior = root / f'smoke_{temperature:.1f}.json'
            if not prior.exists() or json.loads(prior.read_text())['status'] != 'passed':
                raise ValueError('Cached smoke traces lack completed validation; quote a bounded validation retry')
        return dict(temperature=temperature, reused=len(specs), generated=0)
    model, tokenizer, prompts, model_meta = load_model(cache_root) if loaded is None else loaded
    load_seconds = time.monotonic() - begun
    batches = []
    for source in ('wm', 'null'):
        work = [s for s in pending if s['source'] == source]
        for offset in range(0, len(work), DESIGN['generation_batch']):
            batch = work[offset:offset + DESIGN['generation_batch']]
            batch_start = time.monotonic()
            words = None
            if source == 'wm':
                words = np.stack([load_arrays(root / 'codewords' /
                    f"g{s['group']:02d}_p{s['prompt']:02d}.npz", {'fingerprint': fingerprint()})[0]['codeword']
                    for s in batch])
            out = generate(model, [prompts[s['prompt']] for s in batch], tokenizer.pad_token_id, batch, words)
            generation_seconds = time.monotonic() - batch_start
            capture = [] if smoke else None
            out.update(replay(model, out['tokens'], temperature, capture=capture))
            if smoke and not np.array_equal(np.concatenate(capture, axis=1), out['tokens'][:, :-1]):
                raise ValueError('Prompt-free replay input capture failed')
            validate_trace(out, batch[0]['tokens'])
            for i, s in enumerate(batch):
                arrays = {k: v[i] for k, v in out.items()}
                arrays['prompt_ids'] = np.array(prompts[s['prompt']], dtype=np.int64)
                meta = dict(sample=s, fingerprint=fingerprint(), context='completion-only',
                            text=tokenizer.decode(arrays['tokens'].tolist()), model=model_meta,
                            provenance=provenance, batch_ids=[v['id'] for v in batch],
                            first_token='all 18 coordinates abstain',
                            codeword=None if source == 'null' else
                                f"codewords/g{s['group']:02d}_p{s['prompt']:02d}.npz")
                meta['key_sha256'] = None if source == 'null' else sha256(
                    root / 'keys' / f"wm_{s['group']:03d}.npz")
                meta['codeword_sha256'] = None if source == 'null' else sha256(root / meta['codeword'])
                save_arrays(root / 'traces' / (s['id'] + '.npz'), arrays, meta)
            if commit is not None:
                commit()
            batches.append(dict(source=source, N=len(batch), generation_seconds=generation_seconds,
                                total_seconds=time.monotonic() - batch_start,
                                cuda_peak_bytes=torch.cuda.max_memory_allocated()))
            if not smoke:
                done = sum(b['N'] for b in batches)
                projected = load_seconds + sum(b['total_seconds'] for b in batches) * len(pending) / done
                if projected > 3300:
                    write_json(root / f'budget_stop_{temperature:.1f}.json',
                               dict(projected_worker_seconds=projected, completed=done,
                                    reason='Projection exceeds quoted worker allowance; new approval required'))
                    if commit is not None:
                        commit()
                    raise RuntimeError('Projected runtime exceeds quote; saved completed batches and stopped')
    sanity = None
    if smoke:
        sanity = check_prefixes(model, out['tokens'][:1], temperature)
        write_json(root / f'smoke_{temperature:.1f}.json', sanity)
    result = dict(temperature=temperature, reused=len(specs)-len(pending), generated=len(pending),
                  load_seconds=load_seconds, batches=batches, seconds=time.monotonic()-begun,
                  smoke=sanity, model=model_meta, provenance=provenance)
    write_json(root / f'timing_{temperature:.1f}.json', result)
    return result


def check_prefixes(model, tokens, temperature):
    """Bounded smoke: cached versus uncached prefixes; no full extra LM pass."""
    import torch
    from qwen import make_kv_cache
    x = torch.tensor(tokens[:, :16], device='cuda')
    cache = make_kv_cache('static', max_length=16)
    errors = []
    with torch.inference_mode():
        for i in range(16):
            cached = model(x[:, i:i+1], cache=cache)[:, -1]
            if i in (0, 3, 7, 15):
                fresh = model(x[:, :i+1])[:, -1]
                a, b = hierarchy.probabilities(cached, temperature), hierarchy.probabilities(fresh, temperature)
                tv = float((a - b).abs().sum() / 2)
                errors.append(dict(prefix=i+1, max_logit_error=float((cached-fresh).abs().max()),
                                   total_variation=tv))
                # BF16 matmuls vary with query shape; fail on material distribution drift.
                if tv > .02:
                    raise ValueError(f'Cached/uncached prefix drift: {errors[-1]}')
    return dict(status='passed', replay_inputs_verified=True, prefix_checks=errors,
                prefix_tv_tolerance=.02, limitation='Short smoke is not a TPR estimate or batch-80 benchmark')
