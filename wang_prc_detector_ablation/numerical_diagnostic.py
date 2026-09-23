"""Bounded cache/precision diagnostic, only on the already saved failing tokens.

No text generation, no detector evaluation, no changed BF16 acceptance tolerance.
This module requires a separately approved GPU invocation via launch.py.
"""
from pathlib import Path
import json
import time
from .storage import load_arrays, write_json
from .hierarchy import probabilities

SOURCE_FINGERPRINT = '35f5566b766f8be3aea1cef9'
SOURCE_ROOT = 'wang_prc_detector_ablation/qwen3_8b_base/' + SOURCE_FINGERPRINT
SOURCE_SAMPLE = 'null_g00_p00_t1.0'
PREFIXES = (1, 4, 8, 16)


def distance(a, b):
    import torch
    pa, pb = probabilities(a, 1.0), probabilities(b, 1.0)
    return dict(max_logit_error=float((a.double()-b.double()).abs().max()),
                total_variation=float((pa-pb).abs().sum()/2),
                logits_equal=bool(torch.equal(a, b)))


def collect(model, tokens, cache_factory):
    """Identical inputs/positions for static and concatenating cache controls."""
    import torch
    out = {'static': {}, 'concat': {}, 'uncached': {}}
    with torch.inference_mode():
        for name in ('static', 'concat'):
            cache = cache_factory(name)
            for i in range(16):
                logits = model(tokens[:, i:i+1], cache=cache)[:, -1]
                if i+1 in PREFIXES:
                    out[name][i+1] = logits.detach().clone()
        for length in PREFIXES:
            out['uncached'][length] = model(tokens[:, :length])[:, -1].detach().clone()
    return out


def run(cache_root, output, provenance):
    import torch
    from qwen import make_kv_cache
    from .lm import load_model
    started = time.monotonic()
    source = Path('/data') / SOURCE_ROOT / 'sanity/traces' / (SOURCE_SAMPLE + '.npz')
    arrays, metadata = load_arrays(source, {'fingerprint': SOURCE_FINGERPRINT, 'context': 'completion-only'})
    if metadata['sample']['id'] != SOURCE_SAMPLE or metadata['sample']['temperature'] != 1.0:
        raise ValueError('Unexpected diagnostic source identity')
    model, _, _, model_meta = load_model(cache_root)
    if metadata['model']['files'] != model_meta['files']:
        raise ValueError('Diagnostic checkpoint differs from failed smoke checkpoint')
    tokens = torch.tensor(arrays['tokens'][:16][None], device='cuda')
    make = lambda name: make_kv_cache(name, max_length=16)
    bf16 = collect(model, tokens, make)
    # FP32 is a numerical reference only; production precision is unchanged.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model.to(dtype=torch.float32)
    model.cfg['dtype'] = torch.float32
    fp32 = collect(model, tokens, make)
    rows = []
    for length in PREFIXES:
        rows.append(dict(prefix=length,
            bf16_static_vs_concat=distance(bf16['static'][length], bf16['concat'][length]),
            fp32_static_vs_concat=distance(fp32['static'][length], fp32['concat'][length]),
            bf16_static_vs_uncached=distance(bf16['static'][length], bf16['uncached'][length]),
            fp32_static_vs_uncached=distance(fp32['static'][length], fp32['uncached'][length]),
            bf16_vs_fp32_static=distance(bf16['static'][length], fp32['static'][length]),
            bf16_vs_fp32_uncached=distance(bf16['uncached'][length], fp32['uncached'][length])))
    result = dict(status='diagnostic-only', source_root=SOURCE_ROOT, source_sample=SOURCE_SAMPLE,
        source_npz_sha256=json.loads(source.with_suffix('.json').read_text())['sha256'],
        provenance=provenance, model=model_meta, prefixes=list(PREFIXES),
        evaluations=rows, seconds=time.monotonic()-started,
        generated_tokens=0, teacher_forced_token_positions=122,
        batch_size=1, bf16_smoke_tolerance_unchanged=.02,
        controls=dict(static_equals_concat=all(r[p]['logits_equal'] for r in rows for p in
                        ('bf16_static_vs_concat','fp32_static_vs_concat')),
                      fp32_cached_uncached_close=all(r['fp32_static_vs_uncached']['total_variation'] <= 1e-4
                        and r['fp32_static_vs_uncached']['max_logit_error'] <= .002 for r in rows)),
        control_limits=dict(fp32_total_variation=1e-4, fp32_max_logit_error=.002),
        limitation='Does not overwrite the failed smoke gate or authorize production.')
    write_json(output, result)
    return result
