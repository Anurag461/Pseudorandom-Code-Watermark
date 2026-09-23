"""Reuse completed source/cache controls without relabeling the failed BF16 check."""
import json
from .config import HERE, DESIGN, implementation_hashes, sha256, digest_json

POLICY = 'source-and-cache-fp32-controls-v1'


def evidence():
    prior = HERE / 'evidence/sanity-20260923'
    numerical = HERE / 'evidence/numerical-20260923/numerical_diagnostic.json'
    manifest = json.loads((prior / 'manifest.json').read_text())
    source = json.loads((prior / 'source_check.json').read_text())
    control = json.loads(numerical.read_text())
    if digest_json(manifest['design']) != digest_json(DESIGN):
        raise ValueError('Experiment design differs from the completed controls')
    current = implementation_hashes()
    # Only orchestration/approval wiring may change without new scientific checks.
    for name, digest in manifest['implementation'].items():
        if name not in ('wang_prc_detector_ablation/cloud.py',
                        'wang_prc_detector_ablation/launch.py') and current.get(name) != digest:
            raise ValueError(f'Validated implementation changed: {name}')
    if source['status'] != 'passed' or not all(g['passed'] for g in source['groups']):
        raise ValueError('Original/adapted detector source check did not pass')
    if not all(control['controls'].get(k) is True for k in
               ('static_equals_concat', 'fp32_cached_uncached_close')):
        raise ValueError('Cache/FP32 controls did not pass')
    return dict(policy=POLICY, source_sha256=sha256(prior / 'source_check.json'),
                numerical_sha256=sha256(numerical),
                original_bf16_guard='failed; retained in evidence, not rerun or marked passed',
                remaining_short_T1p8_smoke='omitted to proceed to the requested results',
                precision='BF16 generation/replay unchanged',
                limitation='Controls cover one completion at prefixes 1,4,8,16; not batch-80 or long-context equivalence')
