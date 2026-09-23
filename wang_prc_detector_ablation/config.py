"""Frozen design and inexpensive provenance helpers (no model imports)."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
TEMPERATURES = (1.0, 1.2, 1.4, 1.6, 1.8)
MODEL = json.loads((HERE / 'model_manifest.json').read_text())
PROMPTS = json.loads((HERE / 'prompts.json').read_text())
DESIGN = dict(schema=1, model=MODEL, temperatures=TEMPERATURES, vocab=151936,
              bits=18, tokens=1024, n=18432, r=17510, t=3, eta=0.1,
              groups=10, prompts=PROMPTS, calibration_groups=list(range(5)),
              evaluation_groups=list(range(5, 10)), null_keys_per_split=256,
              fpr=0.001, bootstrap=2000, master_seed=251217310,
              generation_batch=80, replay_batch=80, backend='qwen-static-bf16',
              primary_context='completion-only-first-token-abstain',
              probability_arithmetic='fp32-log-softmax-fp64-positive-mass-tree',
              source=json.loads((HERE / 'source_manifest.json').read_text()))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def seed(*parts):
    return int(digest_json([DESIGN['master_seed'], *parts])[:16], 16)


def implementation_hashes():
    paths = list(HERE.glob('*.py')) + [HERE.parent / f for f in ('qwen.py', 'detectors.py', 'prc.py')]
    return {str(p.relative_to(HERE.parent)): sha256(p) for p in sorted(paths)}


def fingerprint():
    return digest_json({'design': DESIGN, 'implementation': implementation_hashes()})[:24]


def relative_root():
    return f'wang_prc_detector_ablation/qwen3_8b_base/{fingerprint()}'


def sample_id(source, group, prompt, temperature):
    return f'{source}_g{group:02d}_p{prompt:02d}_t{temperature:.1f}'


def inventory(smoke=False):
    for temp in ((1.0, 1.8) if smoke else TEMPERATURES):
        for source in ('wm', 'null'):
            for group in range(1 if smoke else 10):
                for prompt in range(2 if smoke else 16):
                    yield dict(id=sample_id(source, group, prompt, temp), source=source,
                               group=group, prompt=prompt, temperature=temp,
                               tokens=64 if smoke else 1024,
                               seed=seed('sample', source, group, prompt, temp),
                               split=('watermarked' if source == 'wm' else
                                      'calibration' if group < 5 else 'evaluation'))
