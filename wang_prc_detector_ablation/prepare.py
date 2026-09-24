"""Paid CPU preparation and optional official-artifact detector validation."""
import contextlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
from .config import DESIGN, HERE, fingerprint, implementation_hashes, inventory, seed, sha256
from .storage import save_arrays, load_arrays, write_json, exists
from .wang import keygen, encode, hard_count, hard_threshold, token_bits


def prepare(root, provenance, smoke=False):
    root = Path(root)
    keys = [('wm', i) for i in range(1 if smoke else 10)]
    if not smoke:
        keys += [(split, i) for split in ('calibration', 'evaluation') for i in range(256)]
    for domain, i in keys:
        path = root / 'keys' / f'{domain}_{i:03d}.npz'
        identity = dict(fingerprint=fingerprint(), domain=domain, group=i,
                        seed=seed('key', domain, i))
        if exists(path):
            key, _ = load_arrays(path, identity)
        else:
            key = keygen(DESIGN['n'], np.random.default_rng(identity['seed']))
            save_arrays(path, key, identity)
        if domain == 'wm':
            for prompt in range(2 if smoke else 16):
                cpath = root / 'codewords' / f'g{i:02d}_p{prompt:02d}.npz'
                meta = dict(fingerprint=fingerprint(), group=i, prompt=prompt,
                            seed=seed('codeword', i, prompt), key_sha256=sha256(path))
                if exists(cpath):
                    load_arrays(cpath, meta)
                else:
                    save_arrays(cpath, encode(key, np.random.default_rng(meta['seed'])), meta)
    manifest = dict(design=DESIGN, fingerprint=fingerprint(), provenance=provenance,
                    implementation=implementation_hashes(), smoke=smoke,
                    inventory=list(inventory(smoke)), keys=[f'{d}_{i:03d}' for d, i in keys])
    write_json(root / 'manifest.json', manifest)
    return manifest


def source_check(archive, output):
    """Call the vendored unmodified Detect; compare every recovered bit/decision."""
    import importlib.util
    from scipy.sparse import csr_matrix
    expected = DESIGN['source']['complete_archive_sha256']
    if sha256(archive) != expected:
        raise ValueError('Official complete archive hash mismatch')
    spec = importlib.util.spec_from_file_location('wang_official', HERE / 'vendor/llm_prc_api.py')
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rows = []
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.json') and '__MACOSX' not in n)
        if len(names) != 10:
            raise ValueError('Expected ten complete T=1.8 groups')
        for name in names:
            d = json.loads(z.read(name))
            key = dict(supports=np.asarray(d['secret_key'], dtype=np.int32),
                       otp=np.asarray(d['one_time_pad'], dtype=np.uint8))
            supports = key['supports']
            r = len(supports)
            pcm = csr_matrix((np.ones(supports.size),
                              (np.repeat(np.arange(r), 3), supports.ravel())),
                             shape=(r, len(key['otp'])))
            dec = (None, pcm, key['otp'], .1, None, 3)
            counts, adapted, references = [], [], []
            recovered = token_bits(d['watermark_sentence_tokens']).reshape(16, -1)
            if not np.array_equal(recovered, np.asarray(d['inv_msg']['val'])):
                raise ValueError(f'Saved recovered bits mismatch: {name}')
            for bits in recovered:
                h = int(hard_count(bits, key))
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    reference = bool(official.Detect(dec, bits))
                original_h = int(re.search(r' sum=(\d+) ', captured.getvalue()).group(1))
                if h != original_h:
                    raise ValueError(f'Original/adapted hard statistics disagree: {name}')
                counts.append(h)
                adapted.append(h <= hard_threshold(r))
                references.append(reference)
            saved = int(d['inv_msg']['succ'])
            passed = adapted == references and sum(adapted) == saved
            rows.append(dict(file=name, N=16, hard_counts=counts, adapted=sum(adapted),
                             original=sum(references), saved=saved, passed=passed))
            if not passed:
                write_json(output, dict(status='failed', groups=rows))
                raise ValueError(f'Original detector convention check failed: {name}')
    result = dict(status='passed', archive_sha256=expected, groups=rows,
                  limitation='DeepSeek complete T=1.8 source check, not Figure 5 or Base-model TPR')
    write_json(output, result)
    return result
