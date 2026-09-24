"""Atomic, hash-checked caches. No pickle; incompatible resumes fail closed."""
import json
import os
from pathlib import Path
import numpy as np
from .config import sha256, digest_json


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save_arrays(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    write_json(path.with_suffix('.json'), {'metadata': metadata, 'sha256': sha256(path),
                                         'identity': digest_json(metadata)})


def load_arrays(path, expected=None):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['sha256'] != sha256(path) or meta['identity'] != digest_json(meta['metadata']):
        raise ValueError(f'Corrupt cache: {path}')
    if expected is not None:
        for k, v in expected.items():
            if meta['metadata'].get(k) != v:
                raise ValueError(f'Cache identity mismatch: {path}: {k}')
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return arrays, meta['metadata']


def exists(path):
    path = Path(path)
    # An interrupted two-file write is not silently overwritten or reused.
    if path.exists() != path.with_suffix('.json').exists():
        raise ValueError(f'Incomplete atomic cache; inspect before retry: {path}')
    return path.exists()
