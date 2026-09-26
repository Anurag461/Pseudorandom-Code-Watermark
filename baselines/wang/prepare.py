from pathlib import Path
import numpy as np
from .wang import keygen, encode
from .config import DESIGN, seed, fingerprint, inventory, implementation_hashes, sha256
from .storage import write_json, save_arrays, load_arrays, exists


def prepare(root, provenance):
    root = Path(root)
    keys = [("wm", i) for i in range(10)]
    keys += [(split, i) for split in ("calibration", "evaluation") for i in range(256)]
    for domain, i in keys:
        path = root / "keys" / f"{domain}_{i:03d}.npz"
        identity = dict(
            fingerprint=fingerprint(),
            domain=domain,
            group=i,
            seed=seed("key", domain, i),
        )
        if exists(path):
            (key, _) = load_arrays(path, identity)
        else:
            key = keygen(DESIGN["n"], np.random.default_rng(identity["seed"]))
            save_arrays(path, key, identity)
        if domain == "wm":
            for prompt in range(16):
                cpath = root / "codewords" / f"g{i:02d}_p{prompt:02d}.npz"
                meta = dict(
                    fingerprint=fingerprint(),
                    group=i,
                    prompt=prompt,
                    seed=seed("codeword", i, prompt),
                    key_sha256=sha256(path),
                )
                if exists(cpath):
                    load_arrays(cpath, meta)
                else:
                    save_arrays(
                        cpath, encode(key, np.random.default_rng(meta["seed"])), meta
                    )
    manifest = dict(
        design=DESIGN,
        fingerprint=fingerprint(),
        provenance=provenance,
        implementation=implementation_hashes(),
        smoke=False,
        inventory=list(inventory()),
        keys=[f"{d}_{i:03d}" for (d, i) in keys],
    )
    write_json(root / "settings.json", manifest)
    return manifest
