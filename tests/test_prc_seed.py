import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

from prc import KeyGen


def _key(seed):
    return KeyGen(
        n=64,
        message_length=0,
        false_positive_rate=0.5,
        t=3,
        r=60,
        noise_rate=0.2,
        seed=seed,
    )


def test_keygen_seed_reproduces_the_same_cross_workspace_key():
    (encoding_a, decoding_a) = _key(12345)
    (encoding_b, decoding_b) = _key(12345)

    assert np.array_equal(encoding_a[0], encoding_b[0])
    assert np.array_equal(encoding_a[1], encoding_b[1])
    assert np.array_equal(encoding_a[2], encoding_b[2])
    assert (decoding_a[1] != decoding_b[1]).nnz == 0


def test_keygen_different_seed_changes_the_key():
    encoding_a, _ = _key(12345)
    encoding_b, _ = _key(54321)

    assert not np.array_equal(encoding_a[0], encoding_b[0])
