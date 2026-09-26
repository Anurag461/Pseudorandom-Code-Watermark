import unittest
from pathlib import Path
import numpy as np
import torch
from prc_watermark.prc import (
    KeyGen,
    OnlinePRCKey,
    materialize_supports,
    target_row_count,
)
from prc_watermark.detectors import (
    detect_hoeffding,
    detect_online_hoeffding,
    _soft_tokens,
)


class PRCTests(unittest.TestCase):

    def test_saved_keys_load(self):
        paths = sorted((Path(__file__).resolve().parents[1] / "data/keys").glob("*.pt"))
        self.assertEqual(len(paths), 23)
        for path in paths:
            with self.subTest(key=path.name):
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                self.assertEqual(artifact["partition"].ndim, 2)
                if "encoding_key" in artifact:
                    self.assertEqual(
                        artifact["encoding_key"][0].shape[0],
                        artifact["decoding_key"][1].shape[1],
                    )
                else:
                    self.assertIn("online_key", artifact)

    def test_fixed_key_reproducibility(self):
        a, da = KeyGen(
            64,
            message_length=0,
            false_positive_rate=0.5,
            t=3,
            r=60,
            noise_rate=0.2,
            seed=12345,
        )
        b, db = KeyGen(
            64,
            message_length=0,
            false_positive_rate=0.5,
            t=3,
            r=60,
            noise_rate=0.2,
            seed=12345,
        )
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        self.assertEqual((da[1] != db[1]).nnz, 0)

    def test_causal_support_prefix(self):
        key = OnlinePRCKey.from_seed(12345, check_weight=3, noise_rate=0.05)
        short = materialize_supports(32, key)
        long = materialize_supports(64, key)
        np.testing.assert_array_equal(short, long[: len(short)])
        self.assertTrue(np.all(short[:, :-1] < short[:, -1:]))
        self.assertEqual([target_row_count(n, key) for n in range(5)], [0, 0, 0, 1, 2])

    def test_evidence_alignment(self):
        bits = np.array([0, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(_soft_tokens(bits, [0.5, 0.5], "map"), [0, -1, 1])
        np.testing.assert_array_equal(_soft_tokens(bits, None, "naive"), [1, -1, 1])
        with self.assertRaises(ValueError):
            _soft_tokens(bits, [0.5, 0.5, 0.5], "map")
        with self.assertRaises(ValueError):
            _soft_tokens(bits, [0.5, np.nan], "entropy")

    def test_empty_evidence_rejects(self):
        key = OnlinePRCKey.from_seed(12345, check_weight=3, noise_rate=0.05)
        partition = torch.tensor([[1, 0], [0, 1]])
        decision, info = detect_online_hoeffding(
            key, torch.tensor([0, 1]), [0.5], partition, fpr=0.001, return_info=True
        )
        self.assertFalse(decision)
        self.assertEqual(info["r"], 0)
        _, fixed = KeyGen(
            64,
            message_length=0,
            false_positive_rate=0.5,
            t=3,
            r=60,
            noise_rate=0.2,
            seed=12345,
        )
        self.assertFalse(
            detect_hoeffding(fixed, torch.tensor([0]), [], partition, fpr=0.001)
        )


if __name__ == "__main__":
    unittest.main()
