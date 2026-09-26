import unittest
from collections import Counter
import torch
from experiments.attacks.substitution import apply_attack
from experiments.attacks.stealing import jsv_boosts


class AttackTests(unittest.TestCase):

    def test_substitution_seed_and_input(self):
        tokens = torch.arange(30)
        attack = {"kind": "substitution", "rate": 0.2, "seed": 0, "vocab_size": 100}
        a = apply_attack(tokens, attack, "wm", 3)
        b = apply_attack(tokens, attack, "wm", 3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(tokens, torch.arange(30)))
        self.assertLessEqual(int((a != tokens).sum()), 6)

    def test_float32_boost_threshold(self):
        wm = Counter({1: 2, 2: 2, 3: 1})
        base = Counter({1: 2, 2: 1, 4: 2})
        boosts = jsv_boosts(wm, base, False)
        self.assertNotIn(3, boosts)
        self.assertGreater(boosts[2], boosts[1])
        self.assertEqual(max(boosts.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
