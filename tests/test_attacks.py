import unittest
import json
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from experiments.attacks.baseline_substitution import score
from experiments.attacks.substitution import apply_attack
from experiments.attacks.stealing import jsv_boosts


class AttackTests(unittest.TestCase):

    def test_reference_order_does_not_change_empirical_pvalue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.json"
            reference.write_text(json.dumps([{"statistic": x} for x in [2, 0, 1]]))
            nulls = root / "nulls.jsonl"
            nulls.write_text(
                json.dumps({"prompt_index": 0, "source": "null", "tokens": [0] * 400})
                + "\n"
            )
            (root / "batch.pt").touch()
            output = root / "scores.json"
            batch = {
                "prompt_idx": [0],
                "tokens": torch.zeros((1, 400), dtype=torch.long),
                "seeds": [1],
            }
            settings = {
                "model_directory": "unused",
                "scheme": "exp",
                "tokens": 400,
                "reference": str(reference),
                "null_completions": str(nulls),
                "generations": str(root),
                "rates": [0],
            }
            tokenizer = SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)
            with (
                patch.dict(
                    "sys.modules",
                    {"transformers": SimpleNamespace(AutoTokenizer=tokenizer)},
                ),
                patch(
                    "experiments.attacks.baseline_substitution.make_scorer",
                    return_value=lambda *args: 0.5,
                ),
                patch(
                    "experiments.attacks.baseline_substitution.torch.load",
                    return_value=batch,
                ),
            ):
                score(settings, output)
            rows = json.loads(output.read_text())
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertAlmostEqual(row["p"], 1 / 3)

    def test_substitution_seed_and_input(self):
        tokens = torch.arange(30)
        attack = {"kind": "substitution", "rate": 0.2, "seed": 0, "vocab_size": 100}
        a = apply_attack(tokens, attack, "wm", 3)
        b = apply_attack(tokens, attack, "wm", 3)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(tokens, torch.arange(30)))
        self.assertLessEqual(int((a != tokens).sum()), 6)

    def test_stealing_boost_threshold(self):
        wm = Counter({1: 2, 2: 2, 3: 1})
        base = Counter({1: 2, 2: 1, 4: 2})
        boosts = jsv_boosts(wm, base, False)
        self.assertNotIn(3, boosts)
        self.assertGreater(boosts[2], boosts[1])
        self.assertEqual(max(boosts.values()), 1.0)
        self.assertEqual(jsv_boosts({1: 4, 2: 8}, {1: 8, 2: 16}, False), {})


if __name__ == "__main__":
    unittest.main()
