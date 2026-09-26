import unittest
import csv
import json
import math
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch
import torch
from experiments.attacks.substitution import apply_attack, score
from experiments.attacks.stealing import jsv_boosts
from experiments.attacks import stealing


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
                    "experiments.attacks.substitution.make_scorer",
                    return_value=lambda *args: 0.5,
                ),
                patch(
                    "experiments.attacks.substitution.torch.load",
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

    def test_stealing_summary_calibration_and_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {
                "calib": list(range(5000, 0, -1)),
                "query500": [5] * 500,
                "pos_N1000_a2": [4, 5, 6, 5] + [6] * 496,
            }
            perplexities = {
                "calib": list(range(1, 501)),
                "query500": [1] * 500,
                "pos_N1000_a2": [475, 475.025, 475, 476] + [1] * 496,
            }
            for scheme in ("exp", "prc"):
                score_dir = root / "scores" / scheme
                ppl_dir = root / "ppl" / scheme
                score_dir.mkdir(parents=True)
                ppl_dir.mkdir(parents=True)
                for name, stats in values.items():
                    for start, stop in ((250, len(stats)), (0, 250)):
                        payload = {"scheme": scheme, "name": name, "start": start}
                        if scheme == "prc":
                            payload["rows"] = [
                                {
                                    "decision": i in (2, 3),
                                    "statistic": math.sqrt(-2 * math.log(x / 10000)),
                                    "V": 1,
                                }
                                for i, x in enumerate(stats[start:stop], start)
                            ]
                        else:
                            payload["stats"] = stats[start:stop]
                        (score_dir / f"{name}_{start}.json").write_text(
                            json.dumps(payload)
                        )
                    (ppl_dir / f"{name}.json").write_text(
                        json.dumps(
                            {"scheme": scheme, "name": name, "ppl": perplexities[name]}
                        )
                    )
            path = Path(stealing.summarize_attack(root, schemes="exp,prc"))
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            for scheme in ("exp", "prc"):
                by_set = {r["set"]: r for r in rows if r["scheme"] == scheme}
                spoof = by_set["pos_N1000_a2"]
                self.assertEqual(spoof["texts"], "500")
                self.assertEqual(spoof["variant"], "pos")
                self.assertEqual(spoof["n_query"], "1000")
                self.assertEqual(spoof["alpha"], "2")
                self.assertEqual(spoof["genuine TPR (query500)"], "100.0%")
                self.assertEqual(spoof["detected@1e-3"], "3/500 (0.6%)")
                self.assertEqual(spoof["detected & ppl-ok"], "2/500 (0.4%)")
                self.assertEqual(spoof["ppl-ok cut (calib p95)"], "475")
                self.assertEqual(by_set["calib"]["detected@1e-3"], "0/500 (0.0%)")
                cutoff = "0.0005" if scheme == "prc" else "5"
                self.assertEqual(
                    spoof["threshold"],
                    f"empirical 1e-3 quantile of 5000 unwatermarked (stat<={cutoff})",
                )
                self.assertEqual(
                    spoof["PRC proven-threshold detected"],
                    "2/500 (0.4%)" if scheme == "prc" else "",
                )
                self.assertEqual(
                    spoof["PRC proven-threshold & ppl-ok"],
                    "1/500 (0.2%)" if scheme == "prc" else "",
                )

    def test_stealing_spoof_generation_settings_and_alphas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = {"model_directory": "unused"}
            prompts = torch.ones((2, 50), dtype=torch.long)
            completions = torch.arange(400).repeat(2, 1)
            model = Mock()
            model.cuda.return_value = model
            model.eval.return_value = model
            model.generate.return_value = torch.cat([prompts, completions], 1)
            factory = Mock(return_value=model)
            processor = object()
            with (
                patch.dict(
                    "sys.modules",
                    {
                        "transformers": SimpleNamespace(
                            AutoModelForCausalLM=SimpleNamespace(
                                from_pretrained=factory
                            ),
                            LogitsProcessorList=list,
                        )
                    },
                ),
                patch.object(stealing, "stolen_table", return_value={}) as learn,
                patch.object(stealing, "_eval_prompts", return_value=prompts),
                patch.object(
                    stealing, "_stolen_processor", return_value=processor
                ) as make_processor,
                patch.object(torch.Tensor, "cuda", lambda self: self),
                patch.object(torch, "manual_seed") as seed,
            ):
                paths = stealing.spoof(settings, root, "exp", "pos", 1000, [2.0, 8.0])
            learn.assert_called_once_with(settings, root, "exp", "pos", 1000)
            factory.assert_called_once_with("unused", torch_dtype=torch.float32)
            self.assertEqual(seed.call_args_list, [call(2000), call(8000)])
            self.assertEqual(
                make_processor.call_args_list,
                [call({}, "pos", 256, 2.0), call({}, "pos", 256, 8.0)],
            )
            self.assertEqual(len(paths), 2)
            self.assertEqual(model.generate.call_count, 2)
            for invocation in model.generate.call_args_list:
                self.assertTrue(torch.equal(invocation.args[0], prompts))
                kwargs = dict(invocation.kwargs)
                self.assertTrue(
                    torch.equal(kwargs.pop("attention_mask"), torch.ones_like(prompts))
                )
                self.assertEqual(
                    kwargs,
                    {
                        "do_sample": True,
                        "max_new_tokens": 400,
                        "min_new_tokens": 400,
                        "top_k": 0,
                        "top_p": 1.0,
                        "temperature": 1.0,
                        "pad_token_id": stealing.EOS,
                        "logits_processor": [processor],
                    },
                )
            for alpha, path in zip((2.0, 8.0), paths):
                payload = torch.load(path, weights_only=False)
                self.assertEqual(payload["alpha"], alpha)
                self.assertTrue(torch.equal(payload["tokens"], completions))
                self.assertEqual(payload["tokens"].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
