import csv
import io
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiments.detection import entropy

ROOT = Path(__file__).resolve().parents[1]


class ResultsTests(unittest.TestCase):

    def test_entropy_runner_reports_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompts = root / "prompts.jsonl"
            prompts.write_text(json.dumps({"prompt_tokens": [0]}) + "\n")
            completions = root / "completions.jsonl"
            completions.write_text(
                json.dumps(
                    {"source": "unwatermarked", "prompt_index": 0, "tokens": [0, 1]}
                )
                + "\n"
            )
            settings = root / "settings.json"
            settings.write_text(
                json.dumps(
                    {
                        "model_directory": "unused",
                        "model_size": "test",
                        "artifact": "unused",
                        "prompts": str(prompts),
                        "completions": str(completions),
                        "batch_size": 1,
                    }
                )
            )
            output = root / "entropy.csv"
            partition = torch.tensor([[1, 0], [0, 1]])
            probabilities = torch.full((1, 2), 0.5, dtype=torch.float64)
            entropies = torch.full((1, 2), np.log(2), dtype=torch.float64)
            tensor = torch.tensor

            def cpu_tensor(data, **kwargs):
                kwargs.pop("device", None)
                return tensor(data, **kwargs)

            with (
                patch(
                    "sys.argv",
                    ["entropy", "--settings", str(settings), "--output", str(output)],
                ),
                patch.object(entropy, "load_model", return_value=(None, None)),
                patch.object(
                    entropy,
                    "teacher_force_partition_entropy_trace_batch",
                    return_value=(probabilities, entropies),
                ),
                patch.object(
                    entropy.torch, "load", return_value={"partition": partition}
                ),
                patch.object(entropy.torch, "tensor", side_effect=cpu_tensor),
            ):
                entropy.main()
            with output.open() as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["token_entropy_bits"]), 1.0)
            self.assertAlmostEqual(float(row["bucket_entropy_bits"]), 1.0)

    def test_fixed_counts_and_rates(self):
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (
                        ROOT / "experiments/detection/results/fixed_results.csv"
                    ).read_text()
                )
            )
        )
        self.assertEqual(len(rows), 57)
        self.assertEqual(len({(r["eta"], r["n"]) for r in rows}), 19)
        for row in rows:
            self.assertEqual(int(row["n_watermarked"]), 500)
            self.assertAlmostEqual(
                float(row["tpr"]),
                int(row["true_positives"]) / int(row["n_watermarked"]),
            )
            self.assertAlmostEqual(
                float(row["fpr"]),
                int(row["false_positives"]) / int(row["n_unwatermarked"]),
            )

    def test_figure_counts_and_intervals(self):
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (
                        ROOT / "experiments/detection/results/figure_results.csv"
                    ).read_text()
                )
            )
        )
        self.assertEqual(len(rows), 308)
        for row in rows:
            rate = 100 * int(row["detected"]) / int(row["n_watermarked"])
            self.assertAlmostEqual(rate, float(row["tpr_percent"]))
            self.assertLessEqual(float(row["ci_lower_percent"]), rate + 1e-10)
            self.assertGreaterEqual(float(row["ci_upper_percent"]), rate - 1e-10)

    def test_benchmark_scope(self):
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (ROOT / "experiments/benchmarks/results/results.csv").read_text()
                )
            )
        )
        self.assertEqual(
            {r["benchmark"] for r in rows},
            {"arc_easy", "gsm8k", "hellaswag", "mmlu", "ifeval"},
        )

    def test_diversity_pairing(self):
        data = json.loads(
            (ROOT / "experiments/diversity/results/paired_inputs.json").read_text()
        )
        self.assertEqual(len(data["contrasts"]), 35)
        self.assertTrue(
            all((len(row["differences"]) == 50 for row in data["contrasts"]))
        )
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (ROOT / "experiments/diversity/results/results.csv").read_text()
                )
            )
        )
        self.assertEqual(len(rows), 39)
        for row in rows:
            self.assertLessEqual(float(row["ci95_lower"]), float(row["mean"]) + 1e-12)
            self.assertGreaterEqual(
                float(row["ci95_upper"]), float(row["mean"]) - 1e-12
            )
            if row["holm_p"]:
                self.assertGreaterEqual(float(row["holm_p"]), float(row["p_value"]))

    def test_substitution_cohorts(self):
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (
                        ROOT / "experiments/attacks/results/substitution_results.csv"
                    ).read_text()
                )
            )
        )
        for row in rows:
            self.assertLessEqual(int(row["true_positives"]), int(row["n_watermarked"]))
            if int(row["tokens"]) == 4096 and float(row["substitution_rate"]) == 0.3:
                self.assertTrue(row["scheme"].startswith("prc"))


if __name__ == "__main__":
    unittest.main()
