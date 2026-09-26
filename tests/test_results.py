import csv
import io
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ResultsTests(unittest.TestCase):

    def test_fixed_counts_and_rates(self):
        rows = list(
            csv.DictReader(
                io.StringIO(
                    (ROOT / "experiments/detection/fixed_results.csv").read_text()
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
                    (ROOT / "experiments/detection/figure_results.csv").read_text()
                )
            )
        )
        self.assertEqual(len(rows), 308)
        for row in rows:
            rate = 100 * int(row["detected"]) / int(row["n_watermarked"])
            self.assertAlmostEqual(rate, float(row["tpr_percent"]))
            self.assertLessEqual(float(row["ci_lower_percent"]), rate + 1e-10)
            self.assertGreaterEqual(float(row["ci_upper_percent"]), rate - 1e-10)

    def test_quality_scope(self):
        rows = list(
            csv.DictReader(
                io.StringIO((ROOT / "experiments/quality/results.csv").read_text())
            )
        )
        self.assertEqual(
            {r["benchmark"] for r in rows},
            {"arc_easy", "gsm8k", "hellaswag", "mmlu", "ifeval"},
        )

    def test_diversity_pairing(self):
        data = json.loads(
            (ROOT / "experiments/diversity/paired_inputs.json").read_text()
        )
        self.assertEqual(len(data["contrasts"]), 35)
        self.assertTrue(
            all((len(row["differences"]) == 50 for row in data["contrasts"]))
        )
        rows = list(
            csv.DictReader(
                io.StringIO((ROOT / "experiments/diversity/results.csv").read_text())
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
                    (ROOT / "experiments/attacks/substitution_results.csv").read_text()
                )
            )
        )
        for row in rows:
            self.assertLessEqual(int(row["true_positives"]), int(row["n_watermarked"]))
            if int(row["tokens"]) == 4096 and float(row["substitution_rate"]) == 0.3:
                self.assertTrue(row["scheme"].startswith("prc"))


if __name__ == "__main__":
    unittest.main()
