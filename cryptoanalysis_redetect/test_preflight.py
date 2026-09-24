"""Small synthetic metadata tests; no experiment scoring or model execution."""
import json
import unittest
from pathlib import Path
import tempfile

from cryptoanalysis_redetect.preflight import (
    TEMPERATURES, inspect_archive, inspect_record, split_groups,
)


def record_bytes(temp=1.0):
    record = dict(origin_sentence=["original"] * 16,
                  watermark_sentence=["watermarked"] * 16,
                  correct_rate=[0.5] * 16, avg_entropy=[1.0] * 16,
                  det=[False] * 16, origin_det=[True] * 16)
    if temp != 1.0:
        record["generation_config"] = dict(temperature=temp, max_new_tokens=1024,
                                            top_k=0, top_p=1.0, do_sample=True)
    return json.dumps(record).encode()


class PreflightTests(unittest.TestCase):
    def test_absent_secrets_block_and_origin_det_is_not_null_fpr(self):
        row = inspect_record("gen_result/temperature_1.0/123.json", record_bytes())
        self.assertEqual(row["saved_det_true"], 0)
        self.assertEqual(set(row["missing_primary_fields"]), {
            "secret_key", "one_time_pad", "origin_sentence_tokens", "watermark_sentence_tokens"})
        self.assertEqual(row["missing_oracle_fields"], ["prompt_tokens"])
        self.assertNotIn("FPR", row)

    def test_temperature_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature mismatch"):
            inspect_record("gen_result/temperature_1.2/123.json", record_bytes(1.4))

    def test_only_temperature_one_can_omit_config(self):
        with self.assertRaisesRegex(ValueError, "missing generation_config"):
            inspect_record("gen_result/temperature_1.2/123.json", record_bytes())

    def test_flags_must_be_booleans(self):
        data = json.loads(record_bytes())
        data["det"][0] = "False"
        with self.assertRaisesRegex(ValueError, "booleans"):
            inspect_record("gen_result/temperature_1.0/123.json", json.dumps(data))

    def test_numeric_sort_then_half_split_independent_of_input_order(self):
        records = [dict(temperature=t, group_id=str(i), file=str(i))
                   for t in reversed(TEMPERATURES) for i in [10, 3, 2, 9, 6, 5, 4, 8, 7, 1]]
        split = split_groups(records, 8)
        for t in TEMPERATURES:
            chosen = [r for r in split if r["temperature"] == t]
            self.assertEqual([r["group_id"] for r in chosen], list(map(str, range(1, 9))))
            self.assertEqual([r["split"] for r in chosen], ["calibration"] * 4 + ["evaluation"] * 4)
        self.assertEqual(split_groups(list(reversed(records)), 8), split)

    def test_odd_limits_and_duplicate_ids_fail(self):
        with self.assertRaisesRegex(ValueError, "even"):
            split_groups([], 3)
        records = [dict(temperature=1.0, group_id="1", file="1")] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            split_groups(records, 2)

    def test_unpinned_archive_is_rejected_before_unpacking(self):
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root) / "data.zip"
            archive.write_bytes(b"not the original archive")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                inspect_archive(archive, Path(root) / "out")
            self.assertFalse((Path(root) / "out").exists())

    def test_path_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected group path"):
            inspect_record("../gen_result/temperature_1.0/123.json", record_bytes())


if __name__ == "__main__":
    unittest.main()
