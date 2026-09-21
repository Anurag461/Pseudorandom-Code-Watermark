"""Pure Python scope/aggregation checks; no Modal, torch or experiment scoring."""
import ast
import copy
import json
from pathlib import Path
import unittest

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
tree = ast.parse((ROOT / "online_8b_eta020_remaining450.py").read_text())
functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
namespace = {"RUN": "online_8b_eta020_T14336_remaining450_v1"}
for name in ("verify_plan", "merge_reports"):
    node = copy.deepcopy(functions[name])
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), "isolated_metadata", "exec"), namespace)


def component(start, end):
    return {"passed": True, "protocol": "completion_only_raw_abstain_v1",
            "settings": {"eta": .2}, "reported_lengths": [14336],
            "trace_shard_sha256": {str(start): "synthetic"},
            "records": [{"prompt_idx": i, "source": "wm", "scores": {
                "14336": {"map": {"decision": i % 2 == 0}, "entropy": {"decision": False}}}}
                for i in range(start, end)]}


class MetadataTests(unittest.TestCase):
    def test_scope_excludes_completed_first_batch_and_extra_work(self):
        plan = json.loads((OUT / "setup.json").read_text())
        namespace["verify_plan"](plan)
        for key, value in (("prompt_indices", list(range(450))), ("null_count", 50),
                           ("small_detector_records", 450), ("allowance_usd", 100),
                           ("max_concurrent_GPUs", 10), ("automatic_retries", 1)):
            changed = copy.deepcopy(plan)
            changed[key] = value
            with self.assertRaises(ValueError, msg=key):
                namespace["verify_plan"](changed)

    def test_aggregation_reuses_existing_scores_without_mutation(self):
        old, new = component(0, 50), component(50, 500)
        saved = copy.deepcopy(old)
        merged = namespace["merge_reports"](old, new)
        self.assertEqual(old, saved)
        self.assertEqual(merged["records"][:50], saved["records"])
        self.assertEqual(merged["counts"]["14336"]["map"]["wm"], {"count": 500, "detected": 250})
        self.assertFalse(merged["reuse"]["first_batch_replayed"])
        self.assertFalse(merged["reuse"]["first_batch_rescored"])

    def test_duplicate_or_missing_prompt_cannot_enter_aggregate(self):
        for new in (component(49, 499), component(50, 499), component(50, 501)):
            with self.assertRaises(ValueError):
                namespace["merge_reports"](component(0, 50), new)

    def test_parallel_resource_decorator_and_native_import_order(self):
        decorator = functions["gpu_replay"].decorator_list[0]
        values = {k.arg: ast.literal_eval(k.value) for k in decorator.keywords
                  if k.arg in {"gpu", "max_containers", "retries", "timeout"}}
        self.assertEqual(values, {"gpu": "H200", "max_containers": 9, "retries": 0, "timeout": 5430})
        child = functions["child"].body
        imported = [a.name for statement in child[:7] if isinstance(statement, ast.Import) for a in statement.names]
        self.assertEqual(imported, ["torch", "scipy.special", "galois", "transformers",
                                    "safetensors.torch", "qwen", "detectors"])


if __name__ == "__main__":
    unittest.main()
