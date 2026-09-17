"""CPU regression against all 2,000 frozen raw-completion pilot candidates."""
import argparse
import json
from pathlib import Path

import torch
import numpy as np

from prompt_free.core import PROTOCOL, score
from prompt_free.manifest import file_sha, source_identity
from prompt_free.storage import json_write, load_pt, token_sha


def audit(artifact_directory, results_directory):
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root/"prompt_free/manifests/pilots.json").read_text())
    references = json.loads((root/"prompt_free/manifests/pilot_regression.json").read_text())
    summary = {}
    torch.set_num_threads(1)
    for case in manifest["cases"]:
        reference = references[case["id"]]
        output = Path(results_directory)/reference["run_id"]
        assert file_sha(output/"full.json") == reference["full_sha256"]
        prior = json.loads((output/"full.json").read_text())
        assert prior["protocol"] == PROTOCOL and prior["prepended_token_count"] == 0
        assert prior["first_coordinate_score"] == 0 and prior["inference_dtype"] == "bfloat16"
        assert prior["inference"]["actual_model_inputs_equal_raw_completion_prefix"]
        assert file_sha(output/"inputs.pt") == prior["input_sha256"]
        tokens = load_pt(output/"inputs.pt")["tokens"]
        artifact_path = Path(artifact_directory)/(case["id"]+".pt")
        assert file_sha(artifact_path) == case["artifact"]["sha256"]
        artifact = load_pt(artifact_path)
        traces = []
        for name, expected in prior["inference"]["trace_shard_sha256"].items():
            assert file_sha(output/name) == expected
            saved = load_pt(output/name)
            assert saved["protocol"] == PROTOCOL and saved["run_id"] == reference["run_id"]
            assert saved["start"] == sum(len(t) for t in traces)
            traces.append(saved["partition_probability_coordinates_2_to_n"])
        traces = torch.cat(traces).numpy()
        assert len(tokens) == len(traces) == len(prior["records"]) == len(case["records"]) == 1000
        counts = {s: {w: 0 for w in case["weights"]} for s in ("wm", "null")}
        maximum_error = 0.
        for row, old, token_ids, p in zip(case["records"], prior["records"], tokens, traces):
            assert (row["source"], row["prompt_idx"]) == (old["source"], old["prompt_idx"])
            assert token_sha(token_ids) == row["tokens_sha256"] == old["tokens_sha256"]
            for weight in case["weights"]:
                result = score(artifact, token_ids, p, construction=case["construction"],
                               fpr=case["fpr"], fpr_policy=case["fpr_policy"], weight=weight)
                info = result["blocks"][0] if "blocks" in result else result
                expected = old["scores"][weight]["raw_abstain"]
                assert result["decision"] == expected["decision"]
                for field in ("statistic", "threshold", "V"):
                    maximum_error = max(maximum_error, abs(info[field]-expected[field]))
                    # Historical aggregation used NumPy 1.26.0 on Linux;
                    # allow only float64 host-math roundoff in local replay.
                    np.testing.assert_allclose(info[field], expected[field], rtol=0, atol=1e-12)
                counts[row["source"]][weight] += result["decision"]
        assert counts == reference["counts"]
        summary[case["id"]] = {"all_1000_candidate_decisions_match_exactly": True, "counts": counts,
                                "max_absolute_statistic_V_threshold_error": maximum_error,
                                "float64_absolute_tolerance": 1e-12,
                                "baseline_full_sha256": reference["full_sha256"]}
    return {"passed": True, "model_inference_calls": 0, "candidates": 2000,
            "method_candidate_comparisons": 4000, "source": source_identity(root), "cases": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--results-directory", type=Path, default=Path("outputs/completion_only"))
    parser.add_argument("--output", type=Path, default=Path("outputs/prompt_free/pilot_regression.json"))
    args = parser.parse_args()
    result = audit(args.artifact_directory, args.results_directory)
    json_write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "source"}, indent=2))
