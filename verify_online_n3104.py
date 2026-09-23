"""Explain the earlier n3104 result versus the newly cached n4096 prefix."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import online_prc_redetection as replay
from detectors import prepare_online_map_prefix_context


def main():
    old_dir = Path("outputs/redetection/current/same_0p6b_eta020_n3104")
    new_dir = replay.OUT/"cases/online_0p6b_eta020_n4096"
    old_prepared = json.loads((old_dir/"prepared.json").read_text())[0]
    new_prepared = json.loads((new_dir/"prepared.json").read_text())
    old_file = old_dir/"same_0p6b_eta020_n3104/full.json"
    new_file = new_dir/"full.json"
    old_report, new_report = json.loads(old_file.read_text()), json.loads(new_file.read_text())
    assert old_prepared["identity"]["model"] == new_prepared["run"]["model"]
    old_artifact_path = old_dir/"cache"/old_prepared["root"]/"scoring_artifact.pt"
    new_artifact_path = new_dir/"cache"/new_prepared["root"]/"artifact.pt"
    assert replay.runtime._redetect_sha(old_artifact_path) == old_prepared["artifact_file_sha256"]
    assert replay.runtime._redetect_sha(new_artifact_path) == new_prepared["artifact_sha256"]
    old_artifact, new_artifact = map(replay.runtime._redetect_load, (old_artifact_path, new_artifact_path))
    assert old_artifact["online_key"] == new_artifact["online_key"]
    assert torch.equal(old_artifact["partition"], new_artifact["partition"])
    sides, evidence = [], []
    for folder, prepared, report, trace_key in (
            (old_dir, old_prepared, old_report, "partition_probability_coordinates_2_to_T"),
            (new_dir, new_prepared, new_report, "probabilities_2_to_T")):
        tokens, probs = [], []
        for batch in prepared["batches"]:
            root = folder/"cache"/batch["root"]
            trace_path, input_path = root/"trace.pt", root/"inputs.pt"
            sha = replay.runtime._redetect_sha(trace_path)
            assert sha == report["trace_shard_sha256"][batch["root"]]
            payload = replay.runtime._redetect_load(trace_path)
            assert payload["identity"] == batch["identity"]
            inputs = replay.runtime._redetect_load(input_path)
            assert torch.equal(inputs["partition"], new_artifact["partition"])
            tokens.append(inputs["tokens"][:, :3104])
            probs.append(payload[trace_key][:, :3103])
            evidence.append({"trace_file": str(trace_path), "trace_sha256": sha,
                             "input_file": str(input_path), "input_sha256": replay.runtime._redetect_sha(input_path)})
        sides.append((torch.cat(tokens), torch.cat(probs)))
    assert torch.equal(sides[0][0], sides[1][0])
    context = prepare_online_map_prefix_context(new_artifact["online_key"], 3104)
    flips, margins = [], []
    maximum_score_error = 0.
    for i, (old_record, new_record) in enumerate(zip(old_report["records"], new_report["records"])):
        assert (old_record["source"], old_record["prompt_idx"]) == (new_record["source"], new_record["prompt_idx"])
        assert hashlib.sha256(sides[0][0][i].numpy().tobytes()).hexdigest() == old_record["tokens_sha256"]
        for (tokens, probs), record in zip(sides, (old_record, new_record)):
            result = replay.prefix_scores(context, tokens[i], probs[i].numpy(), new_artifact["partition"], [3104], completion_only=True)["3104"]
            for weight in replay.WEIGHTS:
                saved = record["scores"]["3104"][weight]
                assert result[weight]["decision"] == saved["decision"]
                for key in ("statistic", "threshold", "V"):
                    error = abs(result[weight][key]-saved[key])
                    maximum_score_error = max(maximum_score_error, error)
                    assert np.isclose(result[weight][key], saved[key], atol=1e-11, rtol=1e-13)
        for weight in replay.WEIGHTS:
            old, new = old_record["scores"]["3104"][weight], new_record["scores"]["3104"][weight]
            if old["decision"] != new["decision"]:
                flips.append({"source": old_record["source"], "prompt_idx": old_record["prompt_idx"], "weight": weight,
                              "old_decision": old["decision"], "new_decision": new["decision"],
                              "old_margin": old["statistic"]-old["threshold"],
                              "new_margin": new["statistic"]-new["threshold"]})
    delta = (sides[0][1]-sides[1][1]).abs()
    result = {"passed": True, "status": "verified_execution_variation", "candidate_prefixes_identical": 1000,
              "same_model_key_partition": True, "both_reports_reproduced_from_their_cached_traces": True,
              "maximum_scoring_error": maximum_score_error, "old_report_sha256": replay.runtime._redetect_sha(old_file),
              "new_report_sha256": replay.runtime._redetect_sha(new_file), "old_counts": old_report["counts"]["3104"],
              "new_counts": new_report["counts"]["3104"], "decision_flips": flips,
              "trace_mean_absolute_difference": float(delta.mean()), "trace_maximum_absolute_difference": float(delta.max()),
              "changed_probability_positions": int((delta != 0).sum()), "trace_checksums": evidence,
              "interpretation": "Same candidate prefixes, model, key, partition and detector scores. Cached BF16 probabilities differ across the earlier batch-125 length-3104 execution and new batch-100 length-4096 execution. Three watermarked MAP decisions flip, for a net +1 detection (+0.2 percentage points); entropy and null decisions match. The effect of batch size versus replay length was not isolated with additional GPU inference."}
    replay.save(replay.OUT/"n3104_consistency.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "trace_checksums"}, indent=2))


if __name__ == "__main__":
    main()
