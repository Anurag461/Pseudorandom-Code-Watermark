"""Scientific equivalence and provenance checks for the online replay campaign."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import online_prc_redetection as replay
from detectors import detect_online_hoeffding, prepare_online_map_prefix_context
from online_prc import OnlinePRCKey


@pytest.mark.parametrize("completion_only", [False, True])
@pytest.mark.parametrize("eta", [.05, .10, .15, .20])
@pytest.mark.parametrize("constant", [False, True])
def test_prefix_scores_exactly_match_scalar_detector(completion_only, eta, constant):
    key = OnlinePRCKey.from_seed(12345, check_weight=3, noise_rate=eta)
    lengths = [1, 2, 3, 16, 251, 256, 512]
    rng = np.random.default_rng(72)
    tokens = torch.tensor(rng.integers(0, 2, 512), dtype=torch.int64)
    partition = torch.eye(2)
    probs = np.full(512-int(completion_only), .5) if constant else rng.uniform(size=512-int(completion_only))
    result = replay.prefix_scores(prepare_online_map_prefix_context(key, 512), tokens,
                                  probs, partition, lengths, completion_only=completion_only)
    for length in lengths:
        for weight in replay.WEIGHTS:
            decision, info = detect_online_hoeffding(
                key, tokens[:length], probs[:length-int(completion_only)], partition,
                fpr=.001, weight=weight, return_info=True, completion_only=completion_only)
            assert result[str(length)][weight] == {"decision": decision, **info}


def test_cached_nulls_reject_model_and_execution_mismatch(tmp_path):
    source = {"root": "old", "run": {"model": {"id": "pinned"}, "execution": {"files": {"code": "sha"}, "gpu": "A10G"}}}
    replay.save(tmp_path/"old/manifest.json", source["run"])
    target = json.loads(json.dumps(source))
    target["run"]["model"]["id"] = "other"
    with pytest.raises(ValueError, match="model/provenance"):
        replay.import_nulls(target, source, "unused", tmp_path)
    target = json.loads(json.dumps(source))
    target["run"]["execution"]["gpu"] = "A100-80GB"
    with pytest.raises(ValueError, match="execution mismatch"):
        replay.import_nulls(target, source, "unused", tmp_path)


@pytest.mark.skipif(not Path("outputs/fixed_0p6b_redetect_setup/cases/fixed_0p6b_eta010_n512/prepared.json").exists(), reason="local frozen source evidence not downloaded")
def test_approved_requests_cover_only_recorded_main_families():
    requests = replay.requests()
    assert [(r["eta"], r["replay_length"], r["batch_size"]) for r in requests] == [
        (.05, 512, 100), (.10, 1024, 125), (.15, 2048, 125), (.20, 4096, 100)]
    assert sum(len(r["reported_lengths"]) for r in requests) == 154
    for request in requests:
        assert len(request["expected_map"]) == len(request["null_refs"]) == 500
        assert len(request["reported_lengths"]) == len(set(request["reported_lengths"]))
        assert all(set(map(int, values)) <= set(request["reported_lengths"])
                   for values in request["expected_map"].values())
