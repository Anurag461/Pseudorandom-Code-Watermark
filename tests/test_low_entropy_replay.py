import numpy as np
import pytest
import torch

from detectors import detect_online_hoeffding
from low_entropy_replay import (
    effective_false_positive_rate,
    prepare_online_map_evidence,
    replay_cached_online_map_record,
    score_online_map_evidence,
)
from online_prc import OnlinePRCEncoder, OnlinePRCKey, derive_document_seed


def _key(seed=12345, weight=3, eta=0.05):
    return OnlinePRCKey.from_seed(
        seed, check_weight=weight, noise_rate=eta
    )


def _partition():
    return torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)


def _record(key, length=64):
    encoder = OnlinePRCEncoder(key, [derive_document_seed(91, 0)])
    noisy = encoder.encode_to_length(length)[0]
    return {
        "prompt_idx": 7,
        "watermark": True,
        "tokens": torch.as_tensor(noisy, dtype=torch.long),
        "p_trace": np.linspace(0.03, 0.97, length),
    }


@pytest.mark.parametrize("fpr_policy", ["one_shot", "alpha_spending_v1"])
def test_phase0_replay_exactly_reproduces_existing_hoeffding(fpr_policy):
    key = _key(811)
    record = _record(key)
    fpr = 1e-3
    legacy_decision, legacy = detect_online_hoeffding(
        key,
        record["tokens"],
        record["p_trace"],
        _partition(),
        fpr=fpr,
        weight="map",
        fpr_policy=fpr_policy,
        return_info=True,
    )
    evidence = prepare_online_map_evidence(
        key,
        record["tokens"],
        record["p_trace"],
        _partition(),
    )
    replay = score_online_map_evidence(
        evidence, false_positive_rate=fpr, fpr_policy=fpr_policy
    )
    baseline = replay["calibrations"]["hoeffding"]
    assert baseline["decision"] == legacy_decision
    assert replay["statistic"] == pytest.approx(legacy["statistic"])
    assert replay["V"] == pytest.approx(legacy["V"])
    assert baseline["threshold"] == pytest.approx(legacy["threshold"])
    assert replay["effective_fpr"] == pytest.approx(legacy["effective_fpr"])


def test_phase1_uses_identical_evidence_and_no_looser_threshold():
    key = _key(912)
    record = _record(key)
    evidence = prepare_online_map_evidence(
        key,
        record["tokens"],
        record["p_trace"],
        _partition(),
    )
    replay = score_online_map_evidence(
        evidence, false_positive_rate=1e-3
    )
    hoeffding = replay["calibrations"]["hoeffding"]
    rademacher = replay["calibrations"]["weighted_rademacher_chernoff"]
    assert rademacher["threshold"] <= hoeffding["threshold"]
    assert replay["threshold_ratio_rademacher_to_hoeffding"] <= 1.0
    assert rademacher["pvalue_upper"] <= 1.0
    assert rademacher["log_pvalue_upper"] <= 0.0


def test_saved_record_adapter_preserves_metadata_and_artifact_defaults():
    key = _key(101)
    record = _record(key)
    artifact = {
        "online_key": key.to_dict(),
        "partition": _partition(),
        "T": len(record["p_trace"]),
        "target_fpr": 1e-3,
    }
    replay = replay_cached_online_map_record(artifact, record)
    assert replay["prompt_idx"] == 7
    assert replay["watermark"] is True
    assert replay["fpr"] == pytest.approx(1e-3)
    assert replay["length"] == len(record["p_trace"])


def test_saved_record_adapter_requires_an_explicit_or_artifact_fpr():
    key = _key(102)
    record = _record(key)
    artifact = {
        "online_key": key.to_dict(),
        "partition": _partition(),
        "T": len(record["p_trace"]),
    }
    with pytest.raises(ValueError, match="false_positive_rate"):
        replay_cached_online_map_record(artifact, record)


def test_effective_fpr_matches_current_alpha_spending_policy():
    fpr = 1e-3
    length = 64
    assert effective_false_positive_rate(fpr, length) == fpr
    assert effective_false_positive_rate(
        fpr, length, "alpha_spending_v1"
    ) == pytest.approx(6.0 * fpr / (np.pi ** 2 * length ** 2))
    with pytest.raises(ValueError, match="policy"):
        effective_false_positive_rate(fpr, length, "unknown")
