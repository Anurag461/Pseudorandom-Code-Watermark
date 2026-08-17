import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from adaptive_parity_basis import (
    bucket_reliability,
    gf2_rank_from_masks,
    parity_row_masks,
    select_reliability_adaptive_basis,
)
from detectors import detect_hoeffding, detect_online_hoeffding
from low_entropy_replay import (
    effective_false_positive_rate,
    prepare_fixed_map_block_evidence,
    prepare_online_map_evidence,
    prepare_reliability_adaptive_fixed_map_block_evidence,
    replay_cached_fixed_map_record,
    replay_cached_fixed_map_record_phase2,
    replay_cached_online_map_record,
    score_online_map_evidence,
)
from online_prc import OnlinePRCEncoder, OnlinePRCKey, derive_document_seed
from prc import Encode, KeyGen


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


def test_fixed_phase0_replay_exactly_reproduces_existing_hoeffding():
    n = 64
    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=3,
        noise_rate=0.05,
        r=61,
        seed=2026,
    )
    codeword_signs = Encode(encoding_key, np.empty(0, dtype=np.uint8))
    codeword_bits = ((1 - codeword_signs.numpy()) / 2).astype(np.int64)
    tokens = torch.as_tensor(codeword_bits, dtype=torch.long)
    probabilities = np.linspace(0.03, 0.97, n)
    fpr = 1e-3
    legacy_decision, legacy = detect_hoeffding(
        decoding_key,
        tokens,
        probabilities,
        _partition(),
        fpr=fpr,
        weight="map",
        return_info=True,
    )
    artifact = {
        "decoding_key": decoding_key,
        "partition": _partition(),
        "n": n,
        "T": n,
    }
    record = {
        "prompt_idx": 1,
        "watermark": True,
        "tokens": tokens,
        "p_trace": probabilities,
    }
    replay = replay_cached_fixed_map_record(
        artifact, record, false_positive_rate=fpr
    )
    evidence = prepare_fixed_map_block_evidence(
        decoding_key, tokens, probabilities, _partition()
    )
    baseline = replay["calibrations"]["hoeffding"]
    assert baseline["decision"] == legacy_decision
    assert evidence["signed_check_values"].sum() == pytest.approx(
        legacy["statistic"]
    )
    assert evidence["squared_check_values"].sum() == pytest.approx(legacy["V"])
    assert baseline["threshold"] == pytest.approx(legacy["threshold"])
    assert replay["num_blocks"] == 1


def test_bucket_reliability_uses_absolute_partition_imbalance():
    probabilities = np.asarray([0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    expected = np.asarray(
        [0.0, 0.01 / 0.99, 1.0 / 3.0, 1.0, 1.0 / 3.0,
         0.01 / 0.99, 0.0]
    )
    assert bucket_reliability(probabilities) == pytest.approx(expected)


def test_adaptive_basis_cancels_a_shared_low_reliability_coordinate():
    parity = csr_matrix(
        np.asarray(
            [
                [1, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    reliabilities = np.asarray([1.0, 1.0, 1.0, 1.0, 1e-12])
    selected = select_reliability_adaptive_basis(
        parity,
        reliabilities,
        noise_rate=0.05,
        erasure_quantiles=(0.0, 0.2),
    )
    original_masks, _ = parity_row_masks(parity)
    transformed_masks = selected["row_masks"]

    assert selected["selection"]["erasure_quantile"] == pytest.approx(0.2)
    assert selected["selection"]["erasure_free_rows"] == 1
    assert gf2_rank_from_masks(transformed_masks) == 2
    assert gf2_rank_from_masks(original_masks + transformed_masks) == 2
    assert any(
        support.tolist() == [0, 1, 2, 3]
        for support in selected["supports"]
    )


def test_phase2_basis_selection_is_independent_of_tokens_and_otp():
    n = 64
    _, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=3,
        noise_rate=0.05,
        r=61,
        seed=2027,
    )
    probabilities = np.linspace(0.01, 0.99, n)
    tokens_a = torch.zeros(n, dtype=torch.long)
    tokens_b = torch.ones(n, dtype=torch.long)
    changed_key = list(decoding_key)
    changed_key[2] = 1 - np.asarray(decoding_key[2], dtype=np.int64)

    evidence_a = prepare_reliability_adaptive_fixed_map_block_evidence(
        decoding_key,
        tokens_a,
        probabilities,
        _partition(),
    )
    evidence_b = prepare_reliability_adaptive_fixed_map_block_evidence(
        tuple(changed_key),
        tokens_b,
        probabilities,
        _partition(),
    )
    selection_a = evidence_a["basis_selection"]
    selection_b = evidence_b["basis_selection"]
    assert selection_a["basis_sha256"] == selection_b["basis_sha256"]
    assert "one_time_pad" in selection_a["selection_excludes"]
    assert "observed_detection_statistic" in selection_a["selection_excludes"]


def test_phase2_zero_erasure_candidate_matches_phase1_rademacher_decision():
    n = 64
    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=3,
        noise_rate=0.05,
        r=61,
        seed=2028,
    )
    codeword_signs = Encode(encoding_key, np.empty(0, dtype=np.uint8))
    codeword_bits = ((1 - codeword_signs.numpy()) / 2).astype(np.int64)
    record = {
        "prompt_idx": 9,
        "watermark": True,
        "tokens": torch.as_tensor(codeword_bits, dtype=torch.long),
        "p_trace": np.linspace(0.03, 0.97, n),
    }
    artifact = {
        "decoding_key": decoding_key,
        "partition": _partition(),
        "n": n,
        "T": n,
    }
    phase1 = replay_cached_fixed_map_record(
        artifact, record, false_positive_rate=1e-3
    )
    phase2 = replay_cached_fixed_map_record_phase2(
        artifact,
        record,
        false_positive_rate=1e-3,
        erasure_quantiles=(0.0,),
    )
    phase1_score = phase1["blocks"][0]
    phase2_score = phase2["blocks"][0]
    attenuation = (1.0 - 2.0 * decoding_key[4]) ** decoding_key[8]
    phase1_rademacher = phase1_score["calibrations"][
        "weighted_rademacher_chernoff"
    ]
    phase2_rademacher = phase2_score["calibrations"][
        "weighted_rademacher_chernoff"
    ]

    assert phase2["method"] == "fixed_map_phase2_adaptive_basis_replay"
    assert phase2_score["basis_selection"]["basis_rank"] == 61
    assert phase2_score["statistic"] == pytest.approx(
        attenuation * phase1_score["statistic"]
    )
    assert phase2_rademacher["threshold"] == pytest.approx(
        attenuation * phase1_rademacher["threshold"]
    )
    assert phase2_rademacher["decision"] == phase1_rademacher["decision"]
    assert phase2_rademacher["log_pvalue_upper"] == pytest.approx(
        phase1_rademacher["log_pvalue_upper"]
    )
