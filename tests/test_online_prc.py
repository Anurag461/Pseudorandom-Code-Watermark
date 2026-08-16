from dataclasses import replace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from detectors import (
    detect_online_hoeffding,
    detect_online_map_prefix_grid,
    map_soft_token,
)
from online_prc import (
    GENERATION_SAMPLER_VERSION,
    OnlinePRCEncoder,
    OnlinePRCKey,
    derive_document_seed,
    document_uniform,
    gf2_rank,
    is_parity_coordinate,
    materialize_supports,
    otp_prefix,
    parent_indices,
    parity_check_dense,
    reconstruct_generator,
    support_sha256,
    target_row_count,
    validate_online_word,
)
from prc import Detect


def _key(seed=12345, weight=3, eta=0.05):
    return OnlinePRCKey.from_seed(
        seed, check_weight=weight, noise_rate=eta
    )


def _partition():
    return torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)


def test_weight_three_startup_and_n256_schedule():
    key = _key()
    assert [target_row_count(length, key) for length in range(5)] == [0, 0, 0, 1, 2]
    assert materialize_supports(0, key).shape == (0, 3)
    assert materialize_supports(1, key).shape == (0, 3)
    assert materialize_supports(2, key).shape == (0, 3)
    np.testing.assert_array_equal(materialize_supports(3, key), [[0, 1, 2]])

    assert target_row_count(256, key) == 253
    free_one_based = [
        position + 1
        for position in range(256)
        if not is_parity_coordinate(position, key)
    ]
    assert free_one_based == [1, 2, 251]


def test_supports_have_exact_weight_are_causal_and_full_rank():
    key = _key()
    supports = materialize_supports(256, key)
    assert supports.shape == (253, 3)
    for row in supports:
        assert len(set(row.tolist())) == 3
        assert np.all(row[:-1] < row[-1])
    matrix = parity_check_dense(256, key)
    assert gf2_rank(matrix) == matrix.shape[0]


def test_schedule_count_growth_and_exact_rounding_boundaries():
    key = _key()
    counts = np.asarray([target_row_count(length, key) for length in range(401)])
    assert set(np.diff(counts)).issubset({0, 1})
    assert counts[250] == 248
    assert counts[251] == 248
    assert counts[349] == 346
    assert counts[350] == 346
    assert counts[400] == 396


def test_direct_materialization_is_prefix_consistent():
    key = _key()
    long_rows = materialize_supports(400, key)
    for length in (0, 1, 2, 3, 4, 17, 251, 256, 399):
        direct = materialize_supports(length, key)
        retained = long_rows[long_rows[:, -1] < length]
        np.testing.assert_array_equal(direct, retained)
        np.testing.assert_array_equal(
            otp_prefix(length, key), otp_prefix(400, key)[:length]
        )


def test_seed_reproducibility_and_support_otp_domain_separation():
    key = _key(77)
    same = _key(77)
    other = _key(78)
    np.testing.assert_array_equal(
        materialize_supports(256, key), materialize_supports(256, same)
    )
    np.testing.assert_array_equal(otp_prefix(256, key), otp_prefix(256, same))
    assert support_sha256(256, key) != support_sha256(256, other)

    otp_only_change = replace(key, otp_key=other.otp_key)
    np.testing.assert_array_equal(
        materialize_supports(256, key), materialize_supports(256, otp_only_change)
    )
    assert not np.array_equal(
        otp_prefix(256, key), otp_prefix(256, otp_only_change)
    )


def test_encoder_algebra_and_posthoc_generator():
    key = _key(99)
    encoder = OnlinePRCEncoder(key, [derive_document_seed(12345, 7)])
    noisy = encoder.encode_to_length(256)[0]
    clean = encoder.clean_array()[0]
    error = encoder.error_array()[0]
    validate_online_word(key, clean, noisy, error)

    generator, free = reconstruct_generator(256, key)
    checks = parity_check_dense(256, key)
    assert generator.shape == (256, 3)
    assert free.tolist() == [0, 1, 250]
    assert not np.any((checks @ generator) % 2)
    assert gf2_rank(generator) == generator.shape[1]
    payload = clean[free]
    np.testing.assert_array_equal((generator @ payload) % 2, clean)


def test_batch_order_and_inactive_member_do_not_change_streams():
    key = _key(314)
    seed_a = derive_document_seed(5, 10)
    seed_b = derive_document_seed(5, 20)

    together = OnlinePRCEncoder(key, [seed_a, seed_b])
    for step in range(64):
        together.next_bits(active=[step < 17, True])

    reversed_batch = OnlinePRCEncoder(key, [seed_b, seed_a])
    reversed_batch.encode_to_length(64)
    assert together.noisy_history[1] == reversed_batch.noisy_history[0]
    assert together.noisy_history[0] == reversed_batch.noisy_history[1][:17]


def test_document_position_uniforms_are_reproducible_and_domain_separated():
    seed = derive_document_seed(12345, 7)
    draws = [document_uniform(seed, "lm-token/v1", position)
             for position in range(32)]
    assert GENERATION_SAMPLER_VERSION == "document_position_inverse_cdf_v1"
    assert draws == [document_uniform(seed, b"lm-token/v1", position)
                     for position in range(32)]
    assert all(0.0 < draw < 1.0 for draw in draws)
    assert len(set(draws)) == len(draws)
    assert draws != [document_uniform(seed, "lm-bucket/v1", position)
                     for position in range(32)]
    assert draws != [document_uniform(seed + 1, "lm-token/v1", position)
                     for position in range(32)]

    with pytest.raises(ValueError, match="domain"):
        document_uniform(seed, b"", 0)
    with pytest.raises(ValueError, match="nonnegative"):
        document_uniform(seed, "lm-token/v1", -1)


def test_online_detector_matches_direct_fixed_matrix_score_without_folding():
    key = _key(1001, eta=0.0)
    length = 64
    encoder = OnlinePRCEncoder(key, [derive_document_seed(42, 0)])
    noisy = encoder.encode_to_length(length)[0]
    tokens = torch.as_tensor(noisy, dtype=torch.long)
    probabilities = np.linspace(0.15, 0.85, length)

    decision, info = detect_online_hoeffding(
        key, tokens, probabilities, _partition(), fpr=1e-3,
        weight="map", return_info=True,
    )
    supports = materialize_supports(length, key)
    rows = np.repeat(np.arange(supports.shape[0]), key.check_weight)
    cols = supports.reshape(-1)
    matrix = csr_matrix(
        (np.ones(cols.shape[0], dtype=np.uint8), (rows, cols)),
        shape=(supports.shape[0], length),
    )
    decoding_key = (
        np.zeros((length, 0), dtype=np.uint8),
        matrix,
        otp_prefix(length, key),
        1e-3,
        key.noise_rate,
        np.empty(0, dtype=np.uint8),
        0,
        0,
        key.check_weight,
    )
    observed = tokens.numpy()
    soft = map_soft_token(observed, probabilities)
    fixed_decision, fixed_info = Detect(
        decoding_key, soft, false_positive_rate=1e-3, return_info=True
    )
    assert decision == fixed_decision
    assert info["statistic"] == pytest.approx(fixed_info["statistic"])
    assert info["V"] == pytest.approx(fixed_info["V"])
    assert info["threshold"] == pytest.approx(fixed_info["threshold"])
    assert info["length"] == length
    assert info["T"] == info["n"] == length


def test_online_map_prefix_grid_matches_individual_detector_calls():
    key = _key(811, eta=0.05)
    maximum = 64
    encoder = OnlinePRCEncoder(key, [derive_document_seed(91, 0)])
    noisy = encoder.encode_to_length(maximum)[0]
    tokens = torch.as_tensor(noisy, dtype=torch.long)
    probabilities = np.linspace(0.03, 0.97, maximum)
    lengths = [64, 48, 32, 3, 2]

    grid = detect_online_map_prefix_grid(
        key,
        tokens,
        probabilities,
        _partition(),
        lengths,
        fpr=1e-3,
    )
    assert [result["length"] for result in grid] == lengths
    for result, length in zip(grid, lengths):
        decision, direct = detect_online_hoeffding(
            key,
            tokens[:length],
            probabilities[:length],
            _partition(),
            fpr=1e-3,
            weight="map",
            return_info=True,
        )
        assert result["decision"] == decision
        assert result["r"] == direct["r"]
        assert result["status"] == direct["status"]
        assert result["statistic"] == pytest.approx(direct["statistic"])
        assert result["V"] == pytest.approx(direct["V"])
        if np.isinf(direct["threshold"]):
            assert np.isinf(result["threshold"])
        else:
            assert result["threshold"] == pytest.approx(direct["threshold"])


def test_online_map_prefix_grid_rejects_invalid_lengths():
    key = _key(812)
    tokens = torch.zeros(8, dtype=torch.long)
    probabilities = np.full(8, 0.5)

    with pytest.raises(ValueError, match="nonempty"):
        detect_online_map_prefix_grid(
            key, tokens, probabilities, _partition(), [], fpr=1e-3
        )
    with pytest.raises(ValueError, match="duplicates"):
        detect_online_map_prefix_grid(
            key, tokens, probabilities, _partition(), [4, 4], fpr=1e-3
        )
    with pytest.raises(ValueError, match="need prefix length 9"):
        detect_online_map_prefix_grid(
            key, tokens, probabilities, _partition(), [9], fpr=1e-3
        )


def test_detector_handles_startup_zero_variance_and_anytime_policy():
    key = _key(44)
    decision, info = detect_online_hoeffding(
        key, torch.tensor([0, 1]), np.array([0.5, 0.5]), _partition(),
        fpr=1e-3, return_info=True,
    )
    assert not decision
    assert info["status"] == "insufficient_evidence_no_checks"
    assert np.isinf(info["threshold"])

    decision, info = detect_online_hoeffding(
        key, torch.tensor([0, 1, 0]), np.array([0.0, 1.0, 0.0]),
        _partition(), fpr=1e-3, weight="entropy", return_info=True,
    )
    assert not decision
    assert info["status"] == "insufficient_evidence_zero_variance"

    tokens = torch.tensor([0, 1, 0, 1] * 16)
    probabilities = np.full(64, 0.5)
    _, one_shot = detect_online_hoeffding(
        key, tokens, probabilities, _partition(), fpr=1e-3,
        return_info=True,
    )
    _, anytime = detect_online_hoeffding(
        key, tokens, probabilities, _partition(), fpr=1e-3,
        fpr_policy="alpha_spending_v1", return_info=True,
    )
    assert anytime["effective_fpr"] < one_shot["effective_fpr"]
    assert anytime["threshold"] > one_shot["threshold"]


def test_strong_synthetic_watermark_detects_at_n256():
    key = _key(2024, eta=0.0)
    encoder = OnlinePRCEncoder(key, [derive_document_seed(8, 0)])
    noisy = encoder.encode_to_length(256)[0]
    decision, info = detect_online_hoeffding(
        key,
        torch.as_tensor(noisy, dtype=torch.long),
        np.full(256, 0.5),
        _partition(),
        fpr=1e-3,
        return_info=True,
    )
    assert decision
    assert info["statistic"] == pytest.approx(253.0)
    assert info["V"] == pytest.approx(253.0)


def test_validation_rejects_bad_configs_and_bad_inputs():
    with pytest.raises(ValueError, match="at least 2"):
        _key(weight=1)
    with pytest.raises(ValueError, match=r"\[0, 0.5\)"):
        _key(eta=0.5)

    key = _key()
    with pytest.raises(ValueError, match="is free"):
        parent_indices(0, key)
    with pytest.raises(ValueError, match="tokens length"):
        detect_online_hoeffding(
            key, torch.tensor([0, 1]), np.array([0.5]), _partition()
        )
    with pytest.raises(ValueError, match="fpr_policy"):
        detect_online_hoeffding(
            key, torch.tensor([0, 1, 0]), np.full(3, 0.5), _partition(),
            fpr_policy="peek_forever",
        )
