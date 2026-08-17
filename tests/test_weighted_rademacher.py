import itertools
import math

import numpy as np
import pytest

from weighted_rademacher import (
    weighted_rademacher_log_tail_bound,
    weighted_rademacher_threshold,
)


def _exact_tail(weights, statistic):
    weights = np.asarray(weights, dtype=np.float64)
    values = [
        np.dot(signs, weights)
        for signs in itertools.product((-1.0, 1.0), repeat=weights.size)
    ]
    return np.mean(np.asarray(values) >= statistic)


@pytest.mark.parametrize(
    "weights",
    [
        [1.0, 1.0, 1.0, 1.0],
        [0.1, 0.3, 0.7, 1.0],
        [1e-4, 0.02, 0.5, 0.9, 1.0],
    ],
)
def test_chernoff_bound_is_conservative_against_exact_enumeration(weights):
    support = float(np.sum(np.abs(weights)))
    for statistic in np.linspace(0.05 * support, support, 12):
        log_bound = weighted_rademacher_log_tail_bound(statistic, weights)
        bound = math.exp(log_bound)
        exact = _exact_tail(weights, statistic)
        assert bound + 1e-14 >= exact


def test_threshold_inverts_tail_bound_and_improves_on_hoeffding():
    weights = np.asarray([0.01, 0.03, 0.08, 0.2, 0.4, 0.7, 1.0] * 20)
    fpr = 1e-3
    threshold = weighted_rademacher_threshold(weights, fpr)
    log_bound = weighted_rademacher_log_tail_bound(threshold, weights)
    hoeffding = math.sqrt(2.0 * np.sum(weights ** 2) * math.log(1.0 / fpr))
    assert log_bound == pytest.approx(math.log(fpr), abs=2e-10)
    assert threshold <= hoeffding


def test_equal_weight_threshold_matches_binomial_cgf_not_gaussian_guess():
    weights = np.ones(100)
    threshold, info = weighted_rademacher_threshold(
        weights, 1e-3, return_info=True
    )
    assert 0.0 < threshold < 100.0
    assert info["optimizer_converged"]
    assert info["nonzero_checks"] == 100
    assert weighted_rademacher_log_tail_bound(threshold, weights) == pytest.approx(
        math.log(1e-3), abs=2e-10
    )


def test_weight_sign_permutation_and_scaling_do_not_change_calibration():
    weights = np.asarray([0.05, -0.2, 0.7, -1.0])
    permuted = weights[[2, 0, 3, 1]]
    scale = 3.25
    threshold = weighted_rademacher_threshold(weights, 0.05)
    assert weighted_rademacher_threshold(permuted, 0.05) == pytest.approx(
        threshold
    )
    assert weighted_rademacher_threshold(scale * weights, 0.05) == pytest.approx(
        scale * threshold
    )
    statistic = 0.6
    assert weighted_rademacher_log_tail_bound(
        statistic, weights
    ) == pytest.approx(
        weighted_rademacher_log_tail_bound(statistic, permuted)
    )
    assert weighted_rademacher_log_tail_bound(
        scale * statistic, scale * weights
    ) == pytest.approx(weighted_rademacher_log_tail_bound(statistic, weights))


def test_edge_cases_and_validation():
    assert weighted_rademacher_log_tail_bound(-1.0, [0.5, 1.0]) == 0.0
    assert weighted_rademacher_log_tail_bound(2.0, [0.5, 1.0]) == float("-inf")
    assert weighted_rademacher_log_tail_bound(1.5, [0.5, 1.0]) == pytest.approx(
        -2.0 * math.log(2.0)
    )
    assert math.isinf(weighted_rademacher_threshold([], 1e-3))
    assert math.isinf(weighted_rademacher_threshold([1.0, 1.0], 0.01))
    with pytest.raises(ValueError, match="finite"):
        weighted_rademacher_log_tail_bound(1.0, [np.nan])
    with pytest.raises(ValueError, match="statistic"):
        weighted_rademacher_log_tail_bound(np.inf, [1.0])
    with pytest.raises(ValueError, match="false_positive_rate"):
        weighted_rademacher_threshold([1.0], 1.0)
