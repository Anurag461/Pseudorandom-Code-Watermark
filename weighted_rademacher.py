"""Rigorous conditional tail bounds for weighted Rademacher sums.

For fixed real weights ``q_a`` and independent uniform signs ``epsilon_a``,

    D = sum_a epsilon_a q_a

has conditional cumulant-generating function

    K(lambda) = sum_a log(cosh(lambda q_a)).

Optimizing ``K(lambda) - lambda * d`` over nonnegative ``lambda`` gives a
Chernoff upper bound on ``P[D >= d | q]``.  The functions below evaluate that
bound and invert it into a detection threshold while retaining a rigorous
conditional false-positive guarantee.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq


_LOG_TWO = math.log(2.0)


def _positive_weights(check_weights) -> np.ndarray:
    weights = np.abs(np.asarray(check_weights, dtype=np.float64).reshape(-1))
    if not np.all(np.isfinite(weights)):
        raise ValueError("check_weights must be finite")
    return weights[weights > 0.0]


def _log_cosh(values: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise ``log(cosh(values))``."""
    return np.logaddexp(values, -values) - _LOG_TWO


def _cgf(lambda_value: float, weights: np.ndarray) -> float:
    return float(np.sum(_log_cosh(float(lambda_value) * weights)))


def _cgf_prime(lambda_value: float, weights: np.ndarray) -> float:
    scaled = float(lambda_value) * weights
    return float(np.sum(weights * np.tanh(scaled)))


def _rate_at_lambda(lambda_value: float, weights: np.ndarray) -> float:
    derivative = _cgf_prime(lambda_value, weights)
    return float(lambda_value * derivative - _cgf(lambda_value, weights))


def _increasing_bracket(function, initial: float) -> tuple[float, bool]:
    """Find an upper point where an increasing function is nonnegative."""
    upper = max(float(initial), np.finfo(np.float64).tiny)
    for _ in range(128):
        value = float(function(upper))
        if np.isfinite(value) and value >= 0.0:
            return upper, True
        upper *= 2.0
        if not np.isfinite(upper):
            break
    return upper / 2.0, False


def weighted_rademacher_log_tail_bound(
    statistic: float,
    check_weights,
    *,
    return_info: bool = False,
):
    """Return a Chernoff upper bound on ``log P[D >= statistic | q]``.

    The signs of ``check_weights`` do not affect the null distribution, so the
    calculation uses their absolute values.  The returned log bound is at most
    zero.  ``-inf`` denotes an impossible event beyond the support of the sum.
    """
    observed = float(statistic)
    if not np.isfinite(observed):
        raise ValueError("statistic must be finite")
    weights = _positive_weights(check_weights)
    support = float(np.sum(weights))
    count = int(weights.size)

    if observed <= 0.0:
        log_bound = 0.0
        lambda_star = 0.0
        status = "nonpositive_statistic"
        converged = True
    elif count == 0 or observed > support:
        log_bound = float("-inf")
        lambda_star = float("inf")
        status = "outside_support"
        converged = True
    elif observed == support:
        log_bound = float(-count * _LOG_TWO)
        lambda_star = float("inf")
        status = "support_boundary"
        converged = True
    else:
        derivative = lambda value: _cgf_prime(value, weights) - observed
        upper, bracketed = _increasing_bracket(
            derivative, 1.0 / float(np.max(weights))
        )
        if bracketed:
            lambda_star = float(brentq(derivative, 0.0, upper))
            status = "ok"
            converged = True
        else:
            # Evaluating the Chernoff expression at any finite lambda remains
            # a valid upper bound.  This conservative fallback is only
            # expected when the requested statistic is numerically
            # indistinguishable from the support boundary.
            lambda_star = float(upper)
            status = "finite_lambda_fallback"
            converged = False
        log_bound = min(
            0.0,
            float(_cgf(lambda_star, weights) - lambda_star * observed),
        )

    info = {
        "method": "weighted_rademacher_chernoff",
        "log_pvalue_upper": float(log_bound),
        "pvalue_upper": (
            0.0 if log_bound == float("-inf") else float(math.exp(log_bound))
        ),
        "lambda_star": float(lambda_star),
        "support_bound": support,
        "nonzero_checks": count,
        "optimizer_converged": bool(converged),
        "status": status,
    }
    return (float(log_bound), info) if return_info else float(log_bound)


def weighted_rademacher_threshold(
    check_weights,
    false_positive_rate: float,
    *,
    return_info: bool = False,
):
    """Invert the optimized Chernoff bound at a target false-positive rate."""
    fpr = float(false_positive_rate)
    if not 0.0 < fpr < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")
    weights = _positive_weights(check_weights)
    count = int(weights.size)
    support = float(np.sum(weights))
    log_inverse_fpr = float(-math.log(fpr))
    maximum_rate = float(count * _LOG_TWO)

    if count == 0 or log_inverse_fpr > maximum_rate:
        threshold = float("inf")
        lambda_star = float("inf")
        status = "target_unattainable"
        converged = True
    elif math.isclose(
        log_inverse_fpr, maximum_rate, rel_tol=1e-13, abs_tol=1e-15
    ):
        threshold = support
        lambda_star = float("inf")
        status = "support_boundary"
        converged = True
    else:
        rate_gap = lambda value: (
            _rate_at_lambda(value, weights) - log_inverse_fpr
        )
        upper, bracketed = _increasing_bracket(
            rate_gap, 1.0 / float(np.max(weights))
        )
        if not bracketed:
            # The target is theoretically attainable, so failure to bracket
            # means floating-point precision has reached the support limit.
            # Returning the support is conservative.
            threshold = support
            lambda_star = float(upper)
            status = "support_fallback"
            converged = False
        else:
            lambda_star = float(brentq(rate_gap, 0.0, upper))
            threshold = float(_cgf_prime(lambda_star, weights))
            status = "ok"
            converged = True

    info = {
        "method": "weighted_rademacher_chernoff",
        "threshold": float(threshold),
        "lambda_star": float(lambda_star),
        "support_bound": support,
        "nonzero_checks": count,
        "false_positive_rate": fpr,
        "optimizer_converged": bool(converged),
        "status": status,
    }
    return (float(threshold), info) if return_info else float(threshold)
