"""Run only in the approved CPU worker: paired permutation tests plus Holm."""
import hashlib
import json
import platform
import time


def holm(pvalues):
    import numpy as np
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind='stable')
    adjusted = np.minimum(1.0, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def analyze(payload, protocol):
    import numpy as np
    import scipy
    from scipy.stats import beta, permutation_test

    started = time.monotonic()
    assert protocol['family_size'] == 35 and protocol['resamples'] == 100000
    assert protocol['batch'] == 2000 and payload['prompts'] == 50
    contrasts = payload['contrasts']
    assert len(contrasts) == 35
    differences = np.asarray([r['differences'] for r in contrasts], dtype=float)
    assert differences.shape == (35, 50) and np.isfinite(differences).all()
    # Small mathematical checks are integrated in this single authorized run.
    assert np.allclose(holm([.001, .01, .04, .2]), [.004, .03, .08, .2])
    toy = permutation_test((np.array([1., 2., 3.]),), np.mean,
                           permutation_type='samples', vectorized=True,
                           n_resamples=np.inf, alternative='two-sided')
    assert toy.pvalue == .25

    active = np.count_nonzero(differences, axis=1)
    exact_indices = [j for j in range(35) if 2 ** int(active[j]) <= protocol['resamples']]
    mc_indices = [j for j in range(35) if j not in exact_indices]
    raw = np.empty(35)
    lower = np.empty(35)
    upper = np.empty(35)
    details = {}
    for j in exact_indices:
        # Zero prompt differences do not change the null distribution; omit them
        # to enumerate all effective method-label swaps exactly.
        nonzero = differences[j, differences[j] != 0]
        if len(nonzero) <= 1:
            pvalue, assignments = 1.0, 2 ** len(nonzero)
        else:
            test = permutation_test((nonzero,), np.mean, permutation_type='samples',
                                    vectorized=True, n_resamples=np.inf,
                                    batch=protocol['batch'], alternative='two-sided')
            pvalue, assignments = float(test.pvalue), 2 ** len(nonzero)
        raw[j] = lower[j] = upper[j] = pvalue
        details[j] = {'calculation': 'exact', 'effective_assignments': assignments,
                      'nonzero_prompt_differences': int(active[j])}

    if mc_indices:
        test = permutation_test((differences[mc_indices],), np.mean,
                                permutation_type='samples', vectorized=True,
                                n_resamples=protocol['resamples'], batch=protocol['batch'],
                                alternative='two-sided', axis=-1,
                                random_state=np.random.default_rng(protocol['seed']))
        null = test.null_distribution
        assert null.shape == (protocol['resamples'], len(mc_indices))
        obs = test.statistic
        tolerance = np.abs(obs) * np.finfo(float).eps * 100
        left_counts = np.count_nonzero(null <= obs + tolerance, axis=0)
        right_counts = np.count_nonzero(null >= obs - tolerance, axis=0)
        recomputed = np.minimum(1., 2 * (np.minimum(left_counts, right_counts) + 1) /
                                (protocol['resamples'] + 1))
        assert np.allclose(recomputed, test.pvalue, rtol=0, atol=1e-15)
        # Simultaneous 99.9% Monte Carlo precision bounds for all two tails.
        # These quantify finite-resampling error, not sampling uncertainty.
        alpha_each = protocol['mc_error_probability'] / (2 * len(mc_indices))
        B = protocol['resamples']

        def cp(k):
            lo = 0.0 if k == 0 else float(beta.ppf(alpha_each / 2, k, B - k + 1))
            hi = 1.0 if k == B else float(beta.ppf(1 - alpha_each / 2, k + 1, B - k))
            return lo, hi

        for k, j in enumerate(mc_indices):
            raw[j] = float(test.pvalue[k])
            left, right = cp(int(left_counts[k])), cp(int(right_counts[k]))
            lower[j] = min(1., 2 * min(left[0], right[0]))
            upper[j] = min(1., 2 * min(left[1], right[1]))
            details[j] = {'calculation': 'monte_carlo', 'resamples': B,
                          'nonzero_prompt_differences': int(active[j]),
                          'left_tail_count': int(left_counts[k]),
                          'right_tail_count': int(right_counts[k])}

    adjusted = holm(raw)
    adjusted_lower, adjusted_upper = holm(lower), holm(upper)
    rows = []
    for j, source in enumerate(contrasts):
        significant = bool(adjusted[j] <= protocol['alpha'])
        robust = (adjusted_upper[j] < protocol['alpha'] if significant
                  else adjusted_lower[j] > protocol['alpha'])
        rows.append({k: v for k, v in source.items() if k != 'differences'} | {
            'mean_prc_minus_comparator': float(differences[j].mean()),
            'raw_p': float(raw[j]), 'holm_p': float(adjusted[j]),
            'unadjusted_test_significant': bool(raw[j] <= protocol['alpha']),
            'holm_significant': significant,
            'bold_changed_from_old_bootstrap': significant != source['old_bootstrap_significant'],
            'mc_p_bounds': [float(lower[j]), float(upper[j])],
            'mc_holm_p_bounds': [float(adjusted_lower[j]), float(adjusted_upper[j])],
            'holm_decision_stable_to_mc_error': bool(robust), **details[j]})
    return {'protocol': protocol, 'rows': rows,
            'old_bootstrap_significant_count': sum(r['old_bootstrap_significant'] for r in rows),
            'new_unadjusted_significant_count': sum(r['unadjusted_test_significant'] for r in rows),
            'holm_significant_count': sum(r['holm_significant'] for r in rows),
            'changed_bold_cells': [[r['right'], r['metric']] for r in rows
                                   if r['bold_changed_from_old_bootstrap']],
            'all_holm_decisions_stable_to_mc_error': all(r['holm_decision_stable_to_mc_error'] for r in rows),
            'exact_tests': len(exact_indices), 'monte_carlo_tests': len(mc_indices),
            'integrated_checks': ['known exact sign-flip p=.25', 'known Holm values',
                                  'SciPy tail counts reproduce all Monte Carlo p-values'],
            'software': {'numpy': np.__version__, 'scipy': scipy.__version__,
                         'python': platform.python_version()},
            'elapsed_seconds': time.monotonic() - started,
            'payload_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()}
