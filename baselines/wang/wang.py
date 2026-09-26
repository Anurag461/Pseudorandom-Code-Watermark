import math
import numpy as np


def keygen(n, rng, t=3, r=None, eta=0.1):
    r = int(0.95 * n) if r is None else r
    g = math.comb(n, t).bit_length() - 1
    if not 0 < r < n or t - 1 > n - r:
        raise ValueError("Invalid Wang parameters")
    generator = rng.integers(0, 2, (n, g), dtype=np.uint8)
    supports = np.empty((r, t), dtype=np.int32)
    for row in range(r):
        parents = rng.choice(n - r, t - 1, replace=False)
        supports[row] = [*parents, n - r + row]
        generator[n - r + row] = np.bitwise_xor.reduce(generator[parents], axis=0)
    otp = rng.integers(0, 2, n, dtype=np.uint8)
    permutation = rng.permutation(n)
    inverse = np.argsort(permutation)
    return dict(
        generator=generator[permutation],
        otp=otp[permutation],
        supports=np.sort(inverse[supports], axis=1).astype(np.int32),
        permutation=permutation,
        eta=np.array(eta),
    )


def encode(key, rng):
    (n, g) = key["generator"].shape
    payload = rng.integers(0, 2, g, dtype=np.uint8)
    noise = rng.binomial(1, float(key["eta"]), n).astype(np.uint8)
    clean = key["generator"] @ payload % 2
    return dict(codeword=clean ^ key["otp"] ^ noise, payload=payload, noise=noise)


def token_bits(tokens, width=18):
    tokens = np.asarray(tokens, dtype=np.int64)
    if np.any(tokens < 0) or np.any(tokens >= 2**width):
        raise ValueError("Token outside binary range")
    return (tokens[..., None] >> np.arange(width - 1, -1, -1) & 1).astype(np.uint8)


def hard_count(bits, key):
    return np.bitwise_xor.reduce(
        (np.asarray(bits, dtype=np.uint8) ^ key["otp"])[..., key["supports"]], axis=-1
    ).sum(axis=-1)


def hard_threshold(r):
    return r / 2 - r**0.75


def soft_evidence(bits, probabilities):
    from prc_watermark.detectors import map_soft_token

    (bits, probabilities) = (
        np.asarray(bits),
        np.asarray(probabilities, dtype=np.float64),
    )
    if bits.shape != probabilities.shape or bits.ndim != 2:
        raise ValueError("Expected matching token × bit arrays")
    start = 1
    if not np.isfinite(probabilities[start:]).all() or np.any(
        (probabilities[start:] < 0) | (probabilities[start:] > 1)
    ):
        raise ValueError("Invalid conditional probabilities")
    result = np.zeros(bits.shape, dtype=np.float64)
    result[start:] = map_soft_token(bits[start:], probabilities[start:])
    return result.ravel()


def soft_score(evidence, key):
    products = np.prod(np.asarray(evidence)[..., key["supports"]], axis=-1)
    otp_sign = np.prod(1 - 2 * key["otp"][key["supports"]].astype(np.int64), axis=-1)
    S = np.sum(products * otp_sign, axis=-1, dtype=np.float64)
    V = np.sum(products * products, axis=-1, dtype=np.float64)
    Z = np.full(np.shape(S), -np.inf, dtype=np.float64)
    np.divide(S, np.sqrt(V), out=Z, where=V > 0)
    tau = np.sqrt(2 * V * np.log(1000))
    return dict(
        S=S, V=V, Z=Z, tau=tau, no_evidence=V == 0, standard=(V > 0) & (S >= tau)
    )


def calibrate(hard, posterior, alpha=0.001):
    hard = np.asarray(hard).ravel()
    posterior = np.asarray(posterior, dtype=np.float64).ravel()
    if (
        len(hard) != len(posterior)
        or not len(hard)
        or np.isnan(posterior).any()
        or np.isposinf(posterior).any()
        or (not 0 <= alpha < 1)
    ):
        raise ValueError("Invalid calibration scores")
    budget = math.floor(alpha * len(hard))
    (levels, counts) = np.unique(hard, return_counts=True)
    i = np.flatnonzero(np.cumsum(counts) > budget)[0]
    hcut = int(levels[i]) - 1
    (zlevels, zcounts) = np.unique(
        posterior[np.isfinite(posterior)], return_counts=True
    )
    (zlevels, zcounts) = (zlevels[::-1], zcounts[::-1])
    forbidden = np.flatnonzero(np.cumsum(zcounts) > budget)
    if len(forbidden):
        j = forbidden[0]
        (boundary, ties) = (float(zlevels[j]), int(zcounts[j]))
        zcut = float(np.nextafter(boundary, np.inf))
        reject_all = not np.isfinite(zcut)
    elif len(zlevels):
        (boundary, ties, zcut, reject_all) = (None, 0, -np.finfo(np.float64).max, False)
    else:
        (boundary, ties, zcut, reject_all) = (None, 0, None, True)
    return dict(
        N=len(hard),
        alpha=alpha,
        allowed=budget,
        hard=dict(
            cutoff=hcut,
            boundary=int(levels[i]),
            ties=int(counts[i]),
            accepted=int(np.sum(hard <= hcut)),
            rule="H <= cutoff",
        ),
        posterior=dict(
            cutoff=None if reject_all else zcut,
            boundary=boundary,
            ties=ties,
            reject_all=reject_all,
            rule="finite Z >= cutoff",
            accepted=(
                0
                if reject_all
                else int(np.sum(np.isfinite(posterior) & (posterior >= zcut)))
            ),
        ),
    )


def decisions(hard, soft, r, thresholds):
    z = thresholds["posterior"]
    return dict(
        wang_published=np.asarray(hard) <= hard_threshold(r),
        posterior_standard=soft["standard"],
        hard_matched=np.asarray(hard) <= thresholds["hard"]["cutoff"],
        posterior_matched=(
            np.zeros_like(soft["standard"], dtype=bool)
            if z["reject_all"]
            else ~soft["no_evidence"] & (soft["Z"] >= z["cutoff"])
        ),
    )
