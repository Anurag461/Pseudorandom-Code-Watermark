import numpy as np
import galois

GF2 = galois.GF(2)


def sample_P(n: int, t: int, r: int, rng: np.random.Generator) -> galois.FieldArray:
    """Sample P ∈ F_2^{r×n} with each row a uniform t-sparse vector."""
    P = GF2.Zeros((r, n))
    for i in range(r):
        idx = rng.choice(n, size=t, replace=False)
        P[i, idx] = 1
    return P


def kernel_basis_gf2(P: galois.FieldArray) -> galois.FieldArray:
    """Return a basis (as columns) for ker(P) over F_2. P is r×n."""
    null = P.null_space()  # rows span ker(P)
    return null.T  # (n, n - rank)


def sample_G(P: galois.FieldArray, g: int, rng: np.random.Generator) -> galois.FieldArray:
    """Sample G ∈ F_2^{n×g} uniformly from ker(P)^g, i.e. PG = 0."""
    basis = kernel_basis_gf2(P)
    n, k = basis.shape
    if k < g:
        raise ValueError(f"ker(P) has dim {k} < g={g}; cannot sample G")
    coeffs = GF2(rng.integers(0, 2, size=(k, g), dtype=np.uint8))
    return basis @ coeffs


def generate_PG(n: int, t: int, r: int, g: int, seed: int | None = None):
    """Sample (P, G) ← LDPC[n, g, t, r] per Definition 3 of Christ–Gunn."""
    rng = np.random.default_rng(seed)
    P = sample_P(n, t, r, rng)
    G = sample_G(P, g, rng)
    assert not np.any(P @ G), "PG != 0"
    return P, G


if __name__ == "__main__":
    n = 100
    t = int(np.log2(n))          # Θ(log n)
    g = int(np.log2(n) ** 2)     # Ω(log² n)
    r = n - g                    # or any r ≤ 0.99n

    P, G = generate_PG(n, t, r, g, seed=0)
    print(f"P: {P.shape}, row weights: {np.asarray(P).sum(axis=1)[:5]}... (all == {t})")
    print(f"G: {G.shape}")
    print(f"PG = 0 ? {not np.any(P @ G)}")
