import numpy as np


def sample_P(n: int, t: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Sample P ∈ F_2^{r×n} with each row a uniform t-sparse vector."""
    P = np.zeros((r, n), dtype=np.uint8)
    for i in range(r):
        idx = rng.choice(n, size=t, replace=False)
        P[i, idx] = 1
    return P


def kernel_basis_gf2(P: np.ndarray) -> np.ndarray:
    """Return a basis (as columns) for ker(P) over F_2. P is r×n."""
    r, n = P.shape
    A = P.copy().astype(np.uint8)
    pivot_col = [-1] * r
    row = 0
    pivot_cols = []
    for col in range(n):
        if row >= r:
            break
        piv = None
        for k in range(row, r):
            if A[k, col] == 1:
                piv = k
                break
        if piv is None:
            continue
        if piv != row:
            A[[row, piv]] = A[[piv, row]]
        for k in range(r):
            if k != row and A[k, col] == 1:
                A[k] ^= A[row]
        pivot_col[row] = col
        pivot_cols.append(col)
        row += 1

    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]

    # ker dim = n - rank
    basis = np.zeros((n, len(free_cols)), dtype=np.uint8)
    for j, fc in enumerate(free_cols):
        basis[fc, j] = 1
        # back-substitute: for each pivot row with pivot_col = pc, x_pc = A[row, fc]
        for row_idx, pc in enumerate(pivot_cols):
            basis[pc, j] = A[row_idx, fc]
    return basis  # shape (n, n - rank)


def sample_G(P: np.ndarray, g: int, rng: np.random.Generator) -> np.ndarray:
    """Sample G ∈ F_2^{n×g} uniformly from ker(P)^g, i.e. PG = 0."""
    basis = kernel_basis_gf2(P)  # (n, k) where k = dim ker P
    n, k = basis.shape
    if k < g:
        raise ValueError(f"ker(P) has dim {k} < g={g}; cannot sample G")
    coeffs = rng.integers(0, 2, size=(k, g), dtype=np.uint8)
    G = (basis @ coeffs) % 2
    return G.astype(np.uint8)


def generate_PG(n: int, t: int, r: int, g: int, seed: int | None = None):
    """Sample (P, G) ← LDPC[n, g, t, r] per Definition 3 of Christ–Gunn."""
    rng = np.random.default_rng(seed)
    P = sample_P(n, t, r, rng)
    G = sample_G(P, g, rng)
    assert np.all((P @ G) % 2 == 0), "PG != 0"
    return P, G


if __name__ == "__main__":
    n = 100
    t = int(np.log2(n))          # Θ(log n)
    g = int(np.log2(n) ** 2)     # Ω(log² n)
    r = n - g                    # or any r ≤ 0.99n

    P, G = generate_PG(n, t, r, g, seed=0)
    print(f"P: {P.shape}, row weights: {P.sum(axis=1)[:5]}... (all == {t})")
    print(f"G: {G.shape}")
    print(f"PG = 0 ? {np.all((P @ G) % 2 == 0)}")
