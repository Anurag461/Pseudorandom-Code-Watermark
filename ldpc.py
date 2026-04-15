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

def sample_t_sparse_numpy(n: int, t: int) -> galois.FieldArray:
    if t > n or t < 0:
        raise ValueError("t must be between 0 and n")
    vector = np.zeros(n, dtype=np.int8)
    indices = np.random.choice(n, size=t, replace=False)
    vector[indices] = 1
    return GF2(vector)

def bernoulli_noise(n: int, eta: float):
    t =int(eta*n)
    return sample_t_sparse_numpy(n, t)

def add_error(vec: galois.FieldArray, eta: float):
    n = vec.shape[0]
    return vec + bernoulli_noise(n, eta)

def weight(P, vec: galois.FieldArray):
    return int(np.array(P@vec, dtype=int).sum())

def detect(P, vec: galois.FieldArray):
    wt = weight(P, vec)
    r = P.shape[0]
    threshold = (0.5 - (1/(r**4)))*r
    return wt < threshold


#def detect_rate()

class LDPC_PRC:
    def __init__(self, n):
        self.n = n
        self.eta = 0.4
        self.t = int(np.log2(n))          # Θ(log n)
        self.g = int(np.log2(n) ** 2)     # Ω(log² n)
        self.r = self.n - self.g                    # or any r ≤ 0.99n
        print(f"n is {self.n}")
        print(f"g is {self.g}")
        print(f"t is {self.t}")
        print(f"r is {self.r}")
        P, G = generate_PG(n, self.t, self.r, self.g, seed=0)

        

if __name__ == "__main__":
    n = 100
    eta = 0.4
    t = int(np.log2(n))          # Θ(log n)
    g = int(np.log2(n) ** 2)     # Ω(log² n)
    r = n - g                    # or any r ≤ 0.99n
    print(f"n is {n}")
    print(f"g is {g}")
    print(f"t is {t}")
    print(f"r is {r}")
    
    P, G = generate_PG(n, t, r, g, seed=0)
    print(f"P: {P.shape}, row weights: {np.asarray(P).sum(axis=1)[:5]}... (all == {t})")
    print(f"G: {G.shape}")
    print(f"PG = 0 ? {not np.any(P @ G)}")

    s = GF2(np.random.binomial(1, 0.5, g))
    x = add_error(G@s, 0.2)
    print(weight(P, x))
    print(detect(P,x))
