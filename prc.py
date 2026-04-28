import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.special import binom, lambertw
from ldpc import bp_decoder
import sys
import galois

GF = galois.GF(2)

def apply_channel_probs(x, channel_probs):
    e = GF(np.random.binomial(1, channel_probs))
    return x + e

### Given a GF(2) matrix, do row elimination and return the first k rows of A that form an invertible matrix
def boolean_row_reduce(A, print_progress=False):
    n, k = A.shape
    A_rr = A.copy()
    perm = np.arange(n)
    for j in range(k):
        idxs = j + np.nonzero(A_rr[j:, j])[0]
        if idxs.size == 0:
            print("The given matrix is not invertible")
            return None
        A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]  # For matrices you have to swap them this way
        (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])  # Weirdly, this is MUCH faster if you swap this way instead of using perm[[i,j]]=perm[[j,i]]
        A_rr[idxs[1:]] += A_rr[j]
        if print_progress and (j%5==0 or j+1==k):
            sys.stdout.write(f'\rDecoding progress: {j + 1} / {k}')
            sys.stdout.flush()
    if print_progress: print()
    return perm[:k]


def str_to_bin(string):
    bin_str = ''.join(format(i, '08b') for i in bytearray(string, encoding ='utf-8'))
    return [int(b) for b in bin_str]

def bin_to_str(bin_list):
    bin_str = ''.join(map(str, bin_list))
    byte_array = bytearray(int(bin_str[i:i+8], 2) for i in range(0, len(bin_str), 8) if bin_str[i:i+8]!='00000000')
    return byte_array.decode('utf-8')

### Key generation algorithm.
## Inputs:
# n - block length (i.e., length of PRC codeword).
# message_length - length of messages you want to encode
# false_positive_rate - the false positive rate you're willing to tolerate
# t - sparsity of parity checks. larger values help pseudorandomness
# g - dimension of random code used. larger values help pseudorandomness
# r - number of parity checks used. smaller values help pseudorandomness
# noise_rate - amount of noise for Encode to add to codewords. larger values help p1seudorandomness
def KeyGen(n, message_length=512, false_positive_rate=1e-9, t=3, g=None, r=None, noise_rate=None):
    # Set basic scheme parameters
    num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
    secpar = int(np.log2(binom(n, t)))
    if g is None: g = secpar
    # if noise_rate is None: noise_rate = np.exp(lambertw(-np.log(2) / secpar, -1)).real
    # if noise_rate is None: noise_rate = 1 - 2**(-(secpar - 3*np.log2(g))/g**2)
    if noise_rate is None: noise_rate = 1 - 2 ** (-secpar / g ** 2)
    k = message_length + g + num_test_bits
    if r is None: r = n - k - secpar

    # Sample n by k generator matrix (all but the first n-r of these will be over-written)
    generator_matrix = GF.Random((n, k))

    # Sample scipy.sparse parity-check matrix together with the last n-r rows of the generator matrix
    row_indices = []
    col_indices = []
    data = []
    for row in range(r):
        chosen_indices = np.random.choice(n - r + row, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, n - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

    # Compute scheme parameters
    max_bp_iter = int(np.log(n) / np.log(t))

    # Sample one-time pad and test bits
    one_time_pad = GF.Random(n)
    test_bits = GF.Random(num_test_bits)

    # Permute bits
    permutation = np.random.permutation(n)
    generator_matrix = generator_matrix[permutation]
    one_time_pad = one_time_pad[permutation]
    parity_check_matrix = parity_check_matrix[:, permutation]

    encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
    decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t)

    return encoding_key, decoding_key


### Encoding algorithm
## Inputs:
# encoding_key - Encoding key output by KeyGen.
# message - Message to encode, as an array of k bits. If none is provided a random message is used.
def Encode(encoding_key, message=None):
    generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
    n, k = generator_matrix.shape

    if message is None:
        payload = np.concatenate((test_bits, GF.Random(k - len(test_bits))))
    else:
        assert len(message) <= k-len(test_bits)-g, "Message is too long"
        payload = np.concatenate((test_bits, GF.Random(g), GF(message), GF.Zeros(k-len(test_bits)-g-len(message))))

    error = GF(np.random.binomial(1, noise_rate, n))

    return 1 - 2 * torch.tensor(payload @ generator_matrix.T + one_time_pad + error, dtype=float)


### Detector
## Inputs:
# decoding_key - Decoding key output by KeyGen.
# posteriors - The posterior expectations of sign(z) as a torch.tensor.
## Returns:
# True/False - Detection result.
def Detect(decoding_key, posteriors, false_positive_rate=None):
    generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter, t = decoding_key
    if false_positive_rate is not None:
        fpr = false_positive_rate
    else:
        fpr = false_positive_rate_key

    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)

    r = parity_check_matrix.shape[0]
    Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
    log_plus = np.log((1 + Pi) / 2)
    log_minus = np.log((1 - Pi) / 2)
    log_prod = log_plus + log_minus

    const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
    threshold = np.sqrt(2 * const * np.log(1 / fpr)) + 0.5 * log_prod.sum()

    return log_plus.sum() >= threshold


### Decoder
## Inputs:
# decoding_key - Decoding key output by KeyGen.
# posteriors - The posterior expectations of sign(z) as a torch.tensor.
## Returns:
# recovered_message - The recovered message. If the test bits are incorrect, outputs None.
def Decode(decoding_key, posteriors, print_progress=False, max_bp_iter=None):
    generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter_key, t = decoding_key
    if max_bp_iter is None:
        max_bp_iter = max_bp_iter_key

    posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
    channel_probs = (1 - np.abs(posteriors)) / 2
    x_recovered = (1 - np.sign(posteriors)) // 2

    # Apply the belief-propagation decoder.
    if print_progress:
        print("Running belief propagation...")
    bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
    x_decoded = bpd.decode(x_recovered)

    # Compute a confidence score.
    bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
    confidences = 2 * np.abs(0.5 - bpd_probs)

    # Order codeword bits by confidence.
    confidence_order = np.argsort(-confidences)
    ordered_generator_matrix = generator_matrix[confidence_order]
    ordered_x_decoded = x_decoded[confidence_order]

    # Find the first (according to the confidence order) linearly independent set of rows of the generator matrix.
    top_invertible_rows = boolean_row_reduce(ordered_generator_matrix, print_progress=print_progress)
    if top_invertible_rows is None:
        return None

    # Solve the system.
    if print_progress:
        print("Solving linear system...")
    recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], GF(ordered_x_decoded[top_invertible_rows]))

    if not (recovered_string[:len(test_bits)] == test_bits).all():
        return None
    return np.array(recovered_string[len(test_bits) + g:])

# import numpy as np
# import galois

# GF2 = galois.GF(2)

# def sample_P(n: int, t: int, r: int, rng: np.random.Generator) -> galois.FieldArray:
#     """Sample P ∈ F_2^{r×n} with each row a uniform t-sparse vector."""
#     P = GF2.Zeros((r, n))
#     for i in range(r):
#         idx = rng.choice(n, size=t, replace=False)
#         P[i, idx] = 1
#     return P


# def kernel_basis_gf2(P: galois.FieldArray) -> galois.FieldArray:
#     """Return a basis (as columns) for ker(P) over F_2. P is r×n."""
#     null = P.null_space()  # rows span ker(P)
#     return null.T  # (n, n - rank)


# def sample_G(P: galois.FieldArray, g: int, rng: np.random.Generator) -> galois.FieldArray:
#     """Sample G ∈ F_2^{n×g} uniformly from ker(P)^g, i.e. PG = 0."""
#     basis = kernel_basis_gf2(P)
#     n, k = basis.shape
#     if k < g:
#         raise ValueError(f"ker(P) has dim {k} < g={g}; cannot sample G")
#     coeffs = GF2(rng.integers(0, 2, size=(k, g), dtype=np.uint8))
#     return basis @ coeffs


# def generate_PG(n: int, t: int, r: int, g: int, seed: int | None = None):
#     """Sample (P, G) ← LDPC[n, g, t, r] per Definition 3 of Christ–Gunn."""
#     rng = np.random.default_rng(seed)
#     P = sample_P(n, t, r, rng)
#     G = sample_G(P, g, rng)
#     assert not np.any(P @ G), "PG != 0"
#     return P, G

# def sample_t_sparse_numpy(n: int, t: int) -> galois.FieldArray:
#     if t > n or t < 0:
#         raise ValueError("t must be between 0 and n")
#     vector = np.zeros(n, dtype=np.int8)
#     indices = np.random.choice(n, size=t, replace=False)
#     vector[indices] = 1
#     return GF2(vector)

# def bernoulli_noise(n: int, eta: float):
#     t =int(eta*n)
#     return sample_t_sparse_numpy(n, t)

# def add_error(vec: galois.FieldArray, eta: float):
#     n = vec.shape[0]
#     return vec + bernoulli_noise(n, eta)

# def weight(P, vec: galois.FieldArray, z: galois.FieldArray):
#     return int(np.array(P@vec + P@z, dtype=int).sum())

# def detect(P, vec: galois.FieldArray, z: galois.FieldArray, fpr = 1e-5):
#     wt = weight(P, vec, z)
#     r = P.shape[0]
#     threshold = (0.5 - (1/(r**4)))*r
#     threshold = r / 2 - np.sqrt(0.5 * r * np.log(1 / fpr))
#     return wt < threshold

# def pad(vec):
#     return vec+z
    
# def sample_codeword(G, eta):
#     s = GF2(np.random.binomial(1, 0.5, G.shape[1]))
#     return add_error(pad(G@s), eta)


# def sample(p,x):
#     if p < 0.5:
#         mod_p  = 2*x*p
#     else:
#         mod_p =  (1 - 2*(1 - x)*(1 - p))
#     print(f"p is {p}, x is {x}, new probability is {mod_p}")
#     t = np.random.binomial(1 , mod_p)
#     return t

# import numpy as np


# def sample_orthogonal_gf2_matrices(n, g, t):
#     """
#     Sample two GF(2) matrices P and G such that P @ G = 0 over GF(2).

#     Parameters:
#         n: dimension parameter
#         g: number of columns in G
#         t: exact number of ones per row of P; each s_i has exactly t-1 ones

#     Returns:
#         P: (0.99n) x n matrix over GF(2)
#         G: n x g matrix over GF(2)
#     """
#     n_small = int(0.01 * n)  # rows in G0
#     n_large = int(0.99 * n)  # number of extra rows / rows in P

#     assert n_small + n_large == n, (
#         f"n={n} doesn't split cleanly: 0.01n={n_small}, 0.99n={n_large}"
#     )
#     assert t - 1 <= n_small, (
#         f"t-1={t-1} exceeds n_small={n_small}; can't place that many ones in s_i"
#     )

#     # Step 1: Sample uniformly random G0 in F_2^{n_small x g}
#     G0 = np.random.randint(0, 2, size=(n_small, g), dtype=np.int8)

#     G = G0.copy()
#     P_rows = []

#     # Step 2: For i = 1, ..., n_large
#     for i in range(1, n_large + 1):
#         # (a) Sample s_i with exactly (t-1) ones in F_2^{n_small}
#         s_i = np.zeros(n_small, dtype=np.int8)
#         positions = np.random.choice(n_small, size=t - 1, replace=False)
#         s_i[positions] = 1

#         # (b) New row = s_i^T @ G0 (mod 2)
#         new_row = (s_i @ G0) % 2
#         G = np.vstack([G, new_row.reshape(1, -1)])

#         # (c) s'_i = [s_i, 0_{i-1}, 1, 0_{n_large - i}]
#         s_prime = np.zeros(n, dtype=np.int8)
#         s_prime[:n_small] = s_i
#         s_prime[n_small + (i - 1)] = 1
#         P_rows.append(s_prime)

#     P = np.array(P_rows, dtype=np.int8)

#     return P, G


# def verify(P, G, t):
#     product = (P @ G) % 2
#     orthogonal = np.all(product == 0)
#     row_weights = P.sum(axis=1)
#     all_t_sparse = np.all(row_weights == t)
#     return orthogonal, all_t_sparse
