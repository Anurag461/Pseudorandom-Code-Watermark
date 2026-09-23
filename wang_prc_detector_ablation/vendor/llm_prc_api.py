import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import binom
import galois

GF = galois.GF(2)

def apply_channel_probs(x, channel_probs):
    e = GF(np.random.binomial(1, channel_probs))
    return x + e

def KeyGen(n, t=3, g=None, r=None, noise_rate=None):
    secpar = int(np.log2(binom(n, t)))
    if g is None:
        g = secpar
    if noise_rate is None:
        noise_rate = 0.1
    k = g
    if r is None:
        r = int(0.95*n)
    print(f"n={n} k={k} r={r} noise rate={noise_rate}")

    generator_matrix = GF.Random((n, k))

    row_indices = []
    col_indices = []
    data = []
    for row in range(r):
        chosen_indices = np.random.choice(n - r, t - 1, replace=False)
        chosen_indices = np.append(chosen_indices, n - r + row)
        row_indices.extend([row] * t)
        col_indices.extend(chosen_indices)
        data.extend([1] * t)
        generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(axis=0)
    parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

    
    one_time_pad = GF.Random(n)
    
    
    permutation = np.random.permutation(n)
    generator_matrix = generator_matrix[permutation]
    one_time_pad = one_time_pad[permutation]
    parity_check_matrix = parity_check_matrix[:, permutation]

    encoding_key = (generator_matrix, one_time_pad, g, noise_rate)
    decoding_key = (generator_matrix, parity_check_matrix, one_time_pad, noise_rate, g, t)

    return encoding_key, decoding_key

def Encode(encoding_key):

    generator_matrix, one_time_pad, g, noise_rate = encoding_key
    n, k = generator_matrix.shape

    payload = GF.Random(k)


    error = GF(np.random.binomial(1, noise_rate, n))

    return (payload @ generator_matrix.T + one_time_pad + error)

def Detect(decoding_key, posteriors):

    generator_matrix, parity_check_matrix, one_time_pad,  noise_rate, g, t = decoding_key

    r = parity_check_matrix.shape[0]

    sum = 0
    rows, cols = parity_check_matrix.nonzero()



    tmp_list = []
    for i in range(r):
        tmp_list.append(0)

    for i, j in zip(rows, cols):
        tmp_list[i] += int(posteriors[j]) + int(one_time_pad[j])
    for i in range(r):
        sum += (tmp_list[i]%2)

    threshold = (1/2-r**(-0.25)) * r

    print(f"r={r} sum={sum} threshold={threshold}")
    return sum <= threshold

