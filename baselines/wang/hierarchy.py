import numpy as np


def reference_path(probabilities, observed):
    p = np.asarray(probabilities, dtype=np.float64)
    width = (len(p) - 1).bit_length()
    lo, hi, out = (0, 2**width, [])
    for shift in range(width - 1, -1, -1):
        mid = (lo + hi) // 2
        den = p[lo : min(hi, len(p))].sum()
        if den <= 0:
            raise ValueError("Observed zero-probability prefix")
        out.append(p[mid : min(hi, len(p))].sum() / den)
        lo, hi = (mid, hi) if observed >> shift & 1 else (lo, mid)
    return np.array(out)


def reference_sample(probabilities, uniforms, codeword=None):
    p = np.asarray(probabilities, dtype=np.float64)
    width = (len(p) - 1).bit_length()
    lo, hi, ps = (0, 2**width, [])
    for depth in range(width):
        mid = (lo + hi) // 2
        prob = p[mid : min(hi, len(p))].sum() / p[lo : min(hi, len(p))].sum()
        ps.append(prob)
        q = (
            prob
            if codeword is None
            else (
                2 * prob * codeword[depth]
                if prob <= 0.5
                else 1 - 2 * (1 - prob) * (1 - codeword[depth])
            )
        )
        lo, hi = (mid, hi) if uniforms[depth] < q else (lo, mid)
    return (lo, np.array(ps))


def probabilities(logits, temperature):
    import torch

    logp = torch.log_softmax(logits.float() / float(temperature), dim=-1)
    p = torch.exp(logp.double())
    return p / p.sum(dim=-1, keepdim=True)


def walk(p, *, observed=None, uniforms=None, codeword=None):
    import torch
    import torch.nn.functional as F

    if (observed is None) == (uniforms is None):
        raise ValueError("Supply observed tokens OR independent sampling uniforms")
    batch, vocab = p.shape
    width = (vocab - 1).bit_length()
    levels = [F.pad(p.double(), (0, 2**width - vocab))]
    cumulative = F.pad(torch.cumsum(levels[0], dim=-1), (1, 0))
    for _ in range(width):
        levels.append(levels[-1].reshape(batch, -1, 2).sum(dim=-1))
    node = torch.zeros(batch, dtype=torch.long, device=p.device)
    rows = torch.arange(batch, device=p.device)
    path, bits = ([], [])
    for depth in range(width):
        parent = levels[width - depth][rows, node]
        upper = levels[width - depth - 1][rows, 2 * node + 1]
        stable = upper / parent
        span = 2 ** (width - depth)
        lo, mid, hi = (node * span, node * span + span // 2, (node + 1) * span)
        denominator = cumulative[rows, hi] - cumulative[rows, lo]
        numerator = cumulative[rows, hi] - cumulative[rows, mid]
        cdf_p1 = numerator / denominator
        tolerance = 1e-12 * torch.minimum(stable, 1 - stable)
        p1 = torch.where(
            torch.isfinite(cdf_p1) & ((cdf_p1 - stable).abs() <= tolerance),
            cdf_p1,
            stable,
        )
        path.append(p1)
        if observed is not None:
            bit = observed >> width - 1 - depth & 1
        else:
            q = p1
            if codeword is not None:
                x = codeword[:, depth]
                q = torch.where(p1 <= 0.5, 2 * p1 * x, 1 - 2 * (1 - p1) * (1 - x))
            bit = (uniforms[:, depth] < q).long()
        bits.append(bit)
        node = 2 * node + bit
    return (node, torch.stack(path, dim=-1))


def entropy(p):
    import torch

    return -torch.special.xlogy(p, p).sum(dim=-1)


def bit_entropy(p):
    p = np.asarray(p, dtype=np.float64)
    result = np.zeros_like(p)
    valid = (p > 0) & (p < 1)
    result[valid] = -(
        p[valid] * np.log2(p[valid]) + (1 - p[valid]) * np.log2(1 - p[valid])
    )
    result[np.isnan(p)] = np.nan
    return result
