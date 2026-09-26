KEY_LENGTH = 256
KTH_COMMIT = "80d4ec8f4280da2a2cada03adfc8940593d1964c"
KTH_REPO = "https://github.com/jthickstun/watermark"


def generate(model, prompts, vocab_size, key_length, length, seeds, *, random_offset):
    from watermarking.generation import generate as upstream_generate
    from watermarking.gumbel.key import gumbel_key_func
    from watermarking.gumbel.sampler import gumbel_sampling

    return upstream_generate(
        model,
        prompts,
        vocab_size,
        key_length,
        length,
        seeds,
        gumbel_key_func,
        gumbel_sampling,
        random_offset=random_offset,
    )


class Scorer:

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        from watermarking.detection import adjacency, phi
        from watermarking.gumbel.key import gumbel_key_func
        from watermarking.gumbel.score import gumbel_score

        self.dist = gumbel_score
        self.key_func, self.adjacency = (gumbel_key_func, adjacency)
        self.phi = lambda tokens, generator, null: phi(
            tokens=tokens,
            n=KEY_LENGTH,
            k=len(tokens),
            generator=generator,
            key_func=gumbel_key_func,
            vocab_size=vocab_size,
            dist=self.dist,
            null=null,
            normalize=False,
        )

    def keyed_fast(self, tokens, generator):
        import torch

        xi, pi = self.key_func(generator, KEY_LENGTH, self.vocab_size)
        if not torch.equal(pi, torch.arange(self.vocab_size)):
            raise ValueError("EXP key permutation must be the identity")
        unique, inverse = torch.unique(tokens, return_inverse=True)
        A = self.adjacency(inverse, xi[:, unique].contiguous(), self.dist, len(tokens))
        return torch.min(torch.min(A, axis=1)[0])

    def __call__(self, tokens, seed, null=False, reference=False):
        import torch

        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if null or reference:
            return float(self.phi(tokens, generator, null))
        return float(self.keyed_fast(tokens, generator))


def exp_pvalue(tokens, seed, vocab_size=151936):
    import torch
    from scipy.special import gammaincc
    from watermarking.gumbel.key import gumbel_key_func

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    xi, _ = gumbel_key_func(generator, KEY_LENGTH, vocab_size)
    m = len(tokens)
    rows = (torch.arange(KEY_LENGTH)[:, None] + torch.arange(m)[None, :]) % KEY_LENGTH
    u = xi[rows, tokens[None, :].expand(KEY_LENGTH, m)].double()
    s_max = float((-torch.log1p(-u)).sum(1).max())
    return (min(1.0, KEY_LENGTH * float(gammaincc(m, s_max))), s_max)
