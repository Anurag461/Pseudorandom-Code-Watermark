KGW_GAMMA, KGW_DELTA = (0.25, 2.0)
EOS = 151643
SYNTHID = dict(
    ngram_len=5,
    keys=[
        654,
        400,
        836,
        123,
        340,
        443,
        597,
        160,
        57,
        29,
        590,
        639,
        13,
        715,
        468,
        990,
        966,
        226,
        324,
        585,
        118,
        504,
        421,
        521,
        129,
        669,
        732,
        225,
        90,
        960,
    ],
    sampling_table_size=65536,
    sampling_table_seed=0,
    context_history_size=1024,
)
ATTACK_VOCAB = 151665
NULL_SEED = 2
KEY_SEED = 1
KEY_LENGTH = 256
NUM_PROMPTS = 500
PROMPT_TOKENS = 50
M = 400
KTH_COMMIT = "80d4ec8f4280da2a2cada03adfc8940593d1964c"
KTH_REPO = "https://github.com/jthickstun/watermark"


def key_seeds():
    import torch

    torch.manual_seed(KEY_SEED)
    return torch.randint(2**32, (NUM_PROMPTS,))


def synthid_processor(device):
    from transformers.generation.logits_process import (
        SynthIDTextWatermarkLogitsProcessor,
    )

    processor = SynthIDTextWatermarkLogitsProcessor(
        **SYNTHID, device=torch_device("cpu")
    )
    processor.keys = processor.keys.to(device)
    processor.sampling_table = processor.sampling_table.to(device)
    processor.device = device
    return processor


def torch_device(name):
    import torch

    return torch.device(name)


def synthid_mean_score(processor, tokens):
    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()
    mask = processor.compute_context_repetition_mask(ids)
    mask = (
        mask * processor.compute_eos_token_mask(ids, EOS)[:, processor.ngram_len - 1 :]
    )
    count = mask.sum() * g.shape[-1]
    return float((g * mask[..., None]).sum() / count) if count else 0.5


class Scorer:

    def __init__(self, scheme, vocab_size, tokenizer=None):
        import torch

        self.scheme, self.vocab_size = (scheme, vocab_size)
        if scheme == "synthid":
            self.processor = synthid_processor(torch.device("cpu"))
        elif scheme == "kgw2":
            from watermarking.kirchenbauer.watermark_processor import WatermarkDetector

            self.detector = WatermarkDetector(
                vocab=list(tokenizer.get_vocab().values()),
                gamma=KGW_GAMMA,
                seeding_scheme="simple_1",
                device=torch.device("cpu"),
                tokenizer=tokenizer,
                z_threshold=1.5,
                normalizers=[],
                ignore_repeated_bigrams=False,
            )
        else:
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

        if self.scheme == "kgw2":
            return -float(self.detector._score_sequence(tokens)["z_score"])
        if self.scheme == "synthid":
            return -synthid_mean_score(self.processor, tokens)
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if null or reference:
            return float(self.phi(tokens, generator, null))
        return float(self.keyed_fast(tokens, generator))


def _cpu_kgw_processor(vocab):
    import torch
    from watermarking.kirchenbauer.watermark_processor import WatermarkLogitsProcessor

    class CPUSeededKGW(WatermarkLogitsProcessor):

        def __call__(self, input_ids, scores):
            if self.rng is None:
                self.rng = torch.Generator()
            ids = [
                self._get_greenlist_ids(row.cpu()).to(scores.device)
                for row in input_ids
            ]
            mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=ids)
            return self._bias_greenlist_logits(
                scores=scores, greenlist_mask=mask, greenlist_bias=self.delta
            )

    return CPUSeededKGW(
        vocab=vocab, gamma=KGW_GAMMA, delta=KGW_DELTA, seeding_scheme="simple_1"
    )


def empirical_p(reference, stat):
    import numpy as np

    return float(np.searchsorted(reference, stat, side="right") / len(reference))


def hoeffding_p(info):
    import math

    S, V = (info["statistic"], info["V"])
    return 1.0 if S is None or V in (None, 0) or S <= 0 else math.exp(-S * S / (2 * V))


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


def synthid_pvalue(processor, tokens):
    import math
    from scipy.stats import norm
    from .kuditipudi import SYNTHID

    ids = tokens.unsqueeze(0)
    g = processor.compute_g_values(ids).float()
    mask = processor.compute_context_repetition_mask(ids)
    mask = (
        mask * processor.compute_eos_token_mask(ids, EOS)[:, SYNTHID["ngram_len"] - 1 :]
    )
    count = float(mask.sum()) * g.shape[-1]
    if not count:
        return (1.0, 0.5)
    mean = float((g * mask[..., None]).sum()) / count
    return (float(norm.sf((mean - 0.5) / math.sqrt(0.25 / count))), mean)
