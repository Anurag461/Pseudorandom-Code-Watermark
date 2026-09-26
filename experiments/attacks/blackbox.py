import hashlib
from types import SimpleNamespace
from baselines import exp
from baselines.config import EOS
from baselines.kgw import _cpu_kgw_processor
from baselines.synthid import attack_processor as synthid_processor

CHAT_MODEL = "Qwen/Qwen3-0.6B"
CHAT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
IM_END = 151645
STOP = (IM_END, EOS)
BBD_SEED = 20260924
EXP_KEY_SEED, EXP_KEY_LENGTH = (42, 256)
WORD_LISTS = {
    "peaches": (["peaches", "plums", "cherries", "apricots"], "strawberries"),
    "mangoes": (["mangoes", "pineapples", "papayas", "kiwis"], "strawberries"),
    "berries": (
        ["strawberries", "blueberries", "raspberries", "blackberries"],
        "apples",
    ),
    "apples": (["apples", "bananas", "oranges", "pears"], "strawberries"),
}
FRUITS, EXAMPLE = WORD_LISTS["peaches"]
FORMAT = ""
PREFIXES = [
    "I ate",
    "I chose",
    "I picked",
    "I selected",
    "I took",
    "I went for",
    "I settled on",
    "I got",
    "I gathered",
    "I harvested",
]
RG_MAX_NEW = 65
FS_PROMPT, FS_MAX_NEW = ("This is the story of", 100)
RG_LIST = "apples"
RG_HS = (4, 5)
RG_VALID, RG_FIRST, RG_TOPUP, RG_ROUNDS = (100, 115, 30, 5)
FS_N = 1000
SCHEMES = ("none", "prc", "kgw2", "synthid", "exp")
HF_BATCHES = {"none": 115, "kgw2": 115, "synthid": 60, "exp": 30}


def rg_prompt(prefix, digit, H, fruits=None, example=None):
    fruits, example = (fruits or FRUITS, example or EXAMPLE)
    k = str(digit) * H
    return f'Complete the sentence "{prefix} {k}" using only and exacty a random word from the list: {fruits}.  Answer in this speific format: {FORMAT} {prefix} {k} {example}. (here I chose an other fruit for the sake of the example, you have to choose among {fruits})'


def identify_fruit(text, candidates=FRUITS):
    found = [(i, text.count(c)) for i, c in enumerate(candidates) if text.count(c) > 0]
    return found[0][0] if len(found) == 1 and found[0][1] == 1 else None


def chat_ids(tokenizer, prompt):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )


def cut(tokens):
    tokens = [int(t) for t in tokens]
    ends = [i for i, t in enumerate(tokens) if t in STOP]
    return tokens[: ends[0]] if ends else tokens


def _label_int(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:6], "big")


class Generator:

    def __init__(self, scheme, model, tokenizer, artifact=None):
        self.scheme = scheme
        self.device = str(next(model.parameters()).device)
        self.model, self.tokenizer = (model, tokenizer)
        if scheme == "prc":
            from prc_watermark.prc import OnlinePRCKey
            from prc_watermark.generation import generate_batch_and_collect_online

            self.key = OnlinePRCKey.from_dict(artifact["online_key"])
            self.we = SimpleNamespace(
                model=model,
                device=self.device,
                partition=artifact["partition"].to(self.device),
                generate_batch_and_collect_online=generate_batch_and_collect_online,
            )

    def __call__(self, prompt_ids, n, max_new, label):
        import torch

        if self.scheme == "prc":
            from prc_watermark.prc import derive_document_seed

            base = _label_int(label) % 2**40 << 20
            seeds = [derive_document_seed(BBD_SEED, base + i) for i in range(n)]
            out = []
            for b in range(0, n, 60):
                ids = torch.tensor(
                    [prompt_ids] * len(seeds[b : b + 60]), device=self.we.device
                )
                tokens, _ = self.we.generate_batch_and_collect_online(
                    self.we.model,
                    ids,
                    max_new,
                    self.key,
                    self.we.partition,
                    watermark=True,
                    document_seeds=seeds[b : b + 60],
                )
                out += [cut(row) for row in tokens.cpu()]
            return out
        from transformers import LogitsProcessorList

        torch.manual_seed(_label_int(label) % 2**31)
        out, batch = ([], HF_BATCHES[self.scheme])
        for b in range(0, n, batch):
            rows = min(batch, n - b)
            ids = torch.tensor([prompt_ids] * rows)
            if self.scheme == "exp":
                vocab = self.model.get_output_embeddings().weight.shape[0]
                gen = exp.generate(
                    self.model,
                    ids,
                    vocab,
                    EXP_KEY_LENGTH,
                    max_new,
                    torch.full((rows,), EXP_KEY_SEED),
                    random_offset=True,
                )
            else:
                processors = {
                    "none": [],
                    "kgw2": [
                        _cpu_kgw_processor(list(self.tokenizer.get_vocab().values()))
                    ],
                    "synthid": [synthid_processor(torch.device(self.device))],
                }[self.scheme]
                gen = self.model.generate(
                    ids.to(self.device),
                    attention_mask=torch.ones_like(ids).to(self.device),
                    do_sample=True,
                    max_new_tokens=max_new,
                    top_k=0,
                    top_p=1.0,
                    temperature=1.0,
                    eos_token_id=list(STOP),
                    pad_token_id=EOS,
                    logits_processor=LogitsProcessorList(processors),
                ).cpu()
            out += [cut(row[len(prompt_ids) :]) for row in gen]
        return out


def rg_statistic(L):
    import numpy as np

    median = np.median(L, axis=1)
    std = np.median(np.std(L, axis=0))
    red = L.T - median < -1.96 * std
    green = L.T - median > 1.96 * std
    red_score, green_score = (red.sum(axis=1), green.sum(axis=1))
    return max(red_score.max(), green_score.max()) - max(
        red_score.min(), green_score.min()
    )


def rg_pvalue(counts, permutations=10000, seed=0):
    import numpy as np

    counts = np.asarray(counts, dtype=float)
    probs = (counts + 1) / (RG_VALID + counts.shape[-1])
    logits = np.log(probs / (1 - probs))
    chosen = int(np.argmax(logits.sum(axis=(0, 1))))
    observed = rg_statistic(logits[:, :, chosen])
    rng = np.random.default_rng(seed)
    flat = logits.reshape(-1, logits.shape[-1])
    null = np.array(
        [
            rg_statistic(rng.permutation(flat).reshape(logits.shape)[:, :, chosen])
            for _ in range(permutations)
        ]
    )
    return (
        float(np.mean(null >= observed)),
        int(observed),
        float(probs.mean(axis=(0, 1)).max()),
    )


def fs_pvalue(digests, trials=500, seed=0):
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(seed)
    ids = np.unique(digests, return_inverse=True)[1]
    curve = np.zeros(len(ids))
    for _ in range(trials):
        seen, order = (set(), rng.permutation(ids))
        for i, x in enumerate(order):
            seen.add(int(x))
            curve[i] += len(seen)
    curve /= trials
    return (
        float(stats.mannwhitneyu(curve, np.arange(len(curve))).pvalue),
        int(len(set(digests))),
    )
