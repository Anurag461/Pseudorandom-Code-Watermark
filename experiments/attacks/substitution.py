import hashlib
import torch

ATTACK_KINDS = ("substitution",)


def _positions(length, rate, generator):
    return torch.randperm(length, generator=generator)[: int(rate * length)]


def substitution_attack(tokens, rate, vocab_size, generator):
    tokens = tokens.clone()
    idx = _positions(len(tokens), rate, generator)
    tokens[idx] = torch.randint(
        vocab_size, (len(idx),), generator=generator, dtype=tokens.dtype
    )
    return tokens


def record_seed(seed, source, prompt_idx):
    digest = hashlib.sha256(
        f"kth-attack:{seed}:{source}:{prompt_idx}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "little")


def validate_attack(attack):
    if (
        set(attack) != {"kind", "rate", "seed", "vocab_size"}
        or attack["kind"] not in ATTACK_KINDS
        or (not isinstance(attack["rate"], float))
        or (not 0 <= attack["rate"] < 1)
        or (type(attack["seed"]) is not int)
        or (type(attack["vocab_size"]) is not int)
        or (attack["vocab_size"] < 2)
    ):
        raise ValueError(
            "attack needs kind, a rate in [0, 1), an integer seed and vocab_size"
        )


def apply_attack(tokens, attack, source, prompt_idx):
    validate_attack(attack)
    generator = torch.Generator().manual_seed(
        record_seed(attack["seed"], source, prompt_idx)
    )
    return substitution_attack(tokens, attack["rate"], attack["vocab_size"], generator)
