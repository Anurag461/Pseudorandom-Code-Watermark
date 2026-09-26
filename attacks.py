"""Token-level corruption attacks from Kuditipudi et al. (2023).

Ported from https://github.com/jthickstun/watermark/blob/main/watermarking/attacks.py
with the default uniform replacement distribution. Differences from the original:
randomness comes from an explicit torch.Generator so every corrupted completion is
reproducible, and inputs are never modified in place.

Each attack edits exactly int(rate * len(tokens)) distinct positions, chosen
uniformly at random, as in the original.
"""
import hashlib

import torch

ATTACK_KINDS = ("substitution", "insertion", "deletion")


def _positions(length, rate, generator):
    return torch.randperm(length, generator=generator)[:int(rate * length)]


def substitution_attack(tokens, rate, vocab_size, generator):
    """Replace a `rate` fraction of positions with uniform random tokens."""
    tokens = tokens.clone()
    idx = _positions(len(tokens), rate, generator)
    tokens[idx] = torch.randint(vocab_size, (len(idx),), generator=generator, dtype=tokens.dtype)
    return tokens


def deletion_attack(tokens, rate, generator):
    """Delete a `rate` fraction of positions."""
    keep = torch.ones(len(tokens), dtype=torch.bool)
    keep[_positions(len(tokens), rate, generator)] = False
    return tokens[keep]


def insertion_attack(tokens, rate, vocab_size, generator):
    """Insert int(rate * len) uniform random tokens, each before a distinct original position."""
    idx = _positions(len(tokens), rate, generator)
    inserted = torch.randint(vocab_size, (len(idx),), generator=generator, dtype=tokens.dtype)
    before = torch.zeros(len(tokens), dtype=torch.bool)
    before[idx] = True
    fill = torch.empty(len(tokens), dtype=tokens.dtype)
    fill[idx] = inserted
    pieces = []
    for i in range(len(tokens)):
        if before[i]:
            pieces.append(fill[i:i+1])
        pieces.append(tokens[i:i+1])
    return torch.cat(pieces) if pieces else tokens.clone()


def record_seed(seed, source, prompt_idx):
    """Independent, reproducible stream per candidate."""
    digest = hashlib.sha256(f"kth-attack:{seed}:{source}:{prompt_idx}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def validate_attack(attack):
    if (set(attack) != {"kind", "rate", "seed", "vocab_size"} or attack["kind"] not in ATTACK_KINDS
            or not isinstance(attack["rate"], float) or not 0 <= attack["rate"] < 1
            or type(attack["seed"]) is not int or type(attack["vocab_size"]) is not int
            or attack["vocab_size"] < 2):
        raise ValueError("attack needs kind, a rate in [0, 1), an integer seed and vocab_size")


def apply_attack(tokens, attack, source, prompt_idx):
    """Corrupt one 1-D completion. Output length is len*(1-rate) for deletion, len*(1+rate) for insertion."""
    validate_attack(attack)
    generator = torch.Generator().manual_seed(record_seed(attack["seed"], source, prompt_idx))
    if attack["kind"] == "substitution":
        return substitution_attack(tokens, attack["rate"], attack["vocab_size"], generator)
    if attack["kind"] == "insertion":
        return insertion_attack(tokens, attack["rate"], attack["vocab_size"], generator)
    return deletion_attack(tokens, attack["rate"], generator)
