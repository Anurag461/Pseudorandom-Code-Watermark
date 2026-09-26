"""Kuditipudi et al. corruption attacks and their redetection integration."""

import pytest
import torch

from attacks import apply_attack, deletion_attack, insertion_attack, substitution_attack
from modal_run import _prepare_redetection, _recover_redetection_batch, _score_redetection
from test_prompt_free import fixture_case, tiny_model


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_substitution_edits_exact_count_and_keeps_input():
    tokens = torch.arange(1000, 1400, dtype=torch.int64)
    original = tokens.clone()
    out = substitution_attack(tokens, 0.25, vocab_size=1000, generator=gen())
    assert torch.equal(tokens, original) and len(out) == 400
    changed = out != original
    assert changed.sum() == 100 and torch.all(out[changed] < 1000)


def test_deletion_keeps_order_of_survivors():
    tokens = torch.arange(400, dtype=torch.int64)
    out = deletion_attack(tokens, 0.1, generator=gen())
    assert len(out) == 360 and torch.all(out[1:] > out[:-1])


def test_insertion_preserves_original_subsequence():
    tokens = torch.arange(1000, 1400, dtype=torch.int64)
    out = insertion_attack(tokens, 0.2, vocab_size=1000, generator=gen())
    assert len(out) == 480
    assert torch.equal(out[out >= 1000], tokens) and (out < 1000).sum() == 80


def test_zero_rate_is_identity():
    tokens = torch.arange(50, dtype=torch.int64)
    for kind in ("substitution", "insertion", "deletion"):
        attack = {"kind": kind, "rate": 0.0, "seed": 1, "vocab_size": 50}
        assert torch.equal(apply_attack(tokens, attack, "wm", 0), tokens)


def test_per_candidate_streams_are_reproducible_and_independent():
    tokens = torch.arange(500, dtype=torch.int64)
    attack = {"kind": "substitution", "rate": 0.3, "seed": 7, "vocab_size": 500}
    a = apply_attack(tokens, attack, "wm", 3)
    assert torch.equal(a, apply_attack(tokens, attack, "wm", 3))
    assert not torch.equal(a, apply_attack(tokens, attack, "null", 3))
    assert not torch.equal(a, apply_attack(tokens, {**attack, "seed": 8}, "wm", 3))


@pytest.mark.parametrize("attack", [
    {"kind": "paraphrase", "rate": 0.1, "seed": 0, "vocab_size": 10},
    {"kind": "deletion", "rate": 1.0, "seed": 0, "vocab_size": 10},
    {"kind": "deletion", "rate": 1, "seed": 0, "vocab_size": 10},
])
def test_invalid_attacks_are_rejected(attack):
    with pytest.raises(ValueError, match="attack"):
        apply_attack(torch.arange(5), attack, "wm", 0)


@pytest.mark.parametrize("construction", ["fixed", "online"])
@pytest.mark.parametrize("kind,length", [("substitution", 9), ("deletion", 7), ("insertion", 11)])
def test_attacked_redetection_replays_corrupted_tokens(tmp_path, construction, kind, length):
    case = fixture_case(tmp_path, construction)
    case["lengths"] = [9]
    case["attack"] = {"kind": kind, "rate": 0.25, "seed": 0, "vocab_size": 2}
    prepared = _prepare_redetection(case, {"id": "Qwen/Qwen3-8B-Base"}, {"gpu": "H100", "git_commit": "test"},
                                    {"data": tmp_path / "data"}, tmp_path / "results")
    assert {b["identity"]["length"] for b in prepared["batches"]} == {length}
    model = tiny_model(2, dtype=torch.bfloat16)
    for batch in prepared["batches"]:
        _recover_redetection_batch(model, batch, tmp_path / "results", validate=True)
    report = _score_redetection(prepared, tmp_path / "results")
    assert report["attack"] == case["attack"] and report["attacked_length"] == min(9, length)
    assert report["counts"]["9"]["map"]["wm"]["count"] == 2


def test_attack_changes_run_identity_and_requires_one_length(tmp_path):
    case = fixture_case(tmp_path, "fixed")
    clean = _prepare_redetection({**case, "lengths": [9]}, {}, {}, {"data": tmp_path / "data"}, tmp_path / "r")
    attacked = {**case, "lengths": [9], "attack": {"kind": "deletion", "rate": 0.25, "seed": 0, "vocab_size": 2}}
    assert _prepare_redetection(attacked, {}, {}, {"data": tmp_path / "data"}, tmp_path / "r")["root"] != clean["root"]
    with pytest.raises(ValueError, match="one completion length"):
        _prepare_redetection({**attacked, "lengths": [3, 9]}, {}, {}, {"data": tmp_path / "data"}, tmp_path / "r")
