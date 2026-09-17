"""Completion-only behavior through the existing model, detectors and Modal app."""

import hashlib
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from detectors import _soft_tokens, detect_hoeffding, detect_online_hoeffding, map_soft_token
from modal_online_run import (
    _prepare_redetection,
    _recover_redetection_batch,
    _redetect_inputs,
    _redetect_load,
    _redetect_trace,
    _score_redetection,
)
from online_prc import OnlinePRCKey, materialize_supports, otp_prefix
from prc import Detect
from qwen import Qwen3Model, completion_only_partition_trace_batch, make_kv_cache


def tiny_model(vocab=31, dtype=torch.float32):
    torch.manual_seed(19)
    return Qwen3Model(
        {
            "vocab_size": vocab,
            "context_length": 32,
            "emb_dim": 16,
            "n_heads": 4,
            "n_layers": 2,
            "hidden_dim": 32,
            "head_dim": 4,
            "qk_norm": True,
            "n_kv_groups": 2,
            "rope_base": 10000.0,
            "dtype": dtype,
        }
    ).eval()


def fixed_key():
    return (
        np.zeros((4, 1)),
        csr_matrix([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]),
        np.arange(4) % 2,
        0.001,
        0.05,
        None,
        0,
        0,
        3,
    )


@pytest.mark.parametrize("cache", ["static", "concat"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_raw_replay_alignment_reference_causality_and_fresh_cache(cache, dtype):
    model = tiny_model(dtype=dtype)
    tokens = torch.tensor([[3, 6, 9, 12, 15], [4, 7, 10, 13, 16]])
    part = torch.tensor([0.0, 1.0] * 15 + [0.0], dtype=dtype)
    received = []
    hook = model.register_forward_pre_hook(lambda m, args: received.append(args[0].clone()))
    actual = completion_only_partition_trace_batch(model, tokens, part, cache)
    hook.remove()
    assert torch.equal(torch.cat(received, 1), tokens[:, :-1])
    kv = make_kv_cache(cache, max_length=4)
    with torch.no_grad():
        expected = torch.stack(
            [(model(tokens[:, i : i + 1], cache=kv)[:, -1].softmax(-1) * part).sum(-1) for i in range(4)], dim=1
        ).float()
    assert torch.equal(actual, expected)
    assert torch.equal(completion_only_partition_trace_batch(model, tokens, part, cache), actual)
    assert torch.equal(completion_only_partition_trace_batch(model, tokens.flip(0), part, cache).flip(0), actual)
    changed = tokens.clone()
    changed[:, 2:] = 24
    assert torch.equal(completion_only_partition_trace_batch(model, changed, part, cache)[:, :2], actual[:, :2])
    assert completion_only_partition_trace_batch(model, tokens[:, :1], part, cache).shape == (2, 0)
    with pytest.raises(TypeError):
        completion_only_partition_trace_batch(model, tokens, part, prompt_ids=torch.tensor([[999]]))


@pytest.mark.parametrize("weight", ["map", "entropy"])
def test_first_coordinate_only_is_zero_and_fixed_statistic_is_unchanged(weight):
    bits, p = np.array([0, 1, 0, 1]), np.array([0.2, 0.7, 0.4])
    soft = _soft_tokens(bits, p, weight, completion_only=True)
    assert soft[0] == 0
    np.testing.assert_array_equal(soft[1:], _soft_tokens(bits[1:], p, weight))
    key = fixed_key()
    supports = key[1].indices.copy()
    expected, info = Detect(key, soft, false_positive_rate=0.001, return_info=True)
    actual, scored = detect_hoeffding(
        key, torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, weight=weight, completion_only=True, return_info=True
    )
    assert actual == expected
    for field in ("statistic", "V", "threshold"):
        assert scored[field] == info[field]
    np.testing.assert_array_equal(key[1].indices, supports)


def test_multiblock_abstains_once_and_preserves_bonferroni_and_trailing_policy():
    bits = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    p = np.linspace(0.1, 0.9, 9)
    soft = np.r_[0.0, map_soft_token(bits[1:], p)]
    assert soft[4] != 0
    expected = [
        Detect(fixed_key(), soft[b * 4 : (b + 1) * 4], false_positive_rate=0.0005, return_info=True) for b in range(2)
    ]
    actual, info = detect_hoeffding(
        fixed_key(), torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, completion_only=True, return_info=True
    )
    assert actual == any(dec for dec, _ in expected)
    assert info["num_blocks"] == 2 and info["block_fpr"] == 0.0005
    best = max((v for _, v in expected), key=lambda v: v["statistic"] - v["threshold"])
    for field in ("statistic", "V", "threshold"):
        assert info[field] == best[field]


def test_short_prefix_preserves_original_coordinate_supports():
    key = (
        np.zeros((5, 1)),
        csr_matrix([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1]]),
        np.array([1, 0, 1, 0, 1]),
        0.001,
        0.05,
        None,
        0,
        0,
        3,
    )
    bits, p = np.array([1, 0, 1, 0]), np.array([0.2, 0.7, 0.4])
    _, info = detect_hoeffding(
        key, torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, completion_only=True, return_info=True
    )
    soft = np.r_[0.0, map_soft_token(bits[1:], p)]
    product = np.prod(soft[[1, 2, 3]])
    otp_sign = np.prod(1 - 2 * key[2][[1, 2, 3]])
    assert info["r_eff"] == 2 and info["statistic"] == otp_sign * product and info["V"] == product**2


@pytest.mark.parametrize("policy", ["one_shot", "alpha_spending_v1"])
def test_online_supports_otp_and_threshold_policy(policy):
    key = OnlinePRCKey.from_seed(91, check_weight=3, noise_rate=0.05)
    bits, p = np.arange(32) % 2, np.linspace(0.1, 0.9, 31)
    soft = np.r_[0.0, map_soft_token(bits[1:], p)]
    supports, otp = materialize_supports(32, key), otp_prefix(32, key).astype(np.int64)
    products = np.prod(soft[supports], axis=1)
    statistic = np.sum(np.prod(1 - 2 * otp[supports], axis=1) * products)
    V = np.sum(products**2)
    alpha = 0.001 if policy == "one_shot" else 6 * 0.001 / (np.pi**2 * 32**2)
    _, info = detect_online_hoeffding(
        key,
        torch.as_tensor(bits),
        p,
        torch.eye(2),
        fpr=0.001,
        fpr_policy=policy,
        completion_only=True,
        return_info=True,
    )
    np.testing.assert_allclose(
        [info["statistic"], info["V"], info["threshold"]],
        [statistic, V, np.sqrt(2 * V * np.log(1 / alpha))],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("length", [1, 4])
def test_zero_evidence_never_detects(length):
    key = list(fixed_key())
    key[1] = csr_matrix([[1, 1, 1, 0]])
    decision, info = detect_hoeffding(
        tuple(key),
        torch.zeros(length, dtype=torch.long),
        np.full(length - 1, 0.5),
        torch.eye(2),
        completion_only=True,
        return_info=True,
    )
    assert not decision and info["V"] == 0 and np.isinf(info["threshold"])


@pytest.mark.parametrize("p", [np.ones(4), np.array([0.2, np.nan, 0.3]), np.array([0.2, 1.1, 0.3])])
def test_refuses_legacy_or_invalid_probability_vectors(p):
    with pytest.raises(ValueError, match="T-1"):
        detect_hoeffding(fixed_key(), torch.zeros(4, dtype=torch.long), p, torch.eye(2), completion_only=True)


def fixture_case(tmp_path, construction):
    data = tmp_path / "data"
    data.mkdir()

    def save(name, value):
        path = data / name
        torch.save(value, path)
        return {
            "volume": "data",
            "path": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    artifact = {"partition": torch.eye(2), "prompt_ids_list": [[999]] * 2}
    if construction == "fixed":
        artifact["decoding_key"] = fixed_key()
    else:
        artifact["online_key"] = OnlinePRCKey.from_seed(71, check_weight=3, noise_rate=0.05).to_dict()
    records = []
    for i, source in enumerate(["wm", "wm", "null", "null"]):
        tokens = torch.tensor([(j + i) % 2 for j in range(9)])
        ref = save(
            f"{i}.pt",
            {
                "tokens": tokens,
                "watermark": source == "wm",
                "prompt_idx": i % 2,
                "prompt_ids": [999],
                "p_trace": np.full(9, np.nan),
            },
        )
        records.append(
            {
                "source": source,
                "prompt_idx": i % 2,
                "file": ref,
                "tokens_sha256": hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            }
        )
    return {
        "id": "test",
        "generation_model": "Qwen3-8B-Base",
        "construction": construction,
        "artifact": save("artifact.pt", artifact),
        "lengths": [3, 4, 9],
        "fpr": 0.001,
        "fpr_policy": "block_or_bonferroni" if construction == "fixed" else "one_shot",
        "weights": ["map", "entropy"],
        "batch_size": 3,
        "cache": "static",
        "records": records,
    }


@pytest.mark.parametrize("construction", ["fixed", "online"])
def test_batch_cache_resume_and_aggregation_exclude_prompts(tmp_path, construction):
    case = fixture_case(tmp_path, construction)
    prepared = _prepare_redetection(case, {"id": "tiny"}, {}, {"data": tmp_path / "data"}, tmp_path / "results")
    assert [b["identity"]["count"] for b in prepared["batches"]] == [3, 1]
    model = tiny_model(2, dtype=torch.bfloat16)
    for batch in prepared["batches"]:
        assert set(_redetect_inputs(batch, tmp_path / "results")) == {"tokens", "partition"}
        _recover_redetection_batch(model, batch, tmp_path / "results", validate=True)
        assert _recover_redetection_batch(model, batch, tmp_path / "results")["cached"]
    report = _score_redetection(prepared, tmp_path / "results")
    assert len(report["records"]) == 4 and report["counts"]["9"]["map"]["wm"]["count"] == 2
    first = prepared["batches"][0]
    path = tmp_path / "results" / first["root"] / "trace.pt"
    payload = _redetect_load(path)
    payload["probabilities_2_to_T"][0, 0] += 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trace"):
        _redetect_trace(path, first["identity"])


def test_changed_sources_fail_before_inference_and_batch_125_is_allowed(tmp_path):
    case = fixture_case(tmp_path, "fixed")
    case["batch_size"] = 125
    prepared = _prepare_redetection(case, {}, {}, {"data": tmp_path / "data"}, tmp_path / "results")
    assert len(prepared["batches"]) == 1
    (tmp_path / "data" / "0.pt").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source changed"):
        _prepare_redetection(case, {}, {}, {"data": tmp_path / "data"}, tmp_path / "results")
