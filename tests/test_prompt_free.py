import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from detectors import map_soft_token
from online_prc import OnlinePRCKey, materialize_supports, otp_prefix
from prc import Detect
from prompt_free.core import PROTOCOL, recover, score, soft_scores
from prompt_free.manifest import plan, validate
from prompt_free.storage import (
    TRACE_FIELD, aggregate, load_gpu_input, load_pt, prepare_case,
    token_sha, trace_payload, validate_trace,
)
from qwen import Qwen3Model


def tiny_model(vocab=31):
    torch.manual_seed(19)
    return Qwen3Model({"vocab_size": vocab, "context_length": 32, "emb_dim": 16,
        "n_heads": 4, "n_layers": 2, "hidden_dim": 32, "head_dim": 4,
        "qk_norm": True, "n_kv_groups": 2, "rope_base": 10000.,
        "dtype": torch.float32}).eval()


def fixed_artifact(n=4):
    dense = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1]]) if n == 4 else np.ones((1, 3))
    key = (np.zeros((n, 1)), csr_matrix(dense), np.arange(n) % 2, .001, .05, None, 0, 0, 3)
    return {"decoding_key": key, "partition": torch.tensor([[1, 0], [0, 1]], dtype=torch.bfloat16)}


@pytest.mark.parametrize("cache", ["static", "concat"])
def test_actual_inputs_have_no_prefix_and_logits_predict_next_coordinate(cache):
    model = tiny_model()
    tokens = torch.tensor([[3, 6, 9, 12, 15], [4, 7, 10, 13, 16]])
    part = torch.tensor([0., 1.]*15+[0.])
    received = []
    hook = model.register_forward_pre_hook(lambda m, args: received.append(args[0].clone()))
    actual = recover(model, tokens, part, cache=cache)
    hook.remove()
    assert torch.equal(torch.cat(received, 1), tokens[:, :-1])
    with torch.no_grad():
        full = (model(tokens).softmax(-1)*part).sum(-1)[:, :-1]
    torch.testing.assert_close(actual, full, rtol=1e-6, atol=1e-7)
    changed = tokens.clone(); changed[:, 2:] = 24
    future = recover(model, changed, part, cache=cache)
    torch.testing.assert_close(actual[:, :2], future[:, :2])
    assert torch.equal(recover(model, tokens, part, cache=cache), actual)
    assert torch.equal(recover(model, tokens.flip(0), part, cache=cache).flip(0), actual)
    assert recover(model, tokens[:, :1], part, cache=cache).shape == (2, 0)


@pytest.mark.parametrize("weight", ["map", "entropy"])
def test_coordinate_one_only_is_zero_and_original_checks_remain(weight):
    artifact = fixed_artifact()
    tokens = torch.tensor([0, 1, 0, 1])
    p = np.array([.2, .7, .4])
    soft = soft_scores(tokens.numpy(), p, weight)
    assert soft[0] == 0 and len(soft) == len(tokens)
    key = artifact["decoding_key"]
    saved = key[1].indices.copy()
    direct, info = Detect(key, soft, false_positive_rate=.001, return_info=True)
    result = score(artifact, tokens, p, construction="fixed", fpr=.001,
                   fpr_policy="block_or_bonferroni", weight=weight)
    assert result["decision"] == direct
    for field in ("statistic", "V", "threshold"):
        assert result["blocks"][0][field] == info[field]
    np.testing.assert_array_equal(key[1].indices, saved)


def test_multiblock_only_abstains_once_preserves_bonferroni_and_trailing_policy():
    artifact = fixed_artifact(); key = artifact["decoding_key"]
    tokens = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    p = np.linspace(.1, .9, 9)
    soft = np.r_[0., map_soft_token(tokens.numpy()[1:], p)]
    result = score(artifact, tokens, p, construction="fixed", fpr=.001, fpr_policy="block_or_bonferroni")
    assert result["num_blocks"] == 2 and result["block_fpr"] == .0005
    assert result["ignored_trailing_tokens"] == 2
    assert soft[4] != 0
    for b in range(2):
        _, expected = Detect(key, soft[b*4:b*4+4], false_positive_rate=.0005, return_info=True)
        for field in ("statistic", "threshold", "V"):
            assert result["blocks"][b][field] == expected[field]


def test_short_prefix_preserves_supported_checks_without_folding_or_index_shift():
    dense = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 0, 1]])
    key = (np.zeros((5, 1)), csr_matrix(dense), np.array([1, 0, 1, 0, 1]), .001, .05, None, 0, 0, 3)
    artifact = {"decoding_key": key, "partition": fixed_artifact()["partition"]}
    tokens = torch.tensor([1, 0, 1, 0]); p = np.array([.2, .7, .4])
    soft = np.r_[0., map_soft_token(tokens.numpy()[1:], p)]
    result = score(artifact, tokens, p, construction="fixed", fpr=.001, fpr_policy="block_or_bonferroni")
    product = np.prod(soft[[1, 2, 3]])
    assert result["r"] == 2  # The third row includes unobserved coordinate 5.
    assert result["statistic"] == -product
    assert result["V"] == product**2
    assert result["threshold"] == np.sqrt(2*product**2*np.log(1000))


@pytest.mark.parametrize("length", [1, 2, 3])
def test_fixed_no_evidence_never_detects_even_when_old_zero_threshold_would(length):
    artifact = fixed_artifact(3)
    result = score(artifact, torch.zeros(length, dtype=torch.int64), np.full(length-1, .5),
                   construction="fixed", fpr=.001, fpr_policy="block_or_bonferroni")
    assert result["decision"] is False
    info = result["blocks"][0] if "blocks" in result else result
    assert info["V"] == 0 and info["threshold"] is None


@pytest.mark.parametrize("policy", ["one_shot", "alpha_spending_v1"])
def test_online_original_supports_otp_and_threshold_policy(policy):
    key = OnlinePRCKey.from_seed(39, check_weight=3, noise_rate=.05)
    artifact = {"online_key": key.to_dict(), "partition": fixed_artifact()["partition"]}
    tokens = torch.tensor([0, 1]*8)
    probabilities = np.linspace(.1, .9, 15)
    soft = np.r_[0., map_soft_token(tokens.numpy()[1:], probabilities)]
    supports = materialize_supports(16, key)
    products = np.prod(soft[supports], axis=1)
    signs = np.prod(1-2*otp_prefix(16, key).astype(np.int64)[supports], axis=1)
    result = score(artifact, tokens, probabilities, construction="online", fpr=.001, fpr_policy=policy)
    alpha = .001 if policy == "one_shot" else 6*.001/(np.pi**2*16**2)
    assert result["statistic"] == np.sum(signs*products)
    assert result["V"] == np.sum(products**2)
    assert result["threshold"] == np.sqrt(2*np.sum(products**2)*np.log(1/alpha))


def test_refuses_legacy_length_invalid_bits_probabilities_and_fpr_policy():
    with pytest.raises(ValueError, match="T-1"):
        soft_scores([0, 1, 0], [.5]*3)
    with pytest.raises(ValueError, match="binary"):
        soft_scores([.2, 1, 0], [.5]*2)
    for p in [[np.nan, .5], [1.1, .5], [-.1, .5]]:
        with pytest.raises(ValueError, match="probabilities"):
            soft_scores([0, 1, 0], p)
    with pytest.raises(ValueError, match="policy"):
        score(fixed_artifact(), torch.tensor([0, 1, 0]), [.5]*2,
              construction="fixed", fpr=.001, fpr_policy="one_shot")


def fixture_manifest(tmp_path, construction="fixed"):
    original = json.loads(Path("prompt_free/manifests/pilots.json").read_text())
    model = original["model"]
    data = tmp_path/"data"; data.mkdir()
    def save(name, value):
        path = data/name; torch.save(value, path)
        raw = path.read_bytes()
        return {"volume": "data", "path": name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    artifact = fixed_artifact()
    if construction == "online":
        artifact = {"partition": artifact["partition"], "online_key": OnlinePRCKey.from_seed(71, check_weight=3, noise_rate=.05).to_dict()}
    artifact["prompt_ids_list"] = [[99, 98]]*2
    records = []
    for i, source in enumerate(["wm", "wm", "null", "null"]):
        tokens = torch.tensor([(j+i) % 2 for j in range(9)])
        ref = save(f"{i}.pt", {"tokens": tokens, "watermark": source == "wm", "prompt_idx": i % 2,
                                "prompt_token_ids": [999], "p_trace": np.full(9, np.nan)})
        records.append({"source": source, "prompt_idx": i % 2, "file": ref, "tokens_sha256": token_sha(tokens)})
    case = {"id": "test", "generation_model": "Qwen3-8B-Base", "construction": construction,
            "artifact": save("artifact.pt", artifact), "lengths": [3, 4, 9], "fpr": .001,
            "fpr_policy": "block_or_bonferroni" if construction == "fixed" else "one_shot",
            "weights": ["map", "entropy"], "batch_size": 3, "cache": "static", "records": records}
    return {"schema_version": 1, "protocol": PROTOCOL, "model": model, "cases": [case]}


@pytest.mark.parametrize("construction", ["fixed", "online"])
def test_end_to_end_cpu_batch_resume_prefixes_and_no_prompt_leak(tmp_path, construction):
    manifest = fixture_manifest(tmp_path, construction)
    assert plan(manifest)["inference_launched"] is False
    case = manifest["cases"][0]
    source = {"sha256": "a"*64, "git_commit": "b"*40}
    roots = {"data": tmp_path/"data"}; results = tmp_path/"results"
    prepared = prepare_case(case, manifest["model"], source, roots, results)
    model = tiny_model(2)
    assert [b["identity"]["actual_batch_size"] for b in prepared["batches"]] == [3, 1]
    for batch in prepared["batches"]:
        inputs = load_gpu_input(batch, results)
        assert set(inputs) == {"tokens", "partition"}
        trace = recover(model, inputs["tokens"], inputs["partition"][1], cache="static")
        payload = trace_payload(trace, batch["identity"])
        path = results/batch["root"]/"trace.pt"; torch.save(payload, path)
        assert validate_trace(load_pt(path), batch["identity"]).shape[1] == 8
        with pytest.raises(ValueError, match="legacy"):
            validate_trace({"p_trace": trace}, batch["identity"])
        changed = copy.deepcopy(payload); changed[TRACE_FIELD][0, 0] += .01
        with pytest.raises(ValueError, match="contents"):
            validate_trace(changed, batch["identity"])
        changed_id = {**batch["identity"], "cache": "concat"}
        with pytest.raises(ValueError, match="identity"):
            validate_trace(payload, changed_id)
    resumed = prepare_case(case, manifest["model"], source, roots, results)
    assert resumed == prepared
    result = aggregate(prepared, results)
    assert result["passed"]
    assert all(result["counts"][str(n)]["map"]["wm"]["count"] == 2 for n in case["lengths"])
    missing = results/prepared["batches"][1]["root"]/"trace.pt"; missing.unlink()
    with pytest.raises(FileNotFoundError):
        aggregate(prepared, results)


def test_manifest_rejects_duplicates_prompt_inputs_and_execution_changes(tmp_path):
    original = fixture_manifest(tmp_path)
    for mutate in [lambda m: m.update(protocol="completion_only_eot_v1"),
                   lambda m: m["model"].update(dtype="float32"),
                   lambda m: m["cases"][0].update(prompt="original prompt"),
                   lambda m: m["cases"][0]["records"].append(m["cases"][0]["records"][0]),
                   lambda m: m["cases"][0]["artifact"].update(path="../outside.pt")]:
        changed = copy.deepcopy(original); mutate(changed)
        with pytest.raises(ValueError):
            validate(changed)


def test_changed_source_bytes_fail_before_any_inference(tmp_path):
    manifest = fixture_manifest(tmp_path)
    (tmp_path/"data/0.pt").write_bytes(b"changed cached record")
    with pytest.raises(ValueError, match="hash/size"):
        prepare_case(manifest["cases"][0], manifest["model"], {"sha256": "a"*64},
                     {"data": tmp_path/"data"}, tmp_path/"results")


def test_source_guard_requires_exact_committed_files(tmp_path, monkeypatch):
    import prompt_free.manifest as module
    monkeypatch.setattr(module, "SOURCE_FILES", ("code.py",))
    path = tmp_path/"code.py"; path.write_text("value = 1\n")
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "code.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "-c", "core.hooksPath=/dev/null", "commit", "--quiet", "-m", "fixture"], cwd=tmp_path, check=True)
    assert len(module.source_identity(tmp_path, require_commit=True)["git_commit"]) == 40
    path.write_text("value = 2\n")
    with pytest.raises(ValueError, match="not committed"):
        module.source_identity(tmp_path, require_commit=True)


def test_modal_entrypoint_imports_without_launching_a_job():
    # In particular, modal.parameter requires concrete runtime annotations.
    import prompt_free.modal_redetect as runner
    assert runner.Detector is not None
