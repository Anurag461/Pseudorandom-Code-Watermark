"""Completion-only behavior through the existing model, detectors and Modal app."""

import hashlib
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix
from detectors import _soft_tokens, detect_hoeffding, detect_online_hoeffding, map_soft_token
from modal_run import (
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
    soft = _soft_tokens(bits, p, weight)
    assert soft[0] == 0
    np.testing.assert_array_equal(soft[1:], _soft_tokens(bits[1:], p, weight, completion_only=False))
    key = fixed_key()
    supports = key[1].indices.copy()
    expected, info = Detect(key, soft, false_positive_rate=0.001, return_info=True)
    actual, scored = detect_hoeffding(
        key, torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, weight=weight, return_info=True
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
    actual, info = detect_hoeffding(fixed_key(), torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, return_info=True)
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
    _, info = detect_hoeffding(key, torch.as_tensor(bits), p, torch.eye(2), fpr=0.001, return_info=True)
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
        return_info=True,
    )
    assert not decision and info["V"] == 0 and np.isinf(info["threshold"])


@pytest.mark.parametrize("p", [np.ones(4), np.array([0.2, np.nan, 0.3]), np.array([0.2, 1.1, 0.3])])
def test_refuses_legacy_or_invalid_probability_vectors(p):
    with pytest.raises(ValueError, match="T-1"):
        detect_hoeffding(fixed_key(), torch.zeros(4, dtype=torch.long), p, torch.eye(2))


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
    import csv
    from modal_run import _append_redetection_csv, REDETECT_CSV_COLUMNS
    case = fixture_case(tmp_path, construction)
    case["old_tpr"] = dict(detector_model="Qwen/Qwen3-8B-Base", source="test historical report",
                           evidence_sha256="a" * 64,
                           counts={str(n): {"map": dict(detected=1, count=2), "entropy": dict(detected=0, count=2)}
                                   for n in case["lengths"]})
    prepared = _prepare_redetection(case, {"id": "Qwen/Qwen3-8B-Base"}, {"gpu": "H100", "git_commit": "test"},
                                    {"data": tmp_path / "data"}, tmp_path / "results")
    assert [b["identity"]["count"] for b in prepared["batches"]] == [3, 1]
    model = tiny_model(2, dtype=torch.bfloat16)
    for batch in prepared["batches"]:
        assert set(_redetect_inputs(batch, tmp_path / "results")) == {"tokens", "partition"}
        _recover_redetection_batch(model, batch, tmp_path / "results", validate=True)
        assert _recover_redetection_batch(None, batch, tmp_path / "results")["cached"]
    report = _score_redetection(prepared, tmp_path / "results")
    assert len(report["records"]) == 4 and report["counts"]["9"]["map"]["wm"]["count"] == 2
    csv_out = tmp_path / "redetected.csv"
    _append_redetection_csv(prepared, report, csv_out)
    _append_redetection_csv(prepared, report, csv_out)
    with csv_out.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == REDETECT_CSV_COLUMNS
    assert len(rows) == 3 and rows[-1]["T"] == "9"
    assert rows[-1]["Entropy Model"] == "Qwen3-8B-Base" and rows[-1]["Naive TPR"] == "skipped"
    assert rows[-1]["eta"] == "0.05" and rows[-1]["t"] == "3"
    detected = sum(r["scores"]["9"]["map"]["decision"] for r in report["records"] if r["source"] == "wm")
    assert rows[-1]["Posterior TPR"] == f"{detected}/2 ({detected/2:.1%})"
    assert rows[-1]["Old Posterior TPR"] == "1/2 (50.0%)"
    assert rows[-1]["Old Entropy Aware TPR"] == "0/2 (0.0%)"
    assert not any("Log Hoeffding" in k or "Map" in k for k in rows[-1])
    first = prepared["batches"][0]
    path = tmp_path / "results" / first["root"] / "trace.pt"
    payload = _redetect_load(path)
    payload["probabilities_2_to_T"][0, 0] += 0.01
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trace"):
        _redetect_trace(path, first["identity"])


def test_redetect_load_preserves_newer_galois_key_when_factory_is_missing(tmp_path, monkeypatch):
    import galois
    from galois._fields import _factory
    if not hasattr(_factory, "_reconstruct_field_class"):
        pytest.skip("fixture requires a newer galois serializer")
    field = galois.GF(2)
    value = {"generator": field([[1, 0], [0, 1], [1, 1]]), "pad": field([0, 1, 1])}
    path = tmp_path / "fixed_key.pt"
    torch.save(value, path)
    monkeypatch.delattr(_factory, "_reconstruct_field_class")
    with pytest.raises(AttributeError, match="_reconstruct_field_class"):
        torch.load(path, weights_only=False, map_location="cpu")
    restored = _redetect_load(path)
    for name, original in value.items():
        assert type(restored[name]) is type(original)
        np.testing.assert_array_equal(restored[name], original)


def test_changed_sources_fail_before_inference_and_batch_125_is_allowed(tmp_path):
    case = fixture_case(tmp_path, "fixed")
    case["batch_size"] = 125
    prepared = _prepare_redetection(case, {}, {}, {"data": tmp_path / "data"}, tmp_path / "results")
    assert len(prepared["batches"]) == 1
    (tmp_path / "data" / "0.pt").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source changed"):
        _prepare_redetection(case, {}, {}, {"data": tmp_path / "data"}, tmp_path / "results")


@pytest.mark.parametrize("construction", ["fixed", "prefix", "online"])
def test_default_rejects_legacy_traces_and_has_no_prompt_argument(construction):
    from detectors import detect_hoeffding_prefix

    detector = {"fixed": detect_hoeffding, "prefix": detect_hoeffding_prefix, "online": detect_online_hoeffding}[
        construction
    ]
    key = OnlinePRCKey.from_seed(91, check_weight=3, noise_rate=0.05) if construction == "online" else fixed_key()
    tokens = torch.tensor([0, 1, 0, 1])
    with pytest.raises(ValueError, match="T-1"):
        detector(key, tokens, np.full(4, 0.5), torch.eye(2))
    with pytest.raises(TypeError, match="prompt"):
        detector(key, tokens, np.full(3, 0.5), torch.eye(2), prompt_ids=[999])
    assert detector(key, tokens, np.full(3, 0.5), torch.eye(2)) == detector(
        key, tokens, np.full(3, 0.5), torch.eye(2), completion_only=True
    )
    # Historical controls remain explicitly available, never selected by default.
    detector(key, tokens, np.full(4, 0.5), torch.eye(2), completion_only=False)


def test_gpu_input_rejects_prompt_even_with_matching_checksum(tmp_path):
    from detectors import semantic_sha256

    case = fixture_case(tmp_path, "fixed")
    prepared = _prepare_redetection(case, {}, {}, {"data": tmp_path / "data"}, tmp_path / "results")
    batch = prepared["batches"][0]
    path = tmp_path / "results" / batch["root"] / "inputs.pt"
    payload = _redetect_load(path)
    payload["prompt_ids"] = torch.tensor([[999]])
    batch["identity"]["input_sha256"] = semantic_sha256(payload)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="only frozen completion tokens"):
        _redetect_inputs(batch, tmp_path / "results")


@pytest.mark.parametrize("mismatch", ["prompted", "eot", "other_run", "wrong_length"])
def test_cache_rejects_legacy_or_unrelated_traces(tmp_path, mismatch):
    from detectors import tensor_sha256
    from modal_run import REDETECT_PROTOCOL

    identity = dict(protocol=REDETECT_PROTOCOL, run="raw-run", count=2, length=5)
    trace = torch.full((2, 4), 0.5)
    payload = {"identity": identity.copy(), "probabilities_2_to_T": trace, "probabilities_sha256": tensor_sha256(trace)}
    if mismatch == "prompted":
        payload = {"p_trace": torch.full((2, 5), 0.5)}
    elif mismatch == "eot":
        payload["identity"]["protocol"] = "completion_only_eot_v1"
    elif mismatch == "other_run":
        payload["identity"]["run"] = "different-model-or-gpu"
    else:
        payload["probabilities_2_to_T"] = torch.full((2, 5), 0.5)
        payload["probabilities_sha256"] = tensor_sha256(payload["probabilities_2_to_T"])
    path = tmp_path / "trace.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="trace"):
        _redetect_trace(path, identity)


@pytest.mark.parametrize("gpu", ["A100-80GB", "H100", "L40S"])
@pytest.mark.parametrize("size", ["0.6B", "8B"])
def test_runner_passes_gpu_and_validates_once_before_eight_batches(tmp_path, monkeypatch, gpu, size):
    """Exercise the real dispatcher with every remote operation replaced locally."""
    import json
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    import modal_run as runner

    monkeypatch.chdir(tmp_path)
    files = runner.EXECUTION_FILES
    for name in files:
        Path(name).write_text("committed-source")
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda command, **kw: "test-commit" if command[1] == "rev-parse" else b"committed-source",
    )
    case = dict(
        id="test", batch_size=125, lengths=[3104], records=[{}] * 1000, generation_model="Qwen3-8B-Base", cache="static"
    )
    manifest = dict(
        protocol=runner.REDETECT_PROTOCOL,
        schema_version=1,
        cases=[case],
        model=detector_spec(size),
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    batches = [dict(root=f"batch-{i}", identity=dict(count=125)) for i in range(8)]
    calls = []
    monkeypatch.setattr(
        runner, "prepare_redetection", SimpleNamespace(remote=lambda *args: dict(root="run", batches=batches))
    )

    def validate(batch, validate=False):
        assert validate
        calls.append(batch)

    def distribute(received):
        assert calls == [batches[0]] and received == batches
        calls.append("distributed")
        return []

    def with_options(**options):
        assert options["gpu"] == gpu and options["max_containers"] == 10
        assert options["memory"] == (65536 if size == "8B" else 8192)
        return lambda **kw: SimpleNamespace(redetect_batch=SimpleNamespace(remote=validate, map=distribute))

    monkeypatch.setattr(runner, "RedetectionModel", SimpleNamespace(with_options=with_options))
    monkeypatch.setattr(runner, "finish_redetection", SimpleNamespace(remote=lambda p: dict(root="run", counts={})))
    monkeypatch.setattr(runner, "redetect_results", SimpleNamespace(read_file=lambda p: [b"{}\n"]))
    monkeypatch.setattr(runner, "_append_redetection_csv", lambda *args: calls.append("csv saved"))
    runner.redetect(str(path), stage="full", gpu=gpu)
    assert calls == [batches[0], "distributed", "csv saved"]


def detector_spec(size):
    spec = dict(id=f"Qwen/Qwen3-{size}-Base", size=size, dtype="bfloat16", revision="a" * 40,
                cache_directory=f"models/Qwen3-{size}-Base", tokenizer_sha256="b" * 64)
    if size == "0.6B":
        spec["weights_sha256"] = "c" * 64
    else:
        spec.update(weight_files={"model-00001-of-00001.safetensors": "c" * 64}, index_sha256="d" * 64)
    return spec


@pytest.mark.parametrize("size", ["0.6B", "8B"])
def test_checkpoint_hashes_revision_and_shard_index_are_verified(tmp_path, size):
    import json
    from modal_run import _verify_redetection_checkpoint
    spec = detector_spec(size)
    cache = tmp_path / spec["cache_directory"]
    metadata = cache / ".cache/huggingface/download"
    metadata.mkdir(parents=True)
    filename = "model.safetensors" if size == "0.6B" else "model-00001-of-00001.safetensors"
    (cache / filename).write_bytes(b"original weights")
    digest = hashlib.sha256(b"original weights").hexdigest()
    (metadata / (filename + ".metadata")).write_text(spec["revision"] + "\n")
    (cache / "tokenizer.json").write_bytes(b"tokenizer")
    spec["tokenizer_sha256"] = hashlib.sha256(b"tokenizer").hexdigest()
    if size == "0.6B":
        spec["weights_sha256"] = digest
    else:
        spec["weight_files"][filename] = digest
        index = cache / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer.weight": filename}}))
        spec["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    _verify_redetection_checkpoint(spec, tmp_path)
    (cache / filename).write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="checkpoint differs"):
        _verify_redetection_checkpoint(spec, tmp_path)
    (cache / filename).write_bytes(b"original weights")
    (metadata / (filename + ".metadata")).write_text("wrong revision\n")
    with pytest.raises(ValueError, match="revision differs"):
        _verify_redetection_checkpoint(spec, tmp_path)
    if size == "8B":
        index.write_text(json.dumps({"weight_map": {"layer.weight": "missing.safetensors"}}))
        spec["index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="index does not match"):
            _verify_redetection_checkpoint(spec, tmp_path)


@pytest.mark.parametrize("command", ["fixed_main", "online_main", "replicate_main"])
def test_old_main_is_retired_before_any_remote_work(command):
    import modal_run as runner
    with pytest.raises(RuntimeError, match="Prompt-dependent detection has been retired"):
        getattr(runner, command)()


@pytest.mark.parametrize("construction", ["fixed", "online"])
def test_generation_only_commands_preserve_cached_inputs_and_do_not_detect(monkeypatch, construction):
    from types import SimpleNamespace
    import modal_run as runner
    plan = dict(wm_missing=[], null_missing=[], null_T=400, null_root="/data/_nulls",
                wm_mode="exact", wm_source_T=400, wm_resume_source_T=0, wm_rejected_candidates=[])
    monkeypatch.setattr(runner, construction + "_build_artifacts", SimpleNamespace(remote=lambda *a: dict(reused=True, artifact_fingerprint="original-key")))
    monkeypatch.setattr(runner, construction + "_plan_generation", SimpleNamespace(remote=lambda *a: plan))
    def unexpected_gpu(**kw):
        raise AssertionError("cached generation must not start a GPU")
    monkeypatch.setattr(runner, construction.title() + "GenerationModel", SimpleNamespace(with_options=unexpected_gpu))
    getattr(runner, "generate_" + construction)(n=400)
    assert plan["wm_missing"] == [] and plan["null_missing"] == []


@pytest.mark.parametrize("construction", ["fixed", "online"])
def test_separate_generation_commands_keep_batch_and_gpu_settings(monkeypatch, construction):
    from types import SimpleNamespace
    import modal_run as runner
    plan = dict(wm_missing=list(range(500)), null_missing=list(range(500)), null_T=3104,
                null_root="/data/_nulls", wm_mode="fresh", wm_source_T=0,
                wm_resume_source_T=0, wm_resume_source_tag="", wm_rejected_candidates=[])
    built = []
    def build(*args):
        built.append(args)
        return dict(reused=True, artifact_fingerprint="original-key")
    monkeypatch.setattr(runner, construction + "_build_artifacts", SimpleNamespace(remote=build))
    monkeypatch.setattr(runner, construction + "_plan_generation", SimpleNamespace(remote=lambda *a: plan))
    options, batches = [], []
    def dispatch(requests):
        chunks = [r["prompt_indices"] if isinstance(r, dict) else r for r in requests]
        batches.extend(chunks)
        return [dict(generated=len(c), batch=len(c)) for c in chunks]
    def with_options(**kw):
        options.append(kw)
        return lambda **kw: SimpleNamespace(
            ready=SimpleNamespace(remote=lambda: dict(generation_model="Qwen3-8B-Base", model_cache_dir="cached")),
            generate_wm=SimpleNamespace(map=dispatch), generate_null=SimpleNamespace(map=dispatch))
    monkeypatch.setattr(runner, construction.title() + "GenerationModel", SimpleNamespace(with_options=with_options))
    getattr(runner, "generate_" + construction)(n=3104, eta=.2, num_prompts=500, batch=125,
                                               generation_model_size="8B", gpu="H100", max_containers=10)
    assert options == [dict(gpu="H100", max_containers=10)]
    assert len(batches) == 8 and all(len(b) == 125 for b in batches)
    assert sorted(i for b in batches for i in b) == sorted(list(range(500)) * 2)
    assert len(built) == 1 and "8B" in built[0]


def test_one_shared_app_and_no_prompted_replay_implementations():
    import ast
    from pathlib import Path
    import modal_run
    source = Path(modal_run.__file__)
    for removed in ("modal_runtime.py", "modal_fixed.py", "modal_online.py", "modal_fixed_replicate_run.py"):
        assert not (source.parent / removed).exists()
    tree = ast.parse(source.read_text())
    called = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)]
    assert "estimate_partition_trace_batch" not in called
    assert "estimate_partition_entropy_trace_batch" not in called
    functions = modal_run.app.registered_functions
    for name in ("fixed_build_artifacts", "online_build_artifacts", "replicate_build_artifacts",
                 "online_plan_generation", "online_audit_continuation", "prepare_redetection", "finish_redetection"):
        assert name in functions
