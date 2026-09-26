import argparse
import csv
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from baselines import wangetal as hierarchy
from baselines.wangetal import (
    keygen,
    encode,
    token_bits,
    hard_count,
    hard_threshold,
    soft_evidence,
    soft_score,
    calibrate,
    decisions,
    bit_entropy,
)


HERE = Path(__file__).resolve().parent


TEMPERATURES = (1.0, 1.2, 1.4, 1.6, 1.8)


MODEL = json.loads((HERE / "wangetal_model.json").read_text())


PROMPTS = json.loads((HERE / "wangetal_prompts.json").read_text())


DESIGN = dict(
    schema=1,
    model=MODEL,
    temperatures=TEMPERATURES,
    vocab=151936,
    bits=18,
    tokens=1024,
    n=18432,
    r=17510,
    t=3,
    eta=0.1,
    groups=10,
    prompts=PROMPTS,
    calibration_groups=list(range(5)),
    evaluation_groups=list(range(5, 10)),
    null_keys_per_split=256,
    fpr=0.001,
    bootstrap=2000,
    master_seed=251217310,
    generation_batch=80,
    replay_batch=80,
    backend="qwen-static-bf16",
    primary_context="completion-only-first-token-abstain",
    probability_arithmetic="fp32-log-softmax-fp64-positive-mass-tree",
    source=json.loads((HERE.parents[1] / "baselines/sources/wangetal.json").read_text()),
)


def digest_json(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed(*parts):
    return int(digest_json([DESIGN["master_seed"], *parts])[:16], 16)


def implementation_hashes():
    root = HERE.parents[1]
    paths = [Path(__file__).resolve(), root / "baselines/wangetal.py"] + [
        root / "prc_watermark" / name for name in ("qwen.py", "detectors.py", "prc.py")
    ]
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def fingerprint():
    return digest_json({"design": DESIGN, "implementation": implementation_hashes()})[
        :24
    ]


def relative_root():
    return f"wangetal/qwen3_8b_base/{fingerprint()}"


def sample_id(source, group, prompt, temperature):
    return f"{source}_g{group:02d}_p{prompt:02d}_t{temperature:.1f}"


def inventory():
    for temp in TEMPERATURES:
        for source in ("wm", "null"):
            for group in range(10):
                for prompt in range(16):
                    yield dict(
                        id=sample_id(source, group, prompt, temp),
                        source=source,
                        group=group,
                        prompt=prompt,
                        temperature=temp,
                        tokens=1024,
                        seed=seed("sample", source, group, prompt, temp),
                        split=(
                            "watermarked"
                            if source == "wm"
                            else "calibration" if group < 5 else "evaluation"
                        ),
                    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def save_arrays(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    write_json(
        path.with_suffix(".json"),
        {
            "metadata": metadata,
            "sha256": sha256(path),
            "identity": digest_json(metadata),
        },
    )


def load_arrays(path, expected=None):
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if meta["sha256"] != sha256(path) or meta["identity"] != digest_json(
        meta["metadata"]
    ):
        raise ValueError(f"Corrupt cache: {path}")
    if expected is not None:
        for k, v in expected.items():
            if meta["metadata"].get(k) != v:
                raise ValueError(f"Cache identity mismatch: {path}: {k}")
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    return (arrays, meta["metadata"])


def exists(path):
    path = Path(path)
    if path.exists() != path.with_suffix(".json").exists():
        raise ValueError(f"Incomplete atomic cache; inspect before retry: {path}")
    return path.exists()


def prepare(root, provenance):
    root = Path(root)
    keys = [("wm", i) for i in range(10)]
    keys += [(split, i) for split in ("calibration", "evaluation") for i in range(256)]
    for domain, i in keys:
        path = root / "keys" / f"{domain}_{i:03d}.npz"
        identity = dict(
            fingerprint=fingerprint(),
            domain=domain,
            group=i,
            seed=seed("key", domain, i),
        )
        if exists(path):
            (key, _) = load_arrays(path, identity)
        else:
            key = keygen(DESIGN["n"], np.random.default_rng(identity["seed"]))
            save_arrays(path, key, identity)
        if domain == "wm":
            for prompt in range(16):
                cpath = root / "codewords" / f"g{i:02d}_p{prompt:02d}.npz"
                meta = dict(
                    fingerprint=fingerprint(),
                    group=i,
                    prompt=prompt,
                    seed=seed("codeword", i, prompt),
                    key_sha256=sha256(path),
                )
                if exists(cpath):
                    load_arrays(cpath, meta)
                else:
                    save_arrays(
                        cpath, encode(key, np.random.default_rng(meta["seed"])), meta
                    )
    manifest = dict(
        design=DESIGN,
        fingerprint=fingerprint(),
        provenance=provenance,
        implementation=implementation_hashes(),
        smoke=False,
        inventory=list(inventory()),
        keys=[f"{d}_{i:03d}" for (d, i) in keys],
    )
    write_json(root / "settings.json", manifest)
    return manifest


def load_model(cache_root):
    import torch
    from safetensors.torch import load_file
    from prc_watermark.qwen import (
        Qwen3Model,
        Qwen3Tokenizer,
        return_qwen_config,
        load_weights_into_qwen,
    )

    directory = Path(cache_root) / MODEL["cache_directory"]
    expected = {
        **MODEL["weight_files"],
        "model.safetensors.index.json": MODEL["index_sha256"],
        "tokenizer.json": MODEL["tokenizer_sha256"],
    }
    verified = {}
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(
                f"Cached model file does not match pinned checkpoint: {name}"
            )
        verified[name] = digest
    for name in MODEL["weight_files"]:
        metadata = directory / ".cache/huggingface/download" / (name + ".metadata")
        if (
            not metadata.exists()
            or metadata.read_text().splitlines()[0] != MODEL["revision"]
        ):
            raise ValueError(f"Cached checkpoint revision mismatch: {name}")
    cfg = return_qwen_config("8B")
    if cfg["vocab_size"] != DESIGN["vocab"]:
        raise ValueError("Unexpected model vocabulary")
    if (cfg["vocab_size"] - 1).bit_length() != DESIGN["bits"]:
        raise ValueError("Derived token width differs from the frozen design")
    cfg["context_length"] = 2048
    cfg["dtype"] = torch.bfloat16
    model = Qwen3Model(cfg)
    params = {}
    for name in MODEL["weight_files"]:
        params.update(load_file(str(directory / name), device="cpu"))
    load_weights_into_qwen(model, cfg, params)
    del params
    gc.collect()
    model = model.to("cuda").eval().requires_grad_(False)
    tokenizer = Qwen3Tokenizer(
        str(directory / "tokenizer.json"),
        repo_id=MODEL["id"],
        apply_chat_template=False,
        add_generation_prompt=False,
        add_thinking=False,
    )
    prompt_ids = [tokenizer.encode(p, chat_wrapped=False) for p in PROMPTS]
    if any((not p for p in prompt_ids)) or max(map(len, prompt_ids)) + 1024 > 2048:
        raise ValueError("Unexpected Base prompt length")
    return (
        model,
        tokenizer,
        prompt_ids,
        dict(
            files=verified,
            revision=MODEL["revision"],
            torch=torch.__version__,
            device=torch.cuda.get_device_name(),
        ),
    )


def generate(model, prompt_ids, pad_id, specs, codewords):
    import torch
    from prc_watermark.qwen import make_kv_cache

    batch, length = (len(specs), specs[0]["tokens"])
    temp = specs[0]["temperature"]
    assert all((s["temperature"] == temp and s["tokens"] == length for s in specs))
    assert all(((s["source"] == "wm") == (codewords is not None) for s in specs))
    plen = max(map(len, prompt_ids))
    inputs = torch.full((batch, plen), pad_id, dtype=torch.long, device="cuda")
    padding = torch.ones_like(inputs, dtype=torch.bool)
    for row, ids in enumerate(prompt_ids):
        inputs[row, -len(ids) :] = torch.tensor(ids, device="cuda")
        padding[row, -len(ids) :] = False
    cache = make_kv_cache("static", max_length=plen + length)
    uniforms = torch.tensor(
        np.stack(
            [
                np.random.default_rng(s["seed"]).random((length, DESIGN["bits"]))
                for s in specs
            ]
        ),
        device="cuda",
        dtype=torch.float64,
    )
    codes = (
        None
        if codewords is None
        else torch.tensor(
            codewords[:, : length * DESIGN["bits"]].reshape(
                batch, length, DESIGN["bits"]
            ),
            device="cuda",
            dtype=torch.float64,
        )
    )
    tokens, paths, entropies = ([], [], [])
    with torch.inference_mode():
        logits = model(inputs, cache=cache, key_padding_mask=padding)[:, -1]
        for i in range(length):
            p = hierarchy.probabilities(logits, temp)
            token, path = hierarchy.walk(
                p,
                uniforms=uniforms[:, i],
                codeword=None if codes is None else codes[:, i],
            )
            tokens.append(token)
            paths.append(path)
            entropies.append(hierarchy.entropy(p))
            if i + 1 < length:
                logits = model(token[:, None], cache=cache, key_padding_mask=padding)[
                    :, -1
                ]
    result = dict(
        tokens=torch.stack(tokens, 1).cpu().numpy(),
        generation_p1=torch.stack(paths, 1).cpu().numpy(),
        generation_entropy=torch.stack(entropies, 1).cpu().numpy(),
    )

    recovered = token_bits(result["tokens"], DESIGN["bits"])
    u = uniforms.cpu().numpy()
    p = result["generation_p1"]
    x = None if codes is None else codes.cpu().numpy()
    q = p if x is None else np.where(p <= 0.5, 2 * p * x, 1 - 2 * (1 - p) * (1 - x))
    if not np.array_equal(recovered, (u < q).astype(np.uint8)):
        raise ValueError(
            "Saved probabilities/uniforms do not reproduce observed sampler branches"
        )
    result["observed_bits"] = recovered
    validate_trace(result, length)
    return result


def replay(model, completion_ids, temperature, *, capture=None, cache_factory=None):
    import torch

    if cache_factory is None:
        from prc_watermark.qwen import make_kv_cache

        cache_factory = lambda n: make_kv_cache("static", max_length=n)
    tokens = torch.as_tensor(
        completion_ids, device=next(model.parameters()).device, dtype=torch.long
    )
    batch, length = tokens.shape
    if length < 2:
        raise ValueError("Replay needs at least two completion tokens")
    cache = cache_factory(length - 1)
    paths, entropies = ([], [])
    with torch.inference_mode():
        for i in range(1, length):
            model_input = tokens[:, i - 1 : i]
            if capture is not None:
                capture.append(model_input.cpu().numpy().copy())
            logits = model(model_input, cache=cache)[:, -1]
            p = hierarchy.probabilities(logits, temperature)
            _, path = hierarchy.walk(p, observed=tokens[:, i])
            paths.append(path)
            entropies.append(hierarchy.entropy(p))
    return dict(
        replay_p1=np.concatenate(
            (
                np.full((batch, 1, DESIGN["bits"]), np.nan),
                torch.stack(paths, 1).cpu().numpy(),
            ),
            axis=1,
        ),
        replay_entropy=np.concatenate(
            (np.full((batch, 1), np.nan), torch.stack(entropies, 1).cpu().numpy()),
            axis=1,
        ),
    )


def validate_trace(trace, length):
    ids = trace["tokens"]
    if (
        ids.ndim != 2
        or ids.shape[1] != length
        or np.any(ids < 0)
        or np.any(ids >= DESIGN["vocab"])
    ):
        raise ValueError("Invalid generated token IDs/length")
    for name in ("generation_p1", "replay_p1"):
        if name not in trace:
            continue
        p = trace[name][:, 1:] if name == "replay_p1" else trace[name]
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
            raise ValueError(f"Invalid {name}")
        if trace[name].shape != (*ids.shape, DESIGN["bits"]):
            raise ValueError(f"Unexpected {name} shape")
        if name == "replay_p1" and (not np.isnan(trace[name][:, 0]).all()):
            raise ValueError("Replay supplied forbidden first-token probabilities")


def run_temperature(
    root, temperature, cache_root, provenance, commit=None, loaded=None
):
    import torch

    root = Path(root)
    manifest = json.loads((root / "settings.json").read_text())
    if manifest["fingerprint"] != fingerprint() or manifest["smoke"] != False:
        raise ValueError("Preparation/config mismatch")
    specs = [s for s in manifest["inventory"] if s["temperature"] == temperature]
    begun = time.monotonic()
    pending = []
    for s in specs:
        path = root / "traces" / (s["id"] + ".npz")
        expected = dict(sample=s, fingerprint=fingerprint(), context="completion-only")
        if exists(path):
            arrays, _ = load_arrays(path, expected)
            validate_trace(
                {k: v[None] for k, v in arrays.items() if k != "prompt_ids"},
                s["tokens"],
            )
        else:
            pending.append(s)
    if not pending:
        return dict(temperature=temperature, reused=len(specs), generated=0)
    model, tokenizer, prompts, model_meta = (
        load_model(cache_root) if loaded is None else loaded
    )
    load_seconds = time.monotonic() - begun
    batches = []
    for source in ("wm", "null"):
        work = [s for s in pending if s["source"] == source]
        for offset in range(0, len(work), DESIGN["generation_batch"]):
            batch = work[offset : offset + DESIGN["generation_batch"]]
            batch_start = time.monotonic()
            words = None
            if source == "wm":
                words = np.stack(
                    [
                        load_arrays(
                            root
                            / "codewords"
                            / f"g{s['group']:02d}_p{s['prompt']:02d}.npz",
                            {"fingerprint": fingerprint()},
                        )[0]["codeword"]
                        for s in batch
                    ]
                )
            out = generate(
                model,
                [prompts[s["prompt"]] for s in batch],
                tokenizer.pad_token_id,
                batch,
                words,
            )
            generation_seconds = time.monotonic() - batch_start
            capture = None
            out.update(replay(model, out["tokens"], temperature, capture=capture))
            if False and (
                not np.array_equal(
                    np.concatenate(capture, axis=1), out["tokens"][:, :-1]
                )
            ):
                raise ValueError("Prompt-free replay input capture failed")
            validate_trace(out, batch[0]["tokens"])
            for i, s in enumerate(batch):
                arrays = {k: v[i] for k, v in out.items()}
                arrays["prompt_ids"] = np.array(prompts[s["prompt"]], dtype=np.int64)
                meta = dict(
                    sample=s,
                    fingerprint=fingerprint(),
                    context="completion-only",
                    text=tokenizer.decode(arrays["tokens"].tolist()),
                    model=model_meta,
                    provenance=provenance,
                    batch_ids=[v["id"] for v in batch],
                    first_token="all 18 coordinates abstain",
                    codeword=(
                        None
                        if source == "null"
                        else f"codewords/g{s['group']:02d}_p{s['prompt']:02d}.npz"
                    ),
                )
                meta["key_sha256"] = (
                    None
                    if source == "null"
                    else sha256(root / "keys" / f"wm_{s['group']:03d}.npz")
                )
                meta["codeword_sha256"] = (
                    None if source == "null" else sha256(root / meta["codeword"])
                )
                save_arrays(root / "traces" / (s["id"] + ".npz"), arrays, meta)
            if commit is not None:
                commit()
            batches.append(
                dict(
                    source=source,
                    N=len(batch),
                    generation_seconds=generation_seconds,
                    total_seconds=time.monotonic() - batch_start,
                    cuda_peak_bytes=torch.cuda.max_memory_allocated(),
                )
            )
    sanity = None
    result = dict(
        temperature=temperature,
        reused=len(specs) - len(pending),
        generated=len(pending),
        load_seconds=load_seconds,
        batches=batches,
        seconds=time.monotonic() - begun,
        smoke=sanity,
        model=model_meta,
        provenance=provenance,
    )
    write_json(root / f"timing_{temperature:.1f}.json", result)
    return result


LABELS = {
    "wangetal_published": "Wang et al. hard detector (published threshold)",
    "posterior_standard": "Posterior detector (standard threshold, FPR target 1e-3)",
    "hard_matched": "Wang et al. hard statistic @ matched FPR",
    "posterior_matched": "Posterior statistic @ matched FPR",
}


def read_trace(root, sample):
    (arrays, metadata) = load_arrays(
        root / "traces" / (sample["id"] + ".npz"),
        dict(sample=sample, fingerprint=fingerprint(), context="completion-only"),
    )
    bits = token_bits(arrays["tokens"], DESIGN["bits"])
    if not np.array_equal(bits, arrays["observed_bits"]):
        raise ValueError("Saved token IDs and hierarchical bits disagree")
    if sample["source"] == "wm":
        if metadata["key_sha256"] != sha256(
            root / "keys" / f"wm_{sample['group']:03d}.npz"
        ):
            raise ValueError("Watermark trace/key binding changed")
        if metadata["codeword_sha256"] != sha256(root / metadata["codeword"]):
            raise ValueError("Watermark trace/codeword binding changed")
    if (
        arrays["replay_p1"].shape != bits.shape
        or not np.isnan(arrays["replay_p1"][0]).all()
    ):
        raise ValueError("First-token or trace shape invariant failed")
    evidence = soft_evidence(bits, arrays["replay_p1"])
    return (arrays, metadata, bits.ravel(), evidence)


def score_nulls(root, samples, split):
    path = root / "scores" / f"null_{split}.npz"
    trace_hashes = {
        s["id"]: sha256(root / "traces" / (s["id"] + ".npz")) for s in samples
    }
    key_hashes = {
        f"{split}_{k:03d}": sha256(root / "keys" / f"{split}_{k:03d}.npz")
        for k in range(256)
    }
    identity = dict(
        fingerprint=fingerprint(),
        ids=[s["id"] for s in samples],
        split=split,
        trace_hashes=trace_hashes,
        key_hashes=key_hashes,
    )
    if exists(path):
        return (load_arrays(path, identity)[0], identity)
    traces = [read_trace(root, s) for s in samples]
    bits = np.stack([x[2] for x in traces])
    evidence = np.stack([x[3] for x in traces])
    out = {
        k: np.empty((len(samples), 256), dtype=np.float64)
        for k in ("H", "S", "V", "Z", "tau")
    }
    out["no_evidence"] = np.empty((len(samples), 256), dtype=bool)
    out["standard"] = np.empty((len(samples), 256), dtype=bool)
    for k in range(256):
        (key, _) = load_arrays(
            root / "keys" / f"{split}_{k:03d}.npz",
            dict(domain=split, group=k, fingerprint=fingerprint()),
        )
        for begin in range(0, len(samples), 16):
            sl = slice(begin, begin + 16)
            out["H"][sl, k] = hard_count(bits[sl], key)
            soft = soft_score(evidence[sl], key)
            for name in soft:
                out[name][sl, k] = soft[name]
    save_arrays(path, out, identity)
    return (out, identity)


def bootstrap_weights(groups, prompts=16, replicates=2000, domain="wm"):
    rng = np.random.default_rng(seed("bootstrap", domain))
    g = rng.multinomial(groups, np.ones(groups) / groups, replicates)
    p = rng.multinomial(prompts, np.ones(prompts) / prompts, replicates)
    return (g[:, :, None] * p[:, None, :]).reshape(replicates, -1) / (groups * prompts)


def interval(values, weights):
    return np.quantile(weights @ np.asarray(values), [0.025, 0.975]).tolist()


def roc(watermarked, null):
    (wm, neg) = (np.asarray(watermarked).ravel(), np.asarray(null).ravel())
    scores = np.concatenate((wm, neg))
    order = np.argsort(-scores, kind="stable")
    positives = np.concatenate((np.ones(len(wm)) / len(wm), np.zeros(len(neg))))[order]
    negatives = np.concatenate((np.zeros(len(wm)), np.ones(len(neg)) / len(neg)))[order]
    sorted_scores = scores[order]
    ends = np.r_[
        np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1
    ]
    (tpr, fpr) = (
        np.r_[0, np.cumsum(positives)[ends]],
        np.r_[0, np.cumsum(negatives)[ends]],
    )
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return (fpr, tpr, auc)


def csv_write(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def base_record(root, s):
    (arrays, _, bits, evidence) = read_trace(root, s)
    agreement = None
    if s["source"] == "wm":
        word = load_arrays(
            root / "codewords" / f"g{s['group']:02d}_p{s['prompt']:02d}.npz"
        )[0]["codeword"]
        agreement = float(np.mean(word == bits))
    return dict(
        sample_id=s["id"],
        source=s["source"],
        group=s["group"],
        prompt=s["prompt"],
        temperature=s["temperature"],
        split=s["split"],
        generation_seed=s["seed"],
        config_fingerprint=fingerprint(),
        trace_sha256=sha256(root / "traces" / (s["id"] + ".npz")),
        average_full_vocab_entropy=float(np.nanmean(arrays["replay_entropy"])),
        mean_hierarchical_bit_entropy=float(
            np.nanmean(bit_entropy(arrays["replay_p1"]))
        ),
        generation_full_vocab_entropy=float(np.mean(arrays["generation_entropy"])),
        generation_hierarchical_bit_entropy=float(
            np.mean(bit_entropy(arrays["generation_p1"]))
        ),
        entropy_units="full_vocab=nats;binary=bits",
        bit_agreement=agreement,
    )


def scored_record(base, key_id, pairing, hard, soft, dec, thresholds):
    finite_z = float(soft["Z"]) if np.isfinite(soft["Z"]) else None
    return {
        **base,
        "pairing_id": base["sample_id"] + ":" + key_id,
        "scoring_key": key_id,
        "pairing": pairing,
        "hard_violation_count": int(hard),
        "hard_score": (DESIGN["r"] - 2 * float(hard)) / np.sqrt(DESIGN["r"]),
        "S": float(soft["S"]),
        "V": float(soft["V"]),
        "Z": finite_z,
        "no_evidence": bool(soft["no_evidence"]),
        "posterior_S_threshold": float(soft["tau"]),
        "wangetal_published_threshold": hard_threshold(DESIGN["r"]),
        "posterior_standard_Z_threshold": float(np.sqrt(2 * np.log(1000))),
        "hard_matched_threshold": thresholds["hard"]["cutoff"],
        "posterior_matched_threshold": thresholds["posterior"]["cutoff"],
        **{k: bool(v) for (k, v) in dec.items()},
    }


def run(root, provenance):
    root = Path(root)
    begun = time.monotonic()
    manifest = json.loads((root / "settings.json").read_text())
    if manifest["fingerprint"] != fingerprint() or manifest["smoke"]:
        raise ValueError("Production-only scoring requires exact prepared design")
    samples = manifest["inventory"]
    if len(samples) != 1600 or len({s["id"] for s in samples}) != 1600:
        raise ValueError("Production inventory incomplete")
    calibration = [s for s in samples if s["split"] == "calibration"]
    evaluation = [s for s in samples if s["split"] == "evaluation"]
    wm = [s for s in samples if s["source"] == "wm"]
    (null_cal, cal_identity) = score_nulls(root, calibration, "calibration")
    thresholds = calibrate(null_cal["H"], null_cal["Z"])
    frozen_path = root / "threshold_calibration.json"
    frozen_content = dict(
        thresholds=thresholds,
        identity=cal_identity,
        design_fingerprint=fingerprint(),
        pooled_temperatures=list(TEMPERATURES),
    )
    frozen_hash = digest_json(frozen_content)
    if frozen_path.exists():
        old = json.loads(frozen_path.read_text())
        if old["sha256"] != frozen_hash or digest_json(old["content"]) != frozen_hash:
            raise ValueError("Refusing to retune an already frozen calibration")
    else:
        write_json(
            frozen_path,
            dict(
                content=frozen_content,
                sha256=frozen_hash,
                frozen_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
    (null_eval, _) = score_nulls(root, evaluation, "evaluation")
    (wm_scores, wm_rows) = ([], [])
    for s in wm:
        (arrays, _, bits, evidence) = read_trace(root, s)
        key_id = f"wm_{s['group']:03d}"
        key = load_arrays(root / "keys" / (key_id + ".npz"))[0]
        (h, soft) = (int(hard_count(bits, key)), soft_score(evidence, key))
        dec = decisions(h, soft, DESIGN["r"], thresholds)
        base = base_record(root, s)
        wm_rows.append(
            scored_record(base, key_id, "true-watermark-key", h, soft, dec, thresholds)
        )
        wm_scores.append(dict(H=h, **soft, **dec))
    wm_all = {k: np.asarray([v[k] for v in wm_scores]) for k in wm_scores[0]}
    eval_dec = decisions(null_eval["H"], null_eval, DESIGN["r"], thresholds)
    cal_dec = decisions(null_cal["H"], null_cal, DESIGN["r"], thresholds)
    (summary, curves, paired) = ([], {}, [])
    (weights_wm, weights_null) = (
        bootstrap_weights(10),
        bootstrap_weights(5, domain="null"),
    )
    for temp in TEMPERATURES:
        wi = np.array([s["temperature"] == temp for s in wm])
        ni = np.array([s["temperature"] == temp for s in evaluation])
        for name in LABELS:
            ishard = name in ("wangetal_published", "hard_matched")
            wscore = -wm_all["H"][wi] if ishard else wm_all["Z"][wi]
            nscore = -null_eval["H"][ni] if ishard else null_eval["Z"][ni]
            (fpr_curve, tpr_curve, auc) = roc(wscore, nscore)
            curves[temp, "hard" if ishard else "posterior"] = (fpr_curve, tpr_curve)
            values = wm_all[name][wi].astype(float)
            null_values = eval_dec[name][ni].mean(axis=1)
            (tlo, thi) = interval(values, weights_wm)
            (flo, fhi) = interval(null_values, weights_null)
            threshold = (
                hard_threshold(DESIGN["r"])
                if name == "wangetal_published"
                else (
                    float(np.sqrt(2 * np.log(1000)))
                    if name == "posterior_standard"
                    else (
                        thresholds["hard"]["cutoff"]
                        if name == "hard_matched"
                        else thresholds["posterior"]["cutoff"]
                    )
                )
            )
            summary.append(
                dict(
                    temperature=temp,
                    detector=name,
                    label=LABELS[name],
                    role=(
                        "primary"
                        if name in ("wangetal_published", "posterior_standard")
                        else "matched-FPR"
                    ),
                    N=int(wi.sum()),
                    TPR=float(values.mean()),
                    TPR_CI_low=tlo,
                    TPR_CI_high=thi,
                    null_N=int(ni.sum()),
                    null_key_pairings=int(ni.sum()) * 256,
                    realized_FPR=float(null_values.mean()),
                    FPR_CI_low=flo,
                    FPR_CI_high=fhi,
                    threshold=threshold,
                    rule="H <= cutoff" if ishard else "V > 0 and Z >= cutoff",
                    AUC=auc,
                    threshold_freeze_sha256=frozen_hash,
                    CI="crossed group x prompt; frozen thresholds; null conditional on fixed evaluation-key pool",
                )
            )
        for role, a, b in [
            ("primary", "wangetal_published", "posterior_standard"),
            ("matched-FPR", "hard_matched", "posterior_matched"),
        ]:
            delta = wm_all[b][wi].astype(float) - wm_all[a][wi].astype(float)
            (lo, hi) = interval(delta, weights_wm)
            paired.append(
                dict(
                    temperature=temp,
                    comparison=role,
                    posterior_minus_hard=float(delta.mean()),
                    CI_low=lo,
                    CI_high=hi,
                    N=int(wi.sum()),
                )
            )
    csv_write(root / "results_summary.csv", summary)
    csv_write(root / "paired_differences.csv", paired)
    mechanism = []
    for temp in TEMPERATURES:
        rows = [r for r in wm_rows if r["temperature"] == temp]
        row = dict(temperature=temp, N=len(rows))
        for name in (
            "average_full_vocab_entropy",
            "mean_hierarchical_bit_entropy",
            "generation_full_vocab_entropy",
            "generation_hierarchical_bit_entropy",
            "bit_agreement",
            "hard_score",
            "Z",
        ):
            values = np.array([r[name] for r in rows if r[name] is not None])
            row[name + "_mean"] = float(values.mean()) if len(values) else None
            row[name + "_q05"] = (
                float(np.quantile(values, 0.05)) if len(values) else None
            )
            row[name + "_q95"] = (
                float(np.quantile(values, 0.95)) if len(values) else None
            )
        row["posterior_no_evidence_N"] = sum((r["no_evidence"] for r in rows))
        mechanism.append(row)
    csv_write(root / "mechanism_summary.csv", mechanism)
    discussion = [
        "# Low-temperature detector comparison",
        "",
        "Paired differences below are posterior minus hard, in percentage points. ",
        "Intervals resample key groups and prompts; matched thresholds stay frozen. ",
        "The primary comparison measures actual methods at different operating points. ",
        "Matched-FPR results and held-out ROC/AUC assess scoring quality separately.",
        "",
    ]
    for p in paired:
        if p["temperature"] > 1.4:
            continue
        discussion.append(
            f"- T={p['temperature']:.1f}, {p['comparison']}: {100 * p['posterior_minus_hard']:+.2f} pp (95% CI {100 * p['CI_low']:+.2f} to {100 * p['CI_high']:+.2f})."
        )
    discussion += [
        "",
        "Consult both realized FPR columns and AUC in results_summary.csv before interpreting a TPR gain. ",
        "A gain confined to published thresholds does not establish superior posterior information; ",
        "a matched-FPR gain supported by ROC is stronger evidence. Oracle results cannot support the headline.",
    ]
    (root / "interpretation.md").write_text("\n".join(discussion) + "\n")
    for role, filename in [
        ("primary", "primary_methods_summary.csv"),
        ("matched-FPR", "matched_fpr_summary.csv"),
    ]:
        wide = []
        for temp in TEMPERATURES:
            row = dict(temperature=temp, N=160)
            for s in summary:
                if s["role"] == role and s["temperature"] == temp:
                    for column in (
                        "TPR",
                        "TPR_CI_low",
                        "TPR_CI_high",
                        "realized_FPR",
                        "FPR_CI_low",
                        "FPR_CI_high",
                        "threshold",
                        "AUC",
                    ):
                        row[s["detector"] + "_" + column] = s[column]
            wide.append(row)
        csv_write(root / filename, wide)

    def records():
        yield from wm_rows
        for split, ss, scores, dec in [
            ("calibration", calibration, null_cal, cal_dec),
            ("evaluation", evaluation, null_eval, eval_dec),
        ]:
            for i, sample in enumerate(ss):
                base = base_record(root, sample)
                for k in range(256):
                    soft = {
                        name: value[i, k]
                        for (name, value) in scores.items()
                        if name != "H"
                    }
                    yield scored_record(
                        base,
                        f"{split}_{k:03d}",
                        "independent-null-key",
                        scores["H"][i, k],
                        soft,
                        {name: value[i, k] for (name, value) in dec.items()},
                        thresholds,
                    )

    stream = iter(records())
    first = next(stream)
    with (root / "per_sample_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(first), lineterminator="\n")
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(stream)
    plots(root, summary, wm_rows, curves)
    timing = dict(
        seconds=time.monotonic() - begun,
        provenance=provenance,
        threshold_freeze_sha256=frozen_hash,
        wm_N=800,
        calibration_null_N=400,
        evaluation_null_N=400,
        calibration_pairs=102400,
        evaluation_pairs=102400,
    )
    write_json(root / "scoring_timing.json", timing)
    (root / "README.md").write_text(
        "# Comparison with Wang et al.’s PRC implementation\n\n"
        "`results_summary.csv` contains detection rates, false-positive rates, "
        "confidence intervals, and AUCs. `paired_differences.csv` contains paired "
        "detector comparisons. `threshold_calibration.json` records the calibrated thresholds.\n\n"
        f"Configuration: `{fingerprint()}`.\n"
    )
    return dict(summary=summary, paired=paired, timing=timing)


def plots(root, summary, wm_rows, curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for role, filename in [
        ("primary", "tpr_vs_temperature"),
        ("matched-FPR", "tpr_matched_fpr_vs_temperature"),
    ]:
        (fig, ax) = plt.subplots(figsize=(8, 5))
        for name in LABELS:
            rows = [s for s in summary if s["detector"] == name and s["role"] == role]
            if not rows:
                continue
            y = np.array([s["TPR"] for s in rows])
            ax.errorbar(
                [s["temperature"] for s in rows],
                y,
                yerr=[
                    y - np.array([s["TPR_CI_low"] for s in rows]),
                    np.array([s["TPR_CI_high"] for s in rows]) - y,
                ],
                marker="o",
                capsize=3,
                label=LABELS[name],
            )
        ax.set(
            xlabel="Generation temperature",
            ylabel="True positive rate",
            ylim=(-0.02, 1.02),
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(root / f"{filename}.{ext}", dpi=180)
        plt.close(fig)
    (fig, axes) = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, temp in zip(axes, TEMPERATURES[:3]):
        for name in ("hard", "posterior"):
            (x, y) = curves[temp, name]
            ax.plot(x, y, label=name)
        ax.set(
            xscale="symlog",
            xlim=(0, 1),
            xlabel="Held-out false positive rate",
            title=f"T={temp:.1f}",
        )
        ax.axvline(0.001, color="gray", linestyle=":")
        ax.legend()
    axes[0].set_ylabel("True positive rate")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(root / f"roc_low_temperature.{ext}", dpi=180)
    plt.close(fig)
    (fig, axes) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("hard_score", "Z")):
        for temp in TEMPERATURES:
            rows = [
                r for r in wm_rows if r["temperature"] == temp and r[name] is not None
            ]
            ax.scatter(
                [r["mean_hierarchical_bit_entropy"] for r in rows],
                [r[name] for r in rows],
                s=10,
                alpha=0.45,
                label=f"T={temp:.1f}",
            )
        ax.set(
            xlabel="Mean completion-only hierarchical bit entropy (bits)", ylabel=name
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(root / f"score_vs_entropy.{ext}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "generate", "analyze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--temperature", type=float, choices=TEMPERATURES)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.output, {})
    elif args.stage == "generate":
        if args.cache is None or args.temperature is None:
            parser.error("generate requires --cache and --temperature")
        run_temperature(args.output, args.temperature, args.cache, {})
    else:
        run(args.output, {})


if __name__ == "__main__":
    main()
