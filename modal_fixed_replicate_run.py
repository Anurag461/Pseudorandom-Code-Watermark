"""Independent fixed-PRC replicates for comparison with online causal runs.

The baseline fixed runner uses one frozen key/cache per (n,t,eta) setting.
This runner adds an experiment seed to an isolated namespace so repeated fixed
keys and watermarked generations can be compared without overwriting baseline
artifacts.  Compatible unwatermarked Qwen3-0.6B forced-length records are
reused from the shared null cache.
"""
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import modal


SCHEME = "fixed_prc_replicate_v1"
SEED = 12345
MODEL_SIZE = "0.6B"
MODEL_DISPLAY = "Qwen3-0.6B-Base"
VOCAB = 151_936
GPU = "A10G"
DEFAULT_BATCH = 64
DEFAULT_MAX_CONTAINERS = 5
CANONICAL_NUM_PROMPTS = 500
RESULT_SCHEMA_VERSION = 1
CSV_COLUMNS = (
    "timestamp_utc", "scheme", "eta", "T", "n", "r value", "r setting",
    "t", "Target FPR", "Generation Model", "num prompts", "batch",
    "experiment seed", "Map TPR", "Map FPR", "Entropy Aware TPR",
    "Entropy FPR", "Naive TPR", "Naive FPR", "null cache T",
    "artifact fingerprint",
)


def _slug(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def config_tag(n, t, eta, experiment_seed):
    return (
        f"{SCHEME}/qwen3_0p6b_base/"
        f"n{int(n)}_T{int(n)}_t{int(t)}_eta{float(eta):.2f}_rr99of100_"
        f"seed{int(experiment_seed)}"
    )


def artifact_path(tag):
    return f"/data/{tag}/artifacts.pt"


def wm_dir(tag):
    return f"/data/{tag}/wm"


def shared_null_dir(length):
    return f"/data/_nulls/T{int(length)}"


def _chunks(values, size):
    return [values[start:start + size] for start in range(0, len(values), size)]


def _format_rate(successes, total):
    return f"{int(successes)}/{int(total)} ({successes / max(total, 1):.1%})"


def _append_csv(path, row):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def _local_code_fingerprint():
    digest = hashlib.sha256()
    for path in (
        "modal_fixed_replicate_run.py", "prc.py", "watermark_expt.py",
        "detectors.py", "qwen.py", "constants.py",
    ):
        digest.update(path.encode())
        with open(path, "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()


def _find_compatible_null_T(prompt_indices, requested_T):
    if not os.path.isdir("/data/_nulls"):
        return None
    candidates = []
    for name in os.listdir("/data/_nulls"):
        match = re.fullmatch(r"T(\d+)", name)
        if match and int(match.group(1)) >= int(requested_T):
            candidates.append(int(match.group(1)))
    for length in sorted(candidates):
        directory = shared_null_dir(length)
        if all(
            os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
            for index in prompt_indices
        ):
            return length
    return None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "tokenizers", "safetensors",
        "huggingface_hub", "scipy", "galois", "numpy",
    )
    .env({
        "HF_HOME": "/cache/hf",
        "HF_HUB_CACHE": "/cache/hf",
        "PRC_MODEL_CACHE_DIR": "/cache/models",
        "PRC_MODEL_SIZE": MODEL_SIZE,
        "PRC_MODEL_VARIANT": "base",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "qwen", "constants", "detectors", "watermark_expt"
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)
app = modal.App("prc-fixed-replicate", image=image)


@app.function(volumes={"/data": data_vol}, timeout=600)
def build_artifacts(n, t, eta, experiment_seed):
    import numpy as np
    import torch
    from detectors import semantic_sha256
    from prc import KeyGen, parity_check_rank_info

    n = int(n)
    t = int(t)
    experiment_seed = int(experiment_seed)
    r = int(round(0.99 * n))
    if n <= 0 or t < 2 or experiment_seed < 0:
        raise ValueError("invalid fixed replicate configuration")
    if n - r < t - 1:
        raise ValueError(f"n-r={n-r} is too small for t={t}")
    tag = config_tag(n, t, eta, experiment_seed)
    path = artifact_path(tag)
    config = {
        "scheme": SCHEME,
        "n": n,
        "T": n,
        "t": t,
        "eta": float(eta),
        "r": r,
        "experiment_seed": experiment_seed,
        "partition_seed": SEED,
        "generation_model": MODEL_DISPLAY,
        "stopping_policy": "forced_length_v1",
    }
    data_vol.reload()
    if os.path.exists(path):
        previous = torch.load(path, weights_only=False, map_location="cpu")
        if previous.get("config_sig") == config:
            return {
                "tag": tag,
                "artifact_fingerprint": previous["artifact_fingerprint"],
                "reused": True,
            }
        raise RuntimeError(f"incompatible artifact already exists at {path}")

    np.random.seed(experiment_seed)
    torch.manual_seed(experiment_seed)
    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=t,
        noise_rate=float(eta),
        r=r,
        seed=experiment_seed,
    )
    rank_info = parity_check_rank_info(decoding_key[1])
    if not rank_info["full_rank"] or rank_info["rank"] != r:
        raise RuntimeError(f"fixed parity matrix is not full rank: {rank_info}")

    permutation = torch.randperm(
        VOCAB, generator=torch.Generator().manual_seed(SEED)
    )
    bucket_zero = torch.zeros(VOCAB, dtype=torch.bfloat16)
    bucket_zero[permutation[:VOCAB // 2]] = 1.0
    partition = torch.stack([bucket_zero, 1 - bucket_zero], dim=0)

    rows = []
    with open("/root/prompts.jsonl") as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= CANONICAL_NUM_PROMPTS:
                break
    if len(rows) < CANONICAL_NUM_PROMPTS:
        raise RuntimeError("prompts.jsonl does not contain 500 prompts")

    artifact = {
        "encoding_key": encoding_key,
        "decoding_key": decoding_key,
        "partition": partition,
        "prompt_ids_list": [row["prompt_tokens"] for row in rows],
        "n": n,
        "T": n,
        "r": r,
        "rank_info": rank_info,
        "experiment_seed": experiment_seed,
        "config_sig": config,
    }
    artifact["artifact_fingerprint"] = semantic_sha256(artifact)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(artifact, path)
    data_vol.commit()
    print(
        f"[build] {tag}: T=n={n}, t={t}, r={r}, "
        f"rank={rank_info['rank']}", flush=True,
    )
    return {
        "tag": tag,
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "reused": False,
    }


@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(tag, prompt_indices, T):
    data_vol.reload()
    missing = [
        index for index in prompt_indices
        if not os.path.exists(os.path.join(wm_dir(tag), f"wm_{index:04d}.pt"))
    ]
    null_T = _find_compatible_null_T(prompt_indices, T)
    if null_T is None:
        null_T = int(T)
        null_missing = [
            index for index in prompt_indices
            if not os.path.exists(
                os.path.join(shared_null_dir(T), f"null_{index:04d}.pt")
            )
        ]
    else:
        null_missing = []
    return {"wm_missing": missing, "null_missing": null_missing, "null_T": null_T}


@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class FixedModel:
    tag: str = modal.parameter()
    code_fingerprint_sha256: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch
        from detectors import semantic_sha256, tensor_sha256

        data_vol.reload()
        artifact = torch.load(
            artifact_path(self.tag), weights_only=False, map_location="cpu"
        )
        self.encoding_key = artifact["encoding_key"]
        self.partition_cpu = artifact["partition"]
        self.partition_fingerprint = tensor_sha256(self.partition_cpu)
        self.key_fingerprint = semantic_sha256(self.encoding_key)
        self.artifact_fingerprint = artifact["artifact_fingerprint"]
        self.experiment_seed = artifact["experiment_seed"]
        self.prompts = artifact["prompt_ids_list"]
        self.n = int(artifact["n"])
        self.T = int(artifact["T"])

        import watermark_expt as we
        self.we = we
        we.partition = self.partition_cpu.to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch
        return torch.tensor(
            [self.prompts[index] for index in indices],
            dtype=torch.long,
            device=self.we.device,
        )

    @modal.method()
    def ready(self):
        return {"model": MODEL_DISPLAY, "T": self.T, "n": self.n}

    @modal.method()
    def generate_wm(self, prompt_indices):
        import time
        import torch

        data_vol.reload()
        directory = wm_dir(self.tag)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"wm_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}
        started = time.time()
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect(
            self.we.model,
            prompt_batch,
            self.T,
            self.encoding_key,
            self.partition,
            watermark=True,
            return_trace_details=True,
        )
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row], tokens[row], p_traces[row],
                self.partition_cpu, self.n, True,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=details["prc_codeword_bits"][row],
                base_lm_entropy=details["base_lm_entropy"][row],
                base_token_logprob=details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": int(index),
                "scheme": SCHEME,
                "generation_model_size": MODEL_SIZE,
                "generation_model": MODEL_DISPLAY,
                "artifact_seed": self.experiment_seed,
                "artifact_fingerprint": self.artifact_fingerprint,
                "code_fingerprint_sha256": self.code_fingerprint_sha256,
            })
            torch.save(record, os.path.join(directory, f"wm_{index:04d}.pt"))
        data_vol.commit()
        return {
            "generated": len(todo), "cached": len(prompt_indices) - len(todo),
            "batch": len(todo), "seconds": time.time() - started,
        }

    @modal.method()
    def generate_null(self, prompt_indices):
        import torch

        data_vol.reload()
        directory = shared_null_dir(self.T)
        os.makedirs(directory, exist_ok=True)
        todo = [
            index for index in prompt_indices
            if not os.path.exists(os.path.join(directory, f"null_{index:04d}.pt"))
        ]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices), "batch": 0}
        prompt_batch = self._prompt_batch(todo)
        tokens, p_traces, details = self.we.generate_batch_and_collect(
            self.we.model, prompt_batch, self.T, self.encoding_key,
            self.partition, watermark=False, return_trace_details=True,
        )
        for row, index in enumerate(todo):
            record = self.we.build_prc_generation_record(
                prompt_batch[row], tokens[row], p_traces[row],
                self.partition_cpu, self.n, False,
                encoding_key_fingerprint=self.key_fingerprint,
                prc_codeword_bits=None,
                base_lm_entropy=details["base_lm_entropy"][row],
                base_token_logprob=details["base_token_logprob"][row],
                partition_fingerprint=self.partition_fingerprint,
            )
            record.update({
                "prompt_idx": int(index),
                "generation_model_size": MODEL_SIZE,
                "generation_model": MODEL_DISPLAY,
            })
            torch.save(record, os.path.join(directory, f"null_{index:04d}.pt"))
        data_vol.commit()
        return {"generated": len(todo), "cached": 0, "batch": len(todo)}


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(tag, prompt_indices, null_T, fpr, batch, code_fingerprint_sha256):
    import torch
    from detectors import detect_hoeffding

    data_vol.reload()
    artifact = torch.load(
        artifact_path(tag), weights_only=False, map_location="cpu"
    )
    key = artifact["decoding_key"]
    partition = artifact["partition"]
    T = int(artifact["T"])
    results = []
    for watermark in (True, False):
        directory = wm_dir(tag) if watermark else shared_null_dir(null_T)
        prefix = "wm" if watermark else "null"
        for index in prompt_indices:
            path = os.path.join(directory, f"{prefix}_{index:04d}.pt")
            record = torch.load(path, weights_only=False, map_location="cpu")
            if len(record["tokens"]) < T or len(record["p_trace"]) < T:
                raise ValueError(f"record {path} is shorter than T={T}")
            if watermark and record.get("artifact_fingerprint") != artifact[
                "artifact_fingerprint"
            ]:
                raise ValueError(f"watermarked record {path} has stale artifact")
            scored = {}
            for weight in ("map", "entropy", "naive"):
                decision, info = detect_hoeffding(
                    key, record["tokens"][:T], record["p_trace"][:T],
                    partition, fpr=fpr, weight=weight, return_info=True,
                )
                scored[weight] = {"decision": bool(decision), **info}
            results.append({
                "prompt_idx": int(index), "watermark": watermark,
                "scores": scored,
            })
    wm = [result for result in results if result["watermark"]]
    null = [result for result in results if not result["watermark"]]
    counts = {}
    for weight in ("map", "entropy", "naive"):
        counts[weight] = {
            "tp": sum(result["scores"][weight]["decision"] for result in wm),
            "fp": sum(result["scores"][weight]["decision"] for result in null),
            "watermarked_total": len(wm), "null_total": len(null),
        }
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scheme": SCHEME, "tag": tag, "n": T, "T": T,
        "t": int(artifact["config_sig"]["t"]),
        "eta": float(artifact["config_sig"]["eta"]),
        "r": int(artifact["r"]), "target_fpr": float(fpr),
        "generation_model": MODEL_DISPLAY,
        "num_prompts": len(prompt_indices), "batch": int(batch),
        "null_cache_T": int(null_T),
        "experiment_seed": int(artifact["experiment_seed"]),
        "artifact_fingerprint": artifact["artifact_fingerprint"],
        "code_fingerprint_sha256": code_fingerprint_sha256,
        "counts": counts, "results": results,
    }
    output_dir = f"/data/{tag}/results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        f"fpr-{_slug(f'{float(fpr):.12g}')}_prompts-{len(prompt_indices)}.pt",
    )
    torch.save(payload, output_path)
    data_vol.commit()
    return {"payload": payload, "remote_output_path": output_path}


@app.local_entrypoint()
def main(num_prompts: int = CANONICAL_NUM_PROMPTS,
         n: int = 256, t: int = 3, eta: float = 0.05,
         fpr: float = 1e-3, batch: int = DEFAULT_BATCH,
         experiment_seed: int = 54321,
         max_containers: int = DEFAULT_MAX_CONTAINERS,
         gpu: str = GPU,
         csv_out: str = "fixed_replicate_results_summary.csv"):
    if not 0 < num_prompts <= CANONICAL_NUM_PROMPTS:
        raise ValueError("num_prompts must be in [1,500]")
    if batch <= 0 or max_containers <= 0 or experiment_seed < 0:
        raise ValueError("batch, max_containers, and experiment_seed are invalid")
    prompt_indices = list(range(int(num_prompts)))
    tag = config_tag(n, t, eta, experiment_seed)
    code_fingerprint = _local_code_fingerprint()
    print(
        f"[main] {SCHEME}: T=n={n}, t={t}, eta={eta}, fpr={fpr:g}, "
        f"prompts={num_prompts}, batch={batch}, seed={experiment_seed}, "
        f"GPU={gpu}, max_containers={max_containers}", flush=True,
    )
    build = build_artifacts.remote(n, t, eta, experiment_seed)
    print(
        f"[main] artifact {'reused' if build['reused'] else 'built'}: "
        f"{build['artifact_fingerprint']}", flush=True,
    )
    plan = plan_generation.remote(tag, prompt_indices, n)
    print(
        f"[main] generation plan: wm_missing={len(plan['wm_missing'])}, "
        f"null_missing={len(plan['null_missing'])}, null_T={plan['null_T']}",
        flush=True,
    )
    if plan["wm_missing"] or plan["null_missing"]:
        from concurrent.futures import ThreadPoolExecutor

        model = FixedModel.with_options(
            gpu=gpu, max_containers=max_containers
        )(tag=tag, code_fingerprint_sha256=code_fingerprint)
        print(f"[main] model ready: {model.ready.remote()}", flush=True)
        work = []
        if plan["wm_missing"]:
            work.append(("wm", model.generate_wm, _chunks(plan["wm_missing"], batch)))
        if plan["null_missing"]:
            work.append((
                "null", model.generate_null, _chunks(plan["null_missing"], batch)
            ))

        def run_map(item):
            name, method, chunks = item
            return name, list(method.map(chunks))

        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            mapped = list(pool.map(run_map, work))
        for name, records in mapped:
            print(
                f"[main] {name}: generated="
                f"{sum(record['generated'] for record in records)}, "
                f"batch_sizes={[record['batch'] for record in records if record['batch']]}",
                flush=True,
            )

    detected = detect_all.remote(
        tag, prompt_indices, plan["null_T"], fpr, batch, code_fingerprint
    )
    payload = detected["payload"]
    counts = payload["counts"]
    print("\n=== Fixed PRC replicate summary ===", flush=True)
    for weight in ("map", "entropy", "naive"):
        count = counts[weight]
        print(
            f"{weight:>7}: TPR {_format_rate(count['tp'], num_prompts)}  "
            f"FPR {_format_rate(count['fp'], num_prompts)}", flush=True,
        )

    os.makedirs("outputs", exist_ok=True)
    local_json = os.path.join(
        "outputs",
        f"fixed_replicate_n{n}_t{t}_eta{eta:.2f}_prompts{num_prompts}_"
        f"seed{experiment_seed}.json",
    )
    with open(local_json, "w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    row = {
        "timestamp_utc": payload["timestamp_utc"], "scheme": SCHEME,
        "eta": eta, "T": n, "n": n, "r value": payload["r"],
        "r setting": "0.99n", "t": t, "Target FPR": f"{fpr:.0e}",
        "Generation Model": MODEL_DISPLAY, "num prompts": num_prompts,
        "batch": batch, "experiment seed": experiment_seed,
        "Map TPR": _format_rate(counts["map"]["tp"], num_prompts),
        "Map FPR": _format_rate(counts["map"]["fp"], num_prompts),
        "Entropy Aware TPR": _format_rate(counts["entropy"]["tp"], num_prompts),
        "Entropy FPR": _format_rate(counts["entropy"]["fp"], num_prompts),
        "Naive TPR": _format_rate(counts["naive"]["tp"], num_prompts),
        "Naive FPR": _format_rate(counts["naive"]["fp"], num_prompts),
        "null cache T": payload["null_cache_T"],
        "artifact fingerprint": payload["artifact_fingerprint"],
    }
    _append_csv(csv_out, row)
    print(f"[main] remote result: {detected['remote_output_path']}", flush=True)
    print(f"[main] local result: {local_json}", flush=True)
    print(f"[main] local summary: {csv_out}", flush=True)
