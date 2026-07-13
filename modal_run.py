"""
Run the PRC watermark experiment on Modal with the Hoeffding detector.

Cost-optimized pipeline:
  - build_artifacts (CPU): freeze the PRC key, seeded partition map, and
    tokenized RealNews prompts. Written to a Volume per config.
  - Model.generate_wm / generate_null (GPU): batched generation, one batch per
    .map() call.
  - Optional EntropyModel pass (GPU): teacher-force cached generations with a
    different Qwen3 model to estimate P[partition 1] for detection.
  - detect_all (CPU): Hoeffding detection over cached generations/traces.

Null generations depend only on model + prompt + seed, not on the key
(n, t, eta, r), so they live in a shared store keyed by length T and can be
reused across configs with the same exact T.
Current eta=0.1 runs use T=n: one generated length-n code block per prompt.

Usage:
    modal run modal_run.py
    modal run modal_run.py --num-prompts 500 --max-containers 10
    modal run modal_run.py --num-prompts 500 --n 512 --t 3 --eta 0.1 \
      --r-frac 0.99 --fpr 1e-3 --entropy-model-size 4B
"""
import csv
import os
import shutil

import modal

# ---- experiment config ------------------------------------------------------
DEFAULT_N = 400
DEFAULT_T = 3
DEFAULT_ETA = 0.05
DEFAULT_FPR = 1e-3  # 0.1%
DEFAULT_BLOCKS = 1
SEED = 12345
MODEL_SIZE = "0.6B"
VOCAB = 151_936
GPU = "A10G"
DEFAULT_MAX_CONTAINERS = 5
DEFAULT_BATCH = 64
DEFAULT_ENTROPY_BATCH = 8

CSV_COLUMNS = [
    "Target FPR",
    "n",
    "t",
    "eta",
    "T",
    "Map TPR",
    "Entropy Aware TPR",
    "Naive TPR",
    "Log Hoeffding TPR",
    "Map FPR",
    "Entropy FPR",
    "Naive FPR",
    "Log Hoeffding FPR",
    "Entropy Model",
    "Entropy Trace Source",
    "Notes",
]


def normalize_model_size(model_size):
    value = MODEL_SIZE if model_size is None else str(model_size).strip()
    if not value:
        value = MODEL_SIZE
    upper = value.upper()
    if upper.endswith("B"):
        return upper[:-1] + "B"
    if upper.replace(".", "", 1).isdigit():
        return f"{upper}B"
    return value


def model_display(model_size):
    return f"Qwen3-{normalize_model_size(model_size)}-Base"


def entropy_model_tag(model_size):
    size = normalize_model_size(model_size).lower().replace(".", "p")
    return f"qwen3_{size}_base"


def uses_cached_generation_trace(entropy_model_size):
    return normalize_model_size(entropy_model_size) == normalize_model_size(MODEL_SIZE)


def entropy_trace_source(entropy_model_size):
    if uses_cached_generation_trace(entropy_model_size):
        return "cached_generation_p_trace"
    return f"estimated_{normalize_model_size(entropy_model_size)}"


def resolve_r(n, r=0, r_frac=0.0):
    explicit_r = int(r) if r else 0
    explicit_frac = float(r_frac) if r_frac else 0.0
    if explicit_r and explicit_frac:
        raise ValueError("Pass either --r or --r-frac, not both.")
    if explicit_r:
        return explicit_r
    if explicit_frac:
        return int(round(explicit_frac * n))
    return None


def validate_r_for_keygen(n, t, r):
    if r is None:
        return
    if r <= 0:
        raise ValueError(f"r must be positive, got {r}")
    if r > n:
        raise ValueError(f"r must be <= n, got r={r}, n={n}")
    if n - r < t - 1:
        raise ValueError(
            f"r={r} is too large for n={n}, t={t}; need n-r >= t-1"
        )


def experiment_T(n):
    """Generated-token length for new runs: one length-n PRC code block."""
    return DEFAULT_BLOCKS * int(n)


def config_tag(n, t, eta, r=None, T=None):
    """Per-config tag for key-dependent artifacts.

    FPR is excluded because it only affects detection. r is included only when
    explicitly requested so old default-r caches keep their original tags.
    T is included for new runs so T=n caches cannot collide with old T=2n caches.
    """
    base = f"n{n}_t{t}_eta{eta:.2f}"
    if T is not None:
        base = f"{base}_T{int(T)}"
    return f"{base}_r{int(r)}" if r is not None else base


def art_path(n, t, eta, r=None, T=None):
    return f"/data/{config_tag(n, t, eta, r, T)}/artifacts.pt"


def wm_dir(n, t, eta, r=None, T=None):
    return f"/data/{config_tag(n, t, eta, r, T)}/wm"


def null_dir(T):
    return f"/data/_nulls/T{T}"


def wm_entropy_dir(tag, entropy_model_size):
    return f"/data/{tag}/entropy/{entropy_model_tag(entropy_model_size)}/wm"


def null_entropy_dir(T, entropy_model_size):
    return f"/data/_null_entropy/{entropy_model_tag(entropy_model_size)}/T{T}"


def wm_trace_dir(tag, entropy_model_size):
    return f"/data/{tag}/detect_traces/{entropy_model_tag(entropy_model_size)}/wm"


def null_trace_dir(T, entropy_model_size):
    return f"/data/_null_detection_traces/{entropy_model_tag(entropy_model_size)}/T{T}"


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def _format_rate(count, total):
    denom = max(total, 1)
    return f"{count}/{total} ({count / denom:.1%})"


def _ensure_csv_schema(csv_out):
    parent = os.path.dirname(csv_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not os.path.exists(csv_out) or os.path.getsize(csv_out) == 0:
        with open(csv_out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()
        return

    with open(csv_out, newline="") as f:
        reader = csv.DictReader(f)
        old_columns = reader.fieldnames or []
        rows = list(reader)
    if old_columns == CSV_COLUMNS:
        return

    backup = f"{csv_out}.pre_schema_update.bak"
    if not os.path.exists(backup):
        shutil.copy2(csv_out, backup)

    migrated = []
    for row in rows:
        migrated.append({
            "Target FPR": row.get("Target FPR", ""),
            "n": row.get("n", ""),
            "t": row.get("t", ""),
            "eta": row.get("eta", ""),
            "T": row.get("T", ""),
            "Map TPR": row.get("Map TPR", ""),
            "Entropy Aware TPR": row.get("Entropy Aware TPR", ""),
            "Naive TPR": row.get("Naive TPR", ""),
            "Log Hoeffding TPR": row.get("Log Hoeffding TPR", "skipped"),
            "Map FPR": row.get("Map FPR", ""),
            "Entropy FPR": row.get("Entropy FPR", row.get("FPR", "")),
            "Naive FPR": row.get("Naive FPR", row.get("FPR", "")),
            "Log Hoeffding FPR": row.get("Log Hoeffding FPR", "skipped"),
            "Entropy Model": row.get("Entropy Model", model_display(MODEL_SIZE)),
            "Entropy Trace Source": row.get(
                "Entropy Trace Source", "cached_generation_p_trace"
            ),
            "Notes": row.get("Notes", ""),
        })
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(migrated)


def _append_summary_row(csv_out, row):
    _ensure_csv_schema(csv_out)
    with open(csv_out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


# ---- image ------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
        "scipy",
        "galois",
        "numpy",
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "HF_HUB_CACHE": "/cache/hf",
            "PRC_MODEL_SIZE": MODEL_SIZE,
            "PRC_MODEL_VARIANT": "base",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "qwen", "constants", "detectors", "watermark_expt"
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)

app = modal.App("prc-hoeffding", image=image)


# ---- artifact build (CPU only; no model load) -------------------------------
@app.function(volumes={"/data": data_vol}, timeout=600)
def build_artifacts(num_prompts: int, n: int, t: int, eta: float,
                    r: int = 0, fresh: bool = False) -> int:
    import glob
    import json
    import os

    import numpy as np
    import torch
    from prc import KeyGen, parity_check_rank_info

    requested_r = int(r) if r else None
    validate_r_for_keygen(n, t, requested_r)
    max_new_tokens = experiment_T(n)
    ap = art_path(n, t, eta, requested_r, max_new_tokens)
    wmd = wm_dir(n, t, eta, requested_r, max_new_tokens)
    os.makedirs(os.path.dirname(ap), exist_ok=True)

    config_sig = {
        "n": n,
        "t": t,
        "eta": eta,
        "T": max_new_tokens,
        "blocks": DEFAULT_BLOCKS,
        "num_prompts": num_prompts,
        "gen_scheme": "single_codeword_batched",
    }
    if requested_r is not None:
        config_sig["r"] = requested_r

    data_vol.reload()
    if not fresh and os.path.exists(ap):
        prev = torch.load(ap, weights_only=False, map_location="cpu")
        if prev.get("config_sig") == config_sig:
            print(f"[build] reusing frozen key from {ap} (config matches)",
                  flush=True)
            return num_prompts
        print("[build] config changed -> rebuilding key, INVALIDATING wm cache",
              flush=True)

    removed = 0
    for p in glob.glob(os.path.join(wmd, "*.pt")):
        os.remove(p)
        removed += 1
    if removed:
        print(f"[build] cleared {removed} stale watermarked generations",
              flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    encoding_key, decoding_key = KeyGen(
        n=n,
        message_length=0,
        false_positive_rate=0.5,
        t=t,
        noise_rate=eta,
        r=requested_r,
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t_key = decoding_key
    rank_info = parity_check_rank_info(parity_check_matrix)
    actual_r = parity_check_matrix.shape[0]
    print(f"[build] PRC params: n={n} t={t_key} g={g} r={actual_r} "
          f"noise_rate={noise_rate:.4f}", flush=True)
    print(f"[build] parity rank: {rank_info['rank']}/{rank_info['rows']} "
          f"full_rank={rank_info['full_rank']}", flush=True)

    perm = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(SEED))
    v0 = torch.zeros(VOCAB, dtype=torch.bfloat16)
    v0[perm[: VOCAB // 2]] = 1.0
    v1 = 1 - v0
    partition = torch.stack([v0, v1], dim=0)

    rows = []
    with open("/root/prompts.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= num_prompts:
                break
    if len(rows) < num_prompts:
        raise RuntimeError(f"prompts.jsonl has {len(rows)} rows, need {num_prompts}")
    prompt_ids_list = [row["prompt_tokens"] for row in rows]

    torch.save(
        {
            "encoding_key": encoding_key,
            "decoding_key": decoding_key,
            "partition": partition,
            "prompt_ids_list": prompt_ids_list,
            "num_prompts": num_prompts,
            "n": n,
            "T": max_new_tokens,
            "seed": SEED,
            "config_sig": config_sig,
            "parity_check_rank_info": rank_info,
        },
        ap,
    )
    data_vol.commit()
    print(f"[build] wrote artifacts ({num_prompts} prompts) -> {ap}", flush=True)
    return num_prompts


# ---- cache probes (CPU; decide whether GPU work is needed) ------------------
@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(n: int, t: int, eta: float, num_prompts: int,
                    r: int = 0) -> dict:
    import os

    requested_r = int(r) if r else None
    T = experiment_T(n)
    wmd = wm_dir(n, t, eta, requested_r, T)
    data_vol.reload()

    wm_missing = [i for i in range(num_prompts)
                  if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]

    d = null_dir(T)
    null_missing = [i for i in range(num_prompts)
                    if not os.path.exists(os.path.join(d, f"null_{i:04d}.pt"))]

    return {"wm_missing": wm_missing, "null_missing": null_missing,
            "null_T": T, "T": T}


@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_entropy(tag: str, entropy_model_size: str, T: int, null_T: int,
                 num_prompts: int) -> dict:
    import os

    if uses_cached_generation_trace(entropy_model_size):
        return {"wm_missing": [], "null_missing": []}

    data_vol.reload()
    wdir = wm_entropy_dir(tag, entropy_model_size)
    ndir = null_entropy_dir(T, entropy_model_size)
    wm_missing = [i for i in range(num_prompts)
                  if not os.path.exists(os.path.join(wdir, f"wm_{i:04d}.pt"))]
    null_missing = [i for i in range(num_prompts)
                    if not os.path.exists(os.path.join(ndir, f"null_{i:04d}.pt"))]
    return {"wm_missing": wm_missing, "null_missing": null_missing,
            "T": T, "null_T": null_T}


# ---- batched generation (GPU) -----------------------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class Model:
    tag: str = modal.parameter()

    @modal.enter()
    def load(self):
        import torch

        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")

        import watermark_expt as we
        self.we = we
        self.encoding_key = art["encoding_key"]
        self.prompts = art["prompt_ids_list"]
        self.n = art["n"]
        self.T = int(art.get("T", art.get("config_sig", {}).get("T", experiment_T(art["n"]))))
        we.partition = art["partition"].to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch

        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

    @modal.method()
    def generate_wm(self, prompt_indices: list) -> dict:
        import os
        import time

        import torch

        wmd = f"/data/{self.tag}/wm"
        data_vol.reload()
        os.makedirs(wmd, exist_ok=True)
        todo = [i for i in prompt_indices
                if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices)}

        t0 = time.time()
        batch = self._prompt_batch(todo)
        tokens, p_traces = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=True,
        )
        for row, i in enumerate(todo):
            torch.save(
                {"prompt_idx": i, "watermark": True,
                 "tokens": tokens[row].cpu(), "p_trace": p_traces[row]},
                os.path.join(wmd, f"wm_{i:04d}.pt"),
            )
        data_vol.commit()
        return {"generated": len(todo), "cached": len(prompt_indices) - len(todo),
                "dt": time.time() - t0}

    @modal.method()
    def generate_null(self, prompt_indices: list) -> dict:
        import os
        import time

        import torch

        nd = null_dir(self.T)
        data_vol.reload()
        os.makedirs(nd, exist_ok=True)
        todo = [i for i in prompt_indices
                if not os.path.exists(os.path.join(nd, f"null_{i:04d}.pt"))]
        if not todo:
            return {"generated": 0, "cached": len(prompt_indices)}

        t0 = time.time()
        batch = self._prompt_batch(todo)
        tokens, p_traces = self.we.generate_batch_and_collect(
            self.we.model, batch, self.T, self.encoding_key,
            self.partition, watermark=False,
        )
        for row, i in enumerate(todo):
            torch.save(
                {"prompt_idx": i, "watermark": False,
                 "tokens": tokens[row].cpu(), "p_trace": p_traces[row]},
                os.path.join(nd, f"null_{i:04d}.pt"),
            )
        data_vol.commit()
        return {"generated": len(todo), "cached": len(prompt_indices) - len(todo),
                "dt": time.time() - t0}


# ---- optional alternate entropy model (GPU) ---------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=7200,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class EntropyModel:
    tag: str = modal.parameter()
    model_size: str = modal.parameter()
    T: int = modal.parameter()
    null_T: int = modal.parameter()

    @modal.enter()
    def load(self):
        import os

        import torch

        self.model_size = normalize_model_size(self.model_size)
        os.environ["PRC_MODEL_SIZE"] = self.model_size
        os.environ["PRC_MODEL_VARIANT"] = "base"

        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")

        import watermark_expt as we
        self.we = we
        self.prompts = art["prompt_ids_list"]
        self.partition_cpu = art["partition"]
        self.partition = art["partition"].to(we.device)
        self.n = art["n"]
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch

        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

    def _save_trace_payload(self, path, idx, source, tokens, p_trace):
        import numpy as np
        import torch
        from detectors import binary_entropy, fold_soft_token, tokens_to_bits

        bits = tokens_to_bits(tokens, self.partition_cpu)
        entropy = binary_entropy(p_trace) / np.log(2)
        signs = (1 - 2 * bits.astype(np.int64)).astype(np.float64)
        signed_entropy = signs * entropy
        folded = fold_soft_token(bits, p_trace, self.n)
        torch.save(
            {
                "prompt_idx": idx,
                "source": source,
                "entropy_model": model_display(self.model_size),
                "entropy_trace_source": entropy_trace_source(self.model_size),
                "tokens_len": int(tokens.numel()),
                "p_trace": p_trace,
                "entropy_trace": entropy.astype(np.float32),
                "signed_entropy_trace": signed_entropy.astype(np.float32),
                "folded_signed_entropy": folded.astype(np.float32),
            },
            path,
        )

    def _estimate(self, prompt_indices, source):
        import os
        import time

        import torch

        if source == "wm":
            src_dir = f"/data/{self.tag}/wm"
            out_dir = wm_entropy_dir(self.tag, self.model_size)
            prefix = "wm"
        elif source == "null":
            src_dir = null_dir(self.null_T)
            out_dir = null_entropy_dir(self.T, self.model_size)
            prefix = "null"
        else:
            raise ValueError(f"unknown entropy source {source}")

        data_vol.reload()
        os.makedirs(out_dir, exist_ok=True)
        todo = [
            i for i in prompt_indices
            if not os.path.exists(os.path.join(out_dir, f"{prefix}_{i:04d}.pt"))
        ]
        if not todo:
            return {"estimated": 0, "cached": len(prompt_indices), "source": source}

        t0 = time.time()
        records = [
            torch.load(os.path.join(src_dir, f"{prefix}_{i:04d}.pt"),
                       weights_only=False, map_location="cpu")
            for i in todo
        ]
        token_batch = torch.stack([rec["tokens"][:self.T].long() for rec in records])
        prompt_batch = self._prompt_batch(todo)
        p_traces = self.we.estimate_partition_trace_batch(
            self.we.model, prompt_batch, token_batch, self.partition
        )
        for row, i in enumerate(todo):
            tokens = token_batch[row].cpu()
            self._save_trace_payload(
                os.path.join(out_dir, f"{prefix}_{i:04d}.pt"),
                i,
                source,
                tokens,
                p_traces[row],
            )

        data_vol.commit()
        return {"estimated": len(todo), "cached": len(prompt_indices) - len(todo),
                "source": source, "dt": time.time() - t0}

    @modal.method()
    def estimate_wm(self, prompt_indices: list) -> dict:
        return self._estimate(prompt_indices, "wm")

    @modal.method()
    def estimate_null(self, prompt_indices: list) -> dict:
        return self._estimate(prompt_indices, "null")


# ---- detection over cached generations (CPU) --------------------------------
@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(n: int, t: int, eta: float, fpr: float, null_T: int,
               num_prompts: int, r: int = 0,
               entropy_model_size: str = MODEL_SIZE) -> list:
    import os

    import numpy as np
    import torch
    from detectors import (
        binary_entropy,
        detect_hoeffding,
        fold_soft_token,
        tokens_to_bits,
    )
    from prc import parity_check_rank_info

    requested_r = int(r) if r else None
    T = experiment_T(n)
    tag = config_tag(n, t, eta, requested_r, T)
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(model_size)
    source_label = entropy_trace_source(model_size)
    ap = art_path(n, t, eta, requested_r, T)
    wmd = wm_dir(n, t, eta, requested_r, T)
    nd = null_dir(null_T)
    data_vol.reload()
    art = torch.load(ap, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    rank_info = art.get("parity_check_rank_info")
    if rank_info is None:
        rank_info = parity_check_rank_info(decoding_key[1])

    trace_saved = 0

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(record["p_trace"][:T], dtype=np.float64)
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(null_entropy_dir(T, model_size), f"null_{idx:04d}.pt")
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(est["p_trace"][:T], dtype=np.float64)

    def _save_detection_trace(source, idx, tokens, p_trace):
        nonlocal trace_saved
        if source == "wm":
            out_dir = wm_trace_dir(tag, model_size)
            prefix = "wm"
        else:
            out_dir = null_trace_dir(T, model_size)
            prefix = "null"
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{prefix}_{idx:04d}.pt")
        if os.path.exists(path):
            return
        bits = tokens_to_bits(tokens, partition)
        entropy = binary_entropy(p_trace) / np.log(2)
        signs = (1 - 2 * bits.astype(np.int64)).astype(np.float64)
        signed_entropy = signs * entropy
        folded = fold_soft_token(bits, p_trace, n)
        torch.save(
            {
                "prompt_idx": idx,
                "source": source,
                "entropy_model": model_display(model_size),
                "entropy_trace_source": source_label,
                "tokens_len": int(tokens.numel()),
                "entropy_trace": entropy.astype(np.float32),
                "signed_entropy_trace": signed_entropy.astype(np.float32),
                "folded_signed_entropy": folded.astype(np.float32),
            },
            path,
        )
        trace_saved += 1

    def _run(tokens, p_trace, wm_flag, idx):
        dm, im = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="map", return_info=True)
        de, ie = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="entropy", return_info=True)
        out = {
            "prompt_idx": idx,
            "watermark": wm_flag,
            "decision_map": bool(dm),
            "stat_map": float(im["statistic"]),
            "thr_map": float(im["threshold"]),
            "decision_entropy": bool(de),
            "stat_entropy": float(ie["statistic"]),
            "thr_entropy": float(ie["threshold"]),
            "decision_naive": None,
            "stat_naive": None,
            "thr_naive": None,
            "decision_log": None,
            "stat_log": None,
            "thr_log": None,
            "n_tokens": int(tokens.numel()),
            "entropy_model": model_display(model_size),
            "entropy_trace_source": source_label,
            "parity_check_rank_info": rank_info,
        }
        if use_generation_trace:
            dn, ino = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                       fpr=fpr, weight="naive", return_info=True)
            out.update({
                "decision_naive": bool(dn),
                "stat_naive": float(ino["statistic"]),
                "thr_naive": float(ino["threshold"]),
            })
        return out

    out = []
    for i in range(num_prompts):
        gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        wm_tokens = gw["tokens"][:T]
        wm_p = _p_trace("wm", i, gw)
        _save_detection_trace("wm", i, wm_tokens, wm_p)
        out.append(_run(wm_tokens, wm_p, True, i))

        gn = torch.load(os.path.join(nd, f"null_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        null_tokens = gn["tokens"][:T]
        null_p = _p_trace("null", i, gn)
        _save_detection_trace("null", i, null_tokens, null_p)
        out.append(_run(null_tokens, null_p, False, i))

    if trace_saved:
        data_vol.commit()
    return out


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all_any(n: int, t: int, eta: float, fpr: float,
                   num_prompts: int = 500, r: int = 0,
                   entropy_model_size: str = MODEL_SIZE) -> list:
    """Re-detect a config with ALL weight kinds, auto-detecting the cache layout:
      - NEW layout (config_tag/wm/wm_XXXX.pt + shared _nulls/T*) -> batched pipeline
      - OLD layout (config_tag/gens/gen_XXXX.pt, watermark flag inside)
    Free: model-free CPU detection over already-cached generations, no regen."""
    import glob
    import os
    import numpy as np
    import torch
    from detectors import detect_hoeffding, WEIGHT_KINDS

    T = experiment_T(n)
    requested_r = int(r) if r else None
    tag = config_tag(n, t, eta, requested_r, T)
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(model_size)
    ap = art_path(n, t, eta, requested_r, T)
    data_vol.reload()
    art = torch.load(ap, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(record["p_trace"][:T], dtype=np.float64)
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(null_entropy_dir(T, model_size), f"null_{idx:04d}.pt")
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(est["p_trace"][:T], dtype=np.float64)

    def decisions(tokens, p_trace, wm_flag, idx):
        row = {"prompt_idx": idx, "watermark": bool(wm_flag)}
        weight_kinds = WEIGHT_KINDS if use_generation_trace else tuple(
            wname for wname in WEIGHT_KINDS if wname != "naive"
        )
        for wname in weight_kinds:
            row[f"decision_{wname}"] = bool(detect_hoeffding(
                decoding_key, tokens, p_trace, partition, fpr=fpr, weight=wname))
        return row

    wmd = f"/data/{tag}/wm"
    new_layout = os.path.isdir(wmd) and glob.glob(os.path.join(wmd, "wm_*.pt"))

    out = []
    if new_layout:
        # Find the exact-length shared null store for this T.
        root = "/data/_nulls"
        use_T = T
        null_store = os.path.join(root, f"T{T}")
        have_nulls = all(os.path.exists(
            os.path.join(null_store, f"null_{i:04d}.pt"))
            for i in range(num_prompts))
        if not have_nulls:
            raise FileNotFoundError(f"No exact shared null store found for T={T}")
        for i in range(num_prompts):
            gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            wm_tokens = gw["tokens"][:T]
            out.append(decisions(wm_tokens, _p_trace("wm", i, gw), True, i))
            gn = torch.load(os.path.join(root, f"T{use_T}", f"null_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            null_tokens = gn["tokens"][:T]
            out.append(decisions(null_tokens, _p_trace("null", i, gn), False, i))
    else:
        for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
            g = torch.load(path, weights_only=False, map_location="cpu")
            idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
            tokens = g["tokens"][:T]
            out.append(decisions(tokens, _p_trace("wm" if g["watermark"] else "null", idx, g),
                                 bool(g["watermark"]), idx))
    return out


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_map_summary(n: int, t: int, eta: float, fpr: float,
                       num_prompts: int = 500, r: int = 0,
                       entropy_model_size: str = MODEL_SIZE) -> dict:
    """CPU-only cache redetect for the CSV columns we need: map, entropy, naive."""
    import glob
    import os
    import numpy as np
    import torch
    from detectors import detect_hoeffding

    T = experiment_T(n)
    requested_r = int(r) if r else None
    tag = config_tag(n, t, eta, requested_r, T)
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(model_size)
    data_vol.reload()
    art = torch.load(art_path(n, t, eta, requested_r, T), weights_only=False,
                     map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]

    def _p_trace(source, idx, record):
        if use_generation_trace:
            return np.asarray(record["p_trace"][:T], dtype=np.float64)
        legacy_path = os.path.join(
            f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
            f"gen_{idx:04d}.pt",
        )
        if os.path.exists(legacy_path):
            est = torch.load(legacy_path, weights_only=False, map_location="cpu")
            return np.asarray(est["p_trace"][:T], dtype=np.float64)
        if source == "wm":
            path = os.path.join(wm_entropy_dir(tag, model_size), f"wm_{idx:04d}.pt")
        else:
            path = os.path.join(null_entropy_dir(T, model_size), f"null_{idx:04d}.pt")
        est = torch.load(path, weights_only=False, map_location="cpu")
        return np.asarray(est["p_trace"][:T], dtype=np.float64)

    counts = {
        "wm_total": 0,
        "null_total": 0,
        "map_tp": 0,
        "map_fp": 0,
        "entropy_tp": 0,
        "entropy_fp": 0,
        "naive_tp": None if not use_generation_trace else 0,
        "naive_fp": None if not use_generation_trace else 0,
    }

    def _score(tokens, p_trace, watermark):
        dm = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="map")
        de = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="entropy")
        dn = None
        if use_generation_trace:
            dn = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="naive")
        if watermark:
            counts["wm_total"] += 1
            counts["map_tp"] += int(dm)
            counts["entropy_tp"] += int(de)
            if dn is not None:
                counts["naive_tp"] += int(dn)
        else:
            counts["null_total"] += 1
            counts["map_fp"] += int(dm)
            counts["entropy_fp"] += int(de)
            if dn is not None:
                counts["naive_fp"] += int(dn)

    wmd = f"/data/{tag}/wm"
    new_layout = os.path.isdir(wmd) and glob.glob(os.path.join(wmd, "wm_*.pt"))
    if new_layout:
        root = "/data/_nulls"
        use_T = T
        null_store = os.path.join(root, f"T{T}")
        have_nulls = all(os.path.exists(
            os.path.join(null_store, f"null_{i:04d}.pt"))
            for i in range(num_prompts))
        if not have_nulls:
            raise FileNotFoundError(f"No exact shared null store found for T={T}")
        for i in range(num_prompts):
            gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            wm_tokens = gw["tokens"][:T]
            _score(wm_tokens, _p_trace("wm", i, gw), True)
            gn = torch.load(os.path.join(root, f"T{use_T}", f"null_{i:04d}.pt"),
                            weights_only=False, map_location="cpu")
            null_tokens = gn["tokens"][:T]
            _score(null_tokens, _p_trace("null", i, gn), False)
    else:
        for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
            g = torch.load(path, weights_only=False, map_location="cpu")
            idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
            tokens = g["tokens"][:T]
            watermark = bool(g["watermark"])
            source = "wm" if watermark else "null"
            _score(tokens, _p_trace(source, idx, g), watermark)

    counts.update({
        "n": n,
        "t": t,
        "eta": eta,
        "T": T,
        "r": requested_r,
        "entropy_model": model_display(model_size),
        "entropy_trace_source": entropy_trace_source(model_size),
    })
    return counts


@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_legacy_first_block_summary(n: int, t: int, eta: float, fpr: float,
                                      num_prompts: int = 500, r: int = 0,
                                      entropy_model_size: str = MODEL_SIZE,
                                      legacy_token_length: int = 0) -> dict:
    """Redetect old T=2n cache rows using only the first length-n block."""
    import glob
    import os

    import numpy as np
    import torch
    from detectors import detect_hoeffding
    from prc import parity_check_rank_info

    score_T = experiment_T(n)
    cache_T = (
        int(legacy_token_length) if legacy_token_length else 2 * int(n)
    )
    requested_r = int(r) if r else None
    tag = config_tag(n, t, eta, requested_r, None)
    model_size = normalize_model_size(entropy_model_size)
    use_generation_trace = uses_cached_generation_trace(model_size)

    data_vol.reload()
    art = torch.load(f"/data/{tag}/artifacts.pt", weights_only=False,
                     map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    rank_info = art.get("parity_check_rank_info")
    if rank_info is None:
        rank_info = parity_check_rank_info(decoding_key[1])

    def _p_trace(idx, record):
        if use_generation_trace:
            p_trace = np.asarray(record["p_trace"], dtype=np.float64)
        else:
            path = os.path.join(
                f"/data/{tag}/entropy/{entropy_model_tag(model_size)}",
                f"gen_{idx:04d}.pt",
            )
            est = torch.load(path, weights_only=False, map_location="cpu")
            p_trace = np.asarray(est["p_trace"], dtype=np.float64)
        if p_trace.shape[0] < score_T:
            raise ValueError(
                f"{tag} gen_{idx:04d} has p_trace length {p_trace.shape[0]}, "
                f"need at least {score_T}"
            )
        return p_trace[:score_T]

    counts = {
        "wm_total": 0,
        "null_total": 0,
        "map_tp": 0,
        "map_fp": 0,
        "entropy_tp": 0,
        "entropy_fp": 0,
        "naive_tp": None if not use_generation_trace else 0,
        "naive_fp": None if not use_generation_trace else 0,
    }

    def _score(tokens, p_trace, watermark):
        dm = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="map")
        de = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                              fpr=fpr, weight="entropy")
        dn = None
        if use_generation_trace:
            dn = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, weight="naive")
        if watermark:
            counts["wm_total"] += 1
            counts["map_tp"] += int(dm)
            counts["entropy_tp"] += int(de)
            if dn is not None:
                counts["naive_tp"] += int(dn)
        else:
            counts["null_total"] += 1
            counts["map_fp"] += int(dm)
            counts["entropy_fp"] += int(de)
            if dn is not None:
                counts["naive_fp"] += int(dn)

    for path in sorted(glob.glob(f"/data/{tag}/gens/gen_*.pt")):
        g = torch.load(path, weights_only=False, map_location="cpu")
        idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
        watermark = bool(g["watermark"])
        if watermark and counts["wm_total"] >= num_prompts:
            continue
        if not watermark and counts["null_total"] >= num_prompts:
            continue
        tokens = g["tokens"][:score_T]
        if tokens.numel() < score_T:
            raise ValueError(
                f"{tag} gen_{idx:04d} has token length {tokens.numel()}, "
                f"need at least {score_T}"
            )
        _score(tokens, _p_trace(idx, g), watermark)

    counts.update({
        "n": n,
        "t": t,
        "eta": eta,
        "T": score_T,
        "legacy_T": cache_T,
        "legacy_tag": tag,
        "r": requested_r,
        "entropy_model": model_display(model_size),
        "entropy_trace_source": entropy_trace_source(model_size),
        "parity_check_rank_info": rank_info,
    })
    return counts


# ---- driver -----------------------------------------------------------------
@app.local_entrypoint()
def main(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False,
         batch: int = DEFAULT_BATCH,
         entropy_batch: int = DEFAULT_ENTROPY_BATCH,
         r: int = 0, r_frac: float = 0.0,
         entropy_model_size: str = MODEL_SIZE,
         csv_out: str = "hoeffding_results_summary.csv"):
    resolved_r = resolve_r(n, r, r_frac)
    validate_r_for_keygen(n, t, resolved_r)
    entropy_model_size = normalize_model_size(entropy_model_size)
    T = experiment_T(n)
    tag = config_tag(n, t, eta, resolved_r, T)
    r_text = f"r={resolved_r}" if resolved_r is not None else "r=default"
    print(f"[main] config {tag}  FPR_target={fpr:g}  ({num_prompts} prompts, "
          f"batch={batch}, entropy_batch={entropy_batch}, {r_text}, "
          f"entropy_model={model_display(entropy_model_size)}, fresh={fresh}) ...",
          flush=True)

    build_artifacts.remote(num_prompts, n, t, eta, resolved_r or 0, fresh)

    plan = plan_generation.remote(n, t, eta, num_prompts, resolved_r or 0)
    wm_missing, null_missing = plan["wm_missing"], plan["null_missing"]
    null_T = plan["null_T"]
    print(f"[main] to generate: {len(wm_missing)} watermarked, "
          f"{len(null_missing)} null  (exact null store T={null_T})",
          flush=True)

    if wm_missing or null_missing:
        model = Model.with_options(max_containers=max_containers)(tag=tag)
        calls = []
        if wm_missing:
            calls.append(("wm", list(model.generate_wm.map(
                _chunks(wm_missing, batch)))))
        if null_missing:
            calls.append(("null", list(model.generate_null.map(
                _chunks(null_missing, batch)))))
        for kind, metas in calls:
            gen = sum(m.get("generated", 0) for m in metas)
            print(f"[main] {kind}: generated {gen} in {len(metas)} batches",
                  flush=True)
    else:
        print("[main] all generations cached -> skipping generation GPU fleet",
              flush=True)

    if uses_cached_generation_trace(entropy_model_size):
        print("[main] entropy trace: using cached generation p_trace", flush=True)
    else:
        eplan = plan_entropy.remote(tag, entropy_model_size, T, null_T, num_prompts)
        wm_e_missing = eplan["wm_missing"]
        null_e_missing = eplan["null_missing"]
        print(f"[main] entropy traces to estimate with "
              f"{model_display(entropy_model_size)}: "
              f"{len(wm_e_missing)} watermarked, {len(null_e_missing)} null",
              flush=True)
        if wm_e_missing or null_e_missing:
            estimator = EntropyModel.with_options(
                max_containers=max_containers
            )(tag=tag, model_size=entropy_model_size, T=T, null_T=null_T)
            ecalls = []
            if wm_e_missing:
                ecalls.append(("wm entropy", list(estimator.estimate_wm.map(
                    _chunks(wm_e_missing, entropy_batch)))))
            if null_e_missing:
                ecalls.append(("null entropy", list(estimator.estimate_null.map(
                    _chunks(null_e_missing, entropy_batch)))))
            for kind, metas in ecalls:
                est = sum(m.get("estimated", 0) for m in metas)
                print(f"[main] {kind}: estimated {est} in {len(metas)} batches",
                      flush=True)
        else:
            print("[main] all alternate entropy traces cached", flush=True)

    detect_label = "map + entropy-aware + naive"
    if not uses_cached_generation_trace(entropy_model_size):
        detect_label = "map + entropy-aware (naive/log skipped for alternate entropy)"
    print(f"[main] detecting ({detect_label}) ...", flush=True)
    results = detect_all.remote(n, t, eta, fpr, null_T, num_prompts,
                                resolved_r or 0, entropy_model_size)

    wm = sorted([r0 for r0 in results if r0["watermark"]],
                key=lambda r0: r0["prompt_idx"])
    nw = sorted([r0 for r0 in results if not r0["watermark"]],
                key=lambda r0: r0["prompt_idx"])
    tp_m = sum(r0["decision_map"] for r0 in wm)
    fp_m = sum(r0["decision_map"] for r0 in nw)
    tp_e = sum(r0["decision_entropy"] for r0 in wm)
    fp_e = sum(r0["decision_entropy"] for r0 in nw)
    has_naive = any(r0.get("decision_naive") is not None for r0 in results)
    tp_n = sum(r0["decision_naive"] for r0 in wm) if has_naive else 0
    fp_n = sum(r0["decision_naive"] for r0 in nw) if has_naive else 0
    nwm, nnw = max(len(wm), 1), max(len(nw), 1)
    rank_info = results[0].get("parity_check_rank_info", {}) if results else {}

    print("\n=== Summary (Hoeffding detector, proven FPR) ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g}  T={T}  {r_text}",
          flush=True)
    if rank_info:
        print(f"  parity rank: {rank_info.get('rank')}/{rank_info.get('rows')} "
              f"full_rank={rank_info.get('full_rank')}", flush=True)
    print(f"  map (default):  TPR {tp_m}/{len(wm)} ({tp_m/nwm:.1%})   "
          f"FPR {fp_m}/{len(nw)} ({fp_m/nnw:.1%})", flush=True)
    print(f"  entropy-aware:  TPR {tp_e}/{len(wm)} ({tp_e/nwm:.1%})   "
          f"FPR {fp_e}/{len(nw)} ({fp_e/nnw:.1%})", flush=True)
    if has_naive:
        print(f"  naive        :  TPR {tp_n}/{len(wm)} ({tp_n/nwm:.1%})   "
              f"FPR {fp_n}/{len(nw)} ({fp_n/nnw:.1%})", flush=True)
    else:
        print("  naive        :  skipped", flush=True)
    print("  log_hoeffding:  skipped", flush=True)

    rank_note = ""
    if rank_info:
        rank_note = (
            f"r={rank_info.get('rows')} rank={rank_info.get('rank')}/"
            f"{rank_info.get('rows')} full_rank={rank_info.get('full_rank')}"
        )
        if not rank_info.get("full_rank", False):
            rank_note += "; WARNING parity matrix was not full rank"

    notes = "; ".join(part for part in [
        "Qwen3-0.6B-Base generation",
        f"entropy model {model_display(entropy_model_size)}",
        entropy_trace_source(entropy_model_size),
        "map=Bayes-optimal soft-token S_j=E[c|observed bit,p]",
        "batched Modal pipeline with exact-length null cache",
        "single length-n code block (T=n)",
        "Hoeffding threshold tau=sqrt(2V*log(1/F)); one block so block FPR equals target F",
        rank_note,
    ] if part)

    row = {
        "Target FPR": f"{fpr:.0e}",
        "n": n,
        "t": t,
        "eta": eta,
        "T": T,
        "Map TPR": _format_rate(tp_m, len(wm)),
        "Entropy Aware TPR": _format_rate(tp_e, len(wm)),
        "Naive TPR": _format_rate(tp_n, len(wm)) if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _format_rate(fp_m, len(nw)),
        "Entropy FPR": _format_rate(fp_e, len(nw)),
        "Naive FPR": _format_rate(fp_n, len(nw)) if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Model": model_display(entropy_model_size),
        "Entropy Trace Source": entropy_trace_source(entropy_model_size),
        "Notes": notes,
    }
    _append_summary_row(csv_out, row)
    print(f"[main] appended summary row to {csv_out}", flush=True)


@app.local_entrypoint()
def legacy_first_block_redetect(n: int = DEFAULT_N, t: int = DEFAULT_T,
                                eta: float = DEFAULT_ETA,
                                fpr: float = DEFAULT_FPR,
                                num_prompts: int = 500,
                                r: int = 0, r_frac: float = 0.0,
                                entropy_model_size: str = MODEL_SIZE,
                                legacy_token_length: int = 0,
                                csv_out: str = "hoeffding_results_summary.csv"):
    """Append a T=n redetect row from legacy T=2n cached generations."""
    resolved_r = resolve_r(n, r, r_frac)
    entropy_model_size = normalize_model_size(entropy_model_size)
    score_T = experiment_T(n)
    cache_T = (
        int(legacy_token_length) if legacy_token_length else 2 * int(n)
    )
    legacy_tag = config_tag(n, t, eta, resolved_r, None)
    print(f"[legacy_first_block_redetect] cache={legacy_tag} cache_T={cache_T} "
          f"score_T={score_T} FPR_target={fpr:g} "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)

    s = detect_legacy_first_block_summary.remote(
        n, t, eta, fpr, num_prompts, resolved_r or 0, entropy_model_size,
        cache_T,
    )

    wm_total = max(s["wm_total"], 1)
    null_total = max(s["null_total"], 1)
    has_naive = s["naive_tp"] is not None
    rank_info = s.get("parity_check_rank_info", {})

    print("\n=== Legacy First-Block Redetect Summary ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g} "
          f"T={score_T} legacy_T={cache_T} r={resolved_r}", flush=True)
    if rank_info:
        print(f"  parity rank: {rank_info.get('rank')}/{rank_info.get('rows')} "
              f"full_rank={rank_info.get('full_rank')}", flush=True)
    print(f"  entropy model: {s['entropy_model']} ({s['entropy_trace_source']})",
          flush=True)
    print(f"  map    : TPR {s['map_tp']}/{s['wm_total']} "
          f"({s['map_tp']/wm_total:.1%})   FPR {s['map_fp']}/"
          f"{s['null_total']} ({s['map_fp']/null_total:.1%})", flush=True)
    print(f"  entropy: TPR {s['entropy_tp']}/{s['wm_total']} "
          f"({s['entropy_tp']/wm_total:.1%})   FPR {s['entropy_fp']}/"
          f"{s['null_total']} ({s['entropy_fp']/null_total:.1%})", flush=True)
    if has_naive:
        print(f"  naive  : TPR {s['naive_tp']}/{s['wm_total']} "
              f"({s['naive_tp']/wm_total:.1%})   FPR {s['naive_fp']}/"
              f"{s['null_total']} ({s['naive_fp']/null_total:.1%})", flush=True)
    else:
        print("  naive  : skipped", flush=True)

    rank_note = ""
    if rank_info:
        rank_note = (
            f"r={rank_info.get('rows')} rank={rank_info.get('rank')}/"
            f"{rank_info.get('rows')} full_rank={rank_info.get('full_rank')}"
        )
        if not rank_info.get("full_rank", False):
            rank_note += "; WARNING parity matrix was not full rank"

    notes = "; ".join(part for part in [
        "Qwen3-0.6B-Base generation",
        f"entropy model {s['entropy_model']}",
        s["entropy_trace_source"],
        "map=Bayes-optimal soft-token S_j=E[c|observed bit,p]",
        f"legacy redetect from {legacy_tag}",
        f"used first n tokens from legacy T={cache_T} cached generations",
        "appended as new T=n row; old T=2n row retained",
        "Hoeffding threshold tau=sqrt(2V*log(1/F)); one block so block FPR equals target F",
        rank_note,
    ] if part)

    row = {
        "Target FPR": f"{fpr:.0e}",
        "n": n,
        "t": t,
        "eta": eta,
        "T": score_T,
        "Map TPR": _format_rate(s["map_tp"], s["wm_total"]),
        "Entropy Aware TPR": _format_rate(s["entropy_tp"], s["wm_total"]),
        "Naive TPR": _format_rate(s["naive_tp"], s["wm_total"])
        if has_naive else "skipped",
        "Log Hoeffding TPR": "skipped",
        "Map FPR": _format_rate(s["map_fp"], s["null_total"]),
        "Entropy FPR": _format_rate(s["entropy_fp"], s["null_total"]),
        "Naive FPR": _format_rate(s["naive_fp"], s["null_total"])
        if has_naive else "skipped",
        "Log Hoeffding FPR": "skipped",
        "Entropy Model": s["entropy_model"],
        "Entropy Trace Source": s["entropy_trace_source"],
        "Notes": notes,
    }
    _append_summary_row(csv_out, row)
    print(f"[legacy_first_block_redetect] appended summary row to {csv_out}",
          flush=True)


# ---- re-detection sweep over all weight kinds (CPU-only, no regeneration) ---
@app.local_entrypoint()
def redetect(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
             fpr: float = DEFAULT_FPR, num_prompts: int = 500,
             r: int = 0, r_frac: float = 0.0,
             entropy_model_size: str = MODEL_SIZE):
    """Re-detect a config (either cache layout) with ALL weight kinds, ranked.
    Free: model-free CPU detection over already-cached generations."""
    from detectors import WEIGHT_KINDS

    resolved_r = resolve_r(n, r, r_frac)
    entropy_model_size = normalize_model_size(entropy_model_size)
    T = experiment_T(n)
    tag = config_tag(n, t, eta, resolved_r, T)
    print(f"[redetect] {tag}  FPR_target={fpr:g}  "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)
    results = detect_all_any.remote(n, t, eta, fpr, num_prompts,
                                    resolved_r or 0, entropy_model_size)
    wm = [r for r in results if r["watermark"]]
    nw = [r for r in results if not r["watermark"]]
    nwm, nnw = max(len(wm), 1), max(len(nw), 1)
    print(f"\n=== Re-detect (Hoeffding, proven FPR) n={n} t={t} eta={eta} "
          f"F={fpr:g}  T={T} ===", flush=True)
    rows = []
    for name in WEIGHT_KINDS:
        key = f"decision_{name}"
        if not any(key in row for row in results):
            continue
        tp = sum(r.get(key, False) for r in wm)
        fp = sum(r.get(key, False) for r in nw)
        rows.append((tp, fp, name))
    for tp, fp, name in sorted(rows, reverse=True):     # best TPR first
        flag = "  <- baseline" if name == "entropy" else ""
        print(f"  {name:8s}: TPR {tp}/{len(wm)} ({tp/nwm:.1%})   "
              f"FPR {fp}/{len(nw)} ({fp/nnw:.1%}){flag}", flush=True)


@app.local_entrypoint()
def redetect_map(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
                 fpr: float = DEFAULT_FPR, num_prompts: int = 500,
                 r: int = 0, r_frac: float = 0.0,
                 entropy_model_size: str = MODEL_SIZE):
    """CPU-only redetect for map/entropy/naive CSV summary columns."""
    resolved_r = resolve_r(n, r, r_frac)
    entropy_model_size = normalize_model_size(entropy_model_size)
    T = experiment_T(n)
    tag = config_tag(n, t, eta, resolved_r, T)
    print(f"[redetect_map] {tag} FPR_target={fpr:g} "
          f"entropy_model={model_display(entropy_model_size)} ...", flush=True)
    s = detect_map_summary.remote(n, t, eta, fpr, num_prompts,
                                  resolved_r or 0, entropy_model_size)
    wm_total = max(s["wm_total"], 1)
    null_total = max(s["null_total"], 1)
    print("\n=== Map Redetect Summary ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g} T={T} r={resolved_r}",
          flush=True)
    print(f"  entropy model: {s['entropy_model']} ({s['entropy_trace_source']})",
          flush=True)
    print(f"  map    : TPR {s['map_tp']}/{s['wm_total']} "
          f"({s['map_tp']/wm_total:.1%})   FPR {s['map_fp']}/{s['null_total']} "
          f"({s['map_fp']/null_total:.1%})", flush=True)
    print(f"  entropy: TPR {s['entropy_tp']}/{s['wm_total']} "
          f"({s['entropy_tp']/wm_total:.1%})   FPR {s['entropy_fp']}/"
          f"{s['null_total']} ({s['entropy_fp']/null_total:.1%})", flush=True)
    if s["naive_tp"] is None:
        print("  naive  : skipped", flush=True)
    else:
        print(f"  naive  : TPR {s['naive_tp']}/{s['wm_total']} "
              f"({s['naive_tp']/wm_total:.1%})   FPR {s['naive_fp']}/"
              f"{s['null_total']} ({s['naive_fp']/null_total:.1%})", flush=True)


# ---- re-detect a whole set of configs in one Modal session ------------------
# (n, t, eta) for every config we've generated so far. Both cache layouts.
REDETECT_CONFIGS = [
    (256, 3, 0.05), (400, 3, 0.05), (512, 3, 0.05), (1024, 3, 0.05),
    (400, 3, 0.20),
    (400, 5, 0.05), (512, 5, 0.05), (1024, 5, 0.05), (2048, 5, 0.05),
]


@app.local_entrypoint()
def redetect_all(fpr: float = 1e-3, num_prompts: int = 500):
    """Re-detect EVERY cached config with all weight kinds, one Modal session.
    Prints a compact map-vs-entropy line per config plus each full leaderboard."""
    from detectors import WEIGHT_KINDS

    summary = []
    for (n, t, eta) in REDETECT_CONFIGS:
        print(f"\n########## n={n} t={t} eta={eta} ##########", flush=True)
        results = detect_all_any.remote(n, t, eta, fpr, num_prompts)
        wm = [r for r in results if r["watermark"]]
        nw = [r for r in results if not r["watermark"]]
        nwm, nnw = max(len(wm), 1), max(len(nw), 1)
        rows = []
        for name in WEIGHT_KINDS:
            tp = sum(r[f"decision_{name}"] for r in wm)
            fp = sum(r[f"decision_{name}"] for r in nw)
            rows.append((tp, fp, name))
        for tp, fp, name in sorted(rows, reverse=True):
            flag = "  <- baseline" if name == "entropy" else ""
            print(f"  {name:8s}: TPR {tp}/{len(wm)} ({tp/nwm:.1%})   "
                  f"FPR {fp}/{len(nw)} ({fp/nnw:.1%}){flag}", flush=True)
        by = {name: (tp, fp) for tp, fp, name in rows}
        summary.append((n, t, eta, by["map"], by["entropy"], nwm, nnw))

    print("\n================ map vs entropy (all configs) ================",
          flush=True)
    for n, t, eta, (mtp, mfp), (etp, efp), nwm, nnw in summary:
        print(f"  n={n:>4} t={t} eta={eta:<4}: map {mtp/nwm:5.1%} vs entropy "
              f"{etp/nwm:5.1%}  (+{(mtp-etp)/nwm:.1%})   FPR map {mfp}/{nnw} "
              f"ent {efp}/{nnw}", flush=True)
