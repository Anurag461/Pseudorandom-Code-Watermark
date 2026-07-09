"""
Run the PRC watermark experiment on Modal with the Hoeffding detector.

Cost-optimized pipeline:
  - build_artifacts (CPU): freeze the PRC key (OTP + generator), shared seeded
    partition map, tokenized RealNews prompts. Written to a Volume per config.
  - Model.generate_wm / generate_null (GPU): BATCHED generation -- B prompts per
    forward pass (all RealNews prefixes are 50 tokens, so no padding). One
    batch per .map() call.
  - detect_all (CPU, model-free): Hoeffding detection (proven FPR, no calibration).

Two cost levers baked in:
  1. Null (unwatermarked) generations depend only on the model + prompt + seed,
     NOT on the key (n,t,eta). They live in a SHARED store keyed by length T and
     are reused across every config (a longer null store is truncated for a
     shorter T). So nulls are generated once, not once per config.
  2. Batched inference: B sequences per GPU step instead of 1, which is where the
     A10G was idle on a 0.6B model.
  3. Fully-cached configs (e.g. an FPR sweep) never touch the GPU -- a CPU probe
     decides whether any generation is needed before the Model fleet is created.

Usage:
    modal run modal_run.py                                  # smoke: 10 prompts
    modal run modal_run.py --num-prompts 500 --max-containers 25
    modal run modal_run.py --num-prompts 500 --n 1024 --t 5 --eta 0.05 --fpr 1e-3

T = 2n. FPR only changes the detection threshold, so re-detecting a cached
(n,t,eta) at a new FPR is free (CPU-only).
"""
import modal

# ---- experiment config (defaults = n=400 RealNews baseline) -----------------
DEFAULT_N = 400
DEFAULT_T = 3
DEFAULT_ETA = 0.05
DEFAULT_FPR = 2e-5
SEED = 12345
MODEL_SIZE = "0.6B"
VOCAB = 151_936                 # Qwen3 vocab
GPU = "A10G"
DEFAULT_MAX_CONTAINERS = 5      # bound cost on the smoke test; raise for 500
DEFAULT_BATCH = 64             # sequences per GPU forward pass


def config_tag(n, t, eta):
    """Per-config directory tag for the KEY-DEPENDENT (watermarked) artifacts.
    FPR is excluded (doesn't affect generation) so FPR targets share a cache."""
    return f"n{n}_t{t}_eta{eta:.2f}"


def art_path(n, t, eta):
    return f"/data/{config_tag(n, t, eta)}/artifacts.pt"


def wm_dir(n, t, eta):
    return f"/data/{config_tag(n, t, eta)}/wm"


def null_dir(T):
    """Shared, key-INDEPENDENT null store keyed only by output length T."""
    return f"/data/_nulls/T{T}"


def _chunks(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]

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
            # Reduce CUDA allocator fragmentation so large batches (KV cache
            # grows with batch x length) don't OOM on the last few hundred MiB.
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    .add_local_python_source(
        "prc", "qwen", "constants", "ldpc", "detectors", "watermark_expt"
    )
)

hf_cache = modal.Volume.from_name("prc-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("prc-data", create_if_missing=True)

app = modal.App("prc-hoeffding", image=image)


# ---- artifact build (CPU only; no model load) -------------------------------
@app.function(volumes={"/data": data_vol}, timeout=600)
def build_artifacts(num_prompts: int, n: int, t: int, eta: float,
                    fresh: bool = False) -> int:
    import glob
    import json
    import os
    import torch
    import numpy as np
    from prc import KeyGen  # importing prc does NOT load the LM

    ap = art_path(n, t, eta)
    wmd = wm_dir(n, t, eta)
    max_new_tokens = 2 * n
    os.makedirs(os.path.dirname(ap), exist_ok=True)

    # The PRC key is NOT reproducible across processes (galois GF.Random ignores
    # np.random.seed) and every WATERMARKED generation is bound to its key. So we
    # FREEZE the key per (n,t,eta) and reuse it. A rebuild (config change or
    # fresh=True) invalidates ONLY the watermarked cache -- null generations are
    # key-independent and live in a shared store, so they are never wiped here.
    config_sig = {"n": n, "t": t, "eta": eta, "T": max_new_tokens,
                  "num_prompts": num_prompts,
                  "gen_scheme": "fresh_codeword_per_block_batched"}
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
        n=n, message_length=0, false_positive_rate=0.5, t=t, noise_rate=eta
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t_key = decoding_key
    r = parity_check_matrix.shape[0]
    print(f"[build] PRC params: n={n} t={t_key} g={g} r={r} "
          f"noise_rate={noise_rate:.4f}", flush=True)

    # Shared, seeded partition map (identical for every job and every config).
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
            "seed": SEED,
            "config_sig": config_sig,
        },
        ap,
    )
    data_vol.commit()
    print(f"[build] wrote artifacts ({num_prompts} prompts) -> {ap}", flush=True)
    return num_prompts


# ---- cache probes (CPU; decide whether the GPU fleet is needed) -------------
@app.function(volumes={"/data": data_vol}, timeout=300)
def plan_generation(n: int, t: int, eta: float, num_prompts: int) -> dict:
    """Return which watermarked and null prompts still need generating, plus the
    null store T to detect against (reusing a longer null store if present)."""
    import os

    T = 2 * n
    wmd = wm_dir(n, t, eta)
    data_vol.reload()

    wm_missing = [i for i in range(num_prompts)
                  if not os.path.exists(os.path.join(wmd, f"wm_{i:04d}.pt"))]

    # Null reuse: find the smallest existing null store with T' >= T that already
    # holds all prompts. If found, reuse it (detection truncates to T). Else
    # generate the missing nulls into this config's own T store.
    root = "/data/_nulls"
    use_T = None
    if os.path.isdir(root):
        avail = []
        for name in os.listdir(root):
            if not name.startswith("T"):
                continue
            try:
                Tp = int(name[1:])
            except ValueError:
                continue
            if Tp < T:
                continue
            d = os.path.join(root, name)
            have_all = all(
                os.path.exists(os.path.join(d, f"null_{i:04d}.pt"))
                for i in range(num_prompts)
            )
            if have_all:
                avail.append(Tp)
        if avail:
            use_T = min(avail)

    if use_T is not None:
        null_missing = []
    else:
        use_T = T
        d = null_dir(T)
        null_missing = [i for i in range(num_prompts)
                        if not os.path.exists(os.path.join(d, f"null_{i:04d}.pt"))]

    return {"wm_missing": wm_missing, "null_missing": null_missing,
            "null_T": use_T, "T": T}


# ---- batched generation (GPU) -----------------------------------------------
@app.cls(
    gpu=GPU,
    volumes={"/data": data_vol, "/cache": hf_cache},
    timeout=3600,
    max_containers=DEFAULT_MAX_CONTAINERS,
)
class Model:
    tag: str = modal.parameter()  # per-config directory tag (n{n}_t{t}_eta{eta})

    @modal.enter()
    def load(self):
        import torch

        self.ap = f"/data/{self.tag}/artifacts.pt"
        data_vol.reload()
        art = torch.load(self.ap, weights_only=False, map_location="cpu")

        import watermark_expt as we  # loads (first time downloads) the LM
        self.we = we
        self.encoding_key = art["encoding_key"]
        self.prompts = art["prompt_ids_list"]
        self.n = art["n"]
        self.T = 2 * art["n"]
        we.partition = art["partition"].to(we.device)
        self.partition = we.partition
        hf_cache.commit()

    def _prompt_batch(self, indices):
        import torch
        rows = [self.prompts[i] for i in indices]
        return torch.tensor(rows, dtype=torch.long, device=self.we.device)

    @modal.method()
    def generate_wm(self, prompt_indices: list) -> dict:
        """Batched WATERMARKED generation; save one wm_{i}.pt per prompt."""
        import os
        import time
        import torch

        wmd = f"/data/{self.tag}/wm"
        os.makedirs(wmd, exist_ok=True)
        data_vol.reload()
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
        """Batched NULL generation into the shared store; save null_{i}.pt."""
        import os
        import time
        import torch

        nd = null_dir(self.T)
        os.makedirs(nd, exist_ok=True)
        data_vol.reload()
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


# ---- detection over cached generations (CPU, model-free) --------------------
@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(n: int, t: int, eta: float, fpr: float,
               null_T: int, num_prompts: int) -> list:
    """Both Hoeffding detectors over every watermarked + null generation.

    Watermarked come from the per-config store; nulls from the shared store at
    null_T, truncated to this config's T = 2n. No model load."""
    import os
    import torch
    from detectors import detect_hoeffding

    T = 2 * n
    ap = art_path(n, t, eta)
    wmd = wm_dir(n, t, eta)
    nd = null_dir(null_T)
    data_vol.reload()
    art = torch.load(ap, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]

    def _run(tokens, p_trace, wm_flag, idx):
        de, ie = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, entropy_weighted=True, return_info=True)
        dn, ino = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                   fpr=fpr, entropy_weighted=False, return_info=True)
        return {
            "prompt_idx": idx, "watermark": wm_flag,
            "decision_entropy": bool(de), "stat_entropy": float(ie["statistic"]),
            "thr_entropy": float(ie["threshold"]),
            "decision_naive": bool(dn), "stat_naive": float(ino["statistic"]),
            "thr_naive": float(ino["threshold"]),
            "n_tokens": int(tokens.numel()),
        }

    out = []
    for i in range(num_prompts):
        gw = torch.load(os.path.join(wmd, f"wm_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        out.append(_run(gw["tokens"], gw["p_trace"], True, i))

        gn = torch.load(os.path.join(nd, f"null_{i:04d}.pt"),
                        weights_only=False, map_location="cpu")
        # Truncate a (possibly longer) shared null to this config's T.
        tok, ptr = gn["tokens"][:T], gn["p_trace"][:T]
        out.append(_run(tok, ptr, False, i))
    return out


# ---- driver -----------------------------------------------------------------
@app.local_entrypoint()
def main(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False,
         batch: int = DEFAULT_BATCH):
    tag = config_tag(n, t, eta)
    print(f"[main] config {tag}  FPR_target={fpr:g}  ({num_prompts} prompts, "
          f"batch={batch}, fresh={fresh}) ...", flush=True)
    build_artifacts.remote(num_prompts, n, t, eta, fresh)

    plan = plan_generation.remote(n, t, eta, num_prompts)
    wm_missing, null_missing = plan["wm_missing"], plan["null_missing"]
    null_T = plan["null_T"]
    print(f"[main] to generate: {len(wm_missing)} watermarked, "
          f"{len(null_missing)} null  (null store T={null_T}, reuse="
          f"{null_T != plan['T']})", flush=True)

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
        print("[main] all generations cached -> skipping GPU entirely "
              "(detect-only, no LM load)", flush=True)

    print("[main] detecting (entropy-aware + naive) ...", flush=True)
    results = detect_all.remote(n, t, eta, fpr, null_T, num_prompts)

    wm = sorted([r for r in results if r["watermark"]], key=lambda r: r["prompt_idx"])
    nw = sorted([r for r in results if not r["watermark"]], key=lambda r: r["prompt_idx"])
    tp_e = sum(r["decision_entropy"] for r in wm)
    fp_e = sum(r["decision_entropy"] for r in nw)
    tp_n = sum(r["decision_naive"] for r in wm)
    fp_n = sum(r["decision_naive"] for r in nw)
    nwm, nnw = max(len(wm), 1), max(len(nw), 1)
    print("\n=== Summary (Hoeffding detector, proven FPR) ===", flush=True)
    print(f"  n={n} t={t} eta={eta} FPR_target={fpr:g}  T={2 * n}", flush=True)
    print(f"  entropy-aware:  TPR {tp_e}/{len(wm)} ({tp_e/nwm:.1%})   "
          f"FPR {fp_e}/{len(nw)} ({fp_e/nnw:.1%})", flush=True)
    print(f"  naive        :  TPR {tp_n}/{len(wm)} ({tp_n/nwm:.1%})   "
          f"FPR {fp_n}/{len(nw)} ({fp_n/nnw:.1%})", flush=True)
