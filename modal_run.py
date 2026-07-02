"""
Run the PRC watermark experiment on Modal with the Hoeffding detector.

Ports the local run_calibration.py flow to Modal:
  - build_artifacts (CPU): KeyGen, shared partition map, tokenized RealNews
    prompts, and the (prompt, watermark) job list. Written to a Volume.
  - Model.generate (A10G GPU): one job -> generate text with the PRC channel,
    then run detect_hoeffding (proven FPR, no calibration) in the same
    container. Fanned out over all jobs with .map().
  - local_entrypoint: tally TPR / FPR from the returned decisions.

Usage:
    modal run modal_run.py                 # smoke test: 10 prompts (20 jobs)
    modal run modal_run.py --num-prompts 500
    modal run modal_run.py --num-prompts 500 --max-containers 25

The n=400 / t=3 / eta=0.05 / FPR=2e-5 config matches the existing
results_summary.csv RealNews row (T = 2n = 800 tokens).
"""
import modal

# ---- experiment config (defaults = n=400 RealNews baseline) -----------------
# n/t/eta/FPR are CLI-overridable per run (see main). eta changes the PRC key
# (noise_rate) so it needs its own generations; FPR only changes the detection
# threshold, so re-detecting a cached eta at a new FPR is free. T = 2n.
DEFAULT_N = 400
DEFAULT_T = 3
DEFAULT_ETA = 0.05
DEFAULT_FPR = 2e-5
SEED = 12345
MODEL_SIZE = "0.6B"
VOCAB = 151_936                 # Qwen3 vocab
GPU = "A10G"
DEFAULT_MAX_CONTAINERS = 5      # bound cost on the smoke test; raise for 500


def config_tag(n, t, eta):
    """Per-config directory tag. FPR is deliberately excluded (it doesn't affect
    generation), so different FPR targets share one cached generation set."""
    return f"n{n}_t{t}_eta{eta:.2f}"


def config_paths(n, t, eta):
    base = f"/data/{config_tag(n, t, eta)}"
    return f"{base}/artifacts.pt", f"{base}/gens"

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
        }
    )
    .add_local_file("prompts.jsonl", "/root/prompts.jsonl")
    # Local project modules the container needs at import time.
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

    art_path, gens_dir = config_paths(n, t, eta)
    max_new_tokens = 2 * n
    os.makedirs(os.path.dirname(art_path), exist_ok=True)

    # The PRC key (OTP + generator matrix) is NOT reproducible across processes
    # (galois GF.Random ignores np.random.seed), and every generation is
    # cryptographically bound to the key it was produced under. So we FREEZE the
    # key: build it once per (n,t,eta), persist to the Volume, and reuse it on
    # later runs. A rebuild (config change or fresh=True) invalidates the cache.
    config_sig = {"n": n, "t": t, "eta": eta, "T": max_new_tokens,
                  "num_prompts": num_prompts}
    data_vol.reload()
    if not fresh and os.path.exists(art_path):
        prev = torch.load(art_path, weights_only=False, map_location="cpu")
        if prev.get("config_sig") == config_sig:
            n_jobs = len(prev["jobs"])
            print(f"[build] reusing frozen key from {art_path} "
                  f"(config matches; {n_jobs} jobs, cache preserved)", flush=True)
            return n_jobs
        print("[build] config changed -> rebuilding key and INVALIDATING cache",
              flush=True)

    # Fresh key: wipe any stale generations bound to a previous key.
    removed = 0
    for p in glob.glob(os.path.join(gens_dir, "*.pt")):
        os.remove(p)
        removed += 1
    if removed:
        print(f"[build] cleared {removed} stale cached generations", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    encoding_key, decoding_key = KeyGen(
        n=n, message_length=0, false_positive_rate=0.5, t=t, noise_rate=eta
    )
    _, parity_check_matrix, _, _, noise_rate, _, g, _, t_key = decoding_key
    r = parity_check_matrix.shape[0]
    print(f"[build] PRC params: n={n} t={t_key} g={g} r={r} "
          f"noise_rate={noise_rate:.4f}", flush=True)

    # Shared, seeded partition map (identical for every job).
    perm = torch.randperm(VOCAB, generator=torch.Generator().manual_seed(SEED))
    v0 = torch.zeros(VOCAB, dtype=torch.bfloat16)
    v0[perm[: VOCAB // 2]] = 1.0
    v1 = 1 - v0
    partition = torch.stack([v0, v1], dim=0)

    # RealNews prompts (raw token continuation, no chat template).
    rows = []
    with open("/root/prompts.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= num_prompts:
                break
    if len(rows) < num_prompts:
        raise RuntimeError(f"prompts.jsonl has {len(rows)} rows, need {num_prompts}")
    prompt_ids_list = [row["prompt_tokens"] for row in rows]

    jobs = []
    for prompt_idx in range(len(prompt_ids_list)):
        for watermark in (True, False):
            jobs.append({
                "prompt_idx": prompt_idx,
                "watermark": watermark,
                "max_new_tokens": max_new_tokens,
            })

    torch.save(
        {
            "encoding_key": encoding_key,
            "decoding_key": decoding_key,
            "partition": partition,
            "prompt_ids_list": prompt_ids_list,
            "jobs": jobs,
            "n": n,
            "seed": SEED,
            "config_sig": config_sig,
        },
        art_path,
    )
    data_vol.commit()
    print(f"[build] wrote {len(jobs)} jobs ({num_prompts} prompts) -> {art_path}",
          flush=True)
    return len(jobs)


# ---- generation + detection (GPU) -------------------------------------------
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

        self.art_path = f"/data/{self.tag}/artifacts.pt"
        self.gens_dir = f"/data/{self.tag}/gens"
        data_vol.reload()
        art = torch.load(self.art_path, weights_only=False, map_location="cpu")

        # Importing watermark_expt downloads (first time) and loads the LM,
        # then builds a *random* partition; we override it with the shared one.
        import watermark_expt as we

        self.we = we
        self.encoding_key = art["encoding_key"]
        self.decoding_key = art["decoding_key"]
        self.jobs = art["jobs"]
        self.prompts = art["prompt_ids_list"]
        we.partition = art["partition"].to(we.device)
        self.partition = we.partition

        # Persist any weights just downloaded into the HF cache Volume.
        hf_cache.commit()

    @modal.method()
    def generate(self, job_index: int) -> dict:
        """Generate one job and cache (tokens, p_trace) to the data Volume.

        If the cache file already exists (from a prior run), skip generation
        entirely -- this is what makes detector re-runs free.
        """
        import os
        import time
        import torch

        os.makedirs(self.gens_dir, exist_ok=True)
        out_path = os.path.join(self.gens_dir, f"gen_{job_index:04d}.pt")
        data_vol.reload()
        if os.path.exists(out_path):
            return {"job_index": job_index, "cached": True}

        we = self.we
        job = self.jobs[job_index]
        prompt_ids = torch.tensor(
            [self.prompts[job["prompt_idx"]]], dtype=torch.long, device=we.device
        )

        t0 = time.time()
        gen = we.generate_text_watermark_prc(
            we.model,
            prompt_ids,
            max_new_tokens=job["max_new_tokens"],
            encoding_key=self.encoding_key,
            partition_map=self.partition,
            eos_token_id=None,  # want a full-length p-trace (>= n)
            watermark=job["watermark"],
        )
        tokens, p_trace = we.generate_and_collect(gen)

        torch.save(
            {
                "job_index": job_index,
                "prompt_idx": job["prompt_idx"],
                "watermark": bool(job["watermark"]),
                "tokens": tokens.cpu(),
                "p_trace": p_trace,
            },
            out_path,
        )
        data_vol.commit()
        return {
            "job_index": job_index,
            "cached": False,
            "n_tokens": int(tokens.numel()),
            "dt": time.time() - t0,
        }


# ---- detection over cached generations (CPU, model-free) --------------------
@app.function(volumes={"/data": data_vol}, timeout=1800)
def detect_all(n: int, t: int, eta: float, fpr: float) -> list:
    """Run both Hoeffding detectors on every cached generation. No model load."""
    import os
    import torch
    from detectors import detect_hoeffding  # imports prc only, not the LM

    art_path, gens_dir = config_paths(n, t, eta)
    data_vol.reload()
    art = torch.load(art_path, weights_only=False, map_location="cpu")
    decoding_key = art["decoding_key"]
    partition = art["partition"]
    n_jobs = len(art["jobs"])

    out = []
    for idx in range(n_jobs):
        g = torch.load(os.path.join(gens_dir, f"gen_{idx:04d}.pt"),
                       weights_only=False, map_location="cpu")
        tokens, p_trace = g["tokens"], g["p_trace"]
        de, ie = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                  fpr=fpr, entropy_weighted=True, return_info=True)
        dn, ino = detect_hoeffding(decoding_key, tokens, p_trace, partition,
                                   fpr=fpr, entropy_weighted=False, return_info=True)
        out.append({
            "prompt_idx": g["prompt_idx"],
            "watermark": g["watermark"],
            "decision_entropy": bool(de),
            "stat_entropy": float(ie["statistic"]),
            "thr_entropy": float(ie["threshold"]),
            "decision_naive": bool(dn),
            "stat_naive": float(ino["statistic"]),
            "thr_naive": float(ino["threshold"]),
            "n_tokens": int(tokens.numel()),
        })
    return out


# ---- driver -----------------------------------------------------------------
@app.local_entrypoint()
def main(num_prompts: int = 10, max_containers: int = DEFAULT_MAX_CONTAINERS,
         n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
         fpr: float = DEFAULT_FPR, fresh: bool = False):
    tag = config_tag(n, t, eta)
    print(f"[main] config {tag}  FPR_target={fpr:g}  ({num_prompts} prompts, "
          f"fresh={fresh}) ...", flush=True)
    n_jobs = build_artifacts.remote(num_prompts, n, t, eta, fresh)

    # Phase 1: generate + cache (skips jobs already on the Volume).
    print(f"[main] {n_jobs} jobs; generating/caching on {GPU} "
          f"(<= {max_containers} containers) ...", flush=True)
    model = Model.with_options(max_containers=max_containers)(tag=tag)
    gen_meta = list(model.generate.map(range(n_jobs)))
    reused = sum(1 for m in gen_meta if m.get("cached"))
    print(f"[main] generations ready: {len(gen_meta)} "
          f"({reused} reused from cache, {len(gen_meta) - reused} freshly generated)",
          flush=True)

    # Phase 2: detect on cached generations (CPU, model-free, cheap).
    print("[main] detecting (entropy-aware + naive) on cached generations ...",
          flush=True)
    results = detect_all.remote(n, t, eta, fpr)

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


# ---- diagnostics over the cached generations (no regeneration) --------------
@app.local_entrypoint()
def analyze(n: int = DEFAULT_N, t: int = DEFAULT_T, eta: float = DEFAULT_ETA,
            fpr: float = DEFAULT_FPR):
    """Detect-only over cached generations, with token-length + margin histograms."""
    import numpy as np

    results = detect_all.remote(n, t, eta, fpr)
    wm = [r for r in results if r["watermark"]]
    nw = [r for r in results if not r["watermark"]]

    def hist(vals, edges, label):
        vals = np.asarray(vals)
        print(f"  {label}:", flush=True)
        for lo, hi in zip(edges[:-1], edges[1:]):
            c = int(((vals >= lo) & (vals < hi)).sum())
            print(f"    [{lo:>7.1f}, {hi:>7.1f}): {c}", flush=True)

    ntok = [r["n_tokens"] for r in results]
    print(f"\n=== token-length distribution (n={n} t={t} eta={eta}) ===", flush=True)
    print(f"  min={min(ntok)} max={max(ntok)} "
          f"mean={np.mean(ntok):.1f} < T({2 * n}): "
          f"{sum(1 for x in ntok if x < 2 * n)}  "
          f"< n({n}): {sum(1 for x in ntok if x < n)}", flush=True)

    margin_e = [r["stat_entropy"] - r["thr_entropy"] for r in wm]
    print("\n=== watermarked: (entropy stat - threshold) ===", flush=True)
    print(f"  detected (margin>=0): {sum(1 for m in margin_e if m >= 0)}/{len(wm)}", flush=True)
    hist(margin_e, [-50, -20, -10, -5, 0, 5, 10, 20, 50, 200], "margin histogram")

    tp_e = sum(1 for r in wm if r["decision_entropy"])
    tp_n = sum(1 for r in wm if r["decision_naive"])
    fp_e = sum(1 for r in nw if r["decision_entropy"])
    fp_n = sum(1 for r in nw if r["decision_naive"])
    print(f"\n  entropy: TPR {tp_e}/{len(wm)}  FPR {fp_e}/{len(nw)}", flush=True)
    print(f"  naive  : TPR {tp_n}/{len(wm)}  FPR {fp_n}/{len(nw)}", flush=True)
