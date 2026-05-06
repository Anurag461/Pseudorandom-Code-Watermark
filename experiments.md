# PRC watermark calibration — reproduction guide

End-to-end instructions for reproducing the PRC watermark calibration experiments on Qwen3-0.6B-Base, plus the key deviations from the paper that were needed to make detection actually work.

Results live in `results_summary.csv` (one row per `(target FPR, n, t, η)` config).

## Contents

1. [Workspace setup](#workspace-setup)
2. [Differences from the PRC paper](#differences-from-the-prc-paper)
3. [How a single run works (3 phases)](#how-a-single-run-works-3-phases)
4. [Reproducing the headline configs](#reproducing-the-headline-configs)
5. [Reusing cached generations across detectors](#reusing-cached-generations-across-detectors)
6. [Sweeping FPR targets and entropy thresholds](#sweeping-fpr-targets-and-entropy-thresholds)
7. [Producing the results CSV](#producing-the-results-csv)
8. [Source map](#source-map)

---

## Workspace setup

You need CUDA 12.x, Python ≥3.10, and PyTorch 2.6+. Pick whichever path matches your environment.

### Option A — public PyTorch Docker image (recommended)

The official `pytorch/pytorch` images on Docker Hub bundle a working CUDA + PyTorch + cuDNN stack and are what we test against:

```bash
docker run --gpus all -it --rm \
    -v "$PWD":/workspace -w /workspace \
    pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime \
    bash

# inside the container
pip install --no-cache-dir \
    "transformers>=4.40" "tokenizers>=0.21" \
    "huggingface_hub>=0.30" safetensors scipy numpy
```

Any image with PyTorch ≥ 2.6 and CUDA ≥ 12.1 should work (e.g. `nvidia/pytorch:24.10-py3`). The host needs a recent NVIDIA driver compatible with the image's CUDA version.

### Option B — bare metal / venv

If your host already has a working CUDA toolchain (verify with `nvidia-smi` and `nvcc --version`):

```bash
python3 -m venv .venv && source .venv/bin/activate

# Pin PyTorch to a CUDA build that matches your driver.
# CUDA 12.4 wheels:
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install \
    "transformers>=4.40" "tokenizers>=0.21" \
    "huggingface_hub>=0.30" safetensors scipy numpy
```

For CPU-only experimentation (Phase 2/3 only — Phase 1 generation effectively requires a GPU), drop the index URL: `pip install torch==2.6.0`.

### Option C — uv + pyproject.toml (upstream nanochat)

The repo's `pyproject.toml` is configured for upstream nanochat and pulls torch 2.9.1 / CUDA 12.8 via `uv sync --extra gpu`. The watermark scripts work against that stack too — just be aware they were developed and benchmarked on torch 2.6.

### Sanity-check the install

```bash
# Pulls Qwen3-0.6B-Base weights + tokenizer to ./Qwen3-0.6B-Base/ on first run.
# Takes ~1 min over a fast link, ~5 min on home wifi.
python3 -c "import watermark_expt"
```

`watermark_expt` prints the resolved `huggingface_hub`, `tokenizers`, and `torch` versions when it imports — keep an eye on those if you hit version skew.

### Hardware

Phase 1 generation is the only GPU-heavy phase. The orchestrator (`run_calibration.py`) auto-discovers visible GPUs via `nvidia-smi --query-gpu=index` and spawns one worker per GPU with `CUDA_VISIBLE_DEVICES=<i>`. With 8× H100 a `n=4096, T=16384` config takes ~10 min; 1× GPU takes ~80 min for the same. Memory: Qwen3-0.6B-Base with KV cache fits comfortably in <8 GB even at `T=16384`.

Phase 2 (calibration) and Phase 3 (detection) are pure NumPy and run on CPU; no GPU needed for the sweep / backfill scripts (set `CUDA_VISIBLE_DEVICES=` to keep them off the GPUs).

`constants.py` holds the 30 fixed long-form prompts. Each generation is `T = max_new_tokens` tokens, set to a multiple of `n` (typically `T = 4n`) so the detector can run on `T/n` non-overlapping length-`n` blocks.

---

## Differences from the PRC paper

These are the implementation choices that diverge from the PRC paper and matter for reproduction.

### 1. Token-level bucketing instead of prefix-free encoding

The paper assumes a binary channel: at every step the LM emits one bit `b ∈ {0,1}` whose distribution is biased by the PRC codeword bit. To embed this into a multi-token vocabulary, the paper proposes a prefix-free binary encoding of the token stream so that one LM step ≈ one bit, with a known cost in generation length.

We replace this with a **fixed binary partition of the vocabulary** (`partition_map: (2, V)`):

- `v0 = 1[token ∈ random half of V]`, `v1 = 1 − v0`. Built once per run from a seeded `torch.randperm(vocab_size)` (see `run_calibration.py:106-112`).
- At each step we collapse the LM's softmax to a single bit-1 probability `p1 = Σ probs[partition 1]` (`watermark_expt.py:263`).
- Watermarking conditions on `p1`: with codeword bit `xi` we draw `b ~ Bern(bern_p)` where
  ```
  bern_p = where(p1 ≤ 0.5, 2·xi·p1, 1 − 2·(1−xi)·(1−p1))
  ```
  then sample the next token from the masked softmax restricted to half `b` (`watermark_expt.py:265-280`).

Practical consequences:
- **One token = one codeword slot.** No length blow-up, no streaming-decoder bookkeeping.
- The detector observes one bit per token, not one bit per encoded sub-step.
- The watermark signal at each step is bounded by `H₂(p1)`, the binary entropy of the partition split — not by full token entropy. Low-entropy steps (one half is near-impossible) carry near-zero signal regardless of the LM's overall token entropy. This is what motivates the entropy fold below.

### 2. The argmax → multinomial fix (largest single bug)

Low temperature sampling or `argmax` on the masked logits to pick the next token does not work. This deterministically picks the most-probable token in whichever half `b` selected, which collapses the per-step bit posterior to {0, 1} **independent of `b`** — destroying the watermark signal.

- TPR with argmax sampling: 0/30 at every threshold tested.
- TPR after switching to multinomial (`watermark_expt.py:280`): 96.7% at FPR=1e-9 with n=4096, t=3.

If you fork the code, **do not "optimize" sampling back to argmax**. Multinomial is load-bearing.

### 3. Entropy-weighted fold ("entropy fold")

The paper's detector aggregates per-slot observations uniformly. Because some slots will be sampled at near-zero `H₂(p1)` (one half ≈ impossible — common in real LMs), the per-slot posterior gets dragged toward random ±1 by individual deterministic observations.

Our `fold_entropy_weighted` (`watermark_expt.py:306`) downweights each observation by `H₂(p1)/ln 2 ∈ [0, 1]`:

```python
weights = binary_entropy(p_array) / np.log(2)
posterior_per_slot[i] = Σ_j (sign_j · weight_j) / Σ_j weight_j   # cyclic over j ≡ i (mod n)
```

Effects, both visible in the data:
- Suppress corrupted parity checks (smaller, ~10% of TPR swing): a check that touches one near-deterministic slot has an essentially random Π and adds noise.
- Clean up surviving checks at the *observation* level (bigger, ~90% of swing): even within a slot, individual deterministic observations are downweighted toward zero so the slot's posterior is dominated by informative observations.

Both `fold_entropy_weighted` and the equal-weight `fold_naive` are available; pick via the `fold` field on the threshold state (`fit_calibration(..., fold="entropy"|"naive")`).

### 4. Block-OR semantics for syndrome detection

The PRC Theorem 1 detector decides "watermarked" iff the syndrome weight `< (½ − r_eff^{−¼}) · r_eff` for a single length-`n` codeword. We typically generate `T = 4n` tokens.

`detect_syndrome` (`watermark_expt.py:655`) splits the trace into `T/n` consecutive non-overlapping blocks, runs an independent syndrome check per block, and decides watermarked iff **any** block passes. Trailing tokens with `T % n ≠ 0` are ignored. Per-block FPR is bounded analytically by the threshold formula; document FPR ≤ B × per-block FPR (Hoeffding).

### 5. Hard-remove of low-entropy parity checks ("syndrome with entropy filter")

Variant of #4. Within a block, drop any parity check whose `t` token positions include any token with `H₂(p1) < 0.1` bits (`watermark_expt.py:639`). This raises per-check signal-to-noise but reduces `r_eff`, so the analytical threshold tightens in absolute terms but the *fraction-of-violations* needed stays the same. In practice it gains 0–1 prompts over `syndrome_all` at our configs (see `results_summary.csv`).

---

## How a single run works (3 phases)

`run_calibration.py` orchestrates everything:

- **Phase 1: generation.** Parent builds `(encoding_key, decoding_key, partition, prompt_ids, jobs)` once and saves to `WORKDIR/artifacts.pt`. Spawns one worker per GPU (`worker_generate.py`); each handles a `(prompt_idx, watermark)` job back-to-back, saving `WORKDIR/result_NN.pt` (tokens + p_trace + job metadata). Skipped if `REUSE_GENERATIONS=True` and the workdir already has matching artifacts + all 60 result files.
- **Phase 2: calibration.** Only for `entropy_fold` / `naive_fold`. Calls `fit_calibration` on the unwatermarked p_traces — re-samples 2000 simulated null draws (uniform random codeword + Bernoulli channel through real p_traces), folds them, computes the test statistic, and sets `threshold = null_mean + Φ⁻¹(1 − fpr) · null_std`. Saves to `qwen_threshold.json`. Syndrome methods skip Phase 2 (analytical threshold).
- **Phase 3: detect.** Runs the detector on all 60 generations and reports TPR / FPR.

The detector is selected by the top-of-file constant `DETECT_METHOD ∈ {"entropy_fold", "naive_fold", "syndrome_all", "syndrome_entropy"}`.

---

## Reproducing the headline configs

Edit the config block at the top of `run_calibration.py` and run it inside the container:

```bash
docker exec -w /home/anurakas/nanochat nile-nemo-jupyter \
    python3 run_calibration.py 2>&1 | tee run_<tag>.log
```

The four canonical configs are:

| tag | n | t | g | η | T | Notes |
|---|---|---|---|---|---|---|
| `n4096_t6_g144` | 4096 | 6 | 144 | 0.05 | 16384 | Theorem-2 aligned (`t=½log₂n`, `g=log²n`). Best operating point. |
| `n4096_t14`     | 4096 | 14 | None (auto) | 0.0053 | 16384 | More-pseudorandom regime; `g, η` from `KeyGen` defaults. |
| `n512_t4`       | 512 | 4 | None | 0.05 | 2048 | Small but workable; used for the entropy/naive comparison at FPR=2e-10. |
| `n128_t3_eta05` | 128 | 3 | None | 0.05 | 512 | Smallest config; cryptographically broken (`secpar ≈ 18 bits`) but useful for sanity. |

Set `N_CODEWORD`, `T_PARITY`, `G_PARAM`, `NOISE_RATE`, `MAX_NEW_TOKENS`, `WORKDIR`, and `DETECT_METHOD` in `run_calibration.py:60-73` before each run. Phase 1 takes ~10–60 minutes depending on `n` and GPU count; Phase 2+3 take seconds.

**Important**: Changing `NOISE_RATE` (or any other key parameter) invalidates the existing `artifacts.pt`. Set `REUSE_GENERATIONS=False` for the first run at a new config, or move/delete the old `WORKDIR` first.

---

## Reusing cached generations across detectors

The Phase-1 generations are deterministic for a given `(SEED, model, n, t, g, η, prompts)`. To compare detectors on the same generations:

1. Run Phase 1 once with `DETECT_METHOD = "entropy_fold"` (or any choice).
2. Set `REUSE_GENERATIONS = True` and `WORKDIR` to the cached directory.
3. Switch `DETECT_METHOD` and re-run. Phase 1 is skipped; Phase 2 (if applicable) re-fits the threshold against the same null traces; Phase 3 detects.

This is how the `entropy_fold` vs `naive_fold` numbers in `results_summary.csv` were produced — single Phase 1, four Phase-3 passes (one per `DETECT_METHOD`).

For a one-off detection pass without editing `run_calibration.py`, see `backfill_results.py` — it loads `WORKDIR/artifacts.pt` + `result_*.pt`, fits each fold's threshold, and runs all four detectors, printing one CSV row per `(workdir, method, fpr_target)`:

```bash
docker exec -w /home/anurakas/nanochat -e CUDA_VISIBLE_DEVICES= nile-nemo-jupyter \
    python3 backfill_results.py
```

`CUDA_VISIBLE_DEVICES=` keeps it on CPU (Phase 2/3 are pure NumPy / no GPU needed).

---

## Sweeping FPR targets and entropy thresholds

Two helper scripts, both read `WORKDIR` artifacts and don't re-run generation:

- `fpr_sweep.py` / `fpr_sweep_naive.py` — for a fixed fold (entropy / naive), sweep FPR targets `{1e-9, 1e-6, 1e-3, 1e-2, 5e-2, 1e-1}`. Refits the null distribution from watermarked p-traces, sets `threshold = null_mean + Φ⁻¹(1−fpr) · null_std` per target, prints TPR + empirical FPR. Source for the FPR-sweep rows in `results_summary.csv`.
- `sweep_syndrome_entropy.py` — for syndrome detection, sweep entropy filter thresholds `{None, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9}`. Reports TPR / FPR / blocks-fired / average `r_eff` per threshold so you can see the `r_eff` vs detection-power tradeoff.

```bash
docker exec -w /home/anurakas/nanochat -e CUDA_VISIBLE_DEVICES= nile-nemo-jupyter \
    python3 fpr_sweep.py        calib_workdir
docker exec -w /home/anurakas/nanochat -e CUDA_VISIBLE_DEVICES= nile-nemo-jupyter \
    python3 sweep_syndrome_entropy.py calib_workdir
```

---

## Producing the results CSV

`results_summary.csv` is the consolidated table (one row per `(target FPR, n, t, η)` × {entropy fold, naive fold, hard-remove syndrome, no-filter syndrome}). It's hand-assembled from:

- Run logs (`run_*.log` for canonical Phase-2 + Phase-3 results)
- FPR-sweep logs (`fpr_sweep_*.log`) for non-canonical FPR targets
- `backfill_results.log` for cells filled in by post-hoc detection on cached generations

If you re-run anything, regenerate the relevant rows by editing `results_summary.csv` directly. The columns are `Target FPR, n, t, eta, Entropy Aware TPR, Naive TPR, Hard Remove TPR, Syndrome No-Filter TPR, FPR, Notes`. "Entropy Aware" = `entropy_fold`, "Naive" = `naive_fold`, "Hard Remove" = `syndrome_entropy` (drops parity checks involving `H₂ < 0.1` bit tokens), "Syndrome No-Filter" = `syndrome_all` (PRC Theorem 1 over all checks).

---

## Source map

| file | role |
|---|---|
| `prc.py` | LDPC-PRC₀ key generation, encode, decode (paper-aligned, untouched). |
| `qwen.py` | Qwen3 model + tokenizer wrapper. |
| `constants.py` | The 30 fixed `test_prompts`. |
| `watermark_expt.py` | Sampling (`generate_text_watermark_prc`), folds (`fold_naive`, `fold_entropy_weighted`), calibration (`fit_calibration`), and detection (`detect_with_threshold`, `detect_syndrome`). |
| `run_calibration.py` | Three-phase orchestrator. Edit the top-of-file constants to choose a config + detector. |
| `worker_generate.py` | Per-GPU generation worker (one job, saves `result_NN.pt`). |
| `backfill_results.py` | Run all detectors on a cached `WORKDIR` and print CSV rows. |
| `fpr_sweep.py`, `fpr_sweep_naive.py` | FPR-target sweeps for fold detectors. |
| `sweep_syndrome_entropy.py` | Entropy-filter threshold sweep for the syndrome detector. |
| `outlier_analysis.py` | Per-prompt entropy/naive decomposition (used for the `outlier_demo.json` writeup). |
| `qwen_threshold*.json` | Saved threshold states for each headline config. |
| `calib_workdir*/` | Per-config Phase-1 artifacts: `artifacts.pt` + `result_NN.pt × 60` + `worker_NN.log × 60`. |
| `results_summary.csv` | Final consolidated TPR/FPR table. |

For the original results-only writeup (decompositions, headline takeaways, full per-config tables), see `outlier_summary.md` and `n128_fpr_sweep.md`.
