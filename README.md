# PRC watermark calibration — reproduction guide

End-to-end instructions for reproducing the PRC watermark calibration experiments on Qwen3-0.6B-Base, plus the key deviations from the paper that were needed to make detection actually work.

Watermark-detection (TPR/FPR) results live in `hoeffding_results_summary.csv`; benchmark utility results live in `benchmark_utility_results.csv`.

This implementation borrows heavily from the [PRC-Watermark](https://github.com/XuandongZhao/PRC-Watermark) implementation by Sam Gunn, Xuandong Zhao, and Dawn Song.

> **How experiments are run today: Modal.** `modal_run.py` runs the watermark
> detection (TPR/FPR) experiments and `modal_gsm8k.py` runs the benchmark utility
> evals. See [Running experiments (Modal)](#running-experiments-modal) below.

## Running experiments (Modal)

Everything runs server-side on Modal, so a laptop only needs to dispatch the job
(`modal run`) or, better, deploy once and fire jobs that survive disconnects
(`modal deploy` + `.spawn`). Model weights and HF datasets are cached in Modal
Volumes; results land in the `prc-eval-results` Volume.

**Watermark detection experiments** (`modal_run.py`, RealNews prompts) — sweep
`n` / `eta` and report MAP/entropy/naive TPR at a target FPR:

```bash
modal run modal_run.py::main \
    --num-prompts 500 --n 768 --t 3 --eta 0.1 --fpr 1e-3 \
    --r-frac 0.99 --max-containers 10
```

**Benchmark utility evals** (`modal_gsm8k.py`) — measure the task-accuracy cost
of watermarking (watermarked vs unwatermarked) across benchmarks. Small runs:

```bash
modal run modal_gsm8k.py::main --benchmark gsm8k --limit 200 --max-new-tokens 4096
```

Full / large runs are sharded, resumable, and disconnect-proof via deploy+spawn
(each shard commits to the volume; a `SUMMARY.json` is written at the end):

```bash
modal deploy modal_gsm8k.py
# then, fire-and-forget (runs to completion regardless of the client):
python -c "import modal; modal.Function.from_name('prc-gsm8k-eval','orchestrate_full')\
    .spawn('mmlu', 16, 4096, 24, 'reasoning', 'full')"   # (benchmark, batch, max_new_tokens, num_shards, variant, run_tag[, limit, sample, sample_seed])
```

Benchmarks are registered in `benchmarks/registry.py`:
`gsm8k, arc_easy, arc_challenge, hellaswag, mmlu, ag_news, gpqa_diamond, aime24,
aime25, ifeval`. GPQA Diamond is a gated HF dataset — provide an `HF_TOKEN` via a
Modal secret named `huggingface` (`modal secret create huggingface HF_TOKEN=...`).
Utility results are recorded in `benchmark_utility_results.csv`.

## Contents

1. [Running experiments (Modal)](#running-experiments-modal)
2. [Important Implementation Details](#important-implementation-details)
3. [Source map](#source-map)

---

## Important Implementation Details

These are the implementation choices that diverge from the PRC paper and matter for reproduction.

### 1. Token-level bucketing instead of prefix-free encoding

The paper assumes a binary channel: at every step the LM emits one bit `b ∈ {0,1}` whose distribution is biased by the PRC codeword bit. To embed this into a multi-token vocabulary, the paper proposes a prefix-free binary encoding of the token stream so that one LM step ≈ one bit, with a known cost in generation length.

We replace this with a **fixed binary partition of the vocabulary** (`partition_map: (2, V)`):

- `v0 = 1[token ∈ random half of V]`, `v1 = 1 − v0`. Built once per run from a seeded `torch.randperm(vocab_size)` (see `watermark_expt.py`).
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


### 2. Entropy-weighted fold ("entropy fold")

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

### 3. argmax → multinomial sampling

Low temperature sampling or `argmax` on the masked logits to pick the next token does not work. This deterministically picks the most-probable token in whichever half `b` selected, which collapses the per-step bit posterior to {0, 1} **independent of `b`** — destroying the watermark signal.

- TPR with argmax sampling: 0/30 at every threshold tested.
- TPR after switching to multinomial (`watermark_expt.py:280`): 96.7% at FPR=1e-9 with n=4096, t=3.

If you fork the code, **do not "optimize" sampling back to argmax**. Multinomial is load-bearing.

### 4. Block-OR semantics for syndrome detection

The PRC Theorem 1 detector decides "watermarked" iff the syndrome weight `< (½ − r_eff^{−¼}) · r_eff` for a single length-`n` codeword. We typically generate `T = 4n` tokens.

`detect_syndrome` (`watermark_expt.py:655`) splits the trace into `T/n` consecutive non-overlapping blocks, runs an independent syndrome check per block, and decides watermarked iff **any** block passes. Trailing tokens with `T % n ≠ 0` are ignored. Per-block FPR is bounded analytically by the threshold formula; document FPR ≤ B × per-block FPR (Hoeffding).

### 5. Hard-remove of low-entropy parity checks ("syndrome with entropy filter")

Variant of #4. Within a block, drop any parity check whose `t` token positions include any token with `H₂(p1) < 0.1` bits (`watermark_expt.py:639`). This raises per-check signal-to-noise but reduces `r_eff`, so the analytical threshold tightens in absolute terms but the *fraction-of-violations* needed stays the same. In practice it gains 0–1 prompts over `syndrome_all` at our configs (see `results_summary.csv`).


---

## Source map

| file | role |
|---|---|
| `prc.py` | LDPC-PRC₀ key generation, encode, decode (paper-aligned, untouched). |
| `qwen.py` | Qwen3 model + tokenizer wrapper (+ KV caches, batched left-pad `key_padding_mask`). |
| `constants.py` | The 30 fixed `test_prompts`. |
| `modal_run.py` | **(current)** Modal app for watermark detection experiments (RealNews prompts, sharded generation + detection). |
| `modal_gsm8k.py` | **(current)** Modal app for benchmark utility evals: `run_eval` (small), `orchestrate_full` (sharded, resumable, deploy+spawn). |
| `benchmarks/` | Benchmark `Task` classes + `registry.py`. Includes `ifeval_lib/` (vendored Google IFEval verifier). |
| `benchmark_utility_results.csv` | Watermarked-vs-unwatermarked task accuracy across benchmarks. |
| `watermark_expt.py` | Sampling (`generate_text_watermark_prc`), batched eval harness (`chat_eval_benchmark_batched`), folds, calibration (`fit_calibration`), detection (`detect_with_threshold`, `detect_syndrome`, `detect_hoeffding`). |
| `detectors.py` | Model-free detector helpers (folds, `detect_hoeffding`, prefix-column `detect_hoeffding_prefix`, generation-record builder). |
