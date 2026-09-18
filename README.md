# PRC watermark — reproduction guide

**Prompt-free paper redetection** now runs through the existing Modal app:

`modal_run.py` contains the combined PRC Modal implementation for both fixed and
online constructions. Use `::generate_fixed` or `::generate_online` for generation and
`::redetect --manifest ...` for prompt-free detection. The manifest's
`construction` field selects the original key and scoring rules.

The shared baseline code is in `baseline_comparison/comparison_runner.py`
(formerly `smoke_runner.py`). The [TextSeal comparison redetection plan](textseal_prompt_free_redetection_plan.md)
documents the TextSeal entropy correction, PRC cache reuse, upstream-code
requirements, and estimated costs. The [completion-only TextSeal setup](baseline_comparison/README.md)
calls the pinned upstream detector directly; the source preflight is complete.
The completed replay and shared-null alignment are documented there. The
[detectability versus diversity study](self_bleu/README.md) now has its own
package, [plan](self_bleu/plan.md), and [repeat-handling runbook](self_bleu/repeat_handling_ablation.md).

The first cleanup commit, `cf57b46`, retired prompted detection before this
structural move. Old implementations remain in Git history at `61b1739`;
original caches and historical results are unchanged. The old combined
"generate and detect" commands are retired; use the two explicit stages.
Multiple `lengths` in a redetection manifest replace prompted prefix sweeps.

```sh
MODAL_PROFILE=new-prc-watermark python -m modal run --detach \
  modal_run.py::redetect \
  --manifest outputs/redetection/.archive/manifests/same_0p6b_eta020_n3104.json \
  --stage full --gpu A100-80GB --max-containers 10
```

The frozen manifest selects the original candidate files, key, partition,
lengths and batch size (125 for n=3104). Its sources are hash-checked before
inference. Execution code must be committed. Use `--stage preflight` for a
CPU-only input check, or `--stage smoke` for representative validation batches.
Full execution validates one batch per actual shape, then distributes fixed
batches across up to 10 workers. Each worker reuses the existing BF16 model
loader and expandable-segments allocator; completed traces are cached atomically.
`--gpu` is passed through to Modal, including H100. A100-80GB remains the default;
the representative validation runs on the selected GPU with the selected batch.

The implementation has three entry points: completion-only replay in `qwen.py`,
the existing `detectors.py` functions (now prompt-free by default), and the
`redetect` command in `modal_run.py`. The GPU receives only raw completion
tokens and the partition. There is no prepended token; coordinate 1's score is
zero. Original PRC indices and threshold policies are retained. The scorer has
no prompt argument. It requires T-1 response-only probabilities; an old T-length
generation or prompted trace is rejected. `completion_only=False` is an explicit
opt-in for historical/control scoring only. Older experiment commands below
that pass historical probability traces must migrate to this redetection path
before being used for paper detection.

[Results and cache index](outputs/redetection/README.md) cover the completed
runs. Detailed manifests are archived outside Git. On this machine they are
under `outputs/redetection/.archive/manifests/`. For a fresh checkout, retrieve
the implementation archive listed in `outputs/redetection/cache_index.json`
from Modal volume `prc-completion-only`; its `prompt_free/manifests/` members
contain the frozen inputs. The integrated runner uses a separate `integrated/`
cache namespace and leaves previous traces and results intact. Cache reuse
requires matching protocol, run identity (including model, code and GPU), exact
completion/partition hashes, T-1 shape and probability checksum. There is no
fallback to historical generation or EOT caches. These provenance checks and
the audited raw-token replay establish the conditioning context; probability
values alone cannot establish it.

All Modal implementation is in `modal_run.py`, organized into these sections:

| Section | Responsibility |
|---|---|
| Shared runtime | One app, volumes, dependency profiles, model loading, GPU options and batching helpers. |
| Fixed generation | Original fixed keys and candidate caches; `generate_fixed`. |
| Seeded fixed replicates | Isolated seeded keys and shared null-cache reuse; `generate_replicate`. |
| Online generation | Causal keys, generation and continuation; `generate_online`. |
| Redetection | One completion-only recovery, validation and trace-cache pipeline; `redetect`. |

There are no separate `modal_runtime.py`, `modal_fixed.py`, `modal_online.py`,
or `modal_fixed_replicate_run.py` implementation files. Construction-specific
helpers have distinct names in the combined file. Historical cache and report
readers remain available, including shard aggregation and continuation checks.

Generation model sizes, key construction, sampler behavior, cache namespaces,
continuation, null-cache reuse, and batch controls are preserved. The fixed,
online and replicate dependency requirements are also preserved as separate
profiles in the shared runtime, so this move does not change numerical libraries.
Redetection retains its existing BF16 0.6B detector checkpoint restriction;
adding larger detector checkpoints is separate from this cleanup.

Examples (generation launches GPU work; these are instructions, not validation runs):

```sh
modal run modal_run.py::generate_fixed --n 400 --num-prompts 500 --batch 125 --gpu H100
modal run modal_run.py::generate_online --n 3104 --eta 0.2 --batch 125 --gpu H100
modal run modal_run.py::generate_replicate --help
modal run modal_run.py::build_null_cache --help
```

The public app also retains KV-cache diagnostics, seeded replicates, native
quality analysis, and historical shard aggregation. Historical aggregation
reads old reports; it does not recompute prompted detection. Old runbooks below
and elsewhere in this repository are historical references, not current launch
instructions.

Cleanup validation was entirely local: 153 tests passed after retirement and
156 after consolidation. The final merge compared 180 top-level functions and
classes unchanged after resolving their renamed references. Three source
fingerprint helpers now point at the combined file, and duplicate chunking and
replicate display helpers share the existing implementations. The temporary
`generate --construction ...` dispatcher is replaced by the separate generation
commands. Model, sampler and scoring source files are unchanged. All eight
indexed saved result files still match their hashes. CLI help checks cover all
nine public commands. No Modal inference or generation was launched.

Run the focused checks with:

```sh
NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 python -m pytest \
  tests/test_prompt_free.py tests/test_qwen_kv_cache.py tests/test_online_prc.py -q
```

End-to-end instructions for the PRC watermark experiments on Qwen3-0.6B-Base — watermark detection (TPR/FPR) and benchmark utility — plus the key deviations from the paper that were needed to make detection actually work.

Watermark-detection (TPR/FPR) results live in `hoeffding_results_summary.csv`; benchmark utility results live in `benchmark_utility_results.csv`.

This implementation borrows heavily from the [PRC-Watermark](https://github.com/XuandongZhao/PRC-Watermark) implementation by Sam Gunn, Xuandong Zhao, and Dawn Song.

> **Historical workflow (superseded by the commands above).** `modal_run.py` runs the watermark
> detection (TPR/FPR) experiments and `modal_gsm8k.py` runs the benchmark utility
> evals. See [Running experiments (Modal)](#running-experiments-modal) below.

## Running experiments (Modal)

Everything runs server-side on Modal, so a laptop only needs to dispatch the job
(`modal run`) or, better, deploy once and fire jobs that survive disconnects
(`modal deploy` + `.spawn`). Model weights and HF datasets are cached in Modal
Volumes; results land in the `prc-eval-results` Volume.

**Historical watermark detection experiments** (retired `modal_run.py::main`, RealNews prompts) — sweep
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
| `watermark_expt.py` | Sampling (`generate_text_watermark_prc`), batched eval harness (`chat_eval_benchmark_batched`), folds, threshold fitting (`fit_calibration`), detection (`detect_with_threshold`, `detect_syndrome`, `detect_hoeffding`). |
| `detectors.py` | Model-free detector helpers (folds, `detect_hoeffding`, prefix-column `detect_hoeffding_prefix`, generation-record builder). |
