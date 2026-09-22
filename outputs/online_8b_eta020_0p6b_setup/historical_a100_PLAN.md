# Online PRC eta=0.20: 8B generations detected by 0.6B

Prepared September 21, 2026, on `redetection`. **Proposal only; no paid computation has launched.** Expected incremental spending is about **$29**, with a planning range of **$26.10–$34.10** and a proposed **$40 allowance** for the three stages below. Expected elapsed time is **65–80 minutes**, plus any GPU allocation queue or approval wait.

## Exact cohort and reuse

- Use the same **500 watermarked Qwen3-8B-Base completions**, canonical prompt IDs 0–499, at the longest saved length **T=n=14336**, online PRC eta=0.20.
- Perform exactly one primary completion-only 0.6B replay per completion. Report **MAP and entropy at T=14336 only**, and compare against saved native 8B results: MAP 452/500 (90.4%), entropy 433/500 (86.6%). No native replay or recomputation is needed.
- Reuse the existing prompts, original raw token IDs, key, partition, source file/token hashes and model cache. The source aggregate manifest is `online_8b_eta020_T14336_remaining450_v1/combined/8B/prepared.json` in `prc-completion-only`; its 500 frozen source records are copied into `setup.json`.
- Read-only Modal inventory confirmed all 500 source files and the original artifact hash. It inspected 58 integrated completion-only manifests and found **zero 0.6B caches matching this source artifact**. It also verified current model configuration/tokenizer metadata and listed the cached 0.6B weights. Detailed evidence is in `inventory.json`.
- **New generations: 0, cost $0. Null generations/replays/scores: 0, separate cost $0. Empirical FPR is not evaluated.** No benchmark, reference replay, independent validation pass or automatic retry is included.
- Full-length traces will remain reusable for shorter lengths. A shorter-length scoring sweep is not part of this longest-length-only proposal and needs no new GPU replay if requested later.

## Checkpoints and detector protocol

| Role | Exact checkpoint | Model/tokenizer revision |
|---|---|---|
| Saved generator/native comparison | `Qwen/Qwen3-8B-Base` | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` |
| Proposed detector | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |

Both cached tokenizer JSON files are byte-identical, SHA256 `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`; both configurations have 151,936 vocabulary rows and a 32,768-position context limit. T=14336 is supported. Replay uses saved token IDs directly, without decoding/re-tokenizing. The 0.6B weight SHA256 is pinned to `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`; the approved CPU preparation stage will check this existing cached weight file before GPU work, with no new checkpoint download.

Protocol is the established **`completion_only_raw_abstain_v1`**: raw completion IDs with no prompt or added BOS/EOT prefix; coordinate 1 abstains; recover coordinates 2 through T using 0.6B. Score MAP and entropy using the original online key and partition. Do not substitute generation-time probabilities or native 8B traces for 0.6B probabilities.

Settings: eta=0.20, seed=12345, t=3, row rate 99/100 with the existing startup clamp and support schedule, r=14193 at T14336; target FPR=0.001, `one_shot`. Partition SHA256 is `503d1cf93958f0d765606ed3e25aa87a1a777cd08d8f44436b0c6ae9716aa184`. Inference uses BF16, TF32 disabled, static KV, float32 saved probabilities and float64 cloud CPU scores. The saved generations retain their original position-addressed inverse-CDF sampler, temperature 1, unrestricted vocabulary and forced length; no sampling occurs in this task.

## Resources, stages and cost

**Request ten A100 80GB workers concurrently. Each worker loads 0.6B once and processes two distinct batches of 25 sequentially.** This covers twenty batches and 500 records with a maximum of ten GPUs. Each completed batch is saved immediately, and workers shut down after their assigned work. Parallel scheduling reduces elapsed time without duplicating any record.

| Named stage | Work/resources | Expected elapsed time | Expected cost | Allowance |
|---|---|---:|---:|---:|
| Prepare | One CPU worker, 4 physical cores, 16 GiB; verify existing inputs/checkpoint and prepare 20 batch manifests | 1–6 min | $0.01–$0.06 | $0.15 |
| 0.6B replay | 10 concurrent A100 80GB workers; each 4 CPU cores, 16 GiB host RAM, two batches of 25 | 55–70 min | $26–$34 | $39.80 total |
| Score/report | One CPU worker, 4 physical cores, 8 GiB; N=500, T14336, MAP + entropy | 0.5–3 min | $0.003–$0.02 | $0.05 |
| **Total additional** | No generation or native 8B pass | **65–80 min including routine transfers** | **About $29; range $26.10–$34.10** | **$40.00** |

Cost basis: the eight successful eta=0.15 A100 primary batches at T6144/batch50 took 648.85–695.07 seconds (mean 670.90). Scaling by `(25/50)*(14336/6144)^2` projects 1,826 seconds per batch, or 60.9 minutes for two batches per worker. Twenty batches project $28.56 before startup/I/O. This is extrapolation, not a measurement at the new length. Exact evidence and calculations are saved in `timing_cost_basis.json`.

[Modal rates](https://modal.com/pricing), checked September 21: A100 80GB $0.000694/s, physical CPU core $0.0000131/s, memory $0.00000222/GiB/s. The configured replay worker is **$0.00078192/s ($2.814912/hour)** including host CPU and RAM. Queue time before allocation is separate from this runtime estimate.

Batch25 keeps the KV footprint below a prior measured 0.6B A100 configuration: 25×14336=358,400 sequence-tokens versus 100×4096=409,600, which peaked at 51.64 GB allocated. The exact new peak is unmeasured. This choice uses the proven A100 inference path and preserves memory margin; it is not a claim that 25 is the largest feasible batch. Keep the existing 85% memory guard and save-before-margin-check behavior. No paid exploration of larger batches or other GPUs is included.

One attempt per stage/worker, retries=0, durable attempt markers, no fallback launch. Proposed work deadlines are 600s for prepare, 4800s per replay worker (both batches combined), and 300s for scoring; function timeout adds 30s, startup is limited to 60s, and idle shutdown is 2s. The allowance includes startup/shutdown margin; it is an approval envelope rather than a provider-enforced dollar cap. Any failed or interrupted paid attempt requires a new estimate and approval before retrying. Completed batches are preserved.

## Budget, implementation and output

Fresh read-only billing confirms **$66.48862111** for the completed eta=0.20 native N=500 work and prefix sweep. This proposal would bring that cohort's combined native/0.6B spending to approximately **$95.49**, or **$106.48862111** if its entire $40 allowance were used. The usage API does not expose current credits; no reliable remaining balance can be inferred from the old $20/$6.52 figures. This proposal requires up to **$40 of additional approved funds** and does not reuse earlier approvals.

This planning task adds only this plan, a frozen JSON setup, read-only inventory script and evidence. Existing runtime files are unchanged. Before execution, adapt a narrow approval-gated driver from the successful eta=0.20 preparation/scheduling wrapper and eta=0.15 multi-batch 0.6B worker. Reuse `_prepare_redetection`, `_recover_redetection_batch(validate=False)` and the existing cloud CPU scoring/CSV routines. The driver must bind approvals to the setup/source hashes and the three stage allowances, enforce disjoint prompt coverage and check for newly available compatible traces before dispatch.

Retain the corrected loading order: scientific/native imports complete before NumPy pickle compatibility aliases. Keep worker logs under `/tmp` during volume reloads. No numerical or detector algorithm changes are proposed. Exact long-length runtime and GPU availability remain uncertain; no benchmark is needed for this estimate.

After approved execution, save and commit primary traces/manifests before scoring, then save scores, N=500 CSV row, native comparison, timing and actual itemized billing. Preserve the existing CSV schema and old rows; mark empirical FPR skipped. Cache tokens, key/partition, model/source manifests, trace hashes and scores. Extra reproducibility archives remain local. Preserve unrelated work on `redetection`; do not broadly stage or push.

**Await explicit approval for the three named stages and the $40 total allowance before launching anything.**
