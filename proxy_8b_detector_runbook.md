# Qwen3-8B to Qwen3-0.6B cache-only proxy runbook

Status: executed successfully on 2026-08-23. This runbook reproduces the
detector sensitivity analysis from existing caches. It authorizes neither new
text generation nor a 4B/end-to-end proxy experiment.

## Frozen inputs and provenance

- Canonical prompt indices `0..499`, in `prompts.jsonl` order.
- Existing Qwen3-8B PRC caches: eta `.05/.10/.15/.20`, source lengths
  `1280/3072/4096/14336`.
- Selected PRC evaluation lengths: `640/1407/4096/13088`; eta `.15` is a
  censored ceiling (`n90 > 4096`).
- Common prefixes: `128,256,400,512,768,1024`.
- Existing shared Qwen3-8B null cache at T=13088.
- Existing controlled TextSeal outputs from run
  `qwen3-8b-batch50-validation-20260823-v1`.
- Qwen3-8B-Base revision
  `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`.
- Qwen3-0.6B-Base revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; weights etag
  `cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`;
  tokenizer etag `443909a61d429dff23010e5bddd28ff530edda00`.
- Python 3.11 image definition SHA-256
  `e62c9627f72c0448435665ec51304050838f65ef68b8d77766d6176a03745e48`.
  Direct pins are torch 2.4.0, transformers 4.51.3, tokenizers 0.21.1,
  safetensors 0.4.5, huggingface-hub 0.30.2, scipy 1.14.1, galois 0.4.2,
  numba 0.59.1, NumPy 1.26.0, and pytest 8.3.3.
- Finalized integration code fingerprint
  `abe599dbdd5ec91348b508a90fb1f5515620cf3bcf9dad46d594878e0b44c44f`.
- TextSeal commit `c60d0d1da2e59f09a698438e218a07ee779b4616` and
  SynthID-Text commit `addb4a158143c7c6851a1308f78b89fceed59683`.

Model loading is offline from the existing Modal cache. The teacher-forcing
path uses static KV cache v1 and exact causal chunks of 64 tokens. The
historical one-token and chunked results passed exact-equivalence tolerances
before production.

`watermark_expt.py` has a legacy import-time `AutoTokenizer` for the non-Base
0.6B repository. This cache-only replay does not use `prompt_to_ids` or that
tokenizer: all input IDs are validated cached 8B IDs, and the replay weights
are the pinned 0.6B-Base weights. Do not change token construction in a rerun.

## Executed sequence

The read-only inventory is safe and launches no model worker:

```bash
modal run modal_online_run.py::app.proxy-8b --mode plan
```

The first sequential T=13088 benchmark projected `$79.84` and was stopped by
the cost gate. After chunked equivalence passed, the exact benchmark projected
`$3.5123` under the $20 cap:

```bash
modal run modal_online_run.py::app.proxy-8b \
  --mode benchmark \
  --approval-token APPROVE_8B_PROXY_ANALYSES_20_USD
```

The approved cache-only replay and PRC scoring then ran with up to five A10G
workers. It verified all source records before any model load and contains no
sampling or generation entrypoint:

```bash
modal run modal_online_run.py::app.proxy-8b \
  --mode full \
  --approval-token APPROVE_8B_PROXY_ANALYSES_20_USD \
  --gpu A10G \
  --max-containers 5 \
  --detection-max-containers 10
```

TextSeal was rescored in the isolated official-code image. This replaces only
the entropy weights and preserves all native-8B quality fields:

```bash
modal run baseline_comparison/modal_app.py::app.proxy-textseal-score \
  --approval-token APPROVE_8B_PROXY_ANALYSES_20_USD
```

The final native-quality pass reads cached tokens/log-probabilities only; it
loads no model:

```bash
modal run modal_online_run.py::app.proxy-8b-quality \
  --approval-token APPROVE_8B_PROXY_ANALYSES_20_USD
```

## Rebuilding compact local artifacts without replay

Do not invoke the full replay again merely to rebuild tables. Download the
already scored TextSeal shards into a pre-created directory—the trailing slash
on the remote directory is intentional:

```bash
mkdir -p /private/tmp/proxy_ts_shards
modal volume get --force prc-data \
  controlled_baseline_full/qwen3-8b-batch50-validation-20260823-v1/proxy_scored/textseal/ \
  /private/tmp/proxy_ts_shards
```

The native controlled-baseline scored shards must be available at
`/private/tmp/qwen3-8b-controlled-full-scored/scored`. Then run:

```bash
python proxy_8b_finalize.py \
  --native-scored-dir /private/tmp/qwen3-8b-controlled-full-scored/scored \
  --textseal-proxy-dir /private/tmp/proxy_ts_shards/textseal \
  --output-dir outputs
```

The finalizer requires exact coverage: 28,000 proxy PRC prompt rows, 24,000
native controlled-baseline rows, 6,000 TextSeal proxy rows, 2,500 native
quality rows, ten TextSeal validation shards, and all 500 prompt indices. It
recomputes Hoeffding upper bounds, checks every decision against `p<1e-3`,
validates official TextSeal parity, and writes only compact tables/manifests.

## Interpretation and recovery constraints

- Do not call SynthID or Gumbel “0.6B detectors.” Their frequentist tests do
  not use model probabilities and are carried unchanged.
- Do not substitute proxy likelihoods for quality. NLL/perplexity remain under
  Qwen3-8B-Base because the generator text is fixed.
- Do not equate PRC eta with TextSeal alpha.
- Keep eta `.15` labeled as a censored ceiling.
- Treat 0/500 or 1/500 observed false positives as descriptive; 500 nulls do
  not tightly validate nominal 0.1% FPR.
- Prompt-level replay traces and TextSeal proxy outputs remain immutable on
  `prc-data`. If a compact artifact is lost, rerun only the download and local
  finalizer.

Exact app URLs and resource charges are in
`outputs/proxy_8b_cost_ledger.csv`; scientific conclusions are in
`proxy_8b_detector_report.md`.
