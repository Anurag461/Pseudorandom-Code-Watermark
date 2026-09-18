# PRC shared-null alignment: launch review

Status: **completed and validated**. Only the 500 shared PRC nulls were replayed.
See [execution.json](execution.json) and [verification.json](verification.json).
TextSeal remains paused. The frozen plan below records the approved setup.

## Finding

All 500 old T1382 null completions differ from the original full comparison's
T13088 null completions within the first 1,024 tokens. Their FPRs cannot simply
be relabelled as shared-null results. The local source export and historical
comparison hashes verify the required T13088 tokens. See
[cohort_audit.json](cohort_audit.json) and [cache_inventory.json](cache_inventory.json).

All 500 watermarked completions match. Their existing raw-completion traces
reproduce all **6,000 per-prefix score dictionaries exactly**: 500 responses ×
six lengths × two weighting rules. No watermarked inference is needed.

## Frozen job

| Item | Setting |
|---|---|
| New replay | 500 original shared nulls, `_nulls/qwen3_8b_base/T13088` |
| Input | Raw 1,024-token completion IDs and original partition only; no prompt, special-token prefix, or saved generation probabilities |
| Detector | Existing `modal_run.RedetectionModel` and `qwen.completion_only_partition_trace_batch`, unchanged |
| Model | BF16 Qwen/Qwen3-8B-Base, revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; original pinned weight/tokenizer checks |
| Batches | Four null batches of 125, sequentially on one H100; 4 CPU cores, 64 GiB RAM |
| Replay length | n=1024 once per null; 1,023 probabilities per response |
| Prefix scoring | 128, 256, 400, 512, 768, 1024 from the same traces, on local CPU |
| PRC settings | Original online key/partition, eta=.05, t=3, one-shot FPR=.001, posterior and entropy weights |
| First coordinate | Abstain: score coordinate 1 is zero |
| Validation | First null batch must match an independent token-step replay exactly, plus reversed-batch/prefix consistency |
| Failure behavior | Stop on any failure; no automatic retries; 300-second timeout per GPU batch |
| Reuse | All watermarked traces and old prompted TPR controls stay unchanged |
| TextSeal | Remains paused; this command never launches TextSeal |

Prepared inputs use the integrated clean-input contract: the only tensor fields
are `tokens` and `partition`. All numerical PRC/model files checked against the
watermarked parent match. Six local tests pass, covering token-prefix identity,
trace hashes, prompt rejection, unchanged watermarked scores, publication fields,
coverage, launch approval and protection against restoring the old null cohort.
The full-cohort local verification additionally passed on the actual cached data.

Plan: [plan.json](plan.json). Prepared record identities: [prepared.json](prepared.json).
Approval SHA-256 (canonical JSON):
`099777e556dbf6a4dfdc391c55d2851c9f69737bf6205fa2fd20f93f623cf3f2`.

## Cost

Estimate **$1–$2**, using measured earlier native-8B batch times of roughly
111–113 seconds at n=1280 (224 seconds for the independently validated batch).
The four n=1024 batches plus first-batch validation project to about 7–8 minutes
of replay, plus checkpoint loading, image and storage overhead.

At [Modal's current listed rates](https://modal.com/pricing), H100 + 4 CPU cores +
64 GiB costs approximately $0.00129148/second. Allow **$3 for planning**; this is
not an account-level hard billing cap. One container, zero retries and the
per-batch timeout limit compute exposure. No region premium is selected.

## Approved command (completed)

```sh
MODAL_PROFILE=new-prc-watermark NUMBA_DISABLE_JIT=1 OMP_NUM_THREADS=1 \
  python -m baseline_comparison.prc_shared_nulls --stage run \
  --approved-plan-sha256 099777e556dbf6a4dfdc391c55d2851c9f69737bf6205fa2fd20f93f623cf3f2
```

This command was launched after approval. It uploads frozen inputs, the original
artifact, manifest and reused traces to a new cache namespace; only null batches
are sent to the GPU. It downloads the new traces, then uses the unchanged
integrated scorer locally. Every watermarked per-record score must still match
the saved comparison before publication.

Successful publication replaces only the six PRC FPR fields and explanatory
notes in `baseline_comparisons.csv`, updating the adjacent provenance with the
shared-null source and trace hashes. Other methods and every TPR field are
preserved. The original redetection reports/CSV remain untouched. The local
`before.csv` and `before.provenance.json` preserve the previous comparison state.

The comparison CSV now uses the original shared **T13088 null cohort**. Both
posterior and entropy-weighted FPRs are **0/500 at every reported length**.
All 6,000 watermarked score dictionaries and all TPR fields match the previous
comparison exactly. Replay including validation took **374.63 seconds**; estimated
resources for that measured interval were **$0.484**, excluding loading/startup,
image/storage and other billing overhead. No watermarked inference or generation ran.

Completed reports are stored in the same Modal results namespace as the traces.
Use the compressed archive identified by [result_index.json](result_index.json).
[archive_verification.json](archive_verification.json) records successful readback
of the archive and SHA-256 verification of all nine report members.

To recover completed remote results after a local interruption, use
`--stage collect`; it starts no compute. If the comparison table or provenance
has changed since preparation, publication stops for an explicit merge review.
