# Next run: finish the detector comparison

Prepared and approved September 24, 2026; **executed successfully**. See the
[completed report](evidence/experiment-20260924/README.md). The quote and execution
instructions below are retained as the record of the approved package. The user
asked to prioritize the important results. Proceed directly to the existing
ten-key, five-temperature experiment, using the completed source and cache/FP32
checks. Do not spend another run on the remaining short T=1.8 smoke.

## Workload and outputs

Use the pinned cached Qwen3-8B-Base, BF16, reasoning off, and the unchanged Wang
sampler. Generate 800 watermarked and 800 null completions of 1024 tokens over
T=1.0, 1.2, 1.4, 1.6, 1.8: ten keys/seed groups and sixteen prompts at each T.
Replay completion IDs only; first-token posterior evidence is zero. There are
1,638,400 generated tokens and 1,636,800 replay positions, excluding prompt prefill.

Produce the primary published-Wang versus standard-posterior comparison, the
separate matched-FPR ablation, held-out FPR, group/prompt bootstrap intervals,
ROC/AUC and entropy plots. Calibrate on the designated 400 null completions crossed
with 256 independent keys, and evaluate on 400 held-out nulls crossed with 256
disjoint keys. Freeze thresholds before watermarked scoring. The main conclusion
must distinguish a threshold advantage from an advantage in posterior scoring,
especially at T=1.0, 1.2 and 1.4.

## Exact paid calls covered by the proposed approval

| Component | Hardware and batching | Function limit | Estimate |
| --- | --- | --- | --- |
| Preparation: 10 WM keys, 160 codewords, 512 null keys | One CPU worker, 4 cores, 16 GiB | 15 min | $0.10–$0.40 |
| Generation and replay: five temperatures | Five H100 workers, each 4 cores/64 GiB host RAM, batch 80; four generation and four replay batches per worker | 60 min per worker | $9–$22 total |
| All CPU scoring, intervals and report artifacts | One CPU worker, 8 cores, 16 GiB; null keys scored in bounded chunks | 60 min | $0.50–$2 |

**Planning total: $12–$25; about 1–2 hours elapsed**, subject to queues and startup.
The allowance includes startup. Function-time charges at all timeout limits total
about $23.83; the estimate is not a provider-enforced dollar cap. No automatic
retries, additional validation runs, oracle analysis or follow-up experiment.
Preparation precedes the five GPU calls; CPU scoring runs only if all five finish.
On a GPU failure, cancel the sibling calls and stop with saved caches.

Rates checked September 24 at [Modal pricing](https://modal.com/pricing):
H100 $0.001097/s, CPU $0.0000131/core/s, host RAM $0.00000222/GiB/s.
Prior checks have now both posted: $0.14605021 + $0.13855895 = **$0.28460916**.
Against the last stated $35 experiment budget, this package would leave
approximately **$9.72–$22.72**. Unrelated concurrent projects are not counted
against this experiment estimate. Billing evidence is
`evidence/experiment-20260924/billing_review.json`.

## Validation decision

The original hard-detector/source check passed all 160 individual comparisons.
Static and concatenating caches produced identical logits in the saved-token
controls, and FP32 cached/uncached comparisons passed. These are sufficient to
proceed with the specified BF16 experiment; generation, replay and detector code
remain unchanged. `validation.py` checks those implementation hashes and binds
the evidence hashes into this run's quote.

The original 2% BF16 cached/uncached guard **did fail** (maximum measured TV 4.04%).
Its failure remains recorded; this package explicitly uses source/cache/FP32
controls as the prerequisite instead. The controls tested one completion and
prefix lengths 1, 4, 8, 16, and do not establish batch-80 or long-context numerical
equivalence. Report this limit alongside the results. No new paid numerical check
is included, and no success is claimed for the unrun short T=1.8 smoke.

## Commands

Setup must be committed and pushed on `cryptoanalysis-redetection` before launch.
Printing the quote is free and does not import Modal:

```bash
cd /private/tmp/prc-cryptoanalysis-redetection
python -m wang_prc_detector_ablation.launch quote --stage experiment
```

After explicit approval of this entire package, record the actual approval text,
exact quote hash, commit/fingerprint, billing review, and unique run ID in the
ignored approval file. Include `validation_policy: source-and-cache-fp32-controls-v1`
and `included_sequence: [prepare, production x5, score]`; no oracle flag.

```bash
MODAL_PROFILE=new-prc-watermark PYTHONUNBUFFERED=1 \
python -m wang_prc_detector_ablation.launch launch --stage experiment \
  --approval wang_prc_detector_ablation/approvals/experiment-20260924.json
```

The approval is consumed once. Failed or interrupted work does not authorize
another attempt. Download the completed CSV/JSON/PDF/PNG artifacts and reconcile
provider billing after the run; no local model or dataset scoring is required.
