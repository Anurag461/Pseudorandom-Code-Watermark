# Wang temperature-sweep re-detection: source-data gate

**Historical preflight: blocked before hard-detector reproduction. No paid run was launched.**
The user subsequently approved a fresh Qwen3-8B-Base Wang-channel experiment; see
[`../wang_prc_detector_ablation/README.md`](../wang_prc_detector_ablation/README.md).
The findings below remain the record of why the original saved-sweep task stopped.
This directory implements artifact inspection, provenance, schema diagnostics, and
an eight-group preflight. It does **not** implement or claim completed LM replay,
calibration, detector evaluation, or plots. The requested instruction to stop if
the original detector cannot be reproduced takes precedence over continuing with
incomplete source data.

The downloaded official artifact is pinned to
[`1234wangtr/PRC_estimator@8593e86`](https://github.com/1234wangtr/PRC_estimator/tree/8593e86aeb50b5f82d6c88e390b12a30f581dbaa).
The paper is [arXiv:2512.17310v4](https://arxiv.org/html/2512.17310v4).
The specified archive `llm/data/Deepseek_t_3_temp_all.zip` has SHA-256
`f8ac4b3a45a533f0ea31d579d4125b3f36d19e791ad2d4192d6db927a6f40f91`,
31,773,590 compressed bytes, and 56,066,828 bytes across 320 JSON files.
No watermarked text was generated or retokenized.

## Observed JSON schema

The ZIP unpacks to `gen_result/temperature_{T}/{timestamp}.json`. There are exactly
64 files at each of T=1.0, 1.2, 1.4, 1.6, and 1.8, each containing 16 original and
16 watermarked decoded strings. These are **file groups**; their key identities
cannot be checked because keys are absent.

| Field | Type / shape | Meaning supported by the artifact |
| --- | --- | --- |
| `origin_sentence` | 16 strings | Saved original/unwatermarked decoded completions |
| `watermark_sentence` | 16 strings | Saved watermarked decoded completions |
| `correct_rate` | 16 floats | Saved recovered-codeword bit accuracy, consumed by the official plotter |
| `avg_entropy` | 16 floats | Saved generation entropy in bits, consumed by the official plotter |
| `det` | 16 booleans | Saved watermarked detection decisions, consumed by the official plotter |
| `origin_det` | 16 booleans | Auxiliary saved flags; **not used as realized null FPR** |
| `generation_config` | Object, 69 fields | Present in all 256 files at T>1.0; absent in all 64 T=1.0 files |

`generation_config` records temperature, `max_new_tokens=1024`, `top_k=0`,
`top_p=1.0`, `do_sample=true`, and Transformers 4.51.3. For T=1.0, the directory
and the official plotter's default supply the temperature. That default is
recorded explicitly, not treated as a saved generation configuration.

**Absent in all 320 files:** `secret_key`, `one_time_pad`,
`origin_sentence_tokens`, `watermark_sentence_tokens`, `inv_msg`, `msg`,
`public_key`, `prompt`, and `prompt_tokens`.

The pinned official `llm/generation/main.py` writes a richer schema, including
keys and token IDs, but the published temperature archive uses the smaller schema
above. The available source code does not establish the provenance of every
legacy flag. In particular, the current generator's `origin_det` variable refers
to detection of a PRC codeword before embedding; treating the archive's similarly
named field as unwatermarked FPR is unjustified.

The companion `Deepseek_t_3_temp_1.8.zip` contains ten richer groups, with keys of
shape 17510×3, OTPs of length 18432, and token ID lists. None of those ten
watermarked text groups exactly matches a temperature-sweep group. It cannot
supply the missing five-temperature data. Its inventory is in `evidence/preflight.json`.

Decoded text cannot supply the missing random keys or OTPs. Tokenizer re-encoding
also cannot establish the original token stream: the official generator decodes
with `skip_special_tokens=True`, and decoding/re-encoding need not preserve token
segmentation. No substitute keys, strings re-encoded as original IDs, or fabricated
null samples are used.

## Figure 5 sanity-check table

These counts only aggregate existing `det` flags. **They are not independently
reproduced hard-detector results.** The hard-detector sanity gate remains unmet.

| T | Paper description | Saved positives / all N | Saved rate | First 8 files: positives / N | Saved rate | Recomputed hard score |
| --- | --- | --- | --- | --- | --- | --- |
| 1.0 | No detection | 0 / 1024 | 0% | 0 / 128 | 0% | Blocked: keys and IDs absent |
| 1.2 | No detection | 0 / 1024 | 0% | 0 / 128 | 0% | Blocked: keys and IDs absent |
| 1.4 | About 60% | 609 / 1024 | 59.4727% | 82 / 128 | 64.0625% | Blocked: keys and IDs absent |
| 1.6 | Increasing detection | 1021 / 1024 | 99.7070% | 127 / 128 | 99.2188% | Blocked: keys and IDs absent |
| 1.8 | Increasing detection | 1024 / 1024 | 100% | 128 / 128 | 100% | Blocked: keys and IDs absent |

The eight-group preflight inspected all 320 files for schema completeness and
selected the first eight numeric timestamp IDs independently at each temperature.
The first four selected IDs are marked calibration, the next four evaluation.
`evidence/selected_groups.json` records every selected ID and split. This is a
proposed **file** split, pending verification of actual key identities and absence
of cross-split key reuse. No threshold was calibrated or frozen.

## Exact commands used

From the isolated repository worktree:

```bash
# Source retrieval only; no artifact generation scripts are executed.
git clone --depth 1 https://github.com/1234wangtr/PRC_estimator.git /private/tmp/wang-prc-estimator
git -C /private/tmp/wang-prc-estimator rev-parse HEAD
# Expected: 8593e86aeb50b5f82d6c88e390b12a30f581dbaa

python -m unittest cryptoanalysis_redetect.test_preflight -v

python -m cryptoanalysis_redetect.preflight \
  --archive /private/tmp/wang-prc-estimator/llm/data/Deepseek_t_3_temp_all.zip \
  --companion-archive /private/tmp/wang-prc-estimator/llm/data/Deepseek_t_3_temp_1.8.zip \
  --extract-to cryptoanalysis_redetect/data \
  --max-groups-per-temp 8 \
  --output cryptoanalysis_redetect/evidence
# Expected exit status: 2 (blocked_missing_source_data).

MODAL_PROFILE=new-prc-watermark modal billing report \
  --for today --resolution h --show-resources --json
```

If the upstream default branch changes, fetch and check out the pinned commit
before running inspection. The preflight rejects any other archive digest.
Omitting `--max-groups-per-temp` inventories/selects all available files; it never
launches compute. Metadata inspection and eight tiny synthetic unit tests were
performed locally. No dataset scoring or model execution was performed locally.

The evidence files contain schema shapes, source hashes, counts, proposed splits,
and blockers. They contain no generated detector results. `stored_flags_summary.csv`
is deliberately distinct from the requested `results_summary.csv`.

## Required source supplement

Provide a source URL or directory whose records map unambiguously to the sweep's
`temperature`, original filename/group ID, and prompt index 0–15, containing:

1. The corresponding secret parity-check supports (17510 rows, 3 indices each)
   and 18432-bit OTP for every file group.
2. Exact original token IDs for all saved watermarked and unwatermarked
   completions, including any generated special tokens. Recovered `inv_msg.val`
   bits are useful as an independent equality check when available.
3. Saved prompt token IDs/chat-template provenance for the optional oracle only.

Public generator matrices and encoded pre-channel messages are not necessary
for the four requested scores. The required checkpoint is exactly
`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`, revision
`916b56a44061fd5cd7d6a8fb632557ed4f724f60`, also pinned by the official
[`setup/get_llm.sh`](https://github.com/1234wangtr/PRC_estimator/blob/8593e86aeb50b5f82d6c88e390b12a30f581dbaa/setup/get_llm.sh).
Parameters: 1024 completion tokens, vocab 152064, 18 MSB-first bits/token,
n=18432, r=17510, t=3, eta=0.1.

## Run design once source data is available (not implemented or approved)

1. **CPU reproduction gate:** validate shapes, key identities, original token
   lengths, recovered MSB-first bits, and the saved hard decisions. Apply the
   official inclusive decision `violations <= r/2 - r**0.75`. Compare full
   temperature rates with Figure 5, and require per-example agreement where saved
   decisions exist. Stop with diagnostics on mismatch, before GPU allocation.
2. **Primary replay:** teacher-force raw completion IDs using the exact pinned
   checkpoint in BF16, with float32 logits/softmax and float64 accumulation where
   needed. Input for token i contains only completion IDs strictly before i.
   Prepend no original prompt, BOS, EOT, or chat template. Generated special-token
   IDs already in the saved completion are retained. Set all 18 soft coordinates
   at token 0 to zero. Scale logits by each sample's temperature, with no top-k and
   top-p=1. Recover the observed branch's hierarchical conditional probabilities
   using interval intersections with the actual vocabulary; never use bit
   marginals. Cache token IDs, per-bit p1, entropy traces, sample identity,
   context mode, checkpoint/code hashes, precision, and replay metadata.
3. **CPU-only scoring from cache:** original Wang violation count, empirically
   calibrated hard violation count, observed sign × normalized binary entropy,
   and the existing `detectors.py::map_soft_token` formula. For each soft score,
   save the numerator, V, and Z with the specified OTP parity sign and 1e-12
   denominator stabilizer. Unknown/missing coordinates must not be silently
   padded into a full hard score; verify saved completion lengths before fixing
   null-score policy and workload.
4. **Calibration and evaluation:** use only calibration nulls to set a single
   global threshold per calibrated detector at target FPR 0.001. With eight
   groups at each temperature and distinct keys, there are 20 calibration keys
   and 320 calibration null completions. Scoring each against all 20 calibration
   keys yields 6400 calibration pairings without additional LM calls. Keep the
   original Wang threshold fixed. Freeze thresholds before evaluating held-out
   watermarked completions against their own keys. Report held-out true-key null
   FPR and all-held-out-key augmented FPR separately; cross-key pairs are not
   independent observations. Bootstrap key groups, accounting for repeated
   trace/key use in cross-key estimates, not individual pairs. Check key reuse
   across temperatures before accepting the split.
5. **Reporting:** preserve per-example scores, decisions, entropy diagnostics,
   hard violations and source metadata; write `results_summary.csv`,
   `tpr_vs_temperature.pdf/png`, and `score_vs_entropy.pdf/png`. Report paired
   key-group bootstrap differences for MAP at T=1.0, 1.2, 1.4. Any definition of
   “material improvement” must be fixed before viewing those results. The
   optional `--oracle-prompt-context` belongs to a separately labeled diagnostic
   cache and must never replace or tune primary results.

The selected files hold 640 saved watermarked and 640 saved null strings. Exact
LM workload, batching, hardware, and runtime cannot yet be committed because the
original token streams and their lengths are missing. No Modal launcher is
provided while the source-data gate is blocked. After the missing inputs are
validated, implement and test the replay/scoring stages, commit and push them,
then present the exact CPU/GPU workload, estimated time/cost, and remaining-budget
impact for explicit approval. Every new paid run, validation pass, or retry needs
its own approval. The full 64-group experiment must never launch automatically.

## Cost and current conclusion

Actual task usage: **0 Modal runs, 0 GPU seconds, $0 paid compute**. There is no
actual replay run from which to estimate GPU throughput or full-run cost.
`evidence/billing_snapshot.json` records the read-only billing check: gross usage
in `new-prc-watermark` for September 20 UTC totaled $35.08963943 at inspection.
This is workspace usage across other apps, not a credit balance and not spending
attributable to this task. It cannot be subtracted reliably from the user's
stated $20 remaining without knowing the balance's reference time and later
charges. Refresh billing before any paid proposal.

**MAP improvement at T=1.0, 1.2, 1.4 is unknown.** The required experiment was
stopped at source validation, so no TPR, realized FPR, confidence intervals, or
comparison plots are claimed.
