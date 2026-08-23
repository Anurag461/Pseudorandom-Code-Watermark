# Controlled 8B baseline handoff

Status: five-prompt smoke complete; 500-prompt generation not launched.

The integration is in `baseline_comparison/`; focused local tests are in
`tests/test_baseline_comparison.py`. The authoritative compact outputs are:

- `outputs/controlled_baseline_smoke_prompt_level.jsonl` (258 rows);
- `outputs/controlled_baseline_smoke_prefix_summary.csv` (24 rows);
- `outputs/controlled_baseline_smoke_quality.csv` (40 rows);
- `outputs/controlled_baseline_smoke_validation.json`;
- `outputs/controlled_baseline_smoke_seed_validation.json`;
- `outputs/controlled_baseline_smoke_cost_ledger.csv`;
- `outputs/controlled_baseline_smoke_artifact_manifest.json`;
- `baseline_comparison/pinned_sources_manifest.json`.

The follow-up diagnostic adds:

- `controlled_baseline_diagnostic_report.md`;
- `outputs/controlled_baseline_diagnostic_cache_analysis.json`;
- `outputs/controlled_baseline_diagnostic_prefix_rows.jsonl` and summary CSV;
- `outputs/controlled_baseline_diagnostic_generation.json`;
- `outputs/controlled_baseline_diagnostic_generation_prefix_rows.jsonl` and
  summary CSV;
- `outputs/controlled_baseline_diagnostic_logits_parity.json`;
- `outputs/controlled_baseline_diagnostic_cost_ledger.csv`;
- `outputs/controlled_baseline_diagnostic_artifact_manifest.json`.

All four methods scored all six prefixes. Official TextSeal and Google SynthID
checks passed, exact-prefix deltas were zero, and PRC/null cache reuse generated
zero tokens. TextSeal and SynthID replayed fixed seeds and changed across
controlled second seeds. Gumbel was deterministic across seeds at equal batch
shape but sensitive to batch shape. The follow-up diagnostic showed that the
released power form and exact log-space form produce identical tokens at batch
sizes 1 and 5, so the difference is attributable to batch-dependent model
logits/numerical execution rather than power-form underflow.

The quality fields are finite, but TextSeal and Gumbel show materially high
repetition/low distinct-n on several smoke prompts. Treat that as part of the
scientific result, not an adapter failure and not a reason to change settings.
The TextSeal finding reproduced with a second complete seed and remained on
three of five prompts at the paper's `temperature=0.8`, `top_p=0.9`, 400-token
regime. The strict project-Qwen/Hugging-Face batch-5 JSD criterion failed, but
all top-1 tokens agreed and native Hugging Face showed larger batch-shape
variation. Full details and the exact `0.17113157`-dollar diagnostic spend are
in `controlled_baseline_diagnostic_report.md`.

Measured production projection: 14--18 dollars total and 20--35 minutes with
ten GPUs. The exact spend for this integration/smoke campaign is
`1.39371804` dollars. No full-run blocker remains, but explicit approval
is required. After approval, use exactly:

```bash
modal run baseline_comparison/modal_app.py::app.full-run \
  --approval-token APPROVE_500_PROMPT_CONTROLLED_BASELINE \
  --run-id qwen3-8b-controlled-20260823
```

Then run `app.full-score`, download scored shards, and invoke the streaming
finalizer exactly as documented in `controlled_baseline_full_run_runbook.md`.
Do not launch the 50x5 diversity experiment or any 27B/proxy work under this
approval.
