# Online 8B eta=0.15, N=500: shorter-prefix scoring proposal

Status: completed. Added 186 rows at N=500; first native MAP below 90% at T=4656. Actual reported cost $0.00309468. See RESULTS.md.

Reuse the completed T=6144 raw-completion traces for all 500 prompts: five native 8B shards and ten 0.6B shards, spanning the original 100 plus the new 400. No generation, replay, GPU, null, or independent validation pass.

Starting at 6128, score every 16-token prefix in descending order through the first native 8B MAP result strictly below 450/500 (90%). Include that point. Score 0.6B MAP and entropy at exactly those same lengths. Reuse the existing T=6144 scores without recomputation or duplicate CSV rows. This uses prefixes of the completed T=6144 cohort, not the separate historical T=4096 campaign.

Use the established completion_only_raw_abstain_v1 protocol, coordinate-one abstention, original online key/partition, t=3, and one-shot target FPR=0.001. Preserve the corrected native/scientific import order before NumPy aliases. The source manifests and hashes are frozen in setup.json.

One cloud CPU worker, 4 cores and 8 GiB RAM, processes the existing trace batches (8B: 5 x 100; 0.6B: 10 x 50). Native scoring determines the shared stopping length before 0.6B scoring. Use the existing prepared_weights/adaptive_scores and prefix_scores routines through a small isolated wrapper; no detector or numerical changes. No automatic retry. A 600-second work deadline bounds the run.

Expected time: 1–3 minutes. Expected charge: $0.005–0.02; conservative requested allowance: $0.10. Timing references: 500 prompts across 90 prefixes took 17.62 seconds in the existing CPU routine; the latest 800 detector-records at one length took 23.80 seconds. The wider grid and cache I/O are covered by the margin. Previous experiment actual spending is $28.68301936; including the entire allowance gives $28.78301936, leaving $6.32698064 below the prior $35.11 target. This is task-budget headroom, not a claim about account credits.

Append paired native/0.6B rows to the existing CSV, with N=500 and null N=0 explicit, retaining all 431 current rows and the existing schema. Save per-prompt scores, cache provenance, timing and actual billing; commit only the resulting task files on redetection.

AGENTS.md requires an estimate and explicit approval for every new paid CPU scoring run. Approval for this exact CPU-only scope and the $0.10 allowance is recorded in approval.json.
