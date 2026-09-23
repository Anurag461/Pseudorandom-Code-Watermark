# Result archive

This directory records the completed run, its frozen setup, approvals, output file identities, per-record reports, timing and billing. `RESULTS.md` is the entry point. The result rows are also in `../redetection/redetection_results_summary.csv`.

The exact executed runtime is preserved in `execution_sources/`, with SHA256 hashes in `setup.json` and `execution_source_index.json`. These snapshots are authoritative for this working-tree experiment. Root-level working files may contain other ongoing work.

Large tensor inputs, key artifacts and detector traces follow the repository's existing `*.pt` ignore policy and remain in the Modal result volume. Their volume paths, byte sizes and SHA256 values are recorded by the collected/result manifests. Model weights remain in the model cache. JSON reports and manifests are included here.

This publication performed only file and metadata checks. No cloud computation, numerical rescoring, generation or paid validation was launched. Completed-run attempt markers prevent accidental resubmission.
