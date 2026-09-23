# Final repository cleanup

The comparison campaign is closed. This cleanup changes organization and
publication assets; it does not change numerical experiment implementations.

- Added one canonical report/publication package in `reports/comparisons/`,
  with highlighted takeaways, a complete experiment ledger, explicit cohort
  boundaries, all completed settings, corrected null counts and limitations.
- Replaced the chronological `self_bleu/README.md` and 917-line `plan.md` with a
  compact results/source index and a closed ledger. Their prior contents remain
  in Git at `69c7ea2`; completed-run reports and runbooks are untouched.
- Added links from the root README and baseline-comparison README so the final
  account is easy to find from either package.
- Kept generation, detector, worker and historical analysis files at their
  original paths: completed manifests pin their hashes. Moving/merging them
  during publication cleanup would impair reproducibility.
- Kept raw outputs, model traces, keys, archives and unrelated untracked
  manuscripts/experiments unchanged. No cache, result or source snapshot was
  deleted. Only an obsolete table export created during this report build was
  removed when its diagnostics table was split into three clearer tables.
- Consolidated paper export into one offline builder. Figures have matching
  PDF/SVG/PNG files; tables have matching LaTeX/CSV files. Full-precision values,
  source hashes and cross-report consistency checks accompany the report.
- TeX build intermediates and the reproducible convenience ZIP are ignored;
  temporary rendering/QA files are outside the repository. Cleaned trailing
  whitespace in the existing root ignore file without changing its rules.

No new model generation, model replay, scoring experiment or Modal dispatch
was performed. The user closed the study; no further experiment is pending.
