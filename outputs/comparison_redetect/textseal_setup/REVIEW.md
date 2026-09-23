# TextSeal setup: historical reuse pilot

This is the preserved original longest-trace reuse setup. Its pilot stopped
on null/0000 because n=128 entropy and weighted scores did not exactly match
a direct upstream replay. Only one record completed; no full run launched.
The earlier user instruction to pause superseded the original full-run request.

The discrepancy is now diagnosed as BF16 matrix-shape dependence. See the
[diagnostic and resolution](prefix_diagnostic/REVIEW.md) and the
[current direct-prefix setup](direct_prefix/REVIEW.md). The replacement
pilot passed all 60 comparisons with upstream. The subsequently authorized
full cohort is now complete and published; see the current setup above.

The original [manifest](native8b_manifest.json) is retained with canonical SHA
`0b409b197985987c7ac5b14202dd2a9ec922c457525cc50a5c1edceb13dcc07a`.
[execution.json](execution.json) records both the initial import failure
(before inference) and failed prefix-reuse pilot. Production now rejects this
old manifest. The diagnostic folder includes the exact old source snapshot
for historical reproduction in an isolated checkout.

PRC shared-null alignment is already complete on the same original T13088
cohort. Its watermarked results and TPRs were preserved. See the
[PRC verification](../prc_shared_nulls/verification.json). This TextSeal work
has not changed the shared comparison CSV.
