# Both detector manifests prepared

The saved generation batch was converted into 100 existing-format watermarked
records. Two completion-only manifests now reference exactly the same ordered
records, token hashes, key artifact, partition and input hash. Each has one
batch of 100 completions of length 1024. Both use
`completion_only_raw_abstain_v1`, static replay, MAP plus entropy, target
FPR=.001, and no null cohort.

The GPU input exports contain only completion tokens and the partition. They
exclude original prompts and generation-time probabilities. Raw completion
coordinate 1 will abstain; coordinates 2–1024 will be recovered separately by
each detector. Neither detector has executed yet.

- Worker: four CPU cores, 16 GiB RAM, no GPU; 12.324 seconds.
- App: [ap-VC6JnhZbVzB2aElnjMWqei](https://modal.com/apps/new-prc-watermark/main/ap-VC6JnhZbVzB2aElnjMWqei)
- Provider-reported cost: **$0.00229091**.
- All 108 returned files downloaded and SHA-256 verified locally.
- 4B run: `completion_only_raw_abstain_v1/integrated/ca29c3c412c2ea56a5b60c7b`.
- 0.6B run: `completion_only_raw_abstain_v1/integrated/2d9e999b8f2ea462bbcfbf16`.

Files and cloud/local locations are listed in `cache_index.json`. Detailed
inputs and derived records remain in local and Modal caches; the original
primary generation and artifact were already committed in `a9f3e42`. No new
archive was uploaded, no optional checks ran, and no generation was repeated.

The updated provider report `billing_after_freeze.json` gives:

| Completed stage | Actual reported cost |
|---|---:|
| CPU checkpoint/artifact preparation | $0.00988003 |
| 4B generation | $0.17829656 |
| CPU manifest preparation | $0.00229091 |
| Total | **$0.19046750** |

Generation billing settled above its first $0.16279311 snapshot. The remaining
budget estimate is **$6.31**, including the previously identified separate
$0.01946503 diagnostic usage.

Next: one primary 4B completion-only replay on one H100 80GB, four CPU cores,
64 GiB RAM, batch100, estimated 2–4 minutes and $0.16–0.31, with a $0.90
allowance. It will save the primary trace without a reference pass or retry.
This next paid stage is awaiting explicit approval.
