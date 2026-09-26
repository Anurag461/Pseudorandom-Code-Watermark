# Diversity

The comparison uses Qwen3-8B-Base, 50 prompts, two 1,024-token responses per prompt, sampling seeds 12345 and 67890, and fixed watermark keys. `settings.json` specifies the methods and metric definitions.

`results.csv` contains means, 95% confidence intervals, and adjusted significance tests. `paired_inputs.json` contains the prompt-level measurements. `paired_tests_holm.csv` contains all 35 paired tests; `statistics.json` specifies the permutation and Holm procedure.

```sh
python -m experiments.diversity.run --setting synthid_depth30 --model-directory /cache/models/Qwen3-8B-Base --prompts data/prompts.jsonl --batch-size 50 --output /results/diversity/synthid_depth30
```

Use the diversity execution environment in `execution/environments.json`. The run command lists the available settings with `--help`. PRC additionally requires `--artifact` with its saved partition and online key. Generated response pairs are scored with `experiments.diversity.metrics` and `experiments.diversity.detect`; `experiments.diversity.analyze` implements the paired statistical tests. TextSeal detection uses the `textseal` execution profile on H100 with BF16 eager attention and its verified checkpoint.
