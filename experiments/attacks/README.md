# Attacks

`results/blackbox_results.csv` contains red-green tests at H=4 and H=5, their Bonferroni correction, and fixed-sampling tests. `results/substitution_results.csv` contains counts and rates at 400 and 4,096 tokens, including 30% substitution for every method. The 4,096-token baselines use 200 texts per class; PRC uses 500. Settings and source revisions are in `settings.json`.

```sh
python -m experiments.attacks.run substitute --input completions.jsonl --rate 0.1 --seed 0 --output attacked.jsonl
python -m experiments.attacks.run blackbox --settings blackbox.json --output /results/blackbox
python -m experiments.attacks.run analyze-blackbox --output /results/blackbox
```

Substitution selects a fixed number of distinct positions and samples replacement IDs uniformly; a replacement may equal the original token. Score PRC outputs with the detection runner. `python -m experiments.attacks.baseline_substitution` provides the EXP, KGW, and SynthID `generate`, `reference`, and `score` stages. Each takes `--settings` and `--output`; the 400-token reference uses `data/human_reference.jsonl`.

Black-box settings contain `scheme`, `model_directory`, and, for PRC, `artifact`. The schemes are `none`, `prc`, `kgw2`, `synthid`, and `exp`.

`stealing.py` prepares attacker prompts, generates query responses and spoofed texts, scores detection and perplexity, and builds the result table. `results/stealing_results.csv` contains spoofing results by method, context, query budget, and boost strength, including detection counts and counts passing the perplexity filter. Detection thresholds are calibrated on 5,000 unwatermarked texts; the quality cutoff is the 95th percentile of perplexity on 500 calibration texts.

Set the model directories in the `stealing` section of `settings.json`, then run:

```sh
for stage in prepare generate spoof score perplexity summarize; do
  python -m experiments.attacks.stealing "$stage" --settings experiments/attacks/settings.json --output runs/stealing
done
```

The runner uses 400-token completions, 500 evaluation prompts, the full context and boost grid at 10,000 queries, and the selected attacks at 1,000, 3,000, and 30,000 queries. Individual workers can select `--scheme`, `--split`, `--start`, `--stop`, `--name`, `--variant`, `--n-query`, and `--alpha`; `--help` lists the options.
