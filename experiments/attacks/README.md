# Attacks

`results/blackbox_results.csv` contains red-green tests at H=4 and H=5, their Bonferroni correction, and fixed-sampling tests. `results/substitution_results.csv` contains counts and rates at 400 and 4,096 tokens, including 30% substitution for every method. The 4,096-token baselines use 200 texts per class; PRC uses 500. Settings and source revisions are in `settings.json`.

```sh
python -m experiments.attacks.run substitute --input completions.jsonl --rate 0.1 --seed 0 --output attacked.jsonl
python -m experiments.attacks.run blackbox --settings blackbox.json --output /results/blackbox
python -m experiments.attacks.run analyze-blackbox --output /results/blackbox
```

Substitution selects a fixed number of distinct positions and samples replacement IDs uniformly; a replacement may equal the original token. Score PRC outputs with the detection runner. `python -m experiments.attacks.baseline_substitution` provides the EXP, KGW, and SynthID `generate`, `reference`, and `score` stages. Each takes `--settings` and `--output`; the 400-token reference uses `data/human_reference.jsonl`.

Black-box settings contain `scheme`, `model_directory`, and, for PRC, `artifact`. The schemes are `none`, `prc`, `kgw2`, `synthid`, and `exp`.

`stealing.py` implements context-count learning and logit boosts. `results/stealing_results.csv` contains spoofing results by method, context, query budget, and boost strength, including detection counts and counts passing the perplexity filter. Detection thresholds are calibrated on 5,000 unwatermarked texts; the quality cutoff is the 95th percentile of perplexity on 500 calibration texts.
