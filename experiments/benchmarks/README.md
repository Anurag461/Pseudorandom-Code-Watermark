# Benchmarks

The five benchmarks use Qwen3-0.6B with reasoning enabled, temperature 1, full-vocabulary sampling, and fixed-block PRC with n=800, r=792, t=3, eta=0.1. `settings.json` gives the token limits. IFEval uses prompt-level strict scoring and pools two runs of 541 prompts.

```sh
python -m experiments.benchmarks.run --benchmark gsm8k --model-directory /cache/models/Qwen3-0.6B --artifact /data/artifact.pt --batch-size 8 --seed 12345 --output /results/benchmarks/gsm8k
```

The runner saves responses, token IDs, example scores, and the sampled key. `results/results.csv` contains the saved benchmark aggregates, including truncation rates and median output lengths. Model and dataset downloads must be available in the execution environment; IFEval additionally requires NLTK `punkt` and `punkt_tab` data.
