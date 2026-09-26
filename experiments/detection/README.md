# Detection

`fixed_results.csv` contains the 19 fixed-block settings. `figure_results.csv` contains the detection curves, model-size comparison, detector portability, noise-rate contrast, and fixed-versus-online comparison. Counts use 500 watermarked completions per setting. `portability_*` and `construction_*` contain paired outcomes and tests.

`entropy_results.csv` measures vocabulary and bucket entropy on 500 unwatermarked continuations of 1,808 tokens, conditioned on the original prompt. `wangetal_*.csv` and `wangetal_thresholds.json` contain the hierarchical PRC comparison and its calibration.

Generation and scoring run separately:

```sh
python -m experiments.detection.run generate --settings generation.json --output runs/detection
python -m experiments.detection.run score --settings scoring.json --output runs/detection/scored
```

Generation settings specify `artifact`, `prompts`, `model_directory`, `model_size`, `prompt_indices`, `batch_size`, `sampling_seed`, `construction` (`fixed` or `online`), `tokens`, and `kv_cache` (`concat` or `static`). Scoring settings specify `artifact`, `completions`, `detector_directory`, `detector_size`, `batch_size`, `construction`, `lengths`, `fpr`, and `kv_cache`.

Saved keys for the fixed detection sweep and online settings are in `data/keys/`. Artifacts contain the original `partition` and either `online_key` or `encoding_key`/`decoding_key`. Model directories contain the checkpoint shards and tokenizer. Batch geometry and cache implementation are part of the numerical setup.

For the comparison with Wang et al.’s PRC implementation, use `python -m experiments.detection.wangetal_comparison` with `prepare`, `generate`, or `analyze`; `--help` lists the arguments. Its model checksums and prompts are in `wangetal_model.json` and `wangetal_prompts.json`; the design is in `wangetal_comparison.py`.
