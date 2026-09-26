# Pseudorandom Code Watermarks

Implementation and experiments for fixed-block and online pseudorandom code watermarks for language models.

## Setup

Use Python 3.11 and a CUDA environment for generation and model-based detection.

```sh
python -m pip install -e '.[quality,analysis,cloud]'
python -m unittest discover -s tests
```

The execution environments and baseline source revisions are specified in [execution/environments.json](execution/environments.json). Diversity generation and TextSeal detection use their own pinned environments.

## Experiments

| Directory | Experiments |
| --- | --- |
| [detection](experiments/detection/README.md) | Detection power, model size, detector portability, fixed versus online, entropy, Wang comparison |
| [quality](experiments/quality/README.md) | ARC-Easy, GSM8K, HellaSwag, MMLU, IFEval |
| [diversity](experiments/diversity/README.md) | Self-BLEU, repeated 4-grams, Distinct-3, detection, repeat handling |
| [attacks](experiments/attacks/README.md) | Black-box tests, token substitution, watermark stealing |

Each experiment directory contains its settings, numerical result tables, and source checksums. Rates retain their sample counts where available; diversity tables include confidence intervals and Holm-adjusted p-values.

`prc_watermark/` contains the constructions, generation, detectors, and Qwen implementation. `baselines/` contains comparison methods. `data/` contains the tokenized prompts, reference texts, and saved PRC keys.

Generate LaTeX tables from the saved results:

```sh
python paper/build.py --output runs/paper
```

The figures used in the manuscript are in `paper/figs/`. Add `--plots` to render detection curves from the saved counts.

Use `python -m execution.cloud --help` for the cloud runner.
