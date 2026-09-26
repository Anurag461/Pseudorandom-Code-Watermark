# A practical and undetectable watermark for language models using pseudorandom codes

## Setup

Use Python 3.11 and a CUDA environment for generation and model-based detection.

```sh
python -m pip install -e '.[benchmarks,analysis]'
python -m unittest discover -s tests
```

The execution environments and baseline source revisions are specified in [experiments/environments.json](experiments/environments.json). Diversity generation and TextSeal detection use their own pinned environments.

## Experiments

| Directory | Experiments |
| --- | --- |
| [detection](experiments/detection/README.md) | Detection power, model size, detector portability, fixed versus online, entropy, comparison with Wang et al.’s PRC implementation |
| [benchmarks](experiments/benchmarks/README.md) | ARC-Easy, GSM8K, HellaSwag, MMLU, IFEval |
| [diversity](experiments/diversity/README.md) | Self-BLEU, repeated 4-grams, Distinct-3, detection, repeat handling |
| [attacks](experiments/attacks/README.md) | Black-box tests, token substitution, watermark stealing |

Each experiment directory contains its settings and a `results/` folder with numerical tables and source checksums.

```text
.
├── README.md
├── pyproject.toml
├── prc_watermark/
├── baselines/
├── experiments/
│   ├── detection/
│   │   └── results/
│   ├── benchmarks/
│   │   └── results/
│   ├── diversity/
│   │   └── results/
│   └── attacks/
│       └── results/
├── data/
├── paper/
└── tests/
```

Generate LaTeX tables from the saved results:

```sh
python paper/build.py --output runs/paper
```

The figures used in the manuscript are in `paper/figs/`. Add `--plots` to render detection curves from the saved counts.
