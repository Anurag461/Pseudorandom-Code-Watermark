from benchmarks.gsm8k import GSM
from benchmarks.arc_easy import ARCEasy
from benchmarks.arc_challenge import ARCChallenge
from benchmarks.gpqa_diamond import GPQADiamond
from benchmarks.aime import AIME2024, AIME2025
from benchmarks.hellaswag import HellaSwag
from benchmarks.mmlu import MMLU
from benchmarks.ag_news import AGNews
from benchmarks.ifeval import IFEval

# name -> Task subclass. Names are the canonical keys used on the CLI / eval code.
REGISTRY = {
    "gsm8k": GSM,
    "arc_easy": ARCEasy,
    "arc_challenge":ARCChallenge,
    "gpqa_diamond": GPQADiamond,
    "aime24": AIME2024,
    "aime25": AIME2025,
    "hellaswag": HellaSwag,
    "mmlu": MMLU,
    "ag_news": AGNews,
    "ifeval": IFEval,
}

def available_benchmarks():
    """Sorted list of registered benchmark names."""
    return sorted(REGISTRY)

def get_benchmark(name, load=True, **kwargs):
    """Instantiate a benchmark by name.

    Args:
        name: registry key, e.g. "gsm8k" or "arc_easy".
        load: if True, call .load() to fetch the datasets before returning.
        **kwargs: forwarded to the Task constructor (start, stop, step).

    Returns:
        A ready-to-use Task instance.
    """
    key = name.lower()
    if key not in REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {available_benchmarks()}"
        )
    benchmark = REGISTRY[key](**kwargs)
    if load:
        benchmark.load()
    return benchmark
