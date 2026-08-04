from benchmarks.gsm8k import GSM
from benchmarks.arc_easy import ARCEasy
from benchmarks.arc_challenge import ARCChallenge

# name -> Task subclass. Names are the canonical keys used on the CLI / eval code.
REGISTRY = {
    "gsm8k": GSM,
    "arc_easy": ARCEasy,
    "arc_challenge":ARCChallenge,
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
