from importlib import import_module

REGISTRY = {
    "arc_easy": ("arc_easy", "ARCEasy"),
    "gsm8k": ("gsm8k", "GSM"),
    "hellaswag": ("hellaswag", "HellaSwag"),
    "mmlu": ("mmlu", "MMLU"),
    "ifeval": ("ifeval", "IFEval"),
}


def get_benchmark(name, **kwargs):
    module, class_name = REGISTRY[name]
    task = getattr(import_module(f"{__package__}.{module}"), class_name)(**kwargs)
    task.load()
    return task
