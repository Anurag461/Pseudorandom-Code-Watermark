from benchmarks.task import Task, HFDataset
import re
import numpy as np

# AIME answers are integers in [0, 999]. Robust extraction mirrors gsm8k: scope
# to the post-</think> answer, then try the last \boxed{N} -> "answer is N" ->
# the last integer in that segment.
_INT = r"(-?\d{1,4})"
AIME_BOXED_RE = re.compile(r"\\boxed\{\s*" + _INT)
AIME_PHRASE_RE = re.compile(r"answer\b[^0-9\-]{0,20}?" + _INT, re.I)
AIME_NUM_RE = re.compile(_INT)


def _normalize_int(s):
    """Canonicalize to a plain integer string (so '007' == '7', '33' == 33)."""
    if s is None:
        return None
    try:
        return str(int(str(s).strip()))
    except (TypeError, ValueError):
        return None


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]

    boxed = AIME_BOXED_RE.findall(tail) or AIME_BOXED_RE.findall(completion)
    if boxed:
        return _normalize_int(boxed[-1])

    phrase = AIME_PHRASE_RE.findall(tail)
    if phrase:
        return _normalize_int(phrase[-1])

    nums = AIME_NUM_RE.findall(tail)
    if nums:
        return _normalize_int(nums[-1])

    return None


class _AIMEBase(Task):
    """Shared AIME task: integer-answer competition math, zero-shot."""

    DATASET = None          # HF repo id
    CONFIG = None           # HF config name (or None)
    SPLIT = "test"
    PROBLEM_KEY = "problem"
    ANSWER_KEY = "answer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = (
            "You are given a competition mathematics problem from the AIME. The "
            "answer is an integer between 0 and 999. Think step by step, then give "
            "only the final integer inside \\boxed{} .\n"
        )

    def load(self):
        self.ds = HFDataset(self.DATASET, self.CONFIG, split=self.SPLIT)

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        return "\n".join([self.system_instr, "Problem:", row[self.PROBLEM_KEY]])

    def get_example(self, idx: int, fewshot=False):
        prompt = self.create_prompt(self.ds[idx], fewshot=fewshot)
        return {"messages": [{"role": "user", "content": prompt}]}

    def evaluate(self, idx, assistant_response):
        gold = _normalize_int(self.ds[idx][self.ANSWER_KEY])
        pred = extract_answer(assistant_response)
        return int(gold is not None and gold == pred)


class AIME2024(_AIMEBase):
    DATASET = "Maxwell-Jia/AIME_2024"
    CONFIG = None
    SPLIT = "train"          # this repo ships the 30 problems under 'train'
    PROBLEM_KEY = "Problem"
    ANSWER_KEY = "Answer"


class AIME2025(_AIMEBase):
    DATASET = "math-ai/aime25"
    CONFIG = None
    SPLIT = "test"
    PROBLEM_KEY = "problem"
    ANSWER_KEY = "answer"


if __name__ == "__main__":
    for cls in (AIME2024, AIME2025):
        benchmark = cls(start=0, stop=None, step=1)
        benchmark.load()
        scores = [
            benchmark.evaluate(i, f"\\boxed{{{benchmark.ds[i][cls.ANSWER_KEY]}}}")
            for i in range(benchmark.num_examples())
        ]
        print(f"{cls.__name__} oracle score: {np.mean(scores)}  (n={benchmark.num_examples()})")
