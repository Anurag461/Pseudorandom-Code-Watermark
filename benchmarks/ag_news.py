from benchmarks.task import Task, HFDataset
import re
import numpy as np

# AG News topic classification, run generation-style: the four fixed topics are
# options A-D and the model generates the boxed letter. Extraction mirrors
# arc_easy: post-</think> \boxed{X} -> "answer/topic is X" -> leading "X.".
AGN_RE = re.compile(r"\\boxed\{\(?([A-Da-d])\)?\}")
AGN_PHRASE_RE = re.compile(
    r"(?:answer|option|choice|category|topic)\b[^A-Da-d]{0,20}?\(?([A-Da-d])\)?\b",
    re.I,
)
AGN_LABEL_RE = re.compile(r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*[.):-]")


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]

    boxed = AGN_RE.findall(tail) or AGN_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()

    m = AGN_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    m = AGN_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    return None


LETTERS = ["A", "B", "C", "D"]
TOPICS = ["World", "Sports", "Business", "Sci/Tech"]   # matches label ints 0-3


class AGNews(Task):
    """AG News 4-way topic classification (7600 test snippets)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = (
            "You are given a short news snippet. Classify its topic. Think "
            "briefly, then finish your response with the label of the correct "
            "topic inside \\boxed{} .\n"
        )
        self.choices_block = "\n".join(
            f"{LETTERS[i]}. {t}" for i, t in enumerate(TOPICS)
        )

    def load(self):
        self.ds = HFDataset("fancyzhx/ag_news", split="test")

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        return "\n".join(
            [self.system_instr, "News:", row["text"], "Topics:",
             self.choices_block, "Which topic?"]
        )

    def get_example(self, idx: int, fewshot=False):
        prompt = self.create_prompt(self.ds[idx], fewshot=fewshot)
        return {"messages": [{"role": "user", "content": prompt}]}

    def evaluate(self, idx, assistant_response):
        gold = LETTERS[int(self.ds[idx]["label"])]
        pred = extract_answer(assistant_response)
        return int(pred is not None and pred == gold)


if __name__ == "__main__":
    benchmark = AGNews(start=0, stop=None, step=1)
    benchmark.load()
    n = min(300, benchmark.num_examples())
    scores = [
        benchmark.evaluate(i, f"\\boxed{{{LETTERS[int(benchmark.ds[i]['label'])]}}}")
        for i in range(n)
    ]
    print(f"Oracle score:  {np.mean(scores)}  (n={n})")
