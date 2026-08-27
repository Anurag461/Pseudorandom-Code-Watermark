from benchmarks.task import Task, HFDataset
import re
import numpy as np

# Generation-style HellaSwag: instead of ranking endings by log-likelihood, we
# present the four candidate endings as options A-D and have the model generate
# the letter of the most plausible continuation. Extraction mirrors arc_easy:
# scope to the post-</think> answer, then \boxed{X} -> "answer/continuation is X"
# -> a leading "X." label.
HS_RE = re.compile(r"\\boxed\{\(?([A-Da-d])\)?\}")
HS_PHRASE_RE = re.compile(
    r"(?:answer|option|choice|continuation|ending)\b[^A-Da-d]{0,20}?\(?([A-Da-d])\)?\b",
    re.I,
)
HS_LABEL_RE = re.compile(r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*[.):-]")


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]

    boxed = HS_RE.findall(tail) or HS_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()

    m = HS_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    m = HS_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    return None


LETTERS = ["A", "B", "C", "D"]


class HellaSwag(Task):
    """HellaSwag commonsense NLI, run in generation (multiple-choice) style.

    Uses the `validation` split (10042 examples) because the `test` split ships
    with hidden labels. Each row's four `endings` become options A-D and the gold
    letter is LETTERS[int(label)].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = (
            "You are given the start of a short scenario and four possible "
            "continuations. Think briefly, then finish your response with the "
            "label of the most plausible continuation inside \\boxed{} .\n"
        )

    def load(self):
        self.ds = HFDataset("Rowan/hellaswag", split="validation")

    def format_choices(self, endings):
        return "\n".join(f"{LETTERS[i]}. {txt}" for i, txt in enumerate(endings))

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        ctx = f"{row['activity_label']}: {row['ctx']}".strip()
        question = f"{ctx}\n{self.format_choices(row['endings'])}"
        return "\n".join([self.system_instr, "Now choose the best continuation:", question])

    def get_example(self, idx: int, fewshot=False):
        prompt = self.create_prompt(self.ds[idx], fewshot=fewshot)
        return {"messages": [{"role": "user", "content": prompt}]}

    def evaluate(self, idx, assistant_response):
        gold = LETTERS[int(self.ds[idx]["label"])]
        pred = extract_answer(assistant_response)
        return int(pred is not None and pred == gold)


if __name__ == "__main__":
    benchmark = HellaSwag(start=0, stop=None, step=1)
    benchmark.load()
    n = min(300, benchmark.num_examples())
    scores = [
        benchmark.evaluate(i, f"\\boxed{{{LETTERS[int(benchmark.ds[i]['label'])]}}}")
        for i in range(n)
    ]
    print(f"Oracle score:  {np.mean(scores)}  (n={n})")
