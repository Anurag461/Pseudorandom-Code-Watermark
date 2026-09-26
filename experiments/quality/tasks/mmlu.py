from experiments.quality.tasks.task import Task, HFDataset
import re

MMLU_RE = re.compile("\\\\boxed\\{\\(?([A-Da-d])\\)?\\}")
MMLU_PHRASE_RE = re.compile(
    "(?:answer|option|choice)\\b[^A-Da-d]{0,20}?\\(?([A-Da-d])\\)?\\b", re.I
)
MMLU_LABEL_RE = re.compile("(?:^|\\n)\\s*\\(?([A-Da-d])\\)?\\s*[.):-]")


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]
    boxed = MMLU_RE.findall(tail) or MMLU_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()
    m = MMLU_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()
    m = MMLU_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()
    return None


LETTERS = ["A", "B", "C", "D"]


class MMLU(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = "You are given a multiple choice question. Think before you answer and finish your response with the label of the correct option inside \\boxed{} .\n"

    def load(self):
        self.ds = HFDataset("cais/mmlu", "all", split="test")

    def format_choices(self, choices):
        return "\n".join((f"{LETTERS[i]}. {txt}" for i, txt in enumerate(choices)))

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        question = f"{row['question']}\n{self.format_choices(row['choices'])}"
        return "\n".join([self.system_instr, "Now answer this Question:", question])

    def get_example(self, idx: int, fewshot=False):
        prompt = self.create_prompt(self.ds[idx], fewshot=fewshot)
        return {"messages": [{"role": "user", "content": prompt}]}

    def evaluate(self, idx, assistant_response):
        gold = LETTERS[int(self.ds[idx]["answer"])]
        pred = extract_answer(assistant_response)
        return int(pred is not None and pred == gold)
