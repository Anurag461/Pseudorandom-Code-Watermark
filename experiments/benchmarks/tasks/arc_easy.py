from experiments.benchmarks.tasks.task import Task, HFDataset
import re

ARC_RE = re.compile("\\\\boxed\\{\\(?([A-Ea-e1-4])\\)?\\}")
ARC_PHRASE_RE = re.compile(
    "(?:answer|option|choice)\\b[^A-Ea-e1-4]{0,20}?\\(?([A-Ea-e1-4])\\)?\\b", re.I
)
ARC_LABEL_RE = re.compile("(?:^|\\n)\\s*\\(?([A-Ea-e1-4])\\)?\\s*[.):-]")


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]
    boxed = ARC_RE.findall(tail) or ARC_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()
    m = ARC_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()
    m = ARC_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()
    return None


class ARCEasy(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = "You are given a multiple choice science question. Think before you answer and finish your sentence with the label of the correct option inside \\boxed{} .\n"

    def load(self):
        self.ds = HFDataset("allenai/ai2_arc", "ARC-Easy", split="test")

    def format_choices(self, choices):
        return "\n".join(
            (f"{lbl}. {txt}" for lbl, txt in zip(choices["label"], choices["text"]))
        )

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row):
        question = f"{row['question']}\n{self.format_choices(row['choices'])}"
        ls = [self.system_instr]
        ls += ["Now answer this Question:", question]
        return "\n".join(ls)

    def get_example(self, idx: int):
        row = self.ds[idx]
        prompt = self.create_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        return {"messages": messages}

    def evaluate(self, idx, assistant_response):
        gt_answer = self.ds[idx]["answerKey"].strip().upper()
        asst_answer = extract_answer(assistant_response)
        return int(gt_answer == asst_answer)
