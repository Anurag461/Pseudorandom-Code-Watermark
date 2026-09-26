from experiments.benchmarks.tasks.task import Task, HFDataset
import re

GSM_RE = re.compile("#### (\\-?[0-9\\.\\,]+)")
_NUM = "\\\\?\\$?\\s*(-?[0-9][0-9,]*(?:\\.[0-9]+)?)"
GSM_BOXED_RE = re.compile("\\\\boxed\\{\\s*" + _NUM)
GSM_PHRASE_RE = re.compile("answer\\b[^0-9\\-]{0,20}?" + _NUM, re.I)
GSM_NUM_RE = re.compile(_NUM)


def _normalize_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
    except ValueError:
        return None
    return str(int(f)) if f == int(f) else str(f)


def extract_answer(completion):
    match = GSM_RE.search(completion)
    return _normalize_num(match.group(1)) if match else None


def extract_boxed_answer(completion: str):
    tail = completion.rsplit("</think>", 1)[-1]
    boxed = GSM_BOXED_RE.findall(tail) or GSM_BOXED_RE.findall(completion)
    if boxed:
        return _normalize_num(boxed[-1])
    phrase = GSM_PHRASE_RE.findall(tail)
    if phrase:
        return _normalize_num(phrase[-1])
    nums = GSM_NUM_RE.findall(tail)
    if nums:
        return _normalize_num(nums[-1])
    return None


class GSM(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = "You are given a grade school math question. Think before you answer and include only the numerical part of your answer inside \\boxed{}. Exclude the units like $ from your final answer.\n"

    def load(self):
        self.ds = HFDataset("openai/gsm8k", "main", split="test")

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row):
        ls = [self.system_instr]
        ls += ["Question:", row["question"]]
        return "\n".join(ls)

    def get_example(self, idx: int):
        row = self.ds[idx]
        prompt = self.create_prompt(row)
        messages = [{"role": "user", "content": prompt}]
        return {"messages": messages}

    def evaluate(self, idx, assistant_response):
        answer = self.ds[idx]["answer"]
        gt_answer = extract_answer(answer)
        asst_answer = extract_boxed_answer(assistant_response)
        return int(gt_answer is not None and gt_answer == asst_answer)
