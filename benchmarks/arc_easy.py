from benchmarks.task import Task, HFDataset
import re
import numpy as np

# Reasoning-model outputs put the final label in one of several forms, so we try
# them in order of specificity. Only the text AFTER the </think> block is the
# actual answer, so we scope extraction to that to avoid matching option letters
# that appear inside the reasoning.
ARC_RE = re.compile(r"\\boxed\{\(?([A-Ea-e1-4])\)?\}")                 # \boxed{X}
ARC_PHRASE_RE = re.compile(                                            # "answer is X"
    r"(?:answer|option|choice)\b[^A-Ea-e1-4]{0,20}?\(?([A-Ea-e1-4])\)?\b", re.I)
ARC_LABEL_RE = re.compile(r"(?:^|\n)\s*\(?([A-Ea-e1-4])\)?\s*[.):-]")  # leading "X."

def extract_answer(completion):
    # Scope to the post-reasoning answer when a think block is present.
    tail = completion.rsplit("</think>", 1)[-1]

    # 1) \boxed{X} -- take the LAST one (models sometimes box intermediates).
    boxed = ARC_RE.findall(tail) or ARC_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()

    # 2) "The answer is X" / "correct option is (X)".
    m = ARC_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    # 3) A final answer line that just starts with the label, e.g. "B. breathing mask".
    m = ARC_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    return None

class ARCEasy(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = """You are given a multiple choice science question. Think before you answer and finish your sentence with the label of the correct option inside \\boxed{} .\n"""
        # Hand-written rationales for the first two train examples (indices 0 and 1),
        # so the few-shot exemplars demonstrate reasoning before the answer label.
        self.few_shot_rationales = [
            """A fever is the body's immune response to an infection. A bacterial 
            population in the bloodstream is an active infection that triggers this 
            response, raising body temperature. A relaxing muscle, viral particles 
            merely resting on the skin, and digesting carbohydrates do not cause a fever.""",
            """The green algae are the photosynthetic partner, so they make food (sugars) 
            by photosynthesis and supply it to the fungi. The fungi in turn provide 
            structure, protection, and water, so the algae's contribution is food.""",
        ]

    def load(self):
        self.ds = HFDataset("allenai/ai2_arc", "ARC-Easy", split="test")
        self.train_ds = HFDataset("allenai/ai2_arc", "ARC-Easy", split="train")
        
    def format_choices(self, choices):
        return "\n".join(f"{lbl}. {txt}" for lbl, txt in zip(choices["label"], choices["text"]))

    def few_shot_examples(self):
        base_instr = "Here are 2 examples for how you should answer:"
        parts = [base_instr]
        for k in range(2):
            row = self.train_ds[k]
            question = f"Question{k+1}:\n{row['question']}\n{self.format_choices(row['choices'])}"
            answer = f"Solution:\n{self.few_shot_rationales[k]}\n\\boxed{{{row['answerKey']}}}"
            parts.extend([question, answer])
        parts.append("")
        return "\n".join(parts)

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        question = f"{row['question']}\n{self.format_choices(row['choices'])}"
        ls = [self.system_instr]
        if fewshot:
            ls.append(self.few_shot_examples())
        ls += ["Now answer this Question:", question]
        return "\n".join(ls)

    def get_example(self, idx: int, fewshot=False):
        row = self.ds[idx]
        prompt = self.create_prompt(row, fewshot=fewshot)
        messages = [{"role": "user", "content": prompt}]
        return {"messages": messages}

    def evaluate(self, idx, assistant_response):
        gt_answer = self.ds[idx]["answerKey"].strip().upper()
        asst_answer = extract_answer(assistant_response)
        return int(gt_answer == asst_answer)

if __name__ == "__main__":
    benchmark = ARCEasy(start=0, stop=None, step=1)
    benchmark.load()
    scores = []
    for i in range(benchmark.num_examples()):
        gold_response = f"\\boxed{{{benchmark.ds[i]['answerKey']}}}"
        score = benchmark.evaluate(i, gold_response)
        scores.append(score)
    print("Oracle score: ", np.mean(scores))
