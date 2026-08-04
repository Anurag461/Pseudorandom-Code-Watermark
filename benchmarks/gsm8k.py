from benchmarks.task import Task, HFDataset
import re
import numpy as np
import random

GSM_RE= re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return None

GSM_ANSWER_RE = re.compile(r"\\boxed\{\-?([0-9\.\,]+)\}")
def extract_boxed_answer(completion: str):
    match = GSM_ANSWER_RE.search(completion)
    if match:
        ans= match.group(1).strip()
        ans = ans.replace(",", "")
        return ans
    else:
        return None
    
    
class GSM(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_instr = """You are given a grade school math question. Think before you answer and include only the numerical part of your answer inside \\boxed{}. Exclude the units like $ from your final answer.\n"""

    def load(self):
        self.ds = HFDataset("openai/gsm8k", "main",split= "test")
        
    def few_shot_examples(self):
        base_instr = "Here are 2 examples for how you should answer:"
        question1 = "Question1:\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
        answer1 = """Solution:\nNatalia sold 48/2 = 48/2=24 clips in May.
Natalia sold 48+24 = 48+24=72 clips altogether in April and May.\n#### 72"""
        question2 = """Question2:\nWeng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"""
        answer2 = """Solution:\nWeng earns 12/60 = $12/60=0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $0.2*50=10.\n#### 10"""
        return "\n".join([base_instr, question1, answer1, question2, answer2, ""])

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        ls = [self.system_instr]
        if fewshot:
            ls.append(self.few_shot_examples())
        ls += ["Question:", row['question']]
        return "\n".join(ls)

    # def create_prompt(self, row):
    #     prompt = row['question']
    #     messages =  [{"role": "user", "content": prompt}]
    #     return {"messages":messages}
    
    def get_example(self, idx: int, fewshot=False):
        row = self.ds[idx]
        prompt = self.create_prompt(row, fewshot=fewshot)
        messages =  [{"role": "user", "content": prompt}]
        return {"messages": messages}

    def evaluate(self, idx, assistant_response):
        answer = self.ds[idx]["answer"]
        gt_answer = extract_answer(answer)
        asst_answer = extract_boxed_answer(assistant_response)
        return int(gt_answer==asst_answer)

if __name__ == "__main__":
    benchmark = GSM(start=0,stop= None, step=1)
    scores = []
    for i in range(benchmark.num_examples()):
        example = benchmark.get_example(i)
        score = benchmark.evaluate(i, benchmark.ds[i]['answer'])
        scores.append(score)
    print("Oracle score: ", np.mean(scores))
