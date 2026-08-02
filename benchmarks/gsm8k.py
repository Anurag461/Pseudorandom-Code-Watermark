from task import Task, HFDataset
import re
import numpy as np

GSM_RE= re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return None

class GSM(Task):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = HFDataset("openai/gsm8k", "main",split= "test")

    def num_examples(self):
        return len(self.ds)

    def get_example(self, idx):
        row = self.ds[idx]
        messages =  [{"role": "user", "content": row['question']}]
        return {"messages": messages}

    def evaluate(self, idx, assistant_response):
        answer = self.ds[idx]["answer"]
        gt_answer = extract_answer(answer)
        asst_answer = extract_answer(assistant_response)
        return int(gt_answer==asst_answer)

if __name__ == "__main__":
    benchmark = GSM(start=0,stop= None, step=1)
    scores = []
    for i in range(benchmark.num_examples()):
        example = benchmark.get_example(i)
        score = benchmark.evaluate(i, benchmark.ds[i]['answer'])
        scores.append(score)
    print("Oracle score: ", np.mean(scores))
