from benchmarks.task import Task, HFDataset
import re
import random
import numpy as np

# GPQA Diamond is 4-way multiple choice. We present the correct answer plus the
# three distractors as options A-D and grade on the chosen letter, so extraction
# mirrors arc_easy: scope to the post-</think> answer, then try \boxed{X} ->
# "answer is X" -> a leading "X." label.
GPQA_RE = re.compile(r"\\boxed\{\(?([A-Da-d])\)?\}")
GPQA_PHRASE_RE = re.compile(
    r"(?:answer|option|choice)\b[^A-Da-d]{0,20}?\(?([A-Da-d])\)?\b", re.I)
GPQA_LABEL_RE = re.compile(r"(?:^|\n)\s*\(?([A-Da-d])\)?\s*[.):-]")


def extract_answer(completion):
    tail = completion.rsplit("</think>", 1)[-1]

    boxed = GPQA_RE.findall(tail) or GPQA_RE.findall(completion)
    if boxed:
        return boxed[-1].strip().upper()

    m = GPQA_PHRASE_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    m = GPQA_LABEL_RE.search(tail)
    if m:
        return m.group(1).strip().upper()

    return None


LETTERS = ["A", "B", "C", "D"]


class GPQADiamond(Task):
    """GPQA Diamond (198 hard, Google-proof science MCQs).

    NOTE: the canonical dataset `Idavidrein/gpqa` is GATED. To load it you must
    (1) accept the license at https://huggingface.co/datasets/Idavidrein/gpqa and
    (2) provide an HF token (set HF_TOKEN in the environment / Modal secret). The
    non-gated mirrors drop the multiple-choice distractors, so we use the gated
    original to keep the letter-graded MC format.

    Options are built by shuffling [correct, incorrect1, incorrect2, incorrect3]
    with a per-example deterministic seed, so the prompt shown at generation time
    and the gold letter computed at scoring time always agree.
    """

    def __init__(self, shuffle_seed: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.shuffle_seed = shuffle_seed
        self.system_instr = (
            "You are given a hard, graduate-level multiple choice science "
            "question. Think before you answer and finish your response with the "
            "label of the correct option inside \\boxed{} .\n"
        )

    def load(self):
        self.ds = HFDataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    def _options(self, idx):
        """Return (shuffled_option_texts, gold_letter) deterministically for idx."""
        row = self.ds[idx]
        answers = [
            row["Correct Answer"].strip(),
            row["Incorrect Answer 1"].strip(),
            row["Incorrect Answer 2"].strip(),
            row["Incorrect Answer 3"].strip(),
        ]
        order = list(range(4))
        random.Random(self.shuffle_seed + idx).shuffle(order)  # same every call
        shuffled = [answers[i] for i in order]
        gold_letter = LETTERS[order.index(0)]                  # where correct landed
        return shuffled, gold_letter

    def format_choices(self, options):
        return "\n".join(f"{LETTERS[i]}. {txt}" for i, txt in enumerate(options))

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, idx, fewshot=False):
        row = self.ds[idx]
        options, _ = self._options(idx)
        question = f"{row['Question'].strip()}\n{self.format_choices(options)}"
        return "\n".join([self.system_instr, "Now answer this Question:", question])

    def get_example(self, idx: int, fewshot=False):
        prompt = self.create_prompt(idx, fewshot=fewshot)
        return {"messages": [{"role": "user", "content": prompt}]}

    def evaluate(self, idx, assistant_response):
        _, gold_letter = self._options(idx)
        pred = extract_answer(assistant_response)
        return int(pred is not None and pred == gold_letter)


if __name__ == "__main__":
    benchmark = GPQADiamond(start=0, stop=None, step=1)
    benchmark.load()
    scores = []
    for i in range(benchmark.num_examples()):
        _, gold = benchmark._options(i)
        scores.append(benchmark.evaluate(i, f"\\boxed{{{gold}}}"))
    print("Oracle score: ", np.mean(scores))
