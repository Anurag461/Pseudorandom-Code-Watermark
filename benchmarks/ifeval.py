from benchmarks.task import Task, HFDataset
import re

from benchmarks.ifeval_lib import instructions_registry as _reg

# Reasoning models emit a <think>...</think> block that would violate almost every
# format instruction ("all lowercase", "3 paragraphs", "end with phrase"), so we
# grade only the post-reasoning answer.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    # Truncated reasoning: an unclosed <think> with no matching close -> drop it all.
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


class IFEval(Task):
    """IFEval verifiable instruction following (Zhou et al. 2023), 541 prompts.

    Scored with Google's official verifier (vendored in benchmarks/ifeval_lib).
    evaluate() returns PROMPT-LEVEL STRICT: 1 iff *every* instruction in the
    prompt is followed, which is the strictest / most watermark-sensitive metric.
    """

    def load(self):
        self.ds = HFDataset("google/IFEval", split="train")

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        # IFEval prompts are already self-contained instructions.
        return row["prompt"]

    def get_example(self, idx: int, fewshot=False):
        return {"messages": [{"role": "user", "content": self.create_prompt(self.ds[idx])}]}

    def _follow_flags(self, row, response):
        """Per-instruction booleans, mirroring the official strict evaluation."""
        flags = []
        for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
            instruction = _reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for k, v in kw.items() if v is not None}
            instruction.build_description(**kw)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=row["prompt"])
            flags.append(bool(response.strip()) and instruction.check_following(response))
        return flags

    def evaluate(self, idx, assistant_response):
        row = self.ds[idx]
        response = strip_think(assistant_response)
        return int(all(self._follow_flags(row, response)))


if __name__ == "__main__":
    import numpy as np
    benchmark = IFEval(start=0, stop=None, step=1)
    benchmark.load()
    # Oracle sanity: an empty/garbage answer must NOT pass; the verifier must run
    # cleanly over every row without crashing on any instruction type.
    crashes, n = 0, benchmark.num_examples()
    for i in range(n):
        try:
            benchmark.evaluate(i, "This is a placeholder answer.")
        except Exception as e:
            crashes += 1
            if crashes <= 5:
                print(f"CRASH idx={i} ids={benchmark.ds[i]['instruction_id_list']}: {type(e).__name__}: {e}")
    print(f"ran verifier over {n} rows, crashes={crashes}")
