from experiments.quality.tasks.task import Task, HFDataset
import re
from experiments.quality.tasks.ifeval_lib import instructions_registry as _reg

_THINK_BLOCK = re.compile("<think>.*?</think>", re.S)


def strip_think(text: str) -> str:
    text = _THINK_BLOCK.sub("", text)
    if "<think>" in text:
        text = text.split("<think>")[0]
    return text.strip()


class IFEval(Task):

    def load(self):
        self.ds = HFDataset("google/IFEval", split="train")

    def num_examples(self):
        return len(self.ds)

    def create_prompt(self, row, fewshot=False):
        return row["prompt"]

    def get_example(self, idx: int, fewshot=False):
        return {
            "messages": [{"role": "user", "content": self.create_prompt(self.ds[idx])}]
        }

    def _follow_flags(self, row, response):
        flags = []
        for iid, kw in zip(row["instruction_id_list"], row["kwargs"]):
            instruction = _reg.INSTRUCTION_DICT[iid](iid)
            kw = {k: v for (k, v) in kw.items() if v is not None}
            instruction.build_description(**kw)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=row["prompt"])
            flags.append(
                bool(response.strip()) and instruction.check_following(response)
            )
        return flags

    def evaluate(self, idx, assistant_response):
        row = self.ds[idx]
        response = strip_think(assistant_response)
        return int(all(self._follow_flags(row, response)))
