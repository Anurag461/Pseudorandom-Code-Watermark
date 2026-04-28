import torch
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


device = torch.device("cuda")
model, tokenizer, meta = load_model("sft", device, phase="eval")
engine = Engine(model, tokenizer)
