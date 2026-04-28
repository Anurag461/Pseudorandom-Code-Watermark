import torch
from qwen import Qwen3Model, KVCache

torch.manual_seed(0)

cfg = {
    "vocab_size": 256,
    "context_length": 64,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 64,
    "head_dim": 8,
    "qk_norm": True,
    "n_kv_groups": 2,
    "rope_base": 10000.0,
    "dtype": torch.float32,
}

torch.manual_seed(42)
model = Qwen3Model(cfg).eval()

seq = torch.randint(0, cfg["vocab_size"], (1, 10))

with torch.no_grad():
    full_logits = model(seq)

# Prefill 6 tokens, then decode 4 tokens one at a time using the cache.
cache = KVCache()
with torch.no_grad():
    prefill_logits = model(seq[:, :6], cache=cache)
    step_logits = []
    for i in range(6, 10):
        step_logits.append(model(seq[:, i:i+1], cache=cache))

ref_positions = [5, 6, 7, 8, 9]
cached_outs = [prefill_logits[:, -1:], step_logits[0], step_logits[1], step_logits[2], step_logits[3]]
diffs = [(full_logits[:, p:p+1] - c).abs().max().item() for p, c in zip(ref_positions, cached_outs)]
print("prefill+decode max abs diff per position:", diffs)
print("final cache seq len:", cache.get_seq_len())

# Pure single-token decode from empty cache.
cache2 = KVCache()
outs = []
with torch.no_grad():
    for i in range(10):
        outs.append(model(seq[:, i:i+1], cache=cache2))
diffs2 = [(full_logits[:, i:i+1] - outs[i]).abs().max().item() for i in range(10)]
print("per-token decode diffs:", diffs2)
print("final cache2 seq len:", cache2.get_seq_len())

# Sanity: with no cache passed, behavior should be unchanged.
with torch.no_grad():
    full_again = model(seq)
print("no-cache reproducibility max diff:", (full_logits - full_again).abs().max().item())
