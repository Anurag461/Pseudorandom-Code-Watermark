import pytest
import torch
from types import SimpleNamespace

from baseline_comparison.textseal_prefix_diagnostic import difference, raw_entropy, layer_capture


def test_difference_distinguishes_numerical_mismatch_from_shape_mismatch():
    assert difference([1, 2], [1, 2])["exact"]
    actual = difference([1, 2], [1, 2.25])
    assert not actual["exact"] and actual["different_values"] == 1
    assert actual["max_abs_difference"] == .25
    with pytest.raises(ValueError, match="shapes differ"):
        difference([1], [1, 2])


def test_raw_input_observer_rejects_prompt_or_external_kwargs():
    class Model(torch.nn.Module):
        def forward(self, ids, **kwargs):
            return ids
    class Detector:
        _detector = SimpleNamespace(model=Model())
        def _entropies(self, ids):
            self._detector.model(torch.tensor([ids]))
            return [1.] * (len(ids)-1)
    d = Detector()
    assert raw_entropy(d, [1, 2, 3]) == [1., 1.]
    d._entropies = lambda ids: d._detector.model(torch.tensor([[99] + ids]))
    with pytest.raises(ValueError, match="raw token IDs"):
        raw_entropy(d, [1, 2, 3])
    assert not d._detector.model._forward_pre_hooks


def test_layer_hooks_do_not_change_small_hf_qwen_forward():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, attn_implementation="eager", use_cache=False)).eval()
    class Detector:
        _detector = SimpleNamespace(model=model)
        def _entropies(self, ids):
            with torch.no_grad():
                logits=model(torch.tensor([ids])).logits
                lp=torch.log_softmax(logits,dim=-1)
                return (-(lp.exp()*lp).sum(-1))[0,:-1].tolist()
    detector = Detector()
    ids = list(range(24))
    original = raw_entropy(detector, ids)
    actual, values, order = layer_capture(detector, ids, prefix=8)
    assert actual == original and order[0] == "model.embed_tokens"
    assert values["model.rotary_emb"][0].shape[1] == 8
    assert len(values["model.rotary_emb"]) == 2
    assert all(not module._forward_hooks for module in model.modules())
