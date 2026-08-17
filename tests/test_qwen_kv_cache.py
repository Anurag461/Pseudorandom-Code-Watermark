import pytest
import torch

from qwen import (
    KVCache,
    Qwen3Model,
    StaticKVCache,
    kv_cache_version,
    make_kv_cache,
    normalize_kv_cache_implementation,
)


def _tiny_qwen():
    torch.manual_seed(17)
    model = Qwen3Model({
        "vocab_size": 31,
        "context_length": 32,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 32,
        "head_dim": 4,
        "qk_norm": True,
        "n_kv_groups": 2,
        "rope_base": 10_000.0,
        "dtype": torch.float32,
    })
    return model.eval()


def test_static_cache_matches_concat_logits_and_values_exactly():
    model = _tiny_qwen()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    decode_steps = [
        torch.tensor([[7], [8]]),
        torch.tensor([[9], [10]]),
        torch.tensor([[11], [12]]),
    ]
    concat = KVCache()
    static = StaticKVCache(max_length=6)

    with torch.no_grad():
        concat_logits = [model(prompt, cache=concat)]
        static_logits = [model(prompt, cache=static)]
        static_pointers = {
            layer: tuple(tensor.data_ptr() for tensor in pair)
            for layer, pair in static.cache.items()
        }
        for tokens in decode_steps:
            concat_logits.append(model(tokens, cache=concat))
            static_logits.append(model(tokens, cache=static))

    assert concat.get_seq_len() == static.get_seq_len() == 6
    for expected, observed in zip(concat_logits, static_logits):
        assert torch.equal(expected, observed)
    for layer, (concat_k, concat_v) in concat.cache.items():
        static_k, static_v = static.cache[layer]
        assert torch.equal(concat_k, static_k[:, :, :6, :])
        assert torch.equal(concat_v, static_v[:, :, :6, :])
        assert static_pointers[layer] == (
            static_k.data_ptr(), static_v.data_ptr()
        )


def test_static_cache_reset_reuses_storage_and_enforces_capacity():
    cache = StaticKVCache(max_length=4)
    first = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    cache.update(0, first, -first)
    pointers = tuple(tensor.data_ptr() for tensor in cache.cache[0])
    allocated = cache.allocated_bytes()

    cache.reset()
    assert cache.get_seq_len() == 0
    second = torch.ones((1, 2, 4, 3))
    keys, values = cache.update(0, second, second * 2)
    assert cache.get_seq_len() == 4
    assert torch.equal(keys, second)
    assert torch.equal(values, second * 2)
    assert pointers == tuple(tensor.data_ptr() for tensor in cache.cache[0])
    assert cache.allocated_bytes() == allocated

    with pytest.raises(ValueError, match="capacity 4 exceeded"):
        cache.update(0, torch.ones((1, 2, 1, 3)), torch.ones((1, 2, 1, 3)))


def test_cache_factory_is_opt_in_and_validates_configuration():
    assert isinstance(make_kv_cache(), KVCache)
    assert isinstance(make_kv_cache("dynamic"), KVCache)
    assert isinstance(make_kv_cache("static", max_length=8), StaticKVCache)
    assert normalize_kv_cache_implementation("preallocated") == "static"
    assert kv_cache_version("concat") == "concat-v1"
    assert kv_cache_version("static") == "static-v1"

    with pytest.raises(ValueError, match="requires max_length"):
        make_kv_cache("static")
    with pytest.raises(ValueError, match="must be one of"):
        make_kv_cache("paged")
