"""Validate cache identity and ensure only completion IDs cross the boundary."""
import torch
import pytest

from baseline_comparison.comparison_runner import _token_sha256
from baseline_comparison.textseal_preflight import RUN_ID, validate_source


@pytest.mark.parametrize("kind", ["online_prc", "null"])
def test_prc_null_allowlist_and_token_identity(tmp_path, kind):
    tokens = list(range(1100))
    path = tmp_path / "record.pt"
    row = {"prompt_idx": 2, "watermark": kind == "online_prc", "tokens": tokens,
           "prompt_tokens": [99999], "prompt": "POISON", "p_trace": [float("nan")] * 1100}
    torch.save(row, path)
    digest = _token_sha256(tokens[:1024])
    expected = {(kind, 2): digest}
    clean = validate_source(path, kind, 2, expected)
    assert clean == [{"method": kind, "prompt_index": 2, "token_ids": tokens[:1024],
                      "historical_token_sha256": digest}]
    row["tokens"][0] += 1
    torch.save(row, path)
    with pytest.raises(ValueError, match="completion changed"):
        validate_source(path, kind, 2, expected)


def test_textseal_shard_excludes_prompt_and_generation_entropy(tmp_path):
    tokens = list(range(1024))
    path = tmp_path / "shard.pt"
    source = {"prompt_indices": list(range(50)), "run_id": RUN_ID,
              "prompt_tokens": [[99999]] * 50,
              "sequences": {"textseal": [{"token_ids": tokens, "base_entropies": [float("nan")] * 1024,
                                           "past_key_values": "POISON"} for _ in range(50)]}}
    torch.save(source, path)
    expected = {("textseal", i): _token_sha256(tokens) for i in range(50)}
    clean = validate_source(path, "textseal", 0, expected)
    assert len(clean) == 50
    assert all(set(row) == {"method", "prompt_index", "token_ids", "historical_token_sha256"} for row in clean)
    assert all(row["token_ids"] == tokens for row in clean)
    source["prompt_indices"][0] = 99
    torch.save(source, path)
    with pytest.raises(ValueError, match="shard identity differs"):
        validate_source(path, "textseal", 0, expected)
