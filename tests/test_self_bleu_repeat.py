"""Real upstream policy checks and causal isolation on small CPU fixtures."""
from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from self_bleu.config import StudySetting, digest
from self_bleu.generation import generate_response_batch
from self_bleu.repeat import (
    RepeatSetting, SamplerRepeatPolicy, arm_setting, check_sampler_policy,
    check_synthid_policy, fallback_seed, generate_repeat_batch, install_policy,
)


@pytest.fixture(autouse=True)
def upstream():
    from baseline_comparison.textseal_completion import load_upstream_detector
    load_upstream_detector(os.environ.get("TEXTSEAL_SOURCE_ROOT"))


class Model(torch.nn.Module):
    def forward(self, ids, cache=None):
        values = torch.arange(32, device=ids.device).float()/8
        return values.expand(*ids.shape, 32)


PROMPTS = [[i % 31 for i in range(50)], [(i+5) % 31 for i in range(50)]]
EXECUTION = {"fixture": "CPU synthetic logits, not pretrained model validation"}


def test_synthid_toggle_changes_only_repeat_fallback():
    assert all(check_synthid_policy().values())


@pytest.mark.parametrize("method", ["textseal", "gumbel_max"])
def test_sampler_fallback_is_ordinary_sampling_without_advancing_native_rng(method):
    assert all(check_sampler_policy(method).values())


@pytest.mark.parametrize("method,native_enabled", [("synthid_text",True), ("textseal",False), ("gumbel_max",False)])
def test_native_policy_adapter_reproduces_unmodified_generator(method, native_enabled):
    prompts = PROMPTS[:1] if method == "textseal" else PROMPTS
    indices = list(range(len(prompts)))
    old = generate_response_batch(Model(),prompts,indices,setting=StudySetting(method),sampling_seed=12345,
                                  response_index=0,execution=EXECUTION,max_new_tokens=32,device="cpu")
    current = generate_repeat_batch(Model(),prompts,indices,setting=RepeatSetting(method,repeat_fallback=native_enabled),
                                    sampling_seed=12345,response_index=0,execution=EXECUTION,max_new_tokens=32,device="cpu")
    assert [r["token_ids"] for r in old["responses"]] == [r["token_ids"] for r in current["responses"]]
    assert old["manifest"]["batch_id"] != current["manifest"]["batch_id"]


@pytest.mark.parametrize("arm", ["synthid_off", "textseal_on", "gumbel_on"])
def test_policy_history_and_seed_reset_between_response_calls(arm):
    setting = arm_setting(arm)
    prompts = PROMPTS[:1] if arm == "textseal_on" else PROMPTS
    def run(seed, response):
        return generate_repeat_batch(Model(),prompts,list(range(len(prompts))),setting=setting,sampling_seed=seed,
                       response_index=response,execution=EXECUTION,max_new_tokens=48,device="cpu")
    a = run(12345,0)
    run(67890,1)
    b = run(12345,0)
    assert a["responses"] == b["responses"]
    for row in a["responses"]:
        trace = row["generation_diagnostics"]
        assert len(trace["repeated_context"]) == len(trace["fallback_applied"]) == 48
        assert trace["fallback_applied"] == ([False]*48 if arm == "synthid_off" else trace["repeated_context"])
        assert not trace["fallback_applied"][0]


def test_policy_factories_restore_even_if_generation_fails():
    from baseline_comparison import comparison_runner as runner
    original = runner.synthid_processor
    with pytest.raises(RuntimeError):
        with install_policy(arm_setting("synthid_off"), 12345, [0]):
            assert runner.synthid_processor is not original
            raise RuntimeError("fixture failure")
    assert runner.synthid_processor is original


def test_private_fallback_streams_follow_prompt_identity_and_batch_isolation():
    class ConstantSampler:
        def sample_next(self, logits, context, **kwargs):
            return torch.zeros(logits.shape[0],dtype=torch.long)
    a = SamplerRepeatPolicy(ConstantSampler(),True,12345,[0,1])
    b = SamplerRepeatPolicy(ConstantSampler(),True,12345,[1,0])
    logits = torch.arange(32).float()[None].repeat(2,1)/15
    ctx = torch.tensor([[1,2,3],[4,5,6]])
    for _ in range(8):
        assert torch.equal(a.sample_next(logits,ctx,temperature=1.,top_p=1.),
                           b.sample_next(logits.flip(0),ctx.flip(0),temperature=1.,top_p=1.).flip(0))
    assert fallback_seed(12345,0) != fallback_seed(67890,0) != fallback_seed(12345,1)


def test_policy_identity_rejects_unrelated_parameter_changes():
    a = RepeatSetting("synthid_text",repeat_fallback=True)
    assert a.fingerprint != replace(a,repeat_fallback=False).fingerprint
    for args in ({"method":"online_prc"}, {"method":"synthid_text","depth":20},
                 {"method":"textseal","alpha":.5}, {"method":"gumbel_max","repeat_fallback":1}):
        with pytest.raises(ValueError):
            RepeatSetting(**args)


def test_paired_policy_contrast_keeps_prompt_units():
    from self_bleu.repeat import paired_contrast
    draws = np.random.default_rng(20260918).integers(0,50,(2000,50))
    old = np.linspace(0,.1,50)
    result = paired_contrast(old,old,[.5]*50,[.5]*50,draws)
    assert result == {"self_bleu_difference":{"mean":0.,"ci95":[0.,0.]},
                      "tpr_difference":{"mean":0.,"ci95":[0.,0.]}}


def test_repeat_request_rejects_scope_budget_and_changed_source(tmp_path):
    from self_bleu.repeat import ARMS
    from self_bleu.repeat import validate,TIMEOUTS,RATE
    from self_bleu.validation import sha
    (tmp_path/"prompts.jsonl").write_text("fixture")
    def request(previous=5.52):
        v = {"arms":{a:arm_setting(a).identity() for a in ARMS},"prompt_indices":list(range(50)),
             "seeds":[12345,67890],"length":1024,"control_tokens":64,"protocol":"completion_only_raw_abstain_v1",
             "code_sha256":{},"prompt_sha256":sha(tmp_path/"prompts.jsonl"),
             "cost":{"timeouts":TIMEOUTS,"resource_usd_per_second":RATE,"previous_planning_charge_usd":previous,
                     "total_reserved_with_prior_usd":previous+(sum(TIMEOUTS.values())+6)*RATE+.5}}
        v["id"]=digest(v)
        return v
    validate(request(),tmp_path)
    with pytest.raises(ValueError,match="allocation"):
        validate(request(9.),tmp_path)
    bad=request(); bad["seeds"]=[12345,999]; bad["id"]=digest({k:v for k,v in bad.items() if k!="id"})
    with pytest.raises(ValueError,match="scope"):
        validate(bad,tmp_path)
    valid=request(); (tmp_path/"prompts.jsonl").write_text("changed")
    with pytest.raises(ValueError,match="prompts"):
        validate(valid,tmp_path)


@pytest.mark.parametrize("enabled", [True, False])
def test_token_reconstruction_matches_actual_upstream_generation_history(enabled):
    from baseline_comparison.official import synthid_processor
    from self_bleu.repeat import SynthIDRepeatPolicy, synthid_repeat_masks
    tokens = [[0]*16, [1, 2, 3]*5 + [4], list(range(20, 36))]
    keys = StudySetting("synthid_text").synthid_keys
    policy = SynthIDRepeatPolicy(synthid_processor("cpu", keys=keys), enabled)
    for position in range(16):
        previous = torch.tensor([[row[position-1] if position else 7] for row in tokens])
        policy.watermarked_call(previous, torch.zeros((3, 32)))
    expected = np.asarray(policy.repeated).T.tolist()
    actual = synthid_repeat_masks(tokens, keys=keys)
    assert actual == expected
    assert actual[0] == [False] + [True]*15  # includes zero-context warmup
    assert actual[1] == [False]*6 + [True]*10
    assert actual[2] == [False]*16
    assert np.asarray(policy.applied).T.tolist() == (expected if enabled else [[False]*16]*3)
    with pytest.raises(ValueError, match="history capacity"):
        synthid_repeat_masks([[0]*1025], keys=keys)


def trajectory_fixture(old_ids, new_ids, prompt=0):
    from self_bleu.repeat import synthid_repeat_masks, trajectory_pair
    old_mask, new_mask = synthid_repeat_masks([old_ids, new_ids], keys=StudySetting("synthid_text").synthid_keys)
    old = {"token_ids": old_ids, "response_id": f"old/{prompt}", "prompt_index": prompt, "response_index": 0}
    new = {**old, "token_ids": new_ids, "response_id": f"new/{prompt}",
           "generation_diagnostics": {"repeated_context": new_mask, "fallback_applied": [False]*len(new_ids),
                                      "first_fallback_position": None}}
    return trajectory_pair(old, new, old_mask, new_mask)


def test_full_trajectory_guard_accepts_at_repeat_and_rejects_earlier_divergence():
    old = [1, 2, 3]*5 + [4]
    for pos, expected in ((6, True), (5, False)):
        new = old.copy(); new[pos] = 9
        result = trajectory_fixture(old, new)
        assert result["first_token_divergence"] == pos
        assert result["passed"] is expected
        assert result["checks"]["no_divergence_before_first_repeat"] is expected
    assert trajectory_fixture(old, old)["passed"]  # a repeat need not change the draw
    plain = list(range(20, 36))
    identical = trajectory_fixture(plain, plain)
    assert identical["passed"] and identical["original"]["first_repeat_position"] is None
    changed = plain.copy(); changed[10] = 90
    bad = trajectory_fixture(plain, changed)
    assert not bad["passed"] and not bad["checks"]["no_repeat_reference_stays_identical"]


def test_repeat_summary_includes_both_policies_and_correct_prefix_denominators():
    from self_bleu.repeat import trajectory_summary
    old = [1, 2, 3]*5 + [4]
    new = old.copy(); new[6] = 9
    plain = list(range(20, 36))
    pairs = [trajectory_fixture(old, new), trajectory_fixture(plain, plain, prompt=1)]
    short, full = trajectory_summary(pairs, [6, 16])
    assert short["original"]["repeat_count_total"] == short["pairs_with_divergence"] == 0
    assert full["original"]["fraction_responses_with_repeat"] == .5
    assert full["original"]["repeat_count_total"] == full["original"]["fallback_count_total"] == 10
    assert full["modified"]["fallback_count_total"] == 0
    assert full["fraction_pairs_with_divergence"] == .5
    assert full["first_token_divergence_median_among_diverged"] == 6


def test_trajectory_guard_rejects_false_online_diagnostics():
    from self_bleu.repeat import synthid_repeat_masks, trajectory_pair
    tokens = [1, 2, 3]*5 + [4]
    mask = synthid_repeat_masks([tokens], keys=StudySetting("synthid_text").synthid_keys)[0]
    row = {"token_ids": tokens, "response_id": "fixture", "prompt_index": 0, "response_index": 0}
    new = {**row, "generation_diagnostics": {"repeated_context": [False]*16,
                 "fallback_applied": [False]*16, "first_fallback_position": None}}
    result = trajectory_pair(row, new, mask, mask)
    assert not result["passed"] and not result["checks"]["modified_repeat_trace_matches_reconstruction"]
