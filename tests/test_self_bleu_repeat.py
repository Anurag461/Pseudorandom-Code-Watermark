"""Real upstream policy checks and causal isolation on small CPU fixtures."""
from dataclasses import replace
import os

import numpy as np
import pytest
import torch

from baseline_comparison.self_bleu_config import StudySetting, digest
from baseline_comparison.self_bleu_generation import generate_response_batch
from baseline_comparison.self_bleu_repeat import (
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
    from baseline_comparison.self_bleu_repeat import paired_contrast
    draws = np.random.default_rng(20260918).integers(0,50,(2000,50))
    old = np.linspace(0,.1,50)
    result = paired_contrast(old,old,[.5]*50,[.5]*50,draws)
    assert result == {"self_bleu_difference":{"mean":0.,"ci95":[0.,0.]},
                      "tpr_difference":{"mean":0.,"ci95":[0.,0.]}}


def test_repeat_request_rejects_scope_budget_and_changed_source(tmp_path):
    from baseline_comparison.self_bleu_repeat import ARMS
    from baseline_comparison.self_bleu_repeat import validate,TIMEOUTS,RATE
    from baseline_comparison.self_bleu_validation import sha
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
