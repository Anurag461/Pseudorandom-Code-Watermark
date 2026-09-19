"""Finite-support decoder and completion-only contract checks; no GPU dispatch."""
import numpy as np
import pytest
import torch

from self_bleu.topk import (SETTINGS,truncate,partition_probability,prc_draw,
    semantic_checks,generate,completion_trace,repetition,summarize_metrics)
from self_bleu.validation import load_online_sampler,ROOT
from online_prc import OnlinePRCEncoder,derive_document_seed


class ToyModel(torch.nn.Module):
    def __init__(self,filtered=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.filtered = filtered
        self.calls = []

    def forward(self,ids,cache=None):
        self.calls.append((ids.clone(),cache))
        center = (ids+17)%256
        logits = (-((torch.arange(256)-center.unsqueeze(-1)).float()/80).square()).bfloat16()
        if self.filtered:
            return truncate(logits[:,-1])[0][:,None]
        return logits


PROMPTS = [[i for i in range(50)],[(i+3)%256 for i in range(50)]]
MASK = torch.arange(256).remainder(2).float()
ARTIFACT = dict(partition=torch.stack((1-MASK,MASK)),online_key=SETTINGS["prc"].online_key().to_dict())
MANIFEST = dict(id="unit-test",prompt_indices=[0,1],length=24)


def run(setting,seed=12345):
    return generate(ToyModel(),PROMPTS,setting,seed,0,ARTIFACT,MANIFEST,{"test":True})


def test_semantic_oracles_and_forced_native_fallback():
    assert semantic_checks()["passed"]


def test_bf16_model_logits_have_common_fp32_probabilities_and_stable_ties():
    logits = torch.zeros(2,256,dtype=torch.bfloat16)
    filtered,q,mask = truncate(logits)
    assert filtered.dtype==q.dtype==torch.float32
    assert torch.equal(mask.nonzero()[:,1],torch.arange(100).repeat(2))
    assert torch.all(q[:,100:]==0)
    assert torch.allclose(q[:,:100],torch.full((2,100),.01))
    with pytest.raises(ValueError):truncate(torch.full((2,256),float("nan")))


def test_ordinary_and_prc_generation_match_independent_reference_samplers():
    ordinary = run("null")
    torch.manual_seed(12345)
    model = ToyModel(filtered=True);ids = torch.tensor(PROMPTS);expected=[]
    for _ in range(24):
        token = torch.multinomial(model(ids)[:,-1].softmax(-1),1)
        expected.append(token);ids=token
    assert [r["token_ids"] for r in ordinary["responses"]]==torch.cat(expected,dim=1).tolist()
    sampler = load_online_sampler(ROOT/"watermark_expt.py",device=torch.device("cpu"))
    seeds = [derive_document_seed(12345,i) for i in range(2)]
    tokens,trace,details = sampler(ToyModel(filtered=True),torch.tensor(PROMPTS),24,SETTINGS["prc"].online_key(),
        ARTIFACT["partition"],document_seeds=seeds,return_trace_details=True,kv_cache_implementation="static")
    prc = run("prc")
    assert [r["token_ids"] for r in prc["responses"]]==tokens.tolist()
    np.testing.assert_array_equal([r["generation_partition1"] for r in prc["responses"]],trace)
    np.testing.assert_array_equal([r["prc_codeword_bits"] for r in prc["responses"]],details["prc_codeword_bits"])


@pytest.mark.parametrize("setting",list(SETTINGS))
def test_all_generators_reproduce_and_keep_support(setting):
    a,b,c = run(setting),run(setting),run(setting,67890)
    assert a==b
    assert a["identity"]["setting"]==c["identity"]["setting"]
    assert a["responses"][0]["token_ids"]!=c["responses"][0]["token_ids"]
    assert a["verification"]["support_violations"]==0
    model=ToyModel()
    for i,row in enumerate(a["responses"]):
        previous=PROMPTS[i][-1]
        for token in row["token_ids"]:
            _,_,keep=truncate(model(torch.tensor([[previous]]))[:,-1])
            assert keep[0,token]
            previous=token


def test_completion_replay_never_uses_prompt_and_matches_generation_after_same_history():
    tokens=torch.tensor([r["token_ids"] for r in run("prc")["responses"]])
    model=ToyModel();trace,outside=completion_trace(model,tokens,MASK)
    assert len(model.calls)==23 and trace.shape==outside.shape==(2,23)
    caches={id(cache) for _,cache in model.calls};assert len(caches)==1
    for pos,(ids,_) in enumerate(model.calls):
        assert torch.equal(ids,tokens[:,pos:pos+1])
        q=truncate(ToyModel()(ids)[:,-1])[1]
        assert torch.equal(trace[:,pos],partition_probability(q,MASK))
    shorter,_=completion_trace(ToyModel(),tokens.flip(0)[:,:12],MASK)
    assert torch.equal(shorter.flip(0),trace[:,:11])


def test_repetition_counts_and_paired_covariance_are_preserved():
    assert repetition([1,2,3,4,1,2,3,4])==pytest.approx({"repeated_4gram_fraction":.2,"distinct_3":4/6})
    records=[]
    for setting in SETTINGS:
        for n in (400,1024):
            for i in range(50):
                row=dict(setting=setting,length=n,prompt_index=i,self_bleu=i/100+(setting=="prc")*.01,
                    repeated_4gram_fraction=i/100,distinct_3=1-i/100)
                if setting!="null":row.update(detected=[True,True],tpr=1.)
                records.append(row)
    draws=np.random.default_rng(1).integers(0,50,(2000,50))
    _,contrasts=summarize_metrics(records,draws)
    primary=next(r for r in contrasts if r["primary"])
    assert primary["metrics"]["self_bleu"]["mean"]==pytest.approx(.01)
    assert primary["metrics"]["self_bleu"]["ci95"]==pytest.approx([.01,.01])
