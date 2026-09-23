"""New full-vocabulary paths checked against independent existing samplers."""
import numpy as np
import pytest
import torch

from self_bleu import full_vocab as study
from self_bleu.validation import ROOT,load_online_sampler
from online_prc import derive_document_seed
from test_self_bleu_topk import ToyModel,PROMPTS,ARTIFACT,MASK,diagnostic_fixture


@pytest.fixture(autouse=True)
def numerical_upstream_only():
    from baseline_comparison.textseal_completion import load_upstream_detector
    load_upstream_detector()


class FloatModel(ToyModel):
    def forward(self,ids,cache=None):
        return super().forward(ids,cache).float()


def run(setting,seed=12345):
    # Upstream TextSeal's CPU helper accepts one context; GPU validates batch 50.
    prompts=PROMPTS[:1] if setting=="textseal" else PROMPTS
    manifest=dict(id="test-full-vocab",prompt_indices=list(range(len(prompts))),length=24)
    return study.generate(ToyModel(),prompts,setting,seed,0,ARTIFACT,manifest,{"test":True})


def test_full_support_and_probabilities():
    scores,q,keep=study.full_distribution(torch.zeros(2,256,dtype=torch.bfloat16))
    assert scores.dtype==q.dtype==torch.float32 and keep.all()
    assert torch.equal(q,torch.full((2,256),1/256))
    assert study.semantic_checks()["passed"]


def test_ordinary_and_prc_match_existing_samplers():
    actual=run("null");torch.manual_seed(12345)
    model=FloatModel();ids=torch.tensor(PROMPTS);expected=[]
    for _ in range(24):
        token=torch.multinomial(model(ids)[:,-1].softmax(-1),1)
        expected.append(token);ids=token
    assert [r["token_ids"] for r in actual["responses"]]==torch.cat(expected,dim=1).tolist()
    sampler=load_online_sampler(ROOT/"watermark_expt.py",device=torch.device("cpu"))
    tokens,p,details=sampler(FloatModel(),torch.tensor(PROMPTS),24,study.SETTINGS["prc"].online_key(),
        ARTIFACT["partition"],document_seeds=[derive_document_seed(12345,i) for i in range(2)],
        return_trace_details=True,kv_cache_implementation="static")
    actual=run("prc")
    assert [r["token_ids"] for r in actual["responses"]]==tokens.tolist()
    np.testing.assert_array_equal([r["generation_partition1"] for r in actual["responses"]],p)
    np.testing.assert_array_equal([r["prc_codeword_bits"] for r in actual["responses"]],details["prc_codeword_bits"])


@pytest.mark.parametrize("setting",list(study.SETTINGS))
def test_seed_reproduction_and_unchanged_keys(setting):
    a,b,c=run(setting),run(setting),run(setting,67890)
    assert a==b
    assert a["identity"]["setting"]==c["identity"]["setting"]
    assert a["verification"]["support_checks"]==24*len(a["responses"])
    if setting not in ("null","prc"):
        assert all(len(r["repeat_fallback"])==24 for r in a["responses"])


def test_completion_only_alignment_and_prefix_diagnostics():
    tokens=torch.tensor([r["token_ids"] for r in run("prc")["responses"]])
    model=ToyModel();p,zero=study.completion_trace(model,tokens,MASK)
    assert p.shape==zero.shape==(2,23) and len(model.calls)==23
    for i,(ids,cache) in enumerate(model.calls):
        assert torch.equal(ids,tokens[:,i:i+1])
        assert torch.equal(p[:,i],study.partition_probability(FloatModel()(ids)[:,-1].softmax(-1),MASK))
    short,_=study.completion_trace(ToyModel(),tokens.flip(0)[:,:12],MASK)
    assert torch.equal(short.flip(0),p[:,:11])
    p,b,o=diagnostic_fixture()
    short=study.replay_prefix_diagnostics(p,b,o,400)
    full=study.replay_prefix_diagnostics(p,b,o,1024)
    assert short["all"]["counts"]["zero_token_probability"]==4
    assert full["all"]["counts"]["zero_token_probability"]==6
    assert short["all"]["counts"]["endpoint_contradiction"]==2
    assert full["all"]["counts"]["endpoint_contradiction"]==4


def test_six_arm_paired_summary():
    rows=[]
    for setting in study.SETTINGS:
        for n in (400,1024):
            for i in range(50):
                row=dict(setting=setting,length=n,prompt_index=i,self_bleu=i/100+(setting=="prc")*.01,
                    repeated_4gram_fraction=i/100,distinct_3=1-i/100)
                if setting!="null":row.update(detected=[True,False],tpr=.5)
                rows.append(row)
    results,contrasts=study.summarize_metrics(rows,np.random.default_rng(1).integers(0,50,(2000,50)))
    assert len(results)==12 and len(contrasts)==18
    primary=next(r for r in contrasts if r["primary"])
    assert primary["metrics"]["self_bleu"]["ci95"]==pytest.approx([.01,.01])
