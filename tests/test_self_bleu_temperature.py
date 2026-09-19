"""Temperature adapter versus original arithmetic; no model/GPU dispatch."""
import numpy as np
import pytest
import torch
from self_bleu import temperature as study
from self_bleu.validation import ROOT,load_online_sampler
from baseline_comparison.comparison_runner import generate_method
from online_prc import derive_document_seed
from test_self_bleu_topk import ToyModel,PROMPTS,MASK

ARTIFACT=dict(partition=torch.stack((1-MASK,MASK)).bfloat16(),online_key=study.SETTINGS["prc"].online_key().to_dict())
MANIFEST=dict(id="unit-test",prompt_indices=[0,1])


def run(setting,seed=12345):
    return study.generate(ToyModel(),PROMPTS,setting,seed,0,ARTIFACT,MANIFEST,{"test":True},24)


def test_native_temperature_and_fallback_are_exact():
    result=study.synthid_temperature_check()
    assert all(r["native_single_temperature_exact"] and r["ordinary_fallback_exact"] for r in result.values())


def test_scaling_preserves_bf16_and_does_not_scale_twice():
    model=ToyModel();tokens=torch.tensor(PROMPTS)
    raw=model(tokens)
    scaled=study.TemperatureModel(model)(tokens)
    assert scaled.dtype==torch.bfloat16 and torch.equal(scaled,raw/.7)
    assert not torch.equal(scaled,raw/.7/.7)
    assert not torch.equal(scaled.float().softmax(-1),(raw.float()/.7).softmax(-1))
    assert torch.equal(study.TemperatureModel(model,1)(tokens),raw)


@pytest.mark.parametrize("setting",list(study.SETTINGS))
def test_same_seed_reproduction_fixed_keys_and_original_sampler(setting):
    a,b,c=run(setting),run(setting),run(setting,67890)
    assert a["responses"]==b["responses"]
    assert a["identity"]["setting"]==c["identity"]["setting"]
    assert any(x["token_ids"]!=y["token_ids"] for x,y in zip(a["responses"],c["responses"]))
    assert a["anomalies"]["prc_draw_bucket_mismatches"]==0
    config=study.SETTINGS[setting]
    if setting=="prc":
        sampler=load_online_sampler(ROOT/"watermark_expt.py",device="cpu")
        args=(torch.tensor(PROMPTS),24,config.online_key(),ARTIFACT["partition"])
        kw=dict(document_seeds=[derive_document_seed(12345,i) for i in range(2)],return_trace_details=True,kv_cache_implementation="static")
        original=sampler(ToyModel(),*args,**kw)
        identity=sampler(study.TemperatureModel(ToyModel(),1),*args,**kw)
        assert torch.equal(original[0],identity[0]);np.testing.assert_array_equal(original[1],identity[1])
        expected=sampler(study.TemperatureModel(ToyModel()),*args,**kw)
        assert [r["token_ids"] for r in a["responses"]]==expected[0].tolist()
        np.testing.assert_array_equal([r["generation_partition1"] for r in a["responses"]],expected[1])
    else:
        kw=dict(method=config.method,seed=12345,synthid_keys=config.synthid_keys,max_new_tokens=24,device="cpu")
        original,_=generate_method(ToyModel(),PROMPTS,**kw)
        identity,_=generate_method(study.TemperatureModel(ToyModel(),1),PROMPTS,**kw)
        assert original==identity
        expected,_=generate_method(study.TemperatureModel(ToyModel()),PROMPTS,**kw)
        assert [r["token_ids"] for r in a["responses"]]==[r["token_ids"] for r in expected]


def test_completion_only_replay_preserves_original_bf16_calculation():
    tokens=torch.tensor([r["token_ids"] for r in run("prc")["responses"]])
    model=ToyModel();p,zero=study.replay(model,tokens,MASK.bfloat16())
    assert p.shape==zero.shape==(2,23)
    for i,(ids,cache) in enumerate(model.calls):
        assert torch.equal(ids,tokens[:,i:i+1])
        expected=((ToyModel()(ids)[:,-1]/.7).softmax(-1)*MASK.bfloat16()).sum(-1)
        assert torch.equal(p[:,i],expected.float())
    short,_=study.replay(ToyModel(),tokens.flip(0)[:,:12],MASK.bfloat16())
    assert torch.equal(short.flip(0),p[:,:11])


def test_primary_endpoint_and_paired_covariance():
    rows=[]
    for t in (.7,1.):
        for setting in study.SETTINGS:
            for n in (400,1024):
                for i in range(50):
                    row=dict(temperature=t,setting=setting,length=n,prompt_index=i,self_bleu=i/100-(setting=="prc")*.01,
                        repeated_4gram_fraction=i/100,distinct_3=1-i/100)
                    if setting!="null":row.update(detected=[True,False],tpr=.5)
                    rows.append(row)
    results,contrasts=study.summarize(rows,np.random.default_rng(1).integers(0,50,(2000,50)))
    assert len(results)==16 and len(contrasts)==20
    primary=[r for r in contrasts if r["primary"]];assert len(primary)==1
    assert primary[0]["temperature"]==.7 and primary[0]["length"]==1024
    assert primary[0]["metrics"]["self_bleu"]["ci95"]==pytest.approx([-.01,-.01])
