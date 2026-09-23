"""Finite-support decoder and completion-only contract checks; no GPU dispatch."""
import numpy as np
import pytest
import torch

from self_bleu.topk import (SETTINGS,truncate,partition_probability,prc_draw,
    semantic_checks,generate,completion_trace,repetition,summarize_metrics,
    replay_prefix_diagnostics,summarize_replay_diagnostics)
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


def diagnostic_fixture():
    p=np.full(1023,.25);bits=np.zeros(1023,dtype=np.uint8);outside=np.zeros(1023,dtype=bool)
    for position in (2,64,65,400,401,1024):outside[position-2]=True
    for position in (2,401):p[position-2]=0;bits[position-2]=1
    for position in (64,1024):p[position-2]=1
    # Consistent endpoints and a rare but possible bucket are not contradictions.
    p[8]=0;p[9]=1;bits[9]=1;p[10]=1e-8;bits[10]=1
    return p,bits,outside


def test_replay_prefix_boundaries_and_endpoint_cases_preserve_scoring():
    from detectors import _soft_tokens
    p,bits,outside=diagnostic_fixture();before=_soft_tokens(np.r_[0,bits],p,"map")
    short=replay_prefix_diagnostics(p,bits,outside,400)
    full=replay_prefix_diagnostics(p,bits,outside,1024)
    assert short["all"]["positions"]==399 and full["all"]["positions"]==1023
    assert short["all"]["counts"]["outside_top100"]==4
    assert full["all"]["counts"]["outside_top100"]==6
    assert short["all"]["counts"]["endpoint_contradiction"]==2
    assert full["all"]["counts"]["endpoint_contradiction"]==4
    assert short["early"]["positions"]==63 and short["later"]["positions"]==336
    assert short["early"]["counts"]["endpoint_contradiction"]==2
    assert short["later"]["counts"]["endpoint_contradiction"]==0
    assert full["later"]["counts"]["endpoint_contradiction"]==2
    assert short["all"]["rates"]["outside_top100"]==4/399
    assert full["all"]["counts"]["outside_without_endpoint"]==2
    for window in full.values():
        assert window["counts"]["endpoint_contradiction"]==window["counts"]["p1_zero_observed_one"]+window["counts"]["p1_one_observed_zero"]
    np.testing.assert_array_equal(before,_soft_tokens(np.r_[0,bits],p,"map"))
    assert before[0]==0 and before[1]==-1 and before[63]==1
    with pytest.raises(ValueError):replay_prefix_diagnostics(p[:-1],bits,outside,400)


def test_worker_saves_prefix_diagnostics_and_primary_detector_unchanged(monkeypatch):
    from self_bleu import topk_modal
    from detectors import detect_online_hoeffding
    from self_bleu.config import digest
    p,bits,outside=diagnostic_fixture();tokens=np.r_[0,bits].tolist()
    rows=[dict(response_id="fixture",completion_sha256=digest(tokens),prompt_index=0,response_index=0,token_ids=tokens)]
    tensor=torch.tensor;move=torch.Tensor.to
    # CPU scoring fixture supplies an already-tested replay trace, no Modal call.
    monkeypatch.setattr(torch,"tensor",lambda *a,**kw:tensor(*a,**(kw|{"device":"cpu"})))
    monkeypatch.setattr(torch.Tensor,"to",lambda self,*a,**kw:move(self,*(('cpu',) if a==('cuda',) else a),**kw))
    monkeypatch.setattr(topk_modal,"replay_checked",lambda *args:(tensor(p[None]),tensor(outside[None])))
    report=topk_modal.score_prc(None,ARTIFACT,{"responses":rows},{})
    row=report["rows"][0]
    assert "replay_outside_top100" not in row  # No ambiguous full-response scalar.
    assert row["diagnostics_by_prefix"]["400"]["all"]["counts"]["outside_top100"]==4
    assert row["diagnostics_by_prefix"]["1024"]["all"]["counts"]["outside_top100"]==6
    for n in (400,1024):
        decision,direct=detect_online_hoeffding(ARTIFACT["online_key"],tensor(tokens[:n]),p[:n-1],ARTIFACT["partition"],fpr=.001,return_info=True)
        assert row["results"][str(n)]["decision"]==decision
        for key in ("V","statistic"):assert row["results"][str(n)][key]==direct[key]


def test_replay_summary_separates_prc_and_null_and_uses_prefix_denominators():
    p,bits,outside=diagnostic_fixture();records=[]
    for cohort in ("watermarked","pilot_null"):
        for n in (400,1024):
            diagnostic=replay_prefix_diagnostics(p,bits,outside if cohort=="watermarked" else np.zeros_like(outside),n)
            for i in range(50):
                for response in (0,1):records.append(dict(setting="prc",cohort=cohort,length=n,prompt_index=i,response_index=response,replay_diagnostics=diagnostic))
    draws=np.random.default_rng(1).integers(0,50,(2000,50))
    summary=summarize_replay_diagnostics(records,draws)
    assert len(summary)==12
    short=next(r for r in summary if (r["source"],r["length"],r["window"])==("prc",400,"all"))
    assert short["positions"]==39900 and short["metrics"]["outside_top100"]["count"]==400
    assert short["metrics"]["outside_top100"]["rate"]["mean"]==pytest.approx(4/399)
    assert all(r["metrics"]["outside_top100"]["count"]==0 for r in summary if r["source"]=="null")
