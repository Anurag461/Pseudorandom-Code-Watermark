"""One matched top-100 study: shared decoder, local manifest and analysis.

Historical samplers/results stay unchanged. No GPU dispatch occurs here.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import torch

from .config import StudySetting, digest
from .pilot import paired_interval
from .repeat import upstream_hashes
from .validation import ROOT, RATE, save, sha

SETUP = ROOT/"outputs/self_bleu_topk/matched_v2"
DECODER = dict(top_k=100,temperature=1.,top_p=1.,probability_dtype="float32",
    order="top-100 base logits, then watermark; native fallback uses the same truncated base",
    tie_break="higher logits first; lower token ID at boundary ties",masked_logit=-1e12)
SETTINGS = {"null":StudySetting("null"),"prc":StudySetting("online_prc",eta=.05),
            **{f"synthid_depth{d}":StudySetting("synthid_text",depth=d) for d in (2,10,30)}}
TIMEOUTS = {"validate":600,"batch":1800}
REPLAY_EARLY_END = 64  # One-based completion positions 2..64 versus 65..n.


def truncate(logits, k=100):
    """Exactly k candidates, deterministic boundary ties, FP32 normalization."""
    scores = logits.float()
    if scores.ndim!=2 or scores.shape[1]<k or not torch.isfinite(scores).all():
        raise ValueError("expected finite B x V logits with V >= k")
    cutoff = torch.topk(scores,k,dim=-1).values[:,-1:]
    above, tied = scores>cutoff, scores==cutoff
    keep = above | (tied & (tied.to(torch.int32).cumsum(-1)<=k-above.sum(-1,keepdim=True)))
    if not torch.all(keep.sum(-1)==k):
        raise ValueError("top-k support size differs")
    filtered = scores.masked_fill(~keep,DECODER["masked_logit"])
    probs = torch.softmax(filtered,dim=-1)
    if torch.any(probs[~keep]!=0) or not torch.isfinite(probs).all():
        raise ValueError("truncation leaked support or produced nonfinite probabilities")
    return filtered,probs,keep


def partition_probability(probs,part1):
    """Normalized bucket mass, including exact empty/full-support endpoints."""
    one = part1.to(device=probs.device,dtype=probs.dtype)
    p1 = (probs*one).sum(-1).clamp(0,1)
    positive = probs>0
    p1 = torch.where((positive & (one==1)).any(-1),p1,torch.zeros_like(p1))
    p1 = torch.where((positive & (one==0)).any(-1),p1,torch.ones_like(p1))
    return p1


def prc_draw(filtered,probs,partition,xi,bucket_uniform,token_uniform):
    """Original online-PRC channel and position-addressed inverse-CDF draw."""
    p1 = partition_probability(probs,partition[1])
    bern = torch.where(p1<=.5,2*xi*p1,1-2*(1-xi)*(1-p1)).clamp(0,1)
    bucket = (bucket_uniform<bern.double()).long()
    conditional = torch.softmax(filtered.masked_fill(partition[bucket]==0,float("-inf")),dim=-1)
    if not torch.isfinite(conditional).all():
        raise ValueError("PRC selected an empty bucket")
    cumulative = conditional.cumsum(-1).clamp_max(1.)
    cumulative[:,-1] = 1.
    token = torch.searchsorted(cumulative,token_uniform[:,None].to(conditional.dtype),right=False).clamp_max(conditional.shape[-1]-1)
    bad = conditional.gather(1,token).squeeze(1)==0
    # Floating-point u==1 / trailing CDF roundoff may select a zero-mass tail.
    # Keep the existing CDF everywhere else; repair only an impossible draw.
    if bad.any():
        last = torch.where(conditional>0,torch.arange(conditional.shape[1],device=conditional.device),-1).max(-1).values
        first = torch.where(conditional>0,torch.arange(conditional.shape[1],device=conditional.device),conditional.shape[1]).min(-1).values
        replacement = torch.where(token_uniform<=0,first,last)
        token[bad,0] = replacement[bad]
    if torch.any(probs.gather(1,token)==0) or not torch.equal(partition[1,token[:,0]].long(),bucket):
        raise ValueError("PRC draw violates truncated support or bucket")
    return token,p1,bucket,int(bad.sum())


def checked_synthid_call(processor,history,filtered,probs,keep):
    old = (torch.zeros((len(history),processor.context_history_size),dtype=torch.long,device=filtered.device)
           if processor.state is None else processor.state.context_history.clone())
    updated,indices,base = processor.watermarked_call(history,filtered)
    output = torch.softmax(updated.float(),dim=-1)
    repeated = (old==processor.state.context_history[:,:1]).any(-1)
    if not torch.equal(indices,torch.arange(filtered.shape[1],device=filtered.device)[None].expand_as(indices)):
        raise ValueError("unexpected SynthID internal truncation")
    if torch.any(output[~keep]!=0) or not torch.isfinite(output).all():
        raise ValueError("SynthID watermark/fallback escapes top-100 support")
    if repeated.any() and not torch.equal(output[repeated],probs[repeated]):
        raise ValueError("native fallback differs from common ordinary distribution")
    return output,repeated


@torch.no_grad()
def generate(model,prompts,setting_name,seed,response,artifact,manifest,execution):
    from qwen import StaticKVCache
    from online_prc import OnlinePRCEncoder,derive_document_seed,document_uniform
    from baseline_comparison.official import synthid_processor
    setting = SETTINGS[setting_name]
    device = next(model.parameters()).device
    ids = torch.tensor(prompts,dtype=torch.long,device=device)
    torch.manual_seed(seed)
    if device.type=="cuda":torch.cuda.manual_seed_all(seed)
    partition = artifact["partition"].to(device)
    document_seeds = [derive_document_seed(seed,i) for i in manifest["prompt_indices"]]
    encoder = OnlinePRCEncoder(setting.online_key(),document_seeds) if setting_name=="prc" else None
    processor = synthid_processor(device,keys=setting.synthid_keys) if setting.method=="synthid_text" else None
    cache = StaticKVCache(max_length=50+manifest["length"])
    logits = model(ids,cache=cache)[:,-1]
    tokens,ptrace,bits,repeats = [],[],[],[]
    corrected = 0
    for pos in range(manifest["length"]):
        filtered,probs,keep = truncate(logits)
        if encoder is not None:
            xi = torch.tensor(encoder.next_bits(),dtype=torch.float32,device=device)
            u = torch.tensor([document_uniform(s,"lm-bucket/v1",pos) for s in document_seeds],dtype=torch.float64,device=device)
            v = torch.tensor([document_uniform(s,"lm-token/v1",pos) for s in document_seeds],dtype=torch.float32,device=device)
            token,p1,_,fixes = prc_draw(filtered,probs,partition,xi,u,v)
            corrected += fixes;ptrace.append(p1.cpu());bits.append(xi.cpu())
        elif processor is not None:
            distribution,repeated = checked_synthid_call(processor,ids,filtered,probs,keep)
            token = torch.multinomial(distribution,1)
            repeats.append(repeated.cpu())
        else:
            token = torch.multinomial(probs,1)
        if not keep.gather(1,token).all():raise ValueError("generated token outside common support")
        tokens.append(token.cpu())
        ids = torch.cat((ids,token),dim=1)
        if pos+1<manifest["length"]:logits = model(token,cache=cache)[:,-1]
    tokens = torch.cat(tokens,dim=1).tolist()
    identity = dict(study_id=manifest["id"],setting=setting.identity(),decoder=DECODER,
        sampling_seed=seed,response_index=response,prompt_indices=manifest["prompt_indices"],
        prompt_sha256=[digest(p) for p in prompts],execution=execution,length=manifest["length"])
    bid = digest(identity)
    rows = []
    for i,seq in enumerate(tokens):
        row = dict(response_id=f"{bid}/p{i:04d}/r{response}",prompt_index=i,response_index=response,
                   sampling_seed=seed,token_ids=seq,completion_sha256=digest(seq))
        if encoder is not None:
            row["generation_partition1"] = torch.stack(ptrace,dim=1)[i].tolist()
            row["prc_codeword_bits"] = torch.stack(bits,dim=1)[i].to(torch.uint8).tolist()
            row["document_seed"] = document_seeds[i]
        if repeats:row["native_repeat_fallback"] = torch.stack(repeats,dim=1)[i].tolist()
        rows.append(row)
    return dict(batch_id=bid,identity=identity,responses=rows,
                verification=dict(support_checks=len(prompts)*manifest["length"],support_violations=0,cdf_boundary_repairs=corrected))


def semantic_checks(device="cpu"):
    """Independent finite-support oracles, including repeats and empty buckets."""
    from baseline_comparison.official import synthid_processor
    from synthid_text.logits_processing import update_scores
    raw = torch.stack((torch.zeros(256),torch.linspace(-4,4,256),torch.arange(256).remainder(7).float())).to(device)
    filtered,q,keep = truncate(raw)
    expected_ids = torch.argsort(raw.float(),descending=True,stable=True)[:,:100]
    expected = torch.zeros_like(keep).scatter(1,expected_ids,True)
    if not torch.equal(keep,expected):raise ValueError("common truncation differs from independent stable-sort oracle")
    partition = torch.stack((torch.arange(256)%2==0,torch.arange(256)%2==1)).float().to(device)
    p1 = partition_probability(q,partition[1])
    reference = (q.double()*partition[1].double()).sum(-1)/q.double().sum(-1)
    if not torch.allclose(p1.double(),reference,atol=2e-7,rtol=0):raise ValueError("PRC truncated bucket mass differs")
    for part in (partition,torch.stack((torch.ones(256),torch.zeros(256))).to(device),torch.stack((torch.zeros(256),torch.ones(256))).to(device)):
        p = partition_probability(q,part[1]);mixture = torch.zeros_like(q)
        for xi in (0.,1.):
            x = torch.full((3,),xi,device=device)
            bprob = torch.where(p<=.5,2*x*p,1-2*(1-x)*(1-p)).clamp(0,1)
            for bucket in (0,1):
                weight = bprob if bucket else 1-bprob
                mass = (q*part[bucket]).sum(-1,keepdim=True)
                conditional = q*part[bucket]/mass.clamp_min(1e-30)
                if torch.any((mass[:,0]==0)&(weight!=0)):raise ValueError("nonzero chance of empty PRC bucket")
                mixture += .5*weight[:,None]*conditional
            for u in (0.,.123,.999999,1.):
                # Bucket uniform remains strictly below one, as in ordinary RNG use.
                token,actual,_,_ = prc_draw(filtered,q,part,x,torch.tensor([.01,.49,.99],device=device,dtype=torch.float64),torch.full((3,),u,device=device))
                if not torch.equal(actual,p) or not keep.gather(1,token).all():raise ValueError("PRC draw/bucket audit failed")
        if not torch.allclose(mixture,q,atol=1e-7,rtol=1e-6):raise ValueError("PRC averaged channel does not preserve truncated base")
    forced = {}
    for depth in (2,10,30):
        processor = synthid_processor(device,keys=StudySetting("synthid_text",depth=depth).synthid_keys)
        counts = 0
        for token in [1,2,3]*4:
            histories = torch.full((3,50),token,device=device,dtype=torch.long)
            output,repeated = checked_synthid_call(processor,histories,filtered,q,keep)
            # Match the official dense candidate layout. A zero-stride expand
            # changes reduction rounding even with identical g-values.
            indices = torch.stack([torch.arange(256,device=device) for _ in range(3)])
            hashes,_ = processor._compute_keys(processor.state.context,indices)
            oracle = torch.softmax(update_scores(filtered,processor.get_gvals(hashes)),dim=-1)
            if not torch.equal(output[~repeated],oracle[~repeated]):raise ValueError("truncation-before-watermarking differs")
            counts += int(repeated.sum())
        if not counts:raise ValueError("probe failed to trigger fallback")
        forced[str(depth)] = counts
    return dict(passed=True,stable_top100_exact=True,common_fp32_base=True,prc_bucket_probability_checked=True,
        empty_full_bucket_endpoints=True,prc_channel_preserves_truncated_base=True,inverse_cdf_support_checked=True,
        native_synthid_fallback_on_truncated_support=True,forced_repeat_counts=forced)


@torch.no_grad()
def completion_trace(model,tokens,part1):
    """Raw completion only; fresh position-zero cache; coordinate one abstains."""
    from qwen import StaticKVCache
    cache = StaticKVCache(max_length=tokens.shape[1]-1)
    trace = [];outside = []
    for pos in range(tokens.shape[1]-1):
        logits = model(tokens[:,pos:pos+1],cache=cache)[:,-1]
        _,probs,keep = truncate(logits)
        trace.append(partition_probability(probs,part1).cpu())
        outside.append((~keep.gather(1,tokens[:,pos+1:pos+2]).squeeze(1)).cpu())
    return torch.stack(trace,dim=1),torch.stack(outside,dim=1)


def validate_manifest(manifest,root=ROOT):
    if digest({k:v for k,v in manifest.items() if k!="id"})!=manifest["id"]:raise ValueError("manifest identity differs")
    if (manifest["decoder"]!=DECODER or manifest["settings"]!={n:s.identity() for n,s in SETTINGS.items()}
            or manifest["prompt_indices"]!=list(range(50)) or manifest["seeds"]!=[12345,67890] or manifest["length"]!=1024
            or manifest["primary_length"]!=1024 or manifest["primary_contrast"]!=["prc","synthid_depth2"]):
        raise ValueError("study scope differs")
    for name,h in manifest["code_sha256"].items():
        if sha(Path(root)/name)!=h:raise ValueError(f"source changed: {name}")
    if sha(Path(root)/"prompts.jsonl")!=manifest["prompt_sha256"]:raise ValueError("prompts changed")
    cost = manifest["cost"]
    if (cost["reserved_total_usd"]>200 or cost["timeouts"]!=TIMEOUTS or cost["resource_usd_per_second"]!=RATE
            or cost["reserved_total_usd"]!=cost["previous_planning_charge_usd"]+(sum(TIMEOUTS.values())+4)*RATE+.5):
        raise ValueError("budget/scope exceeded")


def prepare(setup):
    prior = json.loads((ROOT/"outputs/self_bleu_depth/depth2_30_v1/manifest.json").read_text())
    from .depth import validate
    validate(prior)
    pilot = json.loads((ROOT/"outputs/self_bleu_pilot/stage_a_v2/manifest.json").read_text())
    generation = json.loads((ROOT/"outputs/self_bleu_validation/step3-v4/generation_report.json").read_text())
    bid = next(r for r in generation["settings"] if r["setting"]["method"]=="null")["batches"][0]
    raw = ROOT/f"outputs/self_bleu_validation/raw/743658bc40e7f52c910d9538266bbd0ff461bba949e2a842cc8af36fa32507d8/batches/{bid}.json"
    if sha(raw)!=generation["files"][f"batches/{bid}.json"]:raise ValueError("common-history source changed")
    history = json.loads(raw.read_text())
    previous = json.loads((ROOT/"outputs/self_bleu_depth/short_prefixes/summary.json").read_text())["cumulative_planning_charge_usd"]
    previous_setup = ROOT/"outputs/self_bleu_topk/matched_v1"
    previous_manifest = json.loads((previous_setup/"manifest.json").read_text())
    previous_report = json.loads((previous_setup/"validate_report.json").read_text())
    if not previous_report["passed"] or previous_report["manifest_id"]!=previous_manifest["id"]:
        raise ValueError("prior validation record differs")
    previous += previous_report["resource_estimate_usd"]
    names = sorted(set(prior["code_sha256"])|{"self_bleu/topk.py","self_bleu/topk_modal.py"})
    codeword_hashes = {}
    prc_batches = next(r for r in generation["settings"] if r["setting"]["method"]=="online_prc")["batches"]
    for response,batch_id in enumerate(prc_batches[:2]):
        path = raw.parent/f"{batch_id}.json"
        if sha(path)!=generation["files"][f"batches/{batch_id}.json"]:raise ValueError("PRC reference changed")
        saved = json.loads(path.read_text())
        codeword_hashes[str(response)] = [digest(r["generation_diagnostics"]["prc_codeword_bits"]) for r in saved["responses"]]
    manifest = dict(schema_version=1,source_parent_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        decoder=DECODER,settings={n:s.identity() for n,s in SETTINGS.items()},model=prior["model"],artifact=pilot["artifact"],
        generation_runtime=prior["generation_runtime"],upstream_sha256=upstream_hashes(),prc_codeword_sha256=codeword_hashes,
        protocol="completion_only_raw_abstain_v1",repeat_fallback=True,
        code_sha256={n:sha(ROOT/n) for n in names},test_sha256=sha(ROOT/"tests/test_self_bleu_topk.py"),
        prompt_indices=list(range(50)),prompt_sha256=sha(ROOT/"prompts.jsonl"),seeds=[12345,67890],length=1024,
        primary_length=1024,secondary_lengths=[400],primary_contrast=["prc","synthid_depth2"],
        analysis=dict(bootstrap_seed=20260918,bootstrap_resamples=2000,unit="50 paired prompt clusters",nominal_fpr=.001,
            metrics=["self_bleu","repeated_4gram_fraction","distinct_3","detection"],pilot_null="matched top-100 ordinary responses only",
            replay_diagnostics=dict(prefixes=[400,1024],early_end=REPLAY_EARLY_END,position_indexing="one-based completion; position 1 abstains",
                events=["token outside replay top-100","p1=0 with observed bucket 1 or p1=1 with observed bucket 0"],
                scoring="diagnostic only; no dropping, reindexing or detector change")),
        previous_validation=dict(manifest_id=previous_manifest["id"],resource_estimate_usd=previous_report["resource_estimate_usd"],
            files={str((previous_setup/name).relative_to(ROOT)):sha(previous_setup/name) for name in ("manifest.json","validate_report.json")},
            disposition="validation completed; no responses generated; superseded for prefix-specific diagnostics"),
        common_histories=dict(prompt_indices=[0,7,19,31,49],positions=[0,32,128,400,1023],
            source_local=str(raw.relative_to(ROOT)),source_sha256=sha(raw),
            source_remote=f"self_bleu_validation/{generation['source_manifest_id']}/batches/{bid}.json",
            completion_sha256=[r["completion_sha256"] for r in history["responses"]],
            decoders=["full_vocab_fp32_reference","top100_fp32"],
            measurements=["collision_probability","maximum_probability","top100_retained_mass","base_top100_retained_mass"]),
        cache_policy=dict(reused_responses=0,reason="old full-vocabulary decoder and probability arithmetic are incompatible",
            reuse="checkpoint, tokenizer, canonical prompts and fixed PRC key/partition only; new manifest namespaces for all responses/replay"),
        cost=dict(previous_planning_charge_usd=previous,resource_usd_per_second=RATE,timeouts=TIMEOUTS,
            overhead_allowance_usd=.5,reserved_total_usd=previous+(sum(TIMEOUTS.values())+4)*RATE+.5,
            ceiling_usd=200,rate_basis="frozen planning rate from prior studies, not settled billing"),
        stop_rule="Exactly 500 full responses at these five settings. No temperature/depth/eta expansion or automatic retry.")
    manifest["id"] = digest(manifest);validate_manifest(manifest);save(setup/"manifest.json",manifest)
    return {"id":manifest["id"],"cost":manifest["cost"],"new_responses":500}


def collect(setup,stage,download=False):
    manifest = json.loads((setup/"manifest.json").read_text());validate_manifest(manifest)
    remote = f"self_bleu_topk/{manifest['id']}/{stage}"
    if download:
        import modal
        volume = modal.Volume.from_name("prc-completion-only",create_if_missing=False)
    path = setup/f"{stage}_report.json"
    if download and not path.exists():path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
    report = json.loads(path.read_text())
    if not report["passed"] or report["manifest_id"]!=manifest["id"]:raise ValueError("stage did not pass")
    for name,h in report["files"].items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:raise ValueError("unsafe path")
        local = setup/"raw"/stage/name
        if download and not local.exists():
            local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
        if sha(local)!=h:raise ValueError(f"artifact differs: {name}")
    return manifest,report


def repetition(tokens):
    def count(n):return len({tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)})/(len(tokens)-n+1)
    return {"repeated_4gram_fraction":1-count(4),"distinct_3":count(3)}


def replay_prefix_diagnostics(probabilities,observed_buckets,outside,n):
    """Count aligned raw-completion replay events without modifying evidence.

    Array entry j belongs to completion position j+2 (one-based). Endpoints
    refer to the saved FP32 p1 scalar, including any rounding to 0 or 1.
    """
    p,b,o = np.asarray(probabilities),np.asarray(observed_buckets),np.asarray(outside)
    if (p.ndim!=1 or p.shape!=b.shape or p.shape!=o.shape or not 2<=n<=len(p)+1
            or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or not np.isin(b,[0,1]).all() or not np.isin(o,[False,True]).all()):
        raise ValueError("replay diagnostics require aligned T-1 probabilities, bits and support flags")
    zero_one = (p==0)&(b==1);one_zero = (p==1)&(b==0)
    contradictory = zero_one|one_zero;o=o.astype(bool)
    events = dict(outside_top100=o,endpoint_contradiction=contradictory,
        p1_zero_observed_one=zero_one,p1_one_observed_zero=one_zero,
        outside_and_endpoint=o&contradictory,outside_without_endpoint=o&~contradictory)
    windows = {"all":(2,n),"early":(2,min(n,REPLAY_EARLY_END))}
    if n>REPLAY_EARLY_END:windows["later"]=(REPLAY_EARLY_END+1,n)
    output = {}
    for name,(a,z) in windows.items():
        counts = {key:int(flags[a-2:z-1].sum()) for key,flags in events.items()}
        output[name] = dict(first_position=a,last_position=z,positions=z-a+1,
            counts=counts,rates={key:value/(z-a+1) for key,value in counts.items()})
    return output


def summarize_replay_diagnostics(score_records,draws):
    summaries = []
    for cohort in ("watermarked","pilot_null"):
        for n in (1024,400):
            rows = [r for r in score_records if r["setting"]=="prc" and r["cohort"]==cohort and r["length"]==n]
            if len(rows)!=100 or {(r["prompt_index"],r["response_index"]) for r in rows}!={(i,j) for i in range(50) for j in (0,1)}:
                raise ValueError("replay diagnostic coverage differs")
            for window in ("all","early","later"):
                details = [r["replay_diagnostics"][window] for r in rows]
                positions = sum(r["positions"] for r in details)
                metrics = {}
                for event in details[0]["counts"]:
                    rates = [np.mean([r["replay_diagnostics"][window]["rates"][event] for r in rows if r["prompt_index"]==i]) for i in range(50)]
                    metrics[event] = dict(count=sum(r["counts"][event] for r in details),rate=paired_interval(rates,draws),
                        responses_with_event=sum(r["counts"][event]>0 for r in details))
                summaries.append(dict(source="prc" if cohort=="watermarked" else "null",length=n,window=window,
                    first_position=details[0]["first_position"],last_position=details[0]["last_position"],
                    positions=positions,responses=100,prompts=50,metrics=metrics))
    return summaries


def summarize_metrics(prompt_rows,draws):
    vectors,results = {},[]
    for setting in SETTINGS:
        for n in (1024,400):
            rows = sorted((r for r in prompt_rows if r["setting"]==setting and r["length"]==n),key=lambda r:r["prompt_index"])
            if [r["prompt_index"] for r in rows]!=list(range(50)):raise ValueError("prompt pairs incomplete")
            metrics = ["self_bleu","repeated_4gram_fraction","distinct_3"]+([] if setting=="null" else ["tpr"])
            vectors[setting,n] = {m:np.array([r[m] for r in rows]) for m in metrics}
            results.append(dict(setting=setting,length=n,metrics={m:paired_interval(v,draws) for m,v in vectors[setting,n].items()},
                detected=None if setting=="null" else sum(sum(r["detected"]) for r in rows),responses=100,prompts=50))
    pairs = [("prc",f"synthid_depth{d}") for d in (2,10,30)]
    pairs += [(s,"null") for s in SETTINGS if s!="null"]
    contrasts = [dict(left=a,right=b,length=n,primary=(a,b,n)==("prc","synthid_depth2",1024),
        metrics={m:paired_interval(v-vectors[b,n][m],draws) for m,v in vectors[a,n].items() if m in vectors[b,n]})
        for a,b in pairs for n in (1024,400)]
    return results,contrasts


def analyze(setup):
    import sacrebleu
    from sacrebleu.metrics import BLEU
    from tokenizers import Tokenizer
    from .depth import score_completions
    manifest,report = collect(setup,"batch")
    _,validation = collect(setup,"validate")
    if upstream_hashes()!=manifest["upstream_sha256"] or sacrebleu.__version__!="2.4.3":raise ValueError("analysis implementation differs")
    tokenizer = ROOT/"outputs/self_bleu_pilot/stage_a_v2/raw/tokenizer.json"
    if sha(tokenizer)!=manifest["model"]["tokenizer_sha256"]:raise ValueError("tokenizer changed")
    decoder = Tokenizer.from_file(str(tokenizer))
    bleu = BLEU(tokenize="13a",smooth_method="exp",effective_order=True,lowercase=False)
    prompts = [json.loads(line)["prompt_tokens"] for line in (ROOT/"prompts.jsonl").read_text().splitlines()][:50]
    groups = {};support_checks=repairs=0
    for setting in SETTINGS:
        groups[setting] = []
        for response,seed in enumerate(manifest["seeds"]):
            batch = json.loads((setup/f"raw/batch/batches/{setting}_r{response}.json").read_text())
            identity = dict(study_id=manifest["id"],setting=manifest["settings"][setting],decoder=DECODER,sampling_seed=seed,
                response_index=response,prompt_indices=manifest["prompt_indices"],prompt_sha256=[digest(p) for p in prompts],
                execution=manifest["generation_runtime"],length=1024)
            if batch["identity"]!=identity or batch["batch_id"]!=digest(identity) or len(batch["responses"])!=50:raise ValueError("batch identity differs")
            for i,row in enumerate(batch["responses"]):
                if (row["prompt_index"]!=i or row["response_index"]!=response or row["sampling_seed"]!=seed
                        or len(row["token_ids"])!=1024 or row["completion_sha256"]!=digest(row["token_ids"])
                        or row["response_id"]!=f"{batch['batch_id']}/p{i:04d}/r{response}"):
                    raise ValueError("response identity differs")
            verification = batch["verification"]
            if verification["support_violations"] or verification["support_checks"]!=51200:raise ValueError("support audit incomplete")
            support_checks+=verification["support_checks"];repairs+=verification["cdf_boundary_repairs"]
            groups[setting].extend(batch["responses"])
    draws = np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0,50,(manifest["analysis"]["bootstrap_resamples"],50))
    score_records,score_map,null_counts = [],{},[]
    for setting in SETTINGS:
        if setting=="null":continue
        for cohort,rows in (("watermarked",groups[setting]),("pilot_null",groups["null"])):
            if setting=="prc":
                scored = []
                for response in (0,1):
                    source = "prc" if cohort=="watermarked" else "null"
                    replay = json.loads((setup/f"raw/batch/replay/{source}_r{response}.json").read_text())
                    if replay["decoder"]!=DECODER or replay["raw_completion_only"] is not True:raise ValueError("replay protocol differs")
                    if not (len(replay["rows"])==len(replay["probabilities_2_to_T"])==len(replay["observed_buckets_2_to_T"])==len(replay["replay_outside_top100"])==50):
                        raise ValueError("replay diagnostic vectors incomplete")
                    for j,row in enumerate(replay["rows"]):
                        if any(len(replay[key][j])!=1023 for key in ("probabilities_2_to_T","observed_buckets_2_to_T","replay_outside_top100")):
                            raise ValueError("replay must have exactly 1023 aligned positions")
                        for n in (400,1024):
                            diagnostic = replay_prefix_diagnostics(replay["probabilities_2_to_T"][j],replay["observed_buckets_2_to_T"][j],replay["replay_outside_top100"][j],n)
                            if diagnostic!=row["diagnostics_by_prefix"][str(n)]:raise ValueError("saved prefix diagnostics differ")
                            scored.append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(
                                length=n,score=row["results"][str(n)],replay_diagnostics=diagnostic,
                                replay_outside_top100=diagnostic["all"]["counts"]["outside_top100"],
                                replay_endpoint_contradictions=diagnostic["all"]["counts"]["endpoint_contradiction"]))
            else:scored = score_completions(rows,SETTINGS[setting].depth,[400,1024])
            expected = {r["response_id"]:r for r in rows}
            if len(scored)!=200 or len({(r["response_id"],r["length"]) for r in scored})!=200:raise ValueError("detection coverage differs")
            for row in scored:
                original = expected[row["response_id"]]
                if any(row[k]!=original[k] for k in ("completion_sha256","prompt_index","response_index")):raise ValueError("detector join differs")
                record = {**row,"setting":setting,"cohort":cohort};score_records.append(record)
                score_map[setting,row["response_id"],row["length"]] = bool(row["score"]["decision"])
            if cohort=="pilot_null":
                for n in (1024,400):
                    detected = [np.mean([score_map[setting,r["response_id"],n] for r in rows if r["prompt_index"]==i]) for i in range(50)]
                    null_counts.append(dict(setting=setting,length=n,false_positives=int(round(sum(detected)*2)),responses=100,
                        prompts=50,fpr=paired_interval(detected,draws)))
    prompt_rows,response_metrics,fallback = [],[],[]
    for setting,rows in groups.items():
        by_pair = {(r["prompt_index"],r["response_index"]):r for r in rows}
        for n in (1024,400):
            for i in range(50):
                pair = [by_pair[i,r] for r in (0,1)]
                texts = [decoder.decode(r["token_ids"][:n],skip_special_tokens=True) for r in pair]
                value = (bleu.sentence_score(texts[0],[texts[1]]).score+bleu.sentence_score(texts[1],[texts[0]]).score)/200
                reps = [repetition(r["token_ids"][:n]) for r in pair]
                record = dict(setting=setting,length=n,prompt_index=i,response_ids=[r["response_id"] for r in pair],self_bleu=value,
                    **{m:float(np.mean([r[m] for r in reps])) for m in reps[0]})
                if setting!="null":
                    record["detected"] = [score_map[setting,r["response_id"],n] for r in pair]
                    record["tpr"] = float(np.mean(record["detected"]))
                prompt_rows.append(record)
                for row,rep in zip(pair,reps):
                    response_metrics.append(dict(setting=setting,length=n,response_id=row["response_id"],prompt_index=i,response_index=row["response_index"],**rep))
            if setting.startswith("synthid"):
                counts = [sum(r["native_repeat_fallback"][:n]) for r in rows]
                fallback.append(dict(setting=setting,length=n,responses_with_fallback=sum(c>0 for c in counts),responses=100,
                    total_fallbacks=sum(counts),mean_fallback_count=float(np.mean(counts))))
    measured,contrasts = summarize_metrics(prompt_rows,draws)
    diagnostics = json.loads((setup/"raw/validate/common_histories.json").read_text())
    diagnostic_summary = []
    for decoder_name in manifest["common_histories"]["decoders"]:
        for method in ("ordinary","synthid_depth2","synthid_depth10","synthid_depth30"):
            records = [r for r in diagnostics if r["decoder"]==decoder_name and r["method"]==method]
            if len(records)!=25:raise ValueError("diagnostic coverage differs")
            diagnostic_summary.append(dict(decoder=decoder_name,method=method,histories=25,
                metrics={m:dict(mean=float(np.mean([r[m] for r in records])),min=min(r[m] for r in records),max=max(r[m] for r in records))
                         for m in manifest["common_histories"]["measurements"]}))
    save(setup/"raw/score_records.json",score_records);save(setup/"raw/response_metrics.json",response_metrics)
    save(setup/"prompt_metrics.json",prompt_rows);save(setup/"common_history_metrics.json",diagnostics)
    resource = validation["resource_estimate_usd"]+report["resource_estimate_usd"]
    summary = dict(manifest_id=manifest["id"],settings=manifest["settings"],decoder=DECODER,results=measured,contrasts=contrasts,
        primary_contrast=next(r for r in contrasts if r["primary"]),null_counts=null_counts,fallback=fallback,common_histories=diagnostic_summary,
        replay_diagnostics=summarize_replay_diagnostics(score_records,draws),
        bootstrap=dict(resamples=len(draws),seed=manifest["analysis"]["bootstrap_seed"],draws_sha256=digest(draws.tolist()),
            unit="50 paired prompt clusters, retaining both seeds",interval="95% percentile, marginal"),
        bleu_signature=str(bleu.get_signature()),analysis_versions={p:importlib.metadata.version(p) for p in ("torch","numpy","scipy","sacrebleu","tokenizers","synthid-text")},
        verification=dict(passed=True,new_full_responses=500,support_checks=support_checks,cdf_boundary_repairs=repairs,
            score_records=len(score_records),prompt_records=len(prompt_rows)),
        files={str(p.relative_to(setup)):sha(p) for p in (setup/"manifest.json",setup/"validate_report.json",setup/"batch_report.json",
            setup/"prompt_metrics.json",setup/"raw/score_records.json",setup/"raw/response_metrics.json",setup/"common_history_metrics.json")},
        cost=dict(worker_resource_estimate_usd=resource,overhead_allowance_usd=.5,
            cumulative_planning_charge_usd=manifest["cost"]["previous_planning_charge_usd"]+resource+.5),
        limitations=["Fixed keys, one model and 50 prompts; primary contrast predeclared; other intervals are exploratory without multiplicity correction.",
            "Pilot nulls are 100 responses clustered in 50 prompts; nominal p<.001 is not a matched empirical FPR.",
            "Boundary bootstrap intervals do not establish zero population false positives or perfect detection.",
            "Self-BLEU measures lexical overlap, not semantic quality. Repetition metrics use raw token IDs.",
            "Replay support mismatches are not generation top-k violations. Bucket endpoint contradictions use the saved FP32 p1 scalar; the unchanged detector clips endpoints and can return magnitude-one scores, not a posterior justified for a zero-probability observation.",
            "Both diagnostic decoders normalize in FP32; full-vocabulary reference is not the historical BF16 SynthID probability path."])
    save(setup/"summary.json",summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","collect","analyze"));parser.add_argument("--setup",type=Path,default=SETUP)
    parser.add_argument("--stage",choices=("validate","batch"),default="batch");parser.add_argument("--download",action="store_true")
    args = parser.parse_args()
    result = prepare(args.setup) if args.command=="prepare" else (analyze(args.setup) if args.command=="analyze" else collect(args.setup,args.stage,args.download)[1])
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
