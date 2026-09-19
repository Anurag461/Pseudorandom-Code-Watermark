"""Validate first, then explicitly dispatch exactly one matched top-100 batch."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import modal

from .config import StudySetting,digest
from .topk import SETUP,DECODER,SETTINGS,TIMEOUTS,validate_manifest,truncate,partition_probability,completion_trace,generate,semantic_checks,replay_prefix_diagnostics
from .repeat import upstream_hashes
from .validation import save,sha
from .validation_modal import generation_image,hf_cache,data_volume,results,checkpoint

app = modal.App("prc-self-bleu-matched-top100")


def start(manifest,stage):
    validate_manifest(manifest,"/root");results.reload()
    root = Path("/results/self_bleu_topk")/manifest["id"]
    if stage=="batch":
        prior = json.loads((root/"validate/report.json").read_text())
        if not prior["passed"] or prior["manifest_id"]!=manifest["id"]:raise ValueError("validation has not passed")
        for name,h in prior["files"].items():
            if sha(root/"validate"/name)!=h:raise ValueError("validation artifact changed")
    target = root/stage
    if (target/"started.json").exists():raise FileExistsError("already attempted: collect/account for saved outputs before any retry")
    save(root/"manifest.json",manifest);save(target/"started.json",{"manifest_id":manifest["id"],"stage":stage});results.commit()
    return target


def load(manifest):
    import torch
    from baseline_comparison.comparison_runner import preload_official_runtimes,load_qwen3_8b,_numpy_pickle_compat
    from baseline_comparison.config import PINNED_DEPENDENCIES
    preload_official_runtimes()
    if upstream_hashes()!=manifest["upstream_sha256"]:raise ValueError("upstream changed")
    execution = dict(versions={p.split("==")[0]:importlib.metadata.version(p.split("==")[0]) for p in PINNED_DEPENDENCIES},
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,model_revision=manifest["model"]["revision"],dtype="bfloat16",
        tf32=torch.backends.cuda.matmul.allow_tf32,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if execution!=manifest["generation_runtime"]:raise ValueError("model runtime differs")
    checkpoint(manifest);data_volume.reload()
    path = Path("/data")/manifest["artifact"]["path"]
    if sha(path)!=manifest["artifact"]["sha256"]:raise ValueError("fixed PRC artifact differs")
    _numpy_pickle_compat();artifact = torch.load(path,map_location="cpu",weights_only=False)
    if artifact["online_key"]!=SETTINGS["prc"].online_key().to_dict():raise ValueError("PRC key configuration differs")
    partition = artifact["partition"]
    if not torch.all((partition==0)|(partition==1)) or not torch.all(partition.sum(0)==1):raise ValueError("invalid partition")
    return load_qwen3_8b(),artifact,execution


def persist(root,report,name,value):
    save(root/name,value);report["files"][name] = sha(root/name);results.commit()


def finish(root,report,started,manifest):
    report["seconds"] = time.monotonic()-started
    report["resource_estimate_usd"] = report["seconds"]*manifest["cost"]["resource_usd_per_second"]
    save(root/"report.json",report);results.commit()


def replay_checked(model,tokens,part1):
    calls = []
    def observe(module,args,kwargs):
        pos = len(calls)
        if (set(kwargs)!={"cache"} or len(args)!=1 or not __import__("torch").equal(args[0],tokens[:,pos:pos+1])
                or (pos==0 and kwargs["cache"].get_seq_len()!=0)):
            raise ValueError("replay includes a prompt, added token or reused cache")
        calls.append(pos)
    handle = model.register_forward_pre_hook(observe,with_kwargs=True)
    try:trace,outside = completion_trace(model,tokens,part1)
    finally:handle.remove()
    if len(calls)!=tokens.shape[1]-1:raise ValueError("replay alignment differs")
    return trace,outside


def validate_replay(model,tokens,part1):
    import torch
    from qwen import StaticKVCache
    from detectors import _soft_tokens
    ids = tokens[:,:65]
    trace,_ = replay_checked(model,ids,part1)
    independent = [];cache = StaticKVCache(max_length=64)
    with torch.no_grad():
        for pos in range(64):
            raw = model(ids[:,pos:pos+1],cache=cache)[:,-1].float()
            ordered = torch.argsort(raw,dim=-1,descending=True,stable=True)[:,:100]
            keep = torch.zeros_like(raw,dtype=torch.bool).scatter(1,ordered,True)
            q = torch.softmax(raw.masked_fill(~keep,-1e12),dim=-1)
            p = (q*part1).sum(-1).clamp(0,1)
            has_one = ((q>0)&(part1==1)).any(-1);has_zero = ((q>0)&(part1==0)).any(-1)
            p = torch.where(has_one,p,torch.zeros_like(p));p = torch.where(has_zero,p,torch.ones_like(p))
            independent.append(p.cpu())
    if not torch.equal(trace,torch.stack(independent,dim=1)):raise ValueError("replay differs from independent stable-sort/normalization oracle")
    permuted,_ = replay_checked(model,ids.flip(0)[:,:33],part1)
    if not torch.equal(permuted.flip(0),trace[:,:32]):raise ValueError("replay prefix/order differs")
    for i in range(len(ids)):
        bits = part1[ids[i]].to(torch.int8).cpu().numpy()
        if _soft_tokens(bits,trace[i].numpy(),"map")[0]!=0:raise ValueError("first coordinate does not abstain")
    return dict(passed=True,raw_completion_only=True,first_coordinate_abstains=True,independent_top100_bucket_trace_exact=True,
                prefix_and_order_exact=True,batch_size=len(ids),verified_coordinates=64)


def common_diagnostics(model,prompts,completions,manifest):
    import torch
    from qwen import StaticKVCache
    from baseline_comparison.official import synthid_processor
    selected = manifest["common_histories"]["prompt_indices"]
    positions = manifest["common_histories"]["positions"]
    device = next(model.parameters()).device
    ids = torch.tensor(prompts,device=device);tokens = torch.tensor(completions,device=device)
    cache = StaticKVCache(max_length=1074)
    processors = {(decoder,d):synthid_processor(device,keys=StudySetting("synthid_text",depth=d).synthid_keys)
                  for decoder in ("full_vocab_fp32_reference","top100_fp32") for d in (2,10,30)}
    records = []
    with torch.no_grad():
        logits = model(ids,cache=cache)[:,-1]
        for pos in range(max(positions)+1):
            history = ids[selected]
            if pos in positions:
                raw = logits[selected].float();filtered,q,keep = truncate(raw);full = torch.softmax(raw,dim=-1)
                base_mass = (full*keep).sum(-1)
                for decoder,base,ordinary in (("full_vocab_fp32_reference",raw,full),("top100_fp32",filtered,q)):
                    distributions = [("ordinary",ordinary,[False]*len(selected))]
                    for d in (2,10,30):
                        p = processors[decoder,d]
                        old = (torch.zeros((len(selected),p.context_history_size),dtype=torch.long,device=device)
                               if p.state is None else p.state.context_history.clone())
                        update,index,_ = p.watermarked_call(history,base)
                        probabilities = torch.softmax(update.float(),dim=-1)
                        repeat = (old==p.state.context_history[:,:1]).any(-1)
                        if decoder=="top100_fp32" and torch.any(probabilities[~keep]!=0):raise ValueError("diagnostic support violation")
                        if repeat.any() and not torch.equal(probabilities[repeat],ordinary[repeat]):raise ValueError("diagnostic native fallback differs")
                        distributions.append((f"synthid_depth{d}",probabilities,repeat.tolist()))
                    for method,probability,repeat in distributions:
                        collision = probability.double().square().sum(-1).tolist()
                        maximum = probability.max(-1).values.tolist();mass = (probability*keep).sum(-1).tolist()
                        for j,prompt in enumerate(selected):
                            records.append(dict(prompt_index=prompt,position=pos,history_sha256=digest(history[j].tolist()),decoder=decoder,method=method,
                                collision_probability=collision[j],maximum_probability=maximum[j],top100_retained_mass=mass[j],
                                base_top100_retained_mass=float(base_mass[j]),native_fallback=bool(repeat[j])))
            else:
                dummy = torch.zeros((len(selected),2),device=device)
                for p in processors.values():p.watermarked_call(history,dummy)
            if pos<max(positions):
                token = tokens[:,pos:pos+1];ids = torch.cat((ids,token),dim=1);logits = model(token,cache=cache)[:,-1]
    if len(records)!=200:raise ValueError("diagnostic coverage differs")
    return records


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["validate"],
              max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/data":data_volume,"/results":results})
def validate(manifest):
    import torch
    started = time.monotonic();root = start(manifest,"validate")
    report = dict(manifest_id=manifest["id"],stage="validate",passed=False,files={},checks={})
    try:
        model,artifact,execution = load(manifest);report["execution"] = execution
        source = Path("/results")/manifest["common_histories"]["source_remote"]
        if sha(source)!=manifest["common_histories"]["source_sha256"]:raise ValueError("common histories changed")
        old = json.loads(source.read_text());completions = [r["token_ids"] for r in old["responses"]]
        if [digest(t) for t in completions]!=manifest["common_histories"]["completion_sha256"]:raise ValueError("history completion hash mismatch")
        prompts = [json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        report["checks"]["semantics"] = semantic_checks("cuda")
        from online_prc import OnlinePRCEncoder,derive_document_seed
        for response,seed in enumerate(manifest["seeds"]):
            encoder = OnlinePRCEncoder(SETTINGS["prc"].online_key(),[derive_document_seed(seed,i) for i in range(50)])
            bits = encoder.encode_to_length(1024).tolist()
            if [digest(row) for row in bits]!=manifest["prc_codeword_sha256"][str(response)]:raise ValueError("PRC latent streams changed")
        report["checks"]["historical_prc_codewords_exact"] = True
        report["checks"]["completion_replay"] = validate_replay(model,torch.tensor(completions,device="cuda"),artifact["partition"][1].to("cuda"))
        diagnostics = common_diagnostics(model,prompts,completions,manifest)
        persist(root,report,"common_histories.json",diagnostics)
        report["checks"]["preselected_histories"] = 25;report["checks"]["distribution_records"] = len(diagnostics)
        report["passed"] = True
    except Exception as e:report["error"] = repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


def score_prc(model,artifact,batch,manifest):
    import torch
    from detectors import prepare_online_map_prefix_context,prepare_online_map_prefix_trace,score_prepared_online_map_prefix,detect_online_hoeffding
    tokens = torch.tensor([r["token_ids"] for r in batch["responses"]],device="cuda",dtype=torch.long)
    trace,outside = replay_checked(model,tokens,artifact["partition"][1].to("cuda"))
    observed_buckets = artifact["partition"][1][tokens[:,1:].cpu()].to(torch.uint8).tolist()
    context = prepare_online_map_prefix_context(artifact["online_key"],1024)
    rows = []
    for i,row in enumerate(batch["responses"]):
        ids = row["token_ids"];p = trace[i].numpy()
        prepared = prepare_online_map_prefix_trace(artifact["online_key"],ids,p,artifact["partition"],1024,prepared_context=context,completion_only=True)
        scores = {}
        for n in (400,1024):
            info = score_prepared_online_map_prefix(prepared,n,fpr=.001)
            v,stat = info["V"],info["statistic"]
            logp = -stat**2/(2*v) if v>1e-15 and stat>0 else 0.
            info.update(p_value=math.exp(logp),log_p_upper_bound=logp)
            if not math.isfinite(info["threshold"]):info["threshold"] = None
            if i==0:
                decision,direct = detect_online_hoeffding(artifact["online_key"],torch.tensor(ids[:n]),p[:n-1],artifact["partition"],fpr=.001,return_info=True)
                if decision!=info["decision"] or any(direct[k]!=info[k] for k in ("V","statistic")):raise ValueError("PRC direct/prepared detector differs")
            scores[str(n)] = info
        rows.append(dict(response_id=row["response_id"],completion_sha256=row["completion_sha256"],prompt_index=row["prompt_index"],
                         response_index=row["response_index"],results=scores,
                         diagnostics_by_prefix={str(n):replay_prefix_diagnostics(p,observed_buckets[i],outside[i].numpy(),n) for n in (400,1024)}))
    return dict(rows=rows,probabilities_2_to_T=trace.tolist(),raw_completion_only=True,decoder=DECODER,
                replay_outside_top100=outside.tolist(),observed_buckets_2_to_T=observed_buckets)


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["batch"],
              max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/data":data_volume,"/results":results})
def batch(manifest):
    started = time.monotonic();root = start(manifest,"batch")
    report = dict(manifest_id=manifest["id"],stage="batch",passed=False,files={},batches=[])
    try:
        model,artifact,execution = load(manifest);report["execution"] = execution
        prompts = [json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        for setting in SETTINGS:
            for response,seed in enumerate(manifest["seeds"]):
                t = time.monotonic();generated = generate(model,prompts,setting,seed,response,artifact,manifest,execution)
                if setting=="prc" and [digest(r["prc_codeword_bits"]) for r in generated["responses"]]!=manifest["prc_codeword_sha256"][str(response)]:
                    raise ValueError("generated PRC codewords changed")
                name = f"batches/{setting}_r{response}.json";persist(root,report,name,generated)
                report["batches"].append(dict(setting=setting,response_index=response,path=name,seconds=time.monotonic()-t,batch_id=generated["batch_id"]))
                print(f"[top100] saved {setting}, seed {seed}: 50 full responses",flush=True)
        # Generation is complete and persisted before any completion-only replay.
        for setting in ("null","prc"):
            for response in (0,1):
                generated = json.loads((root/f"batches/{setting}_r{response}.json").read_text())
                persist(root,report,f"replay/{setting}_r{response}.json",score_prc(model,artifact,generated,manifest))
                print(f"[top100] saved completion-only PRC replay for {setting} r{response}",flush=True)
        report["passed"] = len(report["batches"])==10
    except Exception as e:report["error"] = repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


@app.local_entrypoint()
def run(stage:str,setup:str=str(SETUP)):
    if os.environ.get("MODAL_PROFILE")!="new-prc-watermark":raise ValueError("expected existing Modal profile")
    if stage not in ("validate","batch"):raise ValueError("select one explicit stage")
    path = Path(setup);manifest = json.loads((path/"manifest.json").read_text());validate_manifest(manifest)
    report = {"validate":validate,"batch":batch}[stage].remote(manifest)
    save(path/f"{stage}_report.json",report);print(json.dumps(report,indent=2))
