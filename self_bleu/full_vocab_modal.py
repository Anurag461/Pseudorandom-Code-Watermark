"""Explicit gated validation, 600-response generation and completion-only replay."""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

import modal

from .config import digest
from .full_vocab import (SETUP,MODEL,DECODER,SETTINGS,TIMEOUTS,validate_manifest,full_distribution,
    partition_probability,completion_trace,generate,semantic_checks,replay_prefix_diagnostics)
from .repeat import upstream_hashes
from .validation import save,sha
from .validation_modal import generation_image,detector_image,hf_cache,data_volume,results

app=modal.App("prc-self-bleu-0p6b-full-vocab")
generation_image=generation_image.env({"PRC_MODEL_SIZE":"0.6B"})


def start(manifest,stage):
    validate_manifest(manifest,"/root");results.reload()
    root=Path("/results/self_bleu_full_vocab")/manifest["id"]
    dependencies={"batch":["validate","validate_textseal"],"textseal":["validate_textseal","batch"]}
    for previous in dependencies.get(stage,[]):
        prior=json.loads((root/previous/"report.json").read_text())
        if not prior["passed"] or prior["manifest_id"]!=manifest["id"]:
            raise ValueError(f"required stage did not pass: {previous}")
        for name,h in prior["files"].items():
            if sha(root/previous/name)!=h:raise ValueError("prior artifact changed")
    target=root/stage
    if (target/"started.json").exists():raise FileExistsError("already attempted; recover and account before any retry")
    save(root/"manifest.json",manifest)
    save(target/"started.json",dict(manifest_id=manifest["id"],stage=stage));results.commit()
    return target


def checkpoint(manifest):
    spec=manifest["model"];root=Path("/cache")/spec["cache_directory"]
    for name,h in {**spec["weight_files"],"tokenizer.json":spec["tokenizer_sha256"]}.items():
        if sha(root/name)!=h:raise ValueError(f"checkpoint bytes differ: {name}")
        metadata=root/".cache/huggingface/download"/f"{name}.metadata"
        if metadata.read_text().splitlines()[0]!=spec["revision"]:raise ValueError("checkpoint revision differs")
    if sha(Path("/root/self_bleu/qwen3_0p6b_config.json"))!=manifest["model_config_sha256"]:
        raise ValueError("pinned config differs")
    return root


def load(manifest):
    import torch
    from safetensors.torch import load_file
    from qwen import Qwen3Model,return_qwen_config,load_weights_into_qwen
    from baseline_comparison.comparison_runner import preload_official_runtimes,_numpy_pickle_compat
    from baseline_comparison.config import PINNED_DEPENDENCIES
    preload_official_runtimes()
    if upstream_hashes()!=manifest["upstream_sha256"]:raise ValueError("upstream changed")
    execution=dict(versions={p.split("==")[0]:importlib.metadata.version(p.split("==")[0]) for p in PINNED_DEPENDENCIES},
        gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,model_revision=MODEL["revision"],dtype="bfloat16",
        tf32=torch.backends.cuda.matmul.allow_tf32,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if execution!=manifest["generation_runtime"]:raise ValueError("model runtime differs")
    root=checkpoint(manifest);data_volume.reload()
    path=Path("/data")/manifest["artifact"]["path"]
    if sha(path)!=manifest["artifact"]["sha256"]:raise ValueError("fixed PRC artifact differs")
    _numpy_pickle_compat();artifact=torch.load(path,map_location="cpu",weights_only=False)
    if artifact["online_key"]!=SETTINGS["prc"].online_key().to_dict():raise ValueError("PRC key differs")
    partition=artifact["partition"]
    if (partition.shape!=(2,151936) or not torch.all((partition==0)|(partition==1))
            or not torch.all(partition.sum(0)==1)):raise ValueError("invalid partition")
    cfg=return_qwen_config("0.6B");hf=manifest["model_config"]
    for a,b in {"vocab_size":"vocab_size","emb_dim":"hidden_size","hidden_dim":"intermediate_size",
                "n_heads":"num_attention_heads","n_layers":"num_hidden_layers","n_kv_groups":"num_key_value_heads",
                "head_dim":"head_dim","rope_base":"rope_theta"}.items():
        if cfg[a]!=hf[b]:raise ValueError(f"model architecture differs: {a}")
    with torch.device("cuda"):model=Qwen3Model(cfg)
    weights=load_file(str(root/"model.safetensors"),device="cpu")
    load_weights_into_qwen(model,cfg,weights)
    return model.eval().requires_grad_(False),artifact,execution


def load_textseal(manifest):
    import torch
    from transformers import AutoModelForCausalLM,Qwen3Config
    from baseline_comparison.textseal_modal import runtime_identity,DEPENDENCIES
    runtime=runtime_identity({"runtime":{"dependencies":DEPENDENCIES}})
    if runtime!=manifest["textseal_runtime"]:raise ValueError("TextSeal runtime differs")
    root=checkpoint(manifest)
    config=Qwen3Config.from_dict(manifest["model_config"])
    model=AutoModelForCausalLM.from_pretrained(str(root),config=config,torch_dtype=torch.bfloat16,
        attn_implementation="eager",local_files_only=True,trust_remote_code=False).eval().to("cuda")
    model.config.use_cache=False
    if model.config.vocab_size!=151936 or model.config.hidden_size!=1024:raise ValueError("wrong TextSeal model")
    return model,runtime


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
    trace,_=replay_checked(model,tokens,part1)
    independent=[];cache=StaticKVCache(max_length=tokens.shape[1]-1)
    with torch.no_grad():
        for pos in range(tokens.shape[1]-1):
            raw=model(tokens[:,pos:pos+1],cache=cache)[:,-1].float()
            q=torch.softmax(raw,dim=-1)
            independent.append((q*part1).sum(-1).clamp(0,1).cpu())
    if not torch.equal(trace,torch.stack(independent,dim=1)):raise ValueError("independent full-vocabulary trace differs")
    shorter,_=replay_checked(model,tokens.flip(0)[:,:33],part1)
    if not torch.equal(shorter.flip(0),trace[:,:32]):raise ValueError("prefix/order differs")
    for i in range(len(tokens)):
        bits=part1[tokens[i]].to(torch.int8).cpu().numpy()
        if _soft_tokens(bits,trace[i].numpy(),"map")[0]!=0:raise ValueError("first coordinate did not abstain")
    return dict(passed=True,raw_completion_only=True,first_coordinate_abstains=True,
                independent_full_vocab_trace_exact=True,prefix_and_order_exact=True,
                batch_size=len(tokens),verified_coordinates=tokens.shape[1]-1)


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["validate"],
              max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/data":data_volume,"/results":results})
def validate(manifest):
    import torch
    from online_prc import OnlinePRCEncoder,derive_document_seed
    started=time.monotonic();root=start(manifest,"validate")
    report=dict(manifest_id=manifest["id"],stage="validate",passed=False,files={},checks={})
    try:
        model,artifact,execution=load(manifest);report["execution"]=execution
        prompts=[json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        report["checks"]["semantics"]=semantic_checks("cuda")
        for response,seed in enumerate(manifest["seeds"]):
            encoder=OnlinePRCEncoder(SETTINGS["prc"].online_key(),[derive_document_seed(seed,i) for i in range(50)])
            if [digest(row) for row in encoder.encode_to_length(1024).tolist()]!=manifest["prc_codeword_sha256"][str(response)]:
                raise ValueError("PRC codeword streams changed")
        report["checks"]["historical_prc_codewords_exact"]=True
        report["checks"]["completion_replay"]=validate_replay(model,torch.tensor(prompts,device="cuda"),artifact["partition"][1].to("cuda"))
        report["passed"]=True
    except Exception as e:report["error"]=repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


@app.function(image=detector_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["validate_textseal"],
              max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/results":results})
def validate_textseal(manifest):
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    from baseline_comparison.textseal_redetect import run_record
    started=time.monotonic();root=start(manifest,"validate_textseal")
    report=dict(manifest_id=manifest["id"],stage="validate_textseal",passed=False,files={})
    try:
        model,runtime=load_textseal(manifest);report["runtime"]=runtime
        detector=TextSealCompletionDetector(model,alpha=.1)
        rows=[json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:2]
        for i,tokens in enumerate(rows):
            data=run_record(detector,dict(method="null",prompt_index=i,token_ids=tokens),[32,50],True,"direct")
            if not data["validation"]["passed"] or not data["actual_model_inputs_verified"]:raise ValueError("TextSeal preflight failed")
            persist(root,report,f"fixture_{i}.json",data)
        report["passed"]=True
    except Exception as e:report["error"]=repr(e);raise
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
                replay_zero_token_probability=outside.tolist(),observed_buckets_2_to_T=observed_buckets)


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
                print(f"[0.6B full vocabulary] saved {setting}, seed {seed}: 50 full responses",flush=True)
        # Generation is complete and persisted before any completion-only replay.
        for setting in ("null","prc"):
            for response in (0,1):
                generated = json.loads((root/f"batches/{setting}_r{response}.json").read_text())
                persist(root,report,f"replay/{setting}_r{response}.json",score_prc(model,artifact,generated,manifest))
                print(f"[0.6B full vocabulary] saved completion-only PRC replay for {setting} r{response}",flush=True)
        report["passed"] = len(report["batches"])==12
    except Exception as e:report["error"] = repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


@app.function(image=detector_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["textseal"],
              max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/results":results})
def textseal(manifest):
    from baseline_comparison.textseal_completion import TextSealCompletionDetector
    from baseline_comparison.textseal_redetect import run_record
    started=time.monotonic();root=start(manifest,"textseal")
    report=dict(manifest_id=manifest["id"],stage="textseal",passed=False,files={},rows=[])
    try:
        model,runtime=load_textseal(manifest);report["runtime"]=runtime
        detector=TextSealCompletionDetector(model,alpha=.1)
        for setting in ("textseal","null"):
            for response in (0,1):
                batch=json.loads((root.parent/f"batch/batches/{setting}_r{response}.json").read_text())
                for i,row in enumerate(batch["responses"]):
                    if digest(row["token_ids"])!=row["completion_sha256"]:raise ValueError("completion changed")
                    clean=dict(method=setting,prompt_index=row["prompt_index"],token_ids=row["token_ids"])
                    data=run_record(detector,clean,[400,1024],i==0,"direct")
                    if not data["actual_model_inputs_verified"] or (data["validation"]["performed"] and not data["validation"]["passed"]):
                        raise ValueError("TextSeal actual input / upstream check failed")
                    name=f"records/{setting}_r{response}_p{i:04d}.json"
                    save(root/name,data);report["files"][name]=sha(root/name)
                    report["rows"].append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(results=data["results"]))
                results.commit();print(f"[TextSeal 0.6B replay] completed {len(report['rows'])}/200",flush=True)
        report["passed"]=len(report["rows"])==200
    except Exception as e:report["error"]=repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


@app.local_entrypoint()
def run(stage:str,setup:str=str(SETUP)):
    if os.environ.get("MODAL_PROFILE")!="new-prc-watermark":raise ValueError("expected existing Modal profile")
    if stage not in TIMEOUTS:raise ValueError("select an explicit stage")
    path=Path(setup);manifest=json.loads((path/"manifest.json").read_text());validate_manifest(manifest)
    report={"validate":validate,"validate_textseal":validate_textseal,"batch":batch,"textseal":textseal}[stage].remote(manifest)
    save(path/f"{stage}_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("files","rows")},indent=2))
