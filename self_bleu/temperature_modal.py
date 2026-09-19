"""Two explicit stages; a failed validation never dispatches evaluation."""
from __future__ import annotations
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import modal
from .temperature import (SETUP,SETTINGS,TIMEOUTS,TEMPERATURE,PRECISION,TemperatureModel,
    validate_manifest,generate,replay,prefix_diagnostics,synthid_temperature_check)
from .config import digest
from .repeat import upstream_hashes
from .validation import ROOT,save,sha
from .validation_modal import generation_image,hf_cache,data_volume,results,checkpoint

app=modal.App("prc-self-bleu-8b-t07")


def start(manifest,stage):
    validate_manifest(manifest,"/root");results.reload()
    root=Path("/results/self_bleu_temperature")/manifest["id"]
    if stage=="batch":
        prior=json.loads((root/"validate/report.json").read_text())
        if not prior["passed"] or prior["manifest_id"]!=manifest["id"]:raise ValueError("validation failed or missing; full generation forbidden")
        for name,h in prior["files"].items():
            if sha(root/"validate"/name)!=h:raise ValueError("validation artifact changed")
    target=root/stage
    if (target/"started.json").exists():raise FileExistsError("attempt already exists; no automatic retries")
    save(root/"manifest.json",manifest);save(target/"started.json",dict(stage=stage,manifest_id=manifest["id"]))
    results.commit();return target


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
    if partition.dtype!=torch.bfloat16:raise ValueError("original BF16 partition required")
    if not torch.all((partition==0)|(partition==1)) or not torch.all(partition.sum(0)==1):raise ValueError("invalid partition")
    return load_qwen3_8b(),artifact,execution


def persist(root,report,name,value):
    save(root/name,value);report["files"][name] = sha(root/name);results.commit()


def finish(root,report,started,manifest):
    report["seconds"] = time.monotonic()-started
    report["resource_estimate_usd"] = report["seconds"]*manifest["cost"]["rate"]
    save(root/"report.json",report);results.commit()



def validate_alignment(model,artifact,batch,prompts):
    import torch
    from qwen import StaticKVCache,teacher_force_partition_trace_batch
    from detectors import _soft_tokens
    tokens=torch.tensor([r["token_ids"] for r in batch["responses"]],device="cuda")
    part=artifact["partition"][1].to("cuda")
    scaled=TemperatureModel(model)
    prompted=teacher_force_partition_trace_batch(scaled,torch.tensor(prompts,device="cuda"),tokens,part,"static",1)
    saved=torch.tensor([r["generation_partition1"] for r in batch["responses"]])
    if not torch.equal(prompted.cpu(),saved):raise ValueError("generation and same-history replay bucket probabilities differ")
    trace,zero=replay(model,tokens,part)
    cache=StaticKVCache(max_length=tokens.shape[1]-1);oracle=[]
    with torch.no_grad():
        for j in range(tokens.shape[1]-1):
            raw=model(tokens[:,j:j+1],cache=cache)[:,-1]
            q=torch.softmax(raw/.7,dim=-1)
            mass=(q*part).sum(-1)
            if q.dtype!=torch.bfloat16 or mass.dtype!=torch.bfloat16:raise ValueError("bucket arithmetic promoted")
            oracle.append(mass.cpu().float())
    if not torch.equal(trace,torch.stack(oracle,dim=1)):raise ValueError("independent T=.7 completion-only oracle differs")
    short,_=replay(model,tokens.flip(0)[:,:33],part)
    if not torch.equal(short.flip(0),trace[:,:32]):raise ValueError("replay prefix/order differs")
    for i in range(len(tokens)):
        bits=part[tokens[i]].to(torch.int8).cpu().numpy()
        if _soft_tokens(bits,trace[i].numpy(),"map")[0]!=0:raise ValueError("first coordinate did not abstain")
    return dict(passed=True,generation_same_history_exact=True,raw_completion_inputs_verified=True,
        independent_completion_oracle_exact=True,prefix_order_exact=True,first_coordinate_abstains=True,
        zero_probability_positions=int(zero.sum()),positions=int(zero.numel()))


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["validate"],
    max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/data":data_volume,"/results":results})
def validate(manifest):
    import torch
    from qwen import StaticKVCache
    from online_prc import OnlinePRCEncoder,derive_document_seed
    started=time.monotonic();root=start(manifest,"validate")
    report=dict(manifest_id=manifest["id"],stage="validate",passed=False,files={},checks={},evaluation_responses=0)
    try:
        model,artifact,execution=load(manifest);report["execution"]=execution
        prompts=[json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        with torch.no_grad():raw=model(torch.tensor(prompts,device="cuda"),cache=StaticKVCache(max_length=50))[:,-1]
        report["checks"]["forced_repeat_temperature"]=synthid_temperature_check("cuda",raw)
        report["checks"]["precision"]=dict(model_output=str(raw.dtype),scaled_logits=str((raw/.7).dtype),
            ordinary_softmax=str((raw/.7).float().softmax(-1).dtype),prc_bucket_softmax=str((raw/.7).softmax(-1).dtype),
            partition=str(artifact["partition"].dtype),synthid_updates="torch.bfloat16",conditional_prc_softmax="torch.float32")
        for response,seed in enumerate(manifest["seeds"]):
            encoder=OnlinePRCEncoder(SETTINGS["prc"].online_key(),[derive_document_seed(seed,i) for i in range(50)])
            if [digest(r) for r in encoder.encode_to_length(1024).tolist()]!=manifest["prc_codeword_sha256"][str(response)]:
                raise ValueError("PRC latent streams changed")
        report["checks"]["historical_codewords_exact"]=True
        report["smokes"]=[]
        for setting in SETTINGS:
            batches=[]
            for tag,response in (("r0",0),("r0_repeat",0),("r1",1)):
                b=generate(model,prompts,setting,manifest["seeds"][response],response,artifact,manifest,execution,64)
                persist(root,report,f"smokes/{setting}_{tag}.json",b);batches.append(b)
                if b["anomalies"]["prc_draw_bucket_mismatches"]:raise ValueError("PRC inverse-CDF bucket mismatch")
                if setting.startswith("synthid"):
                    parity=b["telemetry"]["synthid_official_smoke_reference"]
                    if not parity["indices_equal"] or parity["max_abs_score_difference"]!=0:raise ValueError("native SynthID batch/single-row update mismatch")
            if batches[0]["responses"]!=batches[1]["responses"]:raise ValueError(f"same-seed reproduction failed: {setting}")
            changed=sum(a["token_ids"]!=b["token_ids"] for a,b in zip(batches[0]["responses"],batches[2]["responses"]))
            if not changed:raise ValueError(f"fresh sampling seed failed to change any response: {setting}")
            report["smokes"].append(dict(setting=setting,batch_size=50,length=64,same_seed_exact=True,
                different_seed_changed=changed,short_responses=150,keys_fixed=True))
            if setting=="prc":report["checks"]["replay_alignment"]=validate_alignment(model,artifact,batches[0],prompts)
            print(f"[T=.7 validation] {setting}: three 50 x 64-token smokes passed",flush=True)
        report["short_responses"]=600;report["passed"]=True
    except Exception as e:report["error"]=repr(e);report["stop_reason"]="Validation failed; evaluation is forbidden.";raise
    finally:finish(root,report,started,manifest)
    return report


def score_prc(model,artifact,batch,manifest):
    import torch
    from detectors import prepare_online_map_prefix_context,prepare_online_map_prefix_trace,score_prepared_online_map_prefix,detect_online_hoeffding
    tokens = torch.tensor([r["token_ids"] for r in batch["responses"]],device="cuda",dtype=torch.long)
    trace,outside = replay(model,tokens,artifact["partition"][1].to("cuda"))
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
                         diagnostics_by_prefix={str(n):prefix_diagnostics(p,observed_buckets[i],outside[i].numpy(),n) for n in (400,1024)}))
    return dict(rows=rows,probabilities_2_to_T=trace.tolist(),raw_completion_only=True,temperature=TEMPERATURE,precision=PRECISION,
                replay_zero_token_probability=outside.tolist(),observed_buckets_2_to_T=observed_buckets)


@app.function(image=generation_image,gpu="H100",cpu=(4,4),memory=(65536,65536),timeout=TIMEOUTS["batch"],
    max_containers=1,retries=0,scaledown_window=2,volumes={"/cache":hf_cache,"/data":data_volume,"/results":results})
def batch(manifest):
    started=time.monotonic();root=start(manifest,"batch")
    report=dict(manifest_id=manifest["id"],stage="batch",passed=False,files={},batches=[])
    try:
        model,artifact,execution=load(manifest);report["execution"]=execution
        prompts=[json.loads(line)["prompt_tokens"] for line in Path("/root/prompts.jsonl").read_text().splitlines()][:50]
        for setting in SETTINGS:
            for response,seed in enumerate(manifest["seeds"]):
                start_time=time.monotonic()
                b=generate(model,prompts,setting,seed,response,artifact,manifest,execution)
                name=f"batches/{setting}_r{response}.json";persist(root,report,name,b)
                smoke=json.loads((root.parent/f"validate/smokes/{setting}_r{response}.json").read_text())
                if [r["token_ids"][:64] for r in b["responses"]]!=[r["token_ids"] for r in smoke["responses"]]:
                    raise ValueError("full-length trajectory disagrees with validated smoke prefix")
                if setting=="prc" and [digest(r["prc_codeword_bits"]) for r in b["responses"]]!=manifest["prc_codeword_sha256"][str(response)]:
                    raise ValueError("PRC codeword stream changed")
                report["batches"].append(dict(setting=setting,response_index=response,path=name,
                    seconds=time.monotonic()-start_time,smoke_prefix_exact=True,anomalies=b["anomalies"]))
                print(f"[T=.7] saved {setting} seed {seed}: 50 evaluation responses",flush=True)
        for setting in ("prc","null"):
            for response in (0,1):
                b=json.loads((root/f"batches/{setting}_r{response}.json").read_text())
                persist(root,report,f"replay/{setting}_r{response}.json",score_prc(model,artifact,b,manifest))
                print(f"[T=.7] saved completion-only PRC replay {setting} r{response}",flush=True)
        report["passed"]=len(report["batches"])==8;report["evaluation_responses"]=400
    except Exception as e:report["error"]=repr(e);raise
    finally:finish(root,report,started,manifest)
    return report


@app.local_entrypoint()
def run(stage:str,setup:str=str(SETUP)):
    if os.environ.get("MODAL_PROFILE")!="new-prc-watermark":raise ValueError("wrong Modal profile")
    if stage not in TIMEOUTS:raise ValueError("select validate or batch explicitly")
    path=Path(setup);manifest=json.loads((path/"manifest.json").read_text());validate_manifest(manifest)
    report={"validate":validate,"batch":batch}[stage].remote(manifest)
    save(path/f"{stage}_report.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("files",)},indent=2))
