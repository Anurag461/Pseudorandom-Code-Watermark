"""One T=.7 sensitivity study preserving the original full-vocabulary arithmetic."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import torch

from .config import StudySetting,digest
from .validation import ROOT,RATE,save,sha,load_online_sampler
from .pilot import paired_interval,RAW as ORIGINAL_RAW
from .repeat import upstream_hashes,synthid_repeat_masks
from .topk import repetition,replay_prefix_diagnostics as _diagnostics,summarize_replay_diagnostics

SETUP=ROOT/"outputs/self_bleu_temperature/t07_v1"
PILOT=ROOT/"outputs/self_bleu_pilot/stage_a_v2"
DEPTH=ROOT/"outputs/self_bleu_depth/depth2_30_v1"
SETTINGS={"null":StudySetting("null"),"prc":StudySetting("online_prc",eta=.05),
    "synthid_depth2":StudySetting("synthid_text",depth=2),"synthid_depth10":StudySetting("synthid_text",depth=10)}
TIMEOUTS={"validate":600,"batch":1800}
TEMPERATURE=.7
PRECISION={
    "model":"existing BF16 weights/forward on H100, static KV cache",
    "temperature":"divide BF16 logits by Python float .7 once, retaining BF16, before existing samplers",
    "ordinary":"scaled BF16 logits -> float32 -> softmax -> existing multinomial",
    "prc":"BF16 softmax, BF16 partition multiplication and bucket sum; FP32 channel; FP64 bucket uniform; FP32 masked conditional softmax/CDF and token uniform",
    "synthid":"internal temperature remains 1; BF16 score updates; FP32 final sampling softmax; native zero context, state reset and repeat fallback",
    "fallback":"softmax(scaled_BF16_logits.float()), exactly ordinary distribution on the same history",
    "replay":"original BF16 completion-only bucket path with the same single BF16 temperature scaling; no clipping/repair of bucket masses",
}


class TemperatureModel(torch.nn.Module):
    """Only new numerical operation: BF16 logit division before original paths."""
    def __init__(self,model,temperature=TEMPERATURE):
        super().__init__();self.model=model;self.temperature=temperature

    def forward(self,*args,**kwargs):
        logits=self.model(*args,**kwargs)
        if logits.dtype!=torch.bfloat16:raise ValueError("historical BF16 model logits required")
        return logits/self.temperature


def synthid_temperature_check(device="cpu",raw=None):
    """Explicit native-T oracle, repeats and ordinary distribution equality."""
    from baseline_comparison.official import synthid_processor
    if raw is None:raw=torch.linspace(-4,4,256,device=device).bfloat16()[None].repeat(2,1)
    if raw.dtype!=torch.bfloat16:raise ValueError("probe requires actual model dtype")
    scaled=raw/TEMPERATURE;ordinary=scaled.float().softmax(-1);checks={}
    for depth in (2,10):
        keys=SETTINGS[f"synthid_depth{depth}"].synthid_keys
        wrapped=synthid_processor(device,keys=keys);native=synthid_processor(device,keys=keys)
        if wrapped.temperature!=1 or wrapped.apply_top_k:raise ValueError("unexpected original SynthID processor")
        native.temperature=TEMPERATURE
        count=0
        for token in [1,2,3]*4:
            history=torch.full((len(raw),50),token,dtype=torch.long,device=device)
            old=(torch.zeros((len(raw),wrapped.context_history_size),dtype=torch.long,device=device)
                 if wrapped.state is None else wrapped.state.context_history.clone())
            output,indices,base=wrapped.watermarked_call(history,scaled)
            reference,ref_indices,ref_base=native.watermarked_call(history,raw)
            if not (torch.equal(output,reference) and torch.equal(indices,ref_indices)
                    and torch.equal(base,scaled) and torch.equal(base,ref_base)):
                raise ValueError("temperature applied incorrectly or numerical path changed")
            if output.dtype!=torch.bfloat16:raise ValueError("SynthID arithmetic promoted unexpectedly")
            repeated=(old==wrapped.state.context_history[:,:1]).any(-1)
            probabilities=output.float().softmax(-1)
            if repeated.any() and not torch.equal(probabilities[repeated],ordinary[repeated]):
                raise ValueError("SynthID repeat fallback differs from T=.7 ordinary sampling")
            if not torch.isfinite(probabilities).all():raise ValueError("nonfinite SynthID probabilities")
            count+=int(repeated.sum())
        if not count:raise ValueError("forced-repeat probe did not repeat")
        checks[str(depth)]=dict(native_single_temperature_exact=True,ordinary_fallback_exact=True,
            repeated_cases=count,processor_temperature=1,score_dtype=str(output.dtype))
    return checks


@torch.no_grad()
def generate(model,prompts,setting_name,seed,response,artifact,manifest,execution,length=1024):
    from baseline_comparison.comparison_runner import generate_method
    from online_prc import derive_document_seed,document_uniform
    setting=SETTINGS[setting_name];device=next(model.parameters()).device
    scaled=TemperatureModel(model)
    if setting_name=="prc":
        if artifact["partition"].dtype!=torch.bfloat16:raise ValueError("original partition dtype changed")
        sampler=load_online_sampler(ROOT/"watermark_expt.py",device=device)
        seeds=[derive_document_seed(seed,i) for i in manifest["prompt_indices"]]
        tokens,p,details=sampler(scaled,torch.tensor(prompts,dtype=torch.long,device=device),length,
            setting.online_key(),artifact["partition"],document_seeds=seeds,return_trace_details=True,kv_cache_implementation="static")
        sequences=tokens.tolist();telemetry={k:details[k] for k in ("online_sampler_version","kv_cache_implementation")}
    else:
        output,telemetry=generate_method(scaled,prompts,method=setting.method,seed=seed,
            synthid_keys=setting.synthid_keys,max_new_tokens=length,device=str(device))
        sequences=[r["token_ids"] for r in output]
    identity=dict(study_id=manifest["id"],setting=setting.identity(),temperature=TEMPERATURE,precision=PRECISION,
        sampling_seed=seed,response_index=response,prompt_indices=manifest["prompt_indices"],
        prompt_sha256=[digest(p) for p in prompts],execution=execution,length=length)
    bid=digest(identity);rows=[];wrong_bucket=0
    for i,tokens in enumerate(sequences):
        if len(tokens)!=length:raise ValueError("forced-length generation differs")
        row=dict(response_id=f"{bid}/p{i:04d}/r{response}",prompt_index=i,response_index=response,
            sampling_seed=seed,token_ids=tokens,completion_sha256=digest(tokens))
        if setting_name=="prc":
            row.update(generation_partition1=p[i].tolist(),prc_codeword_bits=details["prc_codeword_bits"][i].tolist(),document_seed=seeds[i])
            # Observe the historical sampler; never repair or resample its output.
            mass=torch.tensor(p[i],dtype=torch.float32);xi=torch.tensor(row["prc_codeword_bits"],dtype=torch.float32)
            bern=torch.where(mass<=.5,2*xi*mass,1-2*(1-xi)*(1-mass)).clamp(0,1)
            uniforms=torch.tensor([document_uniform(seeds[i],"lm-bucket/v1",j) for j in range(length)],dtype=torch.float64)
            chosen=(uniforms<bern.double()).long()
            wrong_bucket+=int((chosen!=artifact["partition"][1,tokens].long()).sum())
        rows.append(row)
    return dict(batch_id=bid,identity=identity,responses=rows,telemetry=telemetry,
        anomalies=dict(prc_draw_bucket_mismatches=wrong_bucket))


def prefix_diagnostics(p,b,zero,n):
    result=_diagnostics(p,b,zero,n)
    rename={"outside_top100":"zero_token_probability","outside_and_endpoint":"zero_token_and_endpoint",
            "outside_without_endpoint":"zero_token_without_endpoint"}
    for window in result.values():
        for field in ("counts","rates"):window[field]={rename.get(k,k):v for k,v in window[field].items()}
    return result


@torch.no_grad()
def replay(model,tokens,part1):
    """Invoke the unchanged completion-only trace function and observe inputs."""
    from qwen import completion_only_partition_trace_batch
    scaled=TemperatureModel(model);calls=[];zero=[]
    def observe(module,args,kwargs):
        pos=len(calls)
        if (len(args)!=1 or set(kwargs)!={"cache"} or not torch.equal(args[0],tokens[:,pos:pos+1])
                or (pos==0 and kwargs["cache"].get_seq_len()!=0)):
            raise ValueError("replay received prompt, BOS, template, wrong position or reused cache")
        calls.append(pos)
    def capture(module,args,output):
        pos=len(zero)
        q=output[:,-1].softmax(-1)
        zero.append((q.gather(1,tokens[:,pos+1:pos+2]).squeeze(1)==0).cpu())
    before=scaled.register_forward_pre_hook(observe,with_kwargs=True)
    after=scaled.register_forward_hook(capture)
    try:trace=completion_only_partition_trace_batch(scaled,tokens,part1,"static")
    finally:before.remove();after.remove()
    if len(calls)!=tokens.shape[1]-1:raise ValueError("replay coordinate count differs")
    return trace.cpu(),torch.stack(zero,dim=1)


def original_groups():
    inputs=json.loads((PILOT/"inputs.json").read_text())
    groups={label:[r for r in inputs if r["method"]==method] for label,method in
        (("null","null"),("prc","online_prc"),("synthid_depth10","synthid_text"))}
    groups["synthid_depth2"]=sum([json.loads((DEPTH/f"raw/batches/depth2_r{r}.json").read_text())["responses"] for r in (0,1)],[])
    for rows in groups.values():
        if len(rows)!=100 or {(r["prompt_index"],r["response_index"]) for r in rows}!={(i,j) for i in range(50) for j in (0,1)}:
            raise ValueError("original T=1 pairs incomplete")
        if any(len(r["token_ids"])!=1024 or digest(r["token_ids"])!=r["completion_sha256"] for r in rows):
            raise ValueError("original completion changed")
    return groups


def validate_manifest(manifest,root=ROOT):
    if digest({k:v for k,v in manifest.items() if k!="id"})!=manifest["id"]:raise ValueError("manifest identity differs")
    if (manifest["settings"]!={n:s.identity() for n,s in SETTINGS.items()} or manifest["temperature"]!=.7
            or manifest["precision"]!=PRECISION or manifest["model"]["id"]!="Qwen/Qwen3-8B-Base"
            or manifest["prompt_indices"]!=list(range(50)) or manifest["seeds"]!=[12345,67890]
            or manifest["length"]!=1024 or manifest["prefix_lengths"]!=[400,1024]
            or manifest["top_p"]!=1 or manifest["top_k"] is not None or manifest["smoke_length"]!=64
            or manifest["repeat_fallback"] is not True or manifest["primary"]!=["prc","synthid_depth2",1024]):
        raise ValueError("temperature study scope changed")
    for name,h in manifest["code_sha256"].items():
        if sha(Path(root)/name)!=h:raise ValueError(f"source changed: {name}")
    if sha(Path(root)/"prompts.jsonl")!=manifest["prompt_sha256"]:raise ValueError("prompts changed")
    cost=manifest["cost"]
    if (cost["timeouts"]!=TIMEOUTS or cost["rate"]!=RATE or cost["reserved_total_usd"]>200
            or cost["reserved_total_usd"]!=cost["previous_planning_charge_usd"]+(sum(TIMEOUTS.values())+4)*RATE+.5):
        raise ValueError("budget changed")


def prepare(setup):
    from baseline_comparison.comparison_runner import _numpy_pickle_compat
    pilot=json.loads((PILOT/"manifest.json").read_text())
    previous=json.loads((ROOT/"outputs/self_bleu_full_vocab/qwen3_0p6b_v1/summary.json").read_text())["cost"]["cumulative_planning_charge_usd"]
    original=json.loads((ROOT/"outputs/self_bleu_validation/step3-v4/generation_report.json").read_text())
    ref=json.loads((ORIGINAL_RAW/"batches"/(original["settings"][0]["batches"][0]+".json")).read_text())
    hashes=ref["manifest"]["execution"]["code_sha256"]
    numerical=("qwen.py","online_prc.py","watermark_expt.py","baseline_comparison/official.py")
    if any(sha(ROOT/n)!=hashes[n] for n in numerical):raise ValueError("original numerical source changed; stop")
    historical_commit="7bde5c6d54dee444db3b69d96bfce6b09c79ba4c"
    old=subprocess.check_output(["git","show",f"{historical_commit}:baseline_comparison/comparison_runner.py"],text=True)
    fn=lambda text:ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=="generate_method"),include_attributes=False)
    if fn(old)!=fn((ROOT/"baseline_comparison/comparison_runner.py").read_text()):raise ValueError("original generator changed; stop")
    artifact=ROOT/"outputs/redetection/.archive/setup_8b_n1280/artifacts.pt"
    if sha(artifact)!=pilot["artifact"]["sha256"]:raise ValueError("artifact differs")
    _numpy_pickle_compat();key=torch.load(artifact,map_location="cpu",weights_only=False)
    if key["partition"].dtype!=torch.bfloat16 or key["online_key"]!=SETTINGS["prc"].online_key().to_dict():raise ValueError("original key/partition differs")
    sources=[PILOT/n for n in ("inputs.json","manifest.json","prc_report.json")]
    sources += [DEPTH/n for n in ("manifest.json","summary.json","prompt_metrics.json","raw/score_records.json","generation_report.json","raw/batches/depth2_r0.json","raw/batches/depth2_r1.json")]
    sources += [ROOT/"outputs/self_bleu_repeat/paired_comparison/prompt_metrics.json"]
    groups=original_groups()
    top=json.loads((ROOT/"outputs/self_bleu_topk/matched_v2/manifest.json").read_text())
    depth_report=json.loads((DEPTH/"generation_report.json").read_text())
    ordinary_seconds=[];synth10_seconds=[]
    for row in original["settings"]:
        if row["setting"]["method"] in ("null","synthid_text"):
            for bid in row["batches"]:
                path=ORIGINAL_RAW/f"batches/{bid}.json";sources.append(path)
                seconds=json.loads(path.read_text())["telemetry"]["method_seconds"]
                (ordinary_seconds if row["setting"]["method"]=="null" else synth10_seconds).append(seconds)
    prc_seconds=[r["seconds"] for r in json.loads((ROOT/"outputs/self_bleu_topk/matched_v2/batch_report.json").read_text())["batches"] if r["setting"]=="prc"]
    replay_seconds=json.loads((PILOT/"prc_report.json").read_text())["seconds"]
    depth2_seconds=[r["seconds"] for r in depth_report["batches"] if r["depth"]==2]
    expected=sum(ordinary_seconds+prc_seconds+synth10_seconds+depth2_seconds)+replay_seconds+150
    names=sorted(set(top["code_sha256"])|{"self_bleu/temperature.py","self_bleu/temperature_modal.py"})
    manifest=dict(schema_version=1,source_parent_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        model=pilot["model"],artifact=pilot["artifact"],generation_runtime=top["generation_runtime"],
        settings={n:s.identity() for n,s in SETTINGS.items()},temperature=.7,precision=PRECISION,top_p=1,top_k=None,
        prompt_indices=list(range(50)),prompt_sha256=sha(ROOT/"prompts.jsonl"),seeds=[12345,67890],
        length=1024,prefix_lengths=[400,1024],smoke_length=64,repeat_fallback=True,
        protocol="completion_only_raw_abstain_v1",primary=["prc","synthid_depth2",1024],
        code_sha256={n:sha(ROOT/n) for n in names},test_sha256=sha(ROOT/"tests/test_self_bleu_temperature.py"),
        upstream_sha256=upstream_hashes(),prc_codeword_sha256=top["prc_codeword_sha256"],
        original_sources={str(p.relative_to(ROOT)):sha(p) for p in sources},
        original_response_hashes={s:{str(r):[x["completion_sha256"] for x in sorted(rows,key=lambda x:x["prompt_index"]) if x["response_index"]==r] for r in (0,1)} for s,rows in groups.items()},
        audit=dict(numerical_sources_exact={n:hashes[n] for n in numerical},generate_method_ast_exact=True,
            historical_commit=historical_commit,partition_dtype="torch.bfloat16",synthid_internal_temperature_original=1,
            method_precision_preserved=True,temperature_placement="single BF16 model-output adapter; SynthID internal /1 unchanged",
            no_later_fp32_sampler_reuse=True,arithmetic_differences_from_original="temperature division only; original BF16 rounding retained",
            known_differences="Source layout has moved and legacy scoring metadata was corrected; generate_method AST and numerical source files match the original. Depth-2 T=1 responses come from the completed depth follow-up using the same implementation."),
        validation=dict(short_batches_per_arm=3,seeds=[12345,12345,67890],responses_per_batch=50,tokens=64,
            short_responses=600,evaluation_responses=400,full_launch_requires_all_checks=True,
            failure_action="Stop and report; no full generation or automatic validation retry"),
        analysis=dict(bootstrap_seed=20260918,bootstrap_resamples=2000,nominal_fpr=.001,unit="50 paired prompt clusters, both responses and all arms together",
            self_bleu="SacreBLEU 2.4.3, decoded text, 13a, exp smoothing, effective order, symmetric two-reference directions /200",
            nulls="100 NEW T=.7 ordinary completions for all three detectors; no calibration or pooling"),
        cost=dict(previous_planning_charge_usd=previous,rate=RATE,timeouts=TIMEOUTS,
            prior_measured_seconds=dict(ordinary=ordinary_seconds,prc_proxy_from_top100=prc_seconds,synthid_depth2=depth2_seconds,synthid_depth10=synth10_seconds,prc_replay_200=replay_seconds),
            estimate_basis="Prior full-vocabulary 8B generation/replay timings; top-100 PRC timing is a conservative generation proxy. Add 150 seconds for validation/loading and 25% timing margin.",
            estimated_new_worker_usd=1.25*expected*RATE,overhead_allowance_usd=.5,
            reserved_total_usd=previous+(sum(TIMEOUTS.values())+4)*RATE+.5,ceiling_usd=200),
        stop_rule="One 400-response T=.7 batch after passing validation, then analysis and STOP regardless of outcome. No other temperatures/configurations, prompts or threshold changes.")
    manifest["id"]=digest(manifest);validate_manifest(manifest);save(setup/"manifest.json",manifest)
    return dict(id=manifest["id"],validation_short_responses=600,evaluation_responses=400,cost=manifest["cost"])


def collect(setup,stage,download=False):
    from concurrent.futures import ThreadPoolExecutor
    manifest=json.loads((setup/"manifest.json").read_text());validate_manifest(manifest)
    remote=f"self_bleu_temperature/{manifest['id']}/{stage}"
    if download:
        import modal
        volume=modal.Volume.from_name("prc-completion-only",create_if_missing=False)
    path=setup/f"{stage}_report.json"
    if download and not path.exists():path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
    report=json.loads(path.read_text())
    if report["manifest_id"]!=manifest["id"]:raise ValueError("report identity differs")
    def fetch(item):
        name,h=item
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:raise ValueError("unsafe path")
        local=setup/"raw"/stage/name
        if download and not local.exists():
            local.parent.mkdir(parents=True,exist_ok=True);local.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
        if sha(local)!=h:raise ValueError(f"artifact differs: {name}")
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(fetch,report["files"].items()))
    return manifest,report


def summarize(prompt_rows,draws):
    vectors,results={},[]
    for temperature in (.7,1.):
        for setting in SETTINGS:
            for n in (1024,400):
                rows=sorted((r for r in prompt_rows if (r["temperature"],r["setting"],r["length"])==(temperature,setting,n)),key=lambda r:r["prompt_index"])
                if [r["prompt_index"] for r in rows]!=list(range(50)):raise ValueError("prompt pairing incomplete")
                metrics=["self_bleu","repeated_4gram_fraction","distinct_3"]+([] if setting=="null" else ["tpr"])
                vectors[temperature,setting,n]={m:np.array([r[m] for r in rows]) for m in metrics}
                results.append(dict(temperature=temperature,setting=setting,length=n,responses=100,
                    detected=None if setting=="null" else sum(sum(r["detected"]) for r in rows),
                    metrics={m:paired_interval(v,draws) for m,v in vectors[temperature,setting,n].items()}))
    pairs=[("prc","synthid_depth2"),("prc","synthid_depth10")]+[(s,"null") for s in SETTINGS if s!="null"]
    contrasts=[dict(temperature=t,left=a,right=b,length=n,primary=(t,a,b,n)==(.7,"prc","synthid_depth2",1024),
        metrics={m:paired_interval(v-vectors[t,b,n][m],draws) for m,v in vectors[t,a,n].items() if m in vectors[t,b,n]})
        for t in (.7,1.) for a,b in pairs for n in (1024,400)]
    return results,contrasts


def analyze(setup):
    from .depth import score_completions
    from tokenizers import Tokenizer
    from sacrebleu.metrics import BLEU
    manifest,validation=collect(setup,"validate");_,batch_report=collect(setup,"batch")
    if not validation["passed"] or not batch_report["passed"]:raise ValueError("failed run; report failure without ordinary analysis")
    for path,h in manifest["original_sources"].items():
        if sha(ROOT/path)!=h:raise ValueError(f"T=1 reference changed: {path}")
    if upstream_hashes()!=manifest["upstream_sha256"] or importlib.metadata.version("sacrebleu")!="2.4.3":raise ValueError("analysis runtime differs")
    tokenizer=PILOT/"raw/tokenizer.json"
    if sha(tokenizer)!=manifest["model"]["tokenizer_sha256"]:raise ValueError("tokenizer differs")
    decoder=Tokenizer.from_file(str(tokenizer));bleu=BLEU(tokenize="13a",smooth_method="exp",effective_order=True,lowercase=False)
    groups={1.:original_groups(),.7:{s:[] for s in SETTINGS}}
    prompts=[json.loads(line)["prompt_tokens"] for line in (ROOT/"prompts.jsonl").read_text().splitlines()][:50]
    anomalies=[]
    for setting in SETTINGS:
        for response,seed in enumerate(manifest["seeds"]):
            batch=json.loads((setup/f"raw/batch/batches/{setting}_r{response}.json").read_text())
            identity=dict(study_id=manifest["id"],setting=SETTINGS[setting].identity(),temperature=.7,precision=PRECISION,
                sampling_seed=seed,response_index=response,prompt_indices=list(range(50)),prompt_sha256=[digest(p) for p in prompts],
                execution=manifest["generation_runtime"],length=1024)
            if batch["identity"]!=identity or batch["batch_id"]!=digest(identity) or len(batch["responses"])!=50:raise ValueError("generation identity differs")
            for i,row in enumerate(batch["responses"]):
                if (row["response_id"]!=f"{batch['batch_id']}/p{i:04d}/r{response}" or row["prompt_index"]!=i
                        or row["response_index"]!=response or len(row["token_ids"])!=1024
                        or row["completion_sha256"]!=digest(row["token_ids"])):raise ValueError("response identity differs")
            groups[.7][setting].extend(batch["responses"])
            anomalies.append(dict(setting=setting,response_index=response,**batch["anomalies"]))
    # Reuse original scores, while recomputing all decoded-text/repetition metrics identically.
    original_prc=json.loads((PILOT/"prc_report.json").read_text())["rows"]
    original_synth=json.loads((DEPTH/"raw/score_records.json").read_text())
    score_records=[];score_map={};null_counts=[]
    draws=np.random.default_rng(manifest["analysis"]["bootstrap_seed"]).integers(0,50,(2000,50))
    for t in (.7,1.):
        for setting in SETTINGS:
            if setting=="null":continue
            for cohort,rows in (("watermarked",groups[t][setting]),("pilot_null",groups[t]["null"])):
                if t==1.:
                    expected={r["response_id"] for r in rows}
                    if setting=="prc":
                        scored=[{k:r[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(length=n,score=r["results"][str(n)])
                            for r in original_prc if r["response_id"] in expected for n in (400,1024)]
                    else:scored=[r for r in original_synth if r["depth"]==SETTINGS[setting].depth and r["cohort"]==cohort and r["response_id"] in expected]
                elif setting=="prc":
                    scored=[];source="prc" if cohort=="watermarked" else "null"
                    for response in (0,1):
                        replay_data=json.loads((setup/f"raw/batch/replay/{source}_r{response}.json").read_text())
                        if replay_data["temperature"]!=.7 or replay_data["precision"]!=PRECISION or not replay_data["raw_completion_only"]:
                            raise ValueError("replay protocol differs")
                        for i,row in enumerate(replay_data["rows"]):
                            for n in (400,1024):
                                diagnostics=prefix_diagnostics(replay_data["probabilities_2_to_T"][i],replay_data["observed_buckets_2_to_T"][i],replay_data["replay_zero_token_probability"][i],n)
                                if diagnostics!=row["diagnostics_by_prefix"][str(n)]:raise ValueError("prefix diagnostic differs")
                                scored.append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(length=n,score=row["results"][str(n)],replay_diagnostics=diagnostics))
                else:scored=score_completions(rows,SETTINGS[setting].depth,[400,1024])
                if len(scored)!=200 or len({(r["response_id"],r["length"]) for r in scored})!=200:raise ValueError("score coverage differs")
                expected={r["response_id"]:r for r in rows}
                for r in scored:
                    if any(r[k]!=expected[r["response_id"]][k] for k in ("completion_sha256","prompt_index","response_index")):raise ValueError("score join differs")
                    score_records.append({**r,"temperature":t,"setting":setting,"cohort":cohort})
                    score_map[t,setting,r["response_id"],r["length"]]=bool(r["score"]["decision"])
                if cohort=="pilot_null":
                    for n in (1024,400):
                        rates=[np.mean([score_map[t,setting,r["response_id"],n] for r in rows if r["prompt_index"]==i]) for i in range(50)]
                        null_counts.append(dict(temperature=t,setting=setting,length=n,false_positives=int(round(sum(rates)*2)),responses=100,fpr=paired_interval(rates,draws)))
    prompt_rows=[];fallback=[]
    old_metrics={(r["setting"],r["length"],r["prompt_index"]):r for r in json.loads((DEPTH/"prompt_metrics.json").read_text())}
    for t in (.7,1.):
        for setting,rows in groups[t].items():
            pairs={(r["prompt_index"],r["response_index"]):r for r in rows}
            repeat_masks=synthid_repeat_masks([r["token_ids"] for r in rows],keys=SETTINGS[setting].synthid_keys) if setting.startswith("synthid") else None
            for n in (1024,400):
                for i in range(50):
                    pair=[pairs[i,r] for r in (0,1)]
                    texts=[decoder.decode(r["token_ids"][:n],skip_special_tokens=True) for r in pair]
                    value=(bleu.sentence_score(texts[0],[texts[1]]).score+bleu.sentence_score(texts[1],[texts[0]]).score)/200
                    reps=[repetition(r["token_ids"][:n]) for r in pair]
                    row=dict(temperature=t,setting=setting,length=n,prompt_index=i,response_ids=[r["response_id"] for r in pair],self_bleu=value,
                        **{m:float(np.mean([r[m] for r in reps])) for m in reps[0]})
                    if setting!="null":
                        row["detected"]=[score_map[t,setting,r["response_id"],n] for r in pair];row["tpr"]=float(np.mean(row["detected"]))
                    if t==1.:
                        old=old_metrics[setting,n,i]
                        if not np.isclose(value,old["self_bleu"],rtol=0,atol=1e-12) or (setting!="null" and row["detected"]!=old["detected"]):
                            raise ValueError("T=1 reference metrics or detection changed")
                    prompt_rows.append(row)
                if repeat_masks is not None:
                    counts=[sum(mask[:n]) for mask in repeat_masks]
                    fallback.append(dict(temperature=t,setting=setting,length=n,responses_with_repeat=sum(c>0 for c in counts),responses=100,
                        total_fallbacks=sum(counts),per_response=[dict(response_id=r["response_id"],count=c,first_repeat_position=next((i+1 for i,b in enumerate(mask[:n]) if b),None)) for r,c,mask in zip(rows,counts,repeat_masks)]))
    measured,contrasts=summarize(prompt_rows,draws)
    save(setup/"raw/score_records.json",score_records);save(setup/"prompt_metrics.json",prompt_rows)
    resource=validation["resource_estimate_usd"]+batch_report["resource_estimate_usd"]
    summary=dict(manifest_id=manifest["id"],precision=PRECISION,audit=manifest["audit"],results=measured,contrasts=contrasts,
        primary_contrast=next(r for r in contrasts if r["primary"]),null_counts=null_counts,fallback=fallback,generation_anomalies=anomalies,
        replay_diagnostics=summarize_replay_diagnostics([r for r in score_records if r["temperature"]==.7],draws),
        bootstrap=dict(resamples=2000,seed=20260918,unit=manifest["analysis"]["unit"],draws_sha256=digest(draws.tolist()),interval="95% percentile, marginal"),
        bleu_signature=str(bleu.get_signature()),analysis_versions={p:importlib.metadata.version(p) for p in ("torch","numpy","scipy","sacrebleu","tokenizers","synthid-text")},
        verification=dict(passed=True,evaluation_responses=400,short_validation_responses=600,prompt_records=len(prompt_rows),score_records=len(score_records),original_T1_metric_checks=400),
        files={str(p.relative_to(setup)):sha(p) for p in (setup/"manifest.json",setup/"validate_report.json",setup/"batch_report.json",setup/"prompt_metrics.json",setup/"raw/score_records.json")},
        cost=dict(worker_resource_estimate_usd=resource,overhead_allowance_usd=.5,cumulative_planning_charge_usd=manifest["cost"]["previous_planning_charge_usd"]+resource+.5),
        limitations=["Exploratory sensitivity analysis on the existing 50-prompt cohort and fixed keys; no claim of general superiority.",
            "Nominal thresholds unchanged; 100 paired pilot nulls per temperature do not establish matched empirical FPR or permit calibration.",
            "Intervals containing zero do not establish equivalence. Nonprimary intervals are exploratory and unadjusted for multiplicity.",
            "Self-BLEU is decoded-text lexical overlap, not semantic quality. Repetition uses raw token IDs.",
            "At boundary outcomes, bootstrap intervals do not establish zero population FPR or perfect detection.",
            "T=1 results are saved original outputs, not matched controls for T=.7; each temperature has its own ordinary controls. Existing T=1 point metrics were reproduced; intervals use this report's common bootstrap draws.",
            "Original BF16 bucket/score arithmetic is retained. BF16 temperature division happens before existing method-specific casts; this intentionally differs from scaling FP32-cast logits."])
    save(setup/"summary.json",summary);write_report(setup,manifest,summary,validation)
    return summary


def write_report(setup,manifest,summary,validation):
    def interval(v):return f"{v['mean']:+.5f} [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"
    primary=summary["primary_contrast"]["metrics"]["self_bleu"]
    lines=["# Final 8B temperature sensitivity: T=.7", "", "Complete: exactly 400 evaluation responses (four arms × 50 prompts × two seeds), each 1,024 tokens. Temperature .7, full vocabulary, top-p 1, no top-k, original forced-length/EOS policy. No further experiments are queued.", "",
        f"**Predeclared primary:** PRC minus SynthID depth 2 decoded-text Self-BLEU at 1,024 tokens: **{interval(primary)}**. Negative values favor PRC. This is exploratory sensitivity analysis on the existing prompt cohort.", "",
        "## Frozen configurations", "",
        f"Qwen3-8B-Base revision `{manifest['model']['revision']}`; H100, BF16 forward, static KV cache, batch 50, TF32 off, original reduced-precision BF16 reduction setting. Canonical prompts 0–49, exactly 50 stored prompt tokens, unchanged formatting. Sampling seeds 12345 and 67890; fixed watermark keys independent of response seeds. Exactly 1,024 steps, including special/EOS tokens, without early stopping.", "",
        "Ordinary sampling; PRC eta .05, t=3, row rate 99/100, key seed 12345, unchanged position-addressed document RNG; native-fallback SynthID depths 2 and 10, ngram length 4, two leaves, history size 1024, original zero-context initialization and fresh state per replicate.", "",
        f"PRC key fingerprint: `{manifest['artifact']['online_key_fingerprint']}`. Key/partition artifact SHA256: `{manifest['artifact']['sha256']}`. SynthID keys: depth 2 `{list(SETTINGS['synthid_depth2'].synthid_keys)}`; depth 10 `{list(SETTINGS['synthid_depth10'].synthid_keys)}`. Full prompt, artifact, source and runtime hashes are frozen in the linked manifest.", "",
        "## Precision audit and temperature placement", "", *[f"- **{k}:** {v}" for k,v in PRECISION.items()], "",
        manifest["audit"]["known_differences"], "",
        "The original sampler functions and numerical files are preserved. This does not inherit the later top-k/0.6B FP32 bucket or SynthID score paths. SynthID's internal division remains /1; the shared BF16 logit adapter performs the only nontrivial temperature scaling. Its native /0.7 path and ordinary fallback were checked for exact equality on actual model logits.", "",
        "## Results beside saved T=1", "", "Self-BLEU is on the 0–1 scale (lower means less overlap). T=1 values use saved original full-vocabulary outputs; they are not T=.7 controls. Each temperature uses its own ordinary sampling responses.", ""]
    for n in (1024,400):
        lines += [f"### {n} tokens"+(" — primary length" if n==1024 else " — secondary"), "", "| Setting | T | Self-BLEU [95% CI] | Detection | Repeated 4-gram | Distinct-3 |", "|---|---:|---|---:|---:|---:|"]
        for setting in SETTINGS:
            for t in (1.,.7):
                r=next(r for r in summary["results"] if (r["setting"],r["temperature"],r["length"])==(setting,t,n));m=r["metrics"]
                lines.append(f"| {setting} | {t:g} | {interval(m['self_bleu'])} | {'—' if r['detected'] is None else str(r['detected'])+'/100'} | {m['repeated_4gram_fraction']['mean']:.5f} | {m['distinct_3']['mean']:.5f} |")
        lines += ["", "| T | Paired contrast | Self-BLEU difference [95% CI] | TPR difference [95% CI] |", "|---:|---|---|---|"]
        for r in summary["contrasts"]:
            if r["length"]==n:lines.append(f"| {r['temperature']:g} | {r['left']} − {r['right']} | {interval(r['metrics']['self_bleu'])} | {interval(r['metrics']['tpr']) if 'tpr' in r['metrics'] else '—'} |")
        lines += ["", "Pilot-null counts: "+"; ".join(f"T={r['temperature']:g} {r['setting']}: {r['false_positives']}/100" for r in summary["null_counts"] if r["length"]==n)+".", ""]
    lines += ["## Validation, anomalies and reproducibility", "", f"Validation passed: {validation['short_responses']} short responses, 64 tokens each, excluded from the evaluation cohort. Every arm used batch 50, both seeds and a same-seed rerun; all same-seed checks, fresh randomness checks, forced-repeat checks, original-codeword checks and generation/replay alignment passed. All 400 full trajectories reproduce their validated 64-token prefixes.", "",
        f"Observed PRC draw bucket mismatches: {sum(r['prc_draw_bucket_mismatches'] for r in summary['generation_anomalies'])}. No sampler repair, token deletion, coordinate shift or threshold change was performed.", ""]
    for r in summary["replay_diagnostics"]:
        if r["window"]=="all":lines.append(f"- {r['source']}, {r['length']} tokens: {r['metrics']['zero_token_probability']['count']} zero-probability observations; {r['metrics']['endpoint_contradiction']['count']} contradictory saved bucket endpoints / {r['positions']} replay positions.")
    lines += ["", "The unchanged detector clips endpoint probabilities. A contradictory endpoint is not a posterior justified by a model assigning the observation zero probability. Prefix-specific early/later diagnostics, fallback trajectories, absolute TPR intervals and all paired repetition intervals are included in [summary.json](summary.json).", "",
        "All means/contrasts use 2,000 paired resamples of the 50 prompts, retaining both seeds and all arms. [Manifest and source audit](manifest.json) · [Validation](validate_report.json) · [Batch report](batch_report.json) · [Prompt metrics](prompt_metrics.json). Raw completions and GPU replay traces are stored on `prc-completion-only/self_bleu_temperature/"+manifest["id"]+"`; local token-only scores are reproducible from those artifacts.", "",
        "## Cost and stop", "", f"Before launch: expected new worker cost ${manifest['cost']['estimated_new_worker_usd']:.3f}, based on the recorded prior 8B timings with 25% margin. Measured worker time × frozen resource rate: ${summary['cost']['worker_resource_estimate_usd']:.5f}. Cumulative planning charge including $0.50 new overhead allowance: ${summary['cost']['cumulative_planning_charge_usd']:.5f} / $200. These estimates are not a settled Modal invoice.", "",
        *[f"- {note}" for note in summary["limitations"]], "", "STOP: no extra configurations, prompts, lengths, calibration or eta changes are authorized by this run."]
    (setup/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","collect","analyze"))
    p.add_argument("--setup",type=Path,default=SETUP);p.add_argument("--stage",choices=tuple(TIMEOUTS),default="batch");p.add_argument("--download",action="store_true")
    args=p.parse_args();result=prepare(args.setup) if args.command=="prepare" else (analyze(args.setup) if args.command=="analyze" else collect(args.setup,args.stage,args.download)[1])
    print(json.dumps({k:v for k,v in result.items() if k not in ("files","fallback")},indent=2))


if __name__=="__main__":main()
