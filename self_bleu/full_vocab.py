"""Bounded Qwen3-0.6B full-vocabulary study; local setup and analysis only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
import torch

from .config import StudySetting, digest
from .pilot import paired_interval
from .repeat import upstream_hashes, SamplerRepeatPolicy
from .validation import ROOT, RATE, save, sha
from .topk import partition_probability, prc_draw, checked_synthid_call, repetition, summarize_replay_diagnostics
from .topk import replay_prefix_diagnostics as _prefix_diagnostics

SETUP = ROOT/"outputs/self_bleu_full_vocab/qwen3_0p6b_v1"
DECODER = dict(top_k=None, top_p=1., temperature=1., probability_dtype="float32",
    order="full-vocabulary FP32 logits before watermark; repeat fallback uses ordinary full-vocabulary sampling")
SETTINGS = {"null": StudySetting("null"), "prc": StudySetting("online_prc", eta=.05),
    "synthid_depth2": StudySetting("synthid_text", depth=2),
    "synthid_depth10": StudySetting("synthid_text", depth=10),
    "textseal": StudySetting("textseal", alpha=.1), "gumbel": StudySetting("gumbel_max")}
TIMEOUTS = {"validate":600, "validate_textseal":300, "batch":2400, "textseal":600}
MODEL = dict(id="Qwen/Qwen3-0.6B-Base", size="0.6B", dtype="bfloat16",
    revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd", cache_directory="models/Qwen3-0.6B-Base",
    weight_files={"model.safetensors":"cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba"},
    tokenizer_sha256="c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539")


def full_distribution(logits):
    scores = logits.float()
    if scores.ndim!=2 or not torch.isfinite(scores).all():
        raise ValueError("expected finite batch x vocabulary logits")
    probs = scores.softmax(-1)
    return scores, probs, torch.ones_like(scores, dtype=torch.bool)


def replay_prefix_diagnostics(probabilities, buckets, zero_probability, n):
    result = _prefix_diagnostics(probabilities,buckets,zero_probability,n)
    names = dict(outside_top100="zero_token_probability", outside_and_endpoint="zero_token_and_endpoint",
                 outside_without_endpoint="zero_token_without_endpoint")
    for window in result.values():
        for field in ("counts","rates"):
            window[field] = {names.get(k,k):v for k,v in window[field].items()}
    return result


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
    from baseline_comparison.official import textseal_generator, gumbel_generator
    native = (textseal_generator(alpha=setting.alpha) if setting.method=="textseal" else
              gumbel_generator() if setting.method=="gumbel_max" else None)
    sampler = SamplerRepeatPolicy(native,True,seed,manifest["prompt_indices"]) if native is not None else None
    cache = StaticKVCache(max_length=50+manifest["length"])
    logits = model(ids,cache=cache)[:,-1]
    tokens,ptrace,bits,repeats = [],[],[],[]
    corrected = 0
    for pos in range(manifest["length"]):
        filtered,probs,keep = full_distribution(logits)
        if encoder is not None:
            xi = torch.tensor(encoder.next_bits(),dtype=torch.float32,device=device)
            u = torch.tensor([document_uniform(s,"lm-bucket/v1",pos) for s in document_seeds],dtype=torch.float64,device=device)
            v = torch.tensor([document_uniform(s,"lm-token/v1",pos) for s in document_seeds],dtype=torch.float32,device=device)
            token,p1,_,fixes = prc_draw(filtered,probs,partition,xi,u,v)
            corrected += fixes;ptrace.append(p1.cpu());bits.append(xi.cpu())
        elif sampler is not None:
            token = sampler.sample_next(filtered,ids[:,-3:],temperature=1.,top_p=1.).reshape(-1,1)
            repeats.append(torch.tensor(sampler.repeated[-1],dtype=torch.bool))
        elif processor is not None:
            distribution,repeated = checked_synthid_call(processor,ids,filtered,probs,keep)
            token = torch.multinomial(distribution,1)
            repeats.append(repeated.cpu())
        else:
            token = torch.multinomial(probs,1)
        if torch.any(probs.gather(1,token)==0):raise ValueError("generated token outside common support")
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
        if repeats:row["repeat_fallback"] = torch.stack(repeats,dim=1)[i].tolist()
        rows.append(row)
    return dict(batch_id=bid,identity=identity,responses=rows,
                verification=dict(support_checks=len(prompts)*manifest["length"],support_violations=0,cdf_boundary_repairs=corrected))


def semantic_checks(device="cpu"):
    """Independent probability and native update/fallback oracles; no responses."""
    from baseline_comparison.official import synthid_processor
    from synthid_text.logits_processing import update_scores
    raw = torch.stack((torch.zeros(256),torch.linspace(-4,4,256),torch.arange(256).remainder(7).float())).to(device)
    filtered,q,keep = full_distribution(raw)
    if not keep.all() or not torch.equal(q,raw.float().softmax(-1)):
        raise ValueError("full-vocabulary normalization differs")
    partition = torch.stack((torch.arange(256)%2==0,torch.arange(256)%2==1)).float().to(device)
    p1 = partition_probability(q,partition[1])
    reference = (q.double()*partition[1].double()).sum(-1)/q.double().sum(-1)
    if not torch.allclose(p1.double(),reference,atol=2e-7,rtol=0):raise ValueError("PRC full-vocabulary bucket mass differs")
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
        if not torch.allclose(mixture,q,atol=1e-7,rtol=1e-6):raise ValueError("PRC averaged channel does not preserve full-vocabulary base")
    forced = {}
    for depth in (2,10):
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
            if not torch.equal(output[~repeated],oracle[~repeated]):raise ValueError("full-vocabulary watermarking differs")
            counts += int(repeated.sum())
        if not counts:raise ValueError("probe failed to trigger fallback")
        forced[str(depth)] = counts
    from .repeat import check_sampler_policy
    adapters = {m:check_sampler_policy(m,device) for m in ("textseal","gumbel_max")}
    return dict(passed=True, common_fp32_full_vocab=True, prc_bucket_probability_checked=True,
        empty_full_bucket_endpoints=True, prc_channel_preserves_base=True,
        inverse_cdf_support_checked=True, forced_repeat_counts=forced, sampler_policies=adapters)

@torch.no_grad()
def completion_trace(model,tokens,part1):
    """Raw completion only; fresh position-zero cache; coordinate one abstains."""
    from qwen import StaticKVCache
    cache = StaticKVCache(max_length=tokens.shape[1]-1)
    trace = [];outside = []
    for pos in range(tokens.shape[1]-1):
        logits = model(tokens[:,pos:pos+1],cache=cache)[:,-1]
        _,probs,keep = full_distribution(logits)
        trace.append(partition_probability(probs,part1).cpu())
        outside.append((probs.gather(1,tokens[:,pos+1:pos+2]).squeeze(1)==0).cpu())
    return torch.stack(trace,dim=1),torch.stack(outside,dim=1)


def validate_manifest(manifest,root=ROOT):
    if digest({k:v for k,v in manifest.items() if k!="id"})!=manifest["id"]:
        raise ValueError("manifest identity differs")
    if (manifest["model"]!=MODEL or manifest["decoder"]!=DECODER
            or manifest["settings"]!={n:s.identity() for n,s in SETTINGS.items()}
            or manifest["prompt_indices"]!=list(range(50)) or manifest["seeds"]!=[12345,67890]
            or manifest["length"]!=1024 or manifest["prefix_lengths"]!=[400,1024]
            or manifest["repeat_fallback"]!={s:True for s in SETTINGS if s not in ("null","prc")}
            or manifest["protocol"]!="completion_only_raw_abstain_v1"):
        raise ValueError("study scope differs")
    for name,h in manifest["code_sha256"].items():
        if sha(Path(root)/name)!=h:raise ValueError(f"source changed: {name}")
    if sha(Path(root)/"prompts.jsonl")!=manifest["prompt_sha256"]:raise ValueError("prompts changed")
    cost=manifest["cost"]
    if (cost["timeouts"]!=TIMEOUTS or cost["resource_usd_per_second"]!=RATE
            or cost["reserved_total_usd"]!=cost["previous_planning_charge_usd"]+(sum(TIMEOUTS.values())+8)*RATE+.5
            or cost["reserved_total_usd"]>200):
        raise ValueError("budget differs or exceeded")


def prepare(setup):
    prior_path=ROOT/"outputs/self_bleu_topk/matched_v2"
    prior=json.loads((prior_path/"manifest.json").read_text())
    previous=json.loads((prior_path/"summary.json").read_text())
    names=sorted(set(prior["code_sha256"])|{"self_bleu/full_vocab.py","self_bleu/full_vocab_modal.py"})
    config_path=ROOT/"self_bleu/qwen3_0p6b_config.json"
    names.append(str(config_path.relative_to(ROOT)))
    runtime={**prior["generation_runtime"],"model_revision":MODEL["revision"]}
    textseal_manifest=json.loads((ROOT/"outputs/self_bleu_repeat/setup_v4/manifest.json").read_text())
    old_cost=previous["cost"]["cumulative_planning_charge_usd"]
    manifest=dict(schema_version=1,source_parent_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        model=MODEL,model_config=json.loads(config_path.read_text()),model_config_sha256=sha(config_path),
        model_config_source=f"https://huggingface.co/{MODEL['id']}/resolve/{MODEL['revision']}/config.json",
        decoder=DECODER,settings={n:s.identity() for n,s in SETTINGS.items()},
        repeat_fallback={s:True for s in SETTINGS if s not in ("null","prc")},
        protocol="completion_only_raw_abstain_v1",artifact=prior["artifact"],
        generation_runtime=runtime,textseal_runtime=textseal_manifest["textseal_runtime"],
        upstream_sha256=upstream_hashes(),prc_codeword_sha256=prior["prc_codeword_sha256"],
        prompt_indices=list(range(50)),prompt_sha256=sha(ROOT/"prompts.jsonl"),seeds=[12345,67890],
        length=1024,prefix_lengths=[400,1024],primary_contrast=["prc","synthid_depth2",1024],
        code_sha256={n:sha(ROOT/n) for n in names},test_sha256=sha(ROOT/"tests/test_self_bleu_full_vocab.py"),
        analysis=dict(bootstrap_seed=20260918,bootstrap_resamples=2000,nominal_fpr=.001,
            unit="50 paired prompt clusters retaining both seeds",nulls="100 newly generated matched ordinary responses",
            textseal="unmodified upstream weighted v2, BF16 HF eager entropy, independent direct replay at each prefix",
            synthid="official context mask and weighted normal test, explicit depth-specific keys",
            prc="FP32 full-vocabulary completion-only replay; original MAP/Hoeffding; coordinate one abstains"),
        validation_fixture="first 50 canonical prompt-token arrays treated only as raw replay fixtures; no generated validation responses",
        cache_policy="No response/entropy cache reuse. Reuse verified weights, tokenizer, prompts and original fixed PRC key/partition only.",
        cost=dict(previous_planning_charge_usd=old_cost,resource_usd_per_second=RATE,timeouts=TIMEOUTS,
            overhead_allowance_usd=.5,reserved_total_usd=old_cost+(sum(TIMEOUTS.values())+8)*RATE+.5,
            ceiling_usd=200,rate_basis="frozen planning rate, not settled billing"),
        stop_rule="Exactly 600 responses, 100 per setting. No additional settings or automatic retries.")
    manifest["id"]=digest(manifest);validate_manifest(manifest);save(setup/"manifest.json",manifest)
    return dict(id=manifest["id"],responses=600,cost=manifest["cost"])


def collect(setup,stage,download=False):
    manifest=json.loads((setup/"manifest.json").read_text());validate_manifest(manifest)
    remote=f"self_bleu_full_vocab/{manifest['id']}/{stage}"
    if download:
        import modal
        volume=modal.Volume.from_name("prc-completion-only",create_if_missing=False)
    path=setup/f"{stage}_report.json"
    if download and not path.exists():path.write_bytes(b"".join(volume.read_file(f"{remote}/report.json")))
    report=json.loads(path.read_text())
    if not report["passed"] or report["manifest_id"]!=manifest["id"]:raise ValueError("stage did not pass")
    for name,h in report["files"].items():
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:raise ValueError("unsafe path")
        local=setup/"raw"/stage/name
        if download and not local.exists():
            local.parent.mkdir(parents=True,exist_ok=True)
            local.write_bytes(b"".join(volume.read_file(f"{remote}/{name}")))
        if sha(local)!=h:raise ValueError(f"artifact differs: {name}")
    return manifest,report


def summarize_metrics(prompt_rows,draws):
    vectors,results={},[]
    for setting in SETTINGS:
        for n in (1024,400):
            rows=sorted((r for r in prompt_rows if r["setting"]==setting and r["length"]==n),key=lambda r:r["prompt_index"])
            if [r["prompt_index"] for r in rows]!=list(range(50)):raise ValueError("prompt pairs incomplete")
            metrics=["self_bleu","repeated_4gram_fraction","distinct_3"]+([] if setting=="null" else ["tpr"])
            vectors[setting,n]={m:np.array([r[m] for r in rows]) for m in metrics}
            results.append(dict(setting=setting,length=n,metrics={m:paired_interval(v,draws) for m,v in vectors[setting,n].items()},
                detected=None if setting=="null" else sum(sum(r["detected"]) for r in rows),responses=100,prompts=50))
    pairs=[("prc",s) for s in SETTINGS if s not in ("null","prc")]+[(s,"null") for s in SETTINGS if s!="null"]
    contrasts=[dict(left=a,right=b,length=n,primary=(a,b,n)==("prc","synthid_depth2",1024),
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
    _,ts_validation = collect(setup,"validate_textseal")
    _,ts_report = collect(setup,"textseal")
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
    # Reconstruct repeat masks independently from saved histories, without logits.
    from .repeat import synthid_repeat_masks
    for setting,rows in groups.items():
        if setting.startswith("synthid"):
            masks=synthid_repeat_masks([r["token_ids"] for r in rows],keys=SETTINGS[setting].synthid_keys)
        elif setting in ("textseal","gumbel"):
            masks=[]
            for row in rows:
                history=prompts[row["prompt_index"]]+row["token_ids"]
                seen=set();mask=[]
                for pos in range(1024):
                    context=tuple(history[47+pos:50+pos]);mask.append(context in seen);seen.add(context)
                masks.append(mask)
        else:continue
        if masks!=[r["repeat_fallback"] for r in rows]:raise ValueError("saved repeat/fallback trajectory differs")
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
                    if not (len(replay["rows"])==len(replay["probabilities_2_to_T"])==len(replay["observed_buckets_2_to_T"])==len(replay["replay_zero_token_probability"])==50):
                        raise ValueError("replay diagnostic vectors incomplete")
                    for j,row in enumerate(replay["rows"]):
                        if any(len(replay[key][j])!=1023 for key in ("probabilities_2_to_T","observed_buckets_2_to_T","replay_zero_token_probability")):
                            raise ValueError("replay must have exactly 1023 aligned positions")
                        for n in (400,1024):
                            diagnostic = replay_prefix_diagnostics(replay["probabilities_2_to_T"][j],replay["observed_buckets_2_to_T"][j],replay["replay_zero_token_probability"][j],n)
                            if diagnostic!=row["diagnostics_by_prefix"][str(n)]:raise ValueError("saved prefix diagnostics differ")
                            scored.append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(
                                length=n,score=row["results"][str(n)],replay_diagnostics=diagnostic,
                                replay_zero_token_probability=diagnostic["all"]["counts"]["zero_token_probability"],
                                replay_endpoint_contradictions=diagnostic["all"]["counts"]["endpoint_contradiction"]))
            elif setting.startswith("synthid"):
                scored = score_completions(rows,SETTINGS[setting].depth,[400,1024])
            elif setting=="textseal":
                by_id={r["response_id"]:r for r in ts_report["rows"]}
                scored=[]
                for original in rows:
                    row=by_id[original["response_id"]]
                    for n in (400,1024):
                        result=row["results"][str(n)]
                        if result["completion_sha256"]!=digest(original["token_ids"][:n]):raise ValueError("TextSeal prefix differs")
                        scored.append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(
                            length=n,score=result["comparison"],upstream=result["upstream"]))
            else:
                from baseline_comparison.official import official_gumbel_scores
                from baseline_comparison.scoring import deduplicated_positions,gumbel_gamma_test
                scored=[]
                for row in rows:
                    for n in (400,1024):
                        tokens=row["token_ids"][:n];positions=deduplicated_positions(tokens)
                        score=gumbel_gamma_test(official_gumbel_scores(tokens,positions))
                        scored.append({k:row[k] for k in ("response_id","completion_sha256","prompt_index","response_index")}|dict(
                            length=n,score=score,eligible_positions=positions))
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
            if setting not in ("prc","null"):
                counts = [sum(r["repeat_fallback"][:n]) for r in rows]
                fallback.append(dict(setting=setting,length=n,responses_with_fallback=sum(c>0 for c in counts),responses=100,
                    total_fallbacks=sum(counts),mean_fallback_count=float(np.mean(counts)),
                    fraction_with_fallback=sum(c>0 for c in counts)/100,
                    per_response=[dict(response_id=r["response_id"],prompt_index=r["prompt_index"],response_index=r["response_index"],
                        repeat_count=c,fallback_count=c,first_repeat_position=next((j+1 for j,v in enumerate(r["repeat_fallback"][:n]) if v),None))
                        for r,c in zip(rows,counts)]))
    measured,contrasts = summarize_metrics(prompt_rows,draws)
    save(setup/"raw/score_records.json",score_records);save(setup/"raw/response_metrics.json",response_metrics)
    save(setup/"prompt_metrics.json",prompt_rows)
    resource = sum(r["resource_estimate_usd"] for r in (validation,ts_validation,report,ts_report))
    summary = dict(manifest_id=manifest["id"],settings=manifest["settings"],decoder=DECODER,results=measured,contrasts=contrasts,
        primary_contrast=next(r for r in contrasts if r["primary"]),null_counts=null_counts,fallback=fallback,
        replay_diagnostics=summarize_replay_diagnostics(score_records,draws),
        bootstrap=dict(resamples=len(draws),seed=manifest["analysis"]["bootstrap_seed"],draws_sha256=digest(draws.tolist()),
            unit="50 paired prompt clusters, retaining both seeds",interval="95% percentile, marginal"),
        bleu_signature=str(bleu.get_signature()),analysis_versions={p:importlib.metadata.version(p) for p in ("torch","numpy","scipy","sacrebleu","tokenizers","synthid-text")},
        verification=dict(passed=True,new_full_responses=600,repeat_trajectories_verified=400,support_checks=support_checks,cdf_boundary_repairs=repairs,
            score_records=len(score_records),prompt_records=len(prompt_rows)),
        files={str(p.relative_to(setup)):sha(p) for p in (setup/"manifest.json",setup/"validate_report.json",setup/"batch_report.json",
            setup/"prompt_metrics.json",setup/"raw/score_records.json",setup/"raw/response_metrics.json",setup/"textseal_report.json",setup/"validate_textseal_report.json")},
        cost=dict(worker_resource_estimate_usd=resource,overhead_allowance_usd=.5,
            cumulative_planning_charge_usd=manifest["cost"]["previous_planning_charge_usd"]+resource+.5),
        limitations=["Fixed keys, one model and 50 prompts; primary contrast predeclared; other intervals are exploratory without multiplicity correction.",
            "Pilot nulls are 100 responses clustered in 50 prompts; nominal p<.001 is not a matched empirical FPR.",
            "Boundary bootstrap intervals do not establish zero population false positives or perfect detection.",
            "Self-BLEU measures lexical overlap, not semantic quality. Repetition metrics use raw token IDs.",
            "Replay diagnostics count zero token probabilities and contradictory saved FP32 bucket endpoints. The unchanged detector clips endpoints; it does not drop tokens or shift coordinates.",
            "All contextual methods have repeat fallback enabled. SynthID initializes zero context; TextSeal/Gumbel start from the last three prompt tokens, preserving established adapters.",
            "Generation and PRC replay use FP32 probability calculations; TextSeal detection preserves upstream BF16 HF eager entropy at each actual prefix length. Historical 8B full-vocabulary probability paths differ and are not exactly matched controls."])
    save(setup/"summary.json",summary)
    write_report(setup,summary)
    return summary


def write_report(setup,summary):
    def interval(x):return f"{x['mean']:.5f} [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]"
    lines=["# Qwen3-0.6B full-vocabulary comparison", "",
        "600 new responses: six settings × the same 50 prompts × two seeds. Temperature 1, top-p 1, no top-k; 1,024 tokens each. Fixed keys. Repeat fallback ON for TextSeal α=.1, Gumbel-max, and SynthID depths 2/10. PRC η=.05 is unchanged.", "",
        "BF16 model execution; FP32 probabilities in generation and PRC replay. TextSeal detection retains upstream BF16 Hugging Face eager entropy, with fresh direct replay at each prefix. All detectors use raw completion tokens only and nominal p < .001.", "",
        "Intervals are 95% percentile intervals from 2,000 paired prompt-cluster bootstrap resamples. Both seeds remain within each prompt. Self-BLEU is on a 0–1 scale; lower indicates more lexical diversity.", ""]
    for n in (1024,400):
        lines += [f"## {n} tokens", "", "| Setting | Detection | Self-BLEU [95% CI] | Repeated 4-gram fraction | Distinct-3 |", "|---|---:|---|---:|---:|"]
        for r in summary["results"]:
            if r["length"]!=n:continue
            m=r["metrics"]
            lines.append(f"| {r['setting']} | {'—' if r['detected'] is None else str(r['detected'])+'/100'} | {interval(m['self_bleu'])} | {m['repeated_4gram_fraction']['mean']:.5f} | {m['distinct_3']['mean']:.5f} |")
        lines += ["", "| Paired contrast (left minus right) | Self-BLEU difference [95% CI] | TPR difference [95% CI] |", "|---|---|---|"]
        for r in summary["contrasts"]:
            if r["length"]==n:
                lines.append(f"| {r['left']} − {r['right']} | {interval(r['metrics']['self_bleu'])} | {interval(r['metrics']['tpr']) if 'tpr' in r['metrics'] else '—'} |")
        lines += ["", "Pilot-null detections: "+", ".join(f"{r['setting']}: {r['false_positives']}/100" for r in summary["null_counts"] if r['length']==n)+".", ""]
    lines += ["## Repeat fallback", "", "| Setting | Length | Responses encountering repeats | Total repeats/fallbacks |", "|---|---:|---:|---:|"]
    for r in summary["fallback"]:lines.append(f"| {r['setting']} | {r['length']} | {r['responses_with_fallback']}/100 | {r['total_fallbacks']} |")
    lines += ["", "Per-response repeat counts, first-repeat positions, absolute detection intervals, paired repetition intervals, and prefix-specific early/later PRC/null replay diagnostics are in summary.json. Raw score records, all completions and detector traces are hash-verified in the existing Modal volume under self_bleu_full_vocab/<manifest-id>.", "",
        "## Verification and cost", "", f"Verification: {summary['verification']}. Worker resource estimate: ${summary['cost']['worker_resource_estimate_usd']:.5f}; cumulative planning charge including allowance: ${summary['cost']['cumulative_planning_charge_usd']:.5f} of $200. This is not settled billing.", "",
        *[f"- {note}" for note in summary['limitations']], "", "The requested batch is complete. No further generation is queued."]
    (setup/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","collect","analyze"))
    parser.add_argument("--setup",type=Path,default=SETUP)
    parser.add_argument("--stage",choices=tuple(TIMEOUTS),default="batch")
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    result=prepare(args.setup) if args.command=="prepare" else (analyze(args.setup) if args.command=="analyze" else collect(args.setup,args.stage,args.download)[1])
    print(json.dumps({k:v for k,v in result.items() if k not in ("files","rows","fallback")},indent=2))


if __name__=="__main__":main()
