"""Freeze local setup metadata only. No cloud dispatch or numerical scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def load(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v): Path(p).write_text(json.dumps(v,indent=2)+'\n')
def result_ref(folder,root,name):
    p=ROOT/'outputs'/folder/'cache/results'/root/name
    return {'volume':'results','path':root+'/'+name,'bytes':p.stat().st_size,'sha256':sha(p)}

def wm_source(folder,root):
    return {
       'prepared':result_ref(folder,root,'prepared.json'),'report':result_ref(folder,root,'full.json')}

inv=load(OUT/'inventory.json')
native=load(ROOT/'outputs/online_8b_eta020_remaining450_setup/setup.json')
eta15=load(ROOT/'outputs/online_8b_eta015_remaining400_execution/setup.json')
keys=[{'id':'eta015','eta':.15,'generation_T':6144,'artifact':eta15['target_artifact'],
       'lengths':list(range(6144,4655,-16)),
       'watermarked_sources':[
          wm_source('online_8b_eta015_remaining400_execution','online_8b_eta015_remaining400_v1/combined/8B'),
          wm_source('online_8b_eta015_prefixes_setup','online_8b_eta015_N500_prefixes_v1/8B')]},
      {'id':'eta020','eta':.2,'generation_T':14336,'artifact':native['artifact'],
       'lengths':list(range(13088,11839,-16)),
       'watermarked_sources':[
          wm_source('online_8b_eta020_prefixes_setup','online_8b_eta020_N500_prefixes_v1/8B')]}]
for spec in keys:
    counts={}
    for source in spec['watermarked_sources']:
        candidates=list((ROOT/'outputs').glob('*/cache/results/'+source['report']['path']))
        assert len(candidates)==1,candidates
        report=load(candidates[0]);assert report['passed'] and report['settings']['eta']==spec['eta']
        counts.update({n:v for n,v in report['counts'].items() if int(n) in spec['lengths']})
    assert set(counts)=={str(n) for n in spec['lengths']}
    spec['reused_watermarked_counts']={n:{w:c['wm'] for w,c in weights.items()} for n,weights in counts.items()}
    assert all(v['count']==500 for weights in spec['reused_watermarked_counts'].values() for v in weights.values())

source_names=['online_8b_shared_null.py','fixed_4b_comparison.py','online_prc_redetection.py','modal_run.py',
              'qwen.py','detectors.py','prc.py','online_prc.py','watermark_expt.py','constants.py','prompts.jsonl']
hashes={n:sha(ROOT/n) for n in source_names}
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
responses=[]
base=ROOT/'outputs/online_8b_eta020_remaining450_setup/cache/results/online_8b_eta020_T14336_remaining450_v1/attempts/replay'
for p in sorted(base.glob('*/response.json')):
    value=load(p)
    responses.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p),'method_seconds':value['replay']['seconds']})
assert len(responses)==9
mean=sum(r['method_seconds'] for r in responses)/9
predicted=mean*(13088/14336)**2
rate=.001261+4*.0000131+64*.00000222
basis={'pricing_url':'https://modal.com/pricing','rates_checked_utc':datetime.now(timezone.utc).isoformat(),
       'native_T14336_batch50_H200_measurements':responses,'mean_method_seconds':mean,
       'attention_length_scale':(13088/14336)**2,'estimated_method_seconds_per_worker':predicted,
       'loading_io_reserve_seconds_per_worker':120,'gpu_worker_rate_usd_per_second':rate,
       'central_replay_cost_usd':10*(predicted+120)*rate,'estimated_total_usd':[55,65],
       'estimated_elapsed_minutes':[70,90],'measured_exact_shape':False,
       'deadline_replay_resource_envelope_usd':10*(4800+92)*rate,
       'budget_scenarios_not_authorized':{'historical_75_allowance':{'worker_deadline_seconds':4800},
       'possible_65_cap_needs_revised_frozen_plan':{'worker_deadline_seconds':4350,'replay_resource_envelope_usd':10*(4350+92)*rate,'cpu_stage_reserve_usd':.3}},
       'kv_cache_gib_batch50':2*inv['config']['num_hidden_layers']*inv['config']['num_key_value_heads']*inv['config']['head_dim']*2*50*13088/1024**3}
save(OUT/'timing_cost_basis.json',basis)
plan={'status':'setup_only_budget_deferred_no_launch_approval','created_utc':datetime.now(timezone.utc).isoformat(),
      'run_id':'online_8b_shared_null_T13088_N500_v1','branch':'redetection','head':head,'profile':'new-prc-watermark',
      'paid_compute_launched':False,'approval_received':False,'budget_confirmed':False,
      'last_user_instruction':'Prepare only; decide the budget later. Retain the original T13088 shared-null scope; ignore the subsequently cancelled extension discussion.',
      'N':500,'T':13088,'batch_size':50,'prompt_indices':list(range(500)),
      'protocol':'completion_only_raw_abstain_v1','weights':['map','entropy'],'target_fpr':.001,'fpr_policy':'one_shot',
      'model':native['model'],'source_manifest_ref':inv['source_manifest_ref'],'source_file_refs':inv['source_file_refs'],
      'historical_file_hash_evidence':inv['historical_file_hash_evidence'],
      'partition_sha256':inv['source_manifest']['partition_sha256'],
      'keys':keys,'generation_count':0,'watermarked_replay_count':0,'reference_replays':0,'benchmarks':0,'automatic_retries':0,
      'detection':{'raw_completion_ids':True,'prompt_prefix':False,'special_token_prefix':False,
                   'first_coordinate_abstention':True,'dtype':'bfloat16','tf32':False,'cache':'static',
                   'trace_dtype':'float32','score_dtype':'float64','max_memory_fraction':.95},
      'stages':{'prepare':{'gpu':None,'workers':1,'cpu':4,'memory_mib':16384,'work_timeout_seconds':600,'allowance_usd':.15},
                'replay':{'gpu':'H200','workers':10,'cpu':4,'memory_mib':65536,'work_timeout_seconds':4350,'allowance_usd':64.7},
                'score':{'gpu':None,'workers':1,'cpu':4,'memory_mib':8192,'work_timeout_seconds':600,'allowance_usd':.15}},
      'proposed_total_allowance_usd':65,'allowance_status':'user go-ahead for quoted55–65 range; capped at65, including all stages',
      'estimated_total_usd':[55,65],'estimated_elapsed_minutes':[70,90],
      'runtime_source_sha256':hashes,'execution':{'git_commit':head,'files':hashes,'gpu':'H200',
                      'source_hashes_authoritative':True,'allocator':'expandable_segments:True'},
      'csv':'outputs/redetection/redetection_results_summary.csv',
      'cache_reuse_decision':load(OUT/'reuse_check/comparison.json'),
      'scope_exclusions':['eta=.20 FPR aboveT13088, includingT14336','new null generations','0.6B replay','eta=.05/.10 rescoring','naive scoring'],
      'preparation_pending_cloud_checks':['all500 current source-file and full-token hashes','original prompt corpus and key/partition compatibility',
                                          'checkpoint weights and metadata hashes','frozen watermarked report identities before GPU dispatch']}
save(OUT/'setup.json',plan);(OUT/'setup.sha256').write_text(sha(OUT/'setup.json')+'\n')
for name in source_names:
    p=OUT/'execution_sources'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
assert not list(OUT.glob('approval*.json')) and not list(OUT.glob('attempt*.json'))
print(json.dumps({'status':plan['status'],'reporting_points':sum(len(k['lengths']) for k in keys),
                  'null_records':len(plan['source_file_refs']),'central_replay_estimate_usd':basis['central_replay_cost_usd'],
                  'kv_cache_gib_batch50':basis['kv_cache_gib_batch50'],'setup_sha256':sha(OUT/'setup.json'),'paid_compute_launched':False}))
