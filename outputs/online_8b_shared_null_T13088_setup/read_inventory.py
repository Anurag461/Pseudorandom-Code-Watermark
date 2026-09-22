"""Read-only volume metadata and billing; never dispatch cloud functions."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from modal import Volume, Workspace

OUT = Path(__file__).resolve().parent
results = Volume.from_name('prc-completion-only', create_if_missing=False)
data = Volume.from_name('prc-data', create_if_missing=False)
hf = Volume.from_name('prc-hf-cache', create_if_missing=False)

def fetch(volume, name, label):
    raw = b''.join(volume.read_file(name))
    path = OUT / 'inventory_cache' / label / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return json.loads(raw), {'volume':label, 'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

root = '_nulls/qwen3_8b_base/T13088'
listing = [{'path':e.path, 'bytes':e.size} for e in data.iterdir(root, recursive=False)]
manifest_name = next(e['path'] for e in listing if e['path'].endswith('.json'))
manifest, manifest_ref = fetch(data, manifest_name, 'data')
roots = list(results.iterdir('completion_only_raw_abstain_v1/integrated', recursive=False))
def inspect(entry):
    run, ref = fetch(results,entry.path+'/manifest.json','results')
    case = run.get('case',{})
    nulls=[r for r in case.get('records',[]) if r.get('source')=='null']
    if not nulls or run.get('model',{}).get('id')!='Qwen/Qwen3-8B-Base':
        return None
    return {'manifest':ref,'lengths':case.get('lengths',[]),'null_records':nulls,
            'trace_files':[{'path':e.path,'bytes':e.size} for e in results.iterdir(entry.path, recursive=True) if e.path.endswith('/trace.pt')]}
with ThreadPoolExecutor(max_workers=8) as pool:
    runs=[v for v in pool.map(inspect,roots) if v]
source=next(r for r in runs if len(r['null_records'])==500 and all(x['file']['path']==f'{root}/null_{i:04d}.pt' for i,x in enumerate(r['null_records'])))
listing_by_path={e['path']:e['bytes'] for e in listing}
assert all(listing_by_path[r['file']['path']]==r['file']['bytes'] for r in source['null_records'])
config,config_ref=fetch(hf,'models/Qwen3-8B-Base/config.json','cache')
checkpoint=[{'path':e.path,'bytes':e.size} for e in hf.iterdir('models/Qwen3-8B-Base',recursive=False)]
now=datetime.now(timezone.utc)
value={'checked_utc':now.isoformat(),'read_only':True,'paid_compute_launched':False,
       'source_manifest':manifest,'source_manifest_ref':manifest_ref,'source_listing':listing,
       'source_file_refs':[r['file'] for r in source['null_records']],
       'historical_file_hash_evidence':source['manifest'],
       'source_file_hash_note':'Historical frozen references; paid CPU preparation must verify actual current bytes and full13088 token hashes before GPU dispatch.',
       'integrated_manifests_inspected':len(roots),'native_null_runs':runs,
       'potential_full_length_reuse':[r['manifest'] for r in runs if max(r['lengths'] or [0])>=13088 and r['trace_files']],
       'config':config,'config_ref':config_ref,'checkpoint_listing':checkpoint}
(OUT/'inventory.json').write_text(json.dumps(value,indent=2)+'\n')
ids=json.loads((OUT.parent/'online_8b_eta020_0p6b_setup/billing_final.json').read_text())['app_ids']
rows=[asdict(r) for r in Workspace.from_context().billing.report(start=datetime(2026,9,22,16,tzinfo=timezone.utc),end=now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1),resolution='h') if r.object_id in ids]
cost=sum((Decimal(str(r['cost'])) for r in rows),Decimal(0))
(OUT/'billing_setup_20260922.json').write_text(json.dumps({'checked_utc':now.isoformat(),'prior_run_app_ids':ids,'prior_run_cost_usd':str(cost),'remaining_from_last35_usd':str(Decimal(35)-cost),'new_run_budget':'user deferred budget decision; preparation only','rows':rows},indent=2,default=str)+'\n')
print(json.dumps({'null_files':len(value['source_file_refs']),'manifests_checked':len(roots),'potential_full_length_reuse':value['potential_full_length_reuse'],'source_manifest':manifest,'prior_run_cost_usd':str(cost)}))
