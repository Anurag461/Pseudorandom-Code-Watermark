"""Approved stage dispatch, collection and Git bookkeeping; no local scoring."""
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import os
import pickletools
import re
import subprocess
import sys
import time
import zipfile
from modal import Workspace

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
PLAN=json.loads((OUT/'setup.json').read_text())
CSV=ROOT/'outputs/redetection/redetection_results_summary.csv'
REFERENCE='user-go-ahead-eta020-batch50-20260920'
STAGES=['prepare','replay','score']


def save(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,default=str)+'\n')


def log(message):
 print(datetime.now(timezone.utc).isoformat(),message,flush=True)


def billing(label):
 now=datetime.now(timezone.utc)
 ids=set()
 for p in OUT.glob('launch_*.log'):
  ids.update(re.findall(r'/apps/[^/]+/main/(ap-[A-Za-z0-9]+)',p.read_text()))
 rows=[asdict(r) for r in Workspace.from_context().billing.report(start=datetime(2026,9,20,tzinfo=timezone.utc),end=now.replace(minute=0,second=0,microsecond=0)+timedelta(hours=1),resolution='h')]
 own=[r for r in rows if r['object_id'] in ids]
 cost=sum((Decimal(str(r['cost'])) for r in own),Decimal(0))
 save('billing_'+label+'.json',{'read_only':True,'checked_utc':now.isoformat(),'app_ids':sorted(ids),'reported_task_cost_usd':str(cost),'task_rows':own})
 return cost


def cache_trace_metadata():
 refs=json.loads((OUT/'collected_replay.json').read_text())['files']
 ref=next(r for r in refs if r['path'].endswith('/trace.pt'))
 path=OUT/'cache'/ref['volume']/ref['path']
 with zipfile.ZipFile(path) as z:
  ops=list(pickletools.genops(z.read(next(n for n in z.namelist() if n.endswith('/data.pkl')))))
 names={'seconds','peak_allocated_bytes','peak_reserved_bytes','total_memory_bytes','memory_limit_fraction','full_validation'}
 values={}
 for i,(op,arg,position) in enumerate(ops):
  if op.name in ('BINUNICODE','SHORT_BINUNICODE') and arg in names:
   for following,value,_ in ops[i+1:]:
    if following.name in ('BINPUT','LONG_BINPUT','MEMOIZE'):continue
    if following.name=='NEWFALSE':value=False
    if following.name=='NEWTRUE':value=True
    values[arg]=value;break
 save('primary_trace_metadata.json',{'inspection':'non-executing scalar pickle-opcode inspection; no tensor loading','trace':ref,'values':values})


def checkpoint(stage):
 if subprocess.check_output(['git','branch','--show-current'],text=True).strip()!='redetection':
  raise RuntimeError('branch changed; preserve outputs before Git changes')
 staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
 if staged:raise RuntimeError('unrelated staging detected; preserve it')
 collected=json.loads((OUT/f'collected_{stage}.json').read_text())
 save(f'checkpoint_{stage}.json',{'stage':stage,'saved_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':hashlib.sha256((OUT/'setup.json').read_bytes()).hexdigest(),'files':collected['files'],'storage':'Raw binaries retained locally and committed to Modal volumes; Git keeps manifests, hashes, scores and evidence.'})
 paths=[]
 for p in OUT.rglob('*.json'):
  rel=p.relative_to(OUT)
  if 'execution_sources' in rel.parts:continue
  if rel.parts[0]=='cache' and rel.parts[1]!='results':continue
  paths.append(str(p.relative_to(ROOT)))
 for name in ['PLAN.md','setup.sha256','test_metadata.py','continue_pipeline.py']:
  paths.append(str((OUT/name).relative_to(ROOT)))
 paths.append('online_8b_eta020_batch50.py')
 if stage=='prepare':
  patch=OUT/'memory_limit.patch'
  subprocess.run(['git','apply','--cached','--check',str(patch)],check=True)
  subprocess.run(['git','apply','--cached',str(patch)],check=True)
 if stage=='score':paths.append(str(CSV.relative_to(ROOT)))
 subprocess.run(['git','add','--',*sorted(set(paths))],check=True)
 staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines())
 allowed=set(paths)|({'modal_run.py'} if stage=='prepare' else set())
 if not staged<=allowed:raise RuntimeError('unexpected staged paths')
 subprocess.run(['git','diff','--cached','--check'],check=True)
 subprocess.run(['git','-c','gc.auto=0','commit','-m',f'Save eta0.20 batch50 {stage} outputs'],check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 save(f'commit_{stage}.json',{'commit':head})
 log(f'{stage} saved and committed: {head}')


def append_csv():
 import modal_run as rt
 prepared=json.loads((OUT/'result_prepare.json').read_text())['prepared']
 report_path=OUT/'cache/results'/prepared['root']/'full.json'
 report=json.loads(report_path.read_text())
 assert report['passed'] and report['reported_lengths']==[14336]
 assert [r['prompt_idx'] for r in report['records']]==list(range(50))
 assert all(r['source']=='wm' for r in report['records'])
 before=(OUT/'csv_before_execution.csv').read_bytes()
 if CSV.read_bytes()!=before:raise RuntimeError('CSV changed; preserve and review before append')
 rt._append_redetection_csv(prepared,report,CSV)
 after=CSV.read_bytes()
 assert after.startswith(before)
 old=list(csv.DictReader(before.decode().splitlines()))
 with CSV.open() as f:rows=list(csv.DictReader(f))
 assert rows[:-1]==old and len(rows)==len(old)+1
 assert 'N=50; null N=0' in rows[-1]['Notes']
 save('csv_verification.json',{'passed':True,'original_rows_preserved':len(old),'rows_added':1,'N':50,'null_N':0,'added_row':rows[-1]})


def main():
 log('Continue only the approved three stages; no retries or extra passes.')
 deadline=time.monotonic()+420
 while not (OUT/'collected_prepare.json').exists():
  text=(OUT/'launch_prepare.log').read_text()
  if 'Traceback (most recent call last)' in text or time.monotonic()>deadline:
   raise RuntimeError('preparation did not complete; inspect saved evidence, no retry')
  time.sleep(5)
 billing('after_prepare')
 if not (OUT/'commit_prepare.json').exists():checkpoint('prepare')
 for stage in ['replay','score']:
  if (OUT/f'attempt_{stage}.json').exists():raise RuntimeError(f'{stage} already attempted')
  spent=billing('before_'+stage)
  remaining=sum(Decimal(str(PLAN['stages'][s]['allowance_usd'])) for s in STAGES[STAGES.index(stage):])
  if spent+remaining>Decimal('9.50'):raise RuntimeError('remaining approved stages exceed allowance')
  log(f'Launching approved {stage}; recorded cost so far ${spent}.')
  with (OUT/f'launch_{stage}.log').open('x') as handle:
   subprocess.run([sys.executable,'-m','modal','run','--detach','online_8b_eta020_batch50.py','--stage',stage,'--approval-reference',REFERENCE],stdout=handle,stderr=subprocess.STDOUT,check=True)
  billing('after_'+stage)
  if stage=='replay':cache_trace_metadata()
  if stage=='score':append_csv()
  checkpoint(stage)
 cost=billing('completed')
 save('pipeline_completed.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'reported_task_cost_usd':str(cost),'N':50,'T':14336,'eta':.2,'null_N':0})
 log(f'Completed all approved work; reported cost ${cost}.')


if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('pipeline_error.json',{'error':str(exc),'recorded_utc':datetime.now(timezone.utc).isoformat()})
  raise
