"""Local JSON/CSV bookkeeping only; all experiment scoring occurred on Modal."""
import csv
import json
from pathlib import Path
import sys
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(OUT/'execution_sources'))
import modal_run as rt
from fixed_4b_comparison import write_json
CSV=ROOT/'outputs/redetection/redetection_results_summary.csv'
before=(OUT/'csv_before_execution.csv').read_bytes()
current=CSV.read_bytes()
if current!=before:
 raise ValueError('CSV changed since approval; preserve and inspect before appending')
reports={}
for size in ['8B','0.6B']:
 folder=OUT/'cache/results/online_8b_eta015_N500_prefixes_v1'/size
 reports[size]=json.loads((folder/'full.json').read_text())
 report=reports[size]
 assert report['passed'] and [r['prompt_idx'] for r in report['records']]==list(range(500))
 assert 6144 not in report['reported_lengths']
 assert report['reuse']=={'T6144_scores_recomputed':False,'primary_traces_replayed':False}
 lengths=report['reported_lengths']
 assert lengths==list(range(6128,lengths[-1]-1,-16))
 for n in lengths:
  for weight in ['map','entropy']:
   count=report['counts'][str(n)][weight]
   assert count['wm']['count']==500 and count['null']['count']==0
native=reports['8B'];lengths=native['reported_lengths']
assert reports['0.6B']['reported_lengths']==lengths
assert native['counts'][str(lengths[-1])]['map']['wm']['detected']<450
assert all(native['counts'][str(n)]['map']['wm']['detected']>=450 for n in lengths[:-1])
for size in ['8B','0.6B']:
 folder=OUT/'cache/results/online_8b_eta015_N500_prefixes_v1'/size
 prepared=json.loads((folder/'prepared.json').read_text())
 rt._append_redetection_csv(prepared,reports[size],CSV)
assert CSV.read_bytes().startswith(before)
with CSV.open() as f: rows=list(csv.DictReader(f))
old=list(csv.DictReader(before.decode().splitlines()))
assert rows[:len(old)]==old and len(rows)==len(old)+2*len(lengths)
added=rows[len(old):]
assert all('N=500; null N=0' in r['Notes'] for r in added)
assert all(int(r['T']) in lengths for r in added)
result={'passed':True,'kind':'JSON/CSV metadata verification, no local scoring',
 'lengths':lengths,'first_below_90_length':lengths[-1],'rows_added':len(added),
 'original_rows_preserved':len(old),'csv_rows_total':len(rows),
 'boundary':{size:reports[size]['counts'][str(lengths[-1])] for size in reports}}
write_json(OUT/'csv_verification.json',result)
print(json.dumps(result))
