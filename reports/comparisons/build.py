"""Build detailed comparison assets from saved results; never launches inference.

REPORT.md is the separately maintained paper summary and is not regenerated.

python reports/comparisons/build.py            # numpy + matplotlib
python reports/comparisons/build.py --pdf      # reportlab, after the first command
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, re, subprocess, textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCES={}
FILES={
 'paired':'outputs/self_bleu_repeat/paired_comparison/summary.json',
 'repetition':'outputs/self_bleu_repeat/matched_repetition/summary.json',
 'repeat':'outputs/self_bleu_repeat/setup_v4/all_analysis.json',
 'depth':'outputs/self_bleu_depth/depth2_30_v1/summary.json',
 'short':'outputs/self_bleu_depth/short_prefixes/summary.json',
 'topk':'outputs/self_bleu_topk/matched_v2/summary.json',
 'small':'outputs/self_bleu_full_vocab/qwen3_0p6b_v1/summary.json',
 'temperature':'outputs/self_bleu_temperature/t07_v1/summary.json',
 'large':'outputs/comparison_redetect/baseline_comparisons.csv',
 'large_repetition':'outputs/comparison_redetect/repetition_audit.json',
 'trajectory':'outputs/self_bleu_repeat/setup_v4/synthid_trajectory.json',
 'followup_trajectory':'outputs/self_bleu_repeat/setup_v4/followup_trajectory.json',
}
NAMES={'null':'Ordinary','prc':'PRC eta=.05','synthid_depth2':'SynthID d=2','synthid_depth10':'SynthID d=10','synthid_depth30':'SynthID d=30','synthid_off':'SynthID d=10 (off)','textseal_off':'TextSeal a=.1 (off)','textseal_on':'TextSeal a=.1 (on)','gumbel_off':'Gumbel-Max (off)','gumbel_on':'Gumbel-Max (on)'}
ALIASES={'synthid_on':'synthid_depth10','textseal':'textseal_on','gumbel':'gumbel_on','online_prc':'prc','synthid_text':'synthid_depth10','gumbel_max':'gumbel_off'}
REGIMES={'8b_full':'8B / full vocabulary / T=1','8b_topk':'8B / top-k=100 / T=1','0p6b_full':'0.6B / full vocabulary / T=1','8b_t07':'8B / full vocabulary / T=.7'}
ORDER=['null','prc','synthid_depth2','synthid_depth10','synthid_depth30','textseal_off','textseal_on','gumbel_off','gumbel_on','synthid_off']
BLOCKS=[];TABLES=[];FIGURES=[]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):
 p=ROOT/FILES.get(name,name);SOURCES[str(p.relative_to(ROOT))]=sha(p)
 return list(csv.DictReader(p.open())) if p.suffix=='.csv' else json.loads(p.read_text())
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def heading(s,level=2):BLOCKS.append(dict(type='heading',text=s,level=level))
def para(s):BLOCKS.append(dict(type='paragraph',text=s))
def bullet(s):BLOCKS.append(dict(type='bullet',text=s))
def number(s):return int(s.split('/')[0])
def fmt(v,scale=1,digits=5):
 if v is None:return '--'
 return f"{v['mean']*scale:.{digits}f} [{v['ci95'][0]*scale:.{digits}f}, {v['ci95'][1]*scale:.{digits}f}]"
def delta(v,scale=1,digits=5):return f"{v['mean']*scale:+.{digits}f} [{v['ci95'][0]*scale:+.{digits}f}, {v['ci95'][1]*scale:+.{digits}f}]"
def tex(s):
 return str(s).replace('\\',r'\textbackslash{}').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_\allowbreak ').replace('#',r'\#').replace('~',r'\textasciitilde{}').replace('−','-').replace('η',r'$\eta$').replace('α',r'$\alpha$').replace('×',r'$\times$').replace('↓',r'$\downarrow$').replace('↑',r'$\uparrow$')
def table(name,caption,headers,rows,note=''):
 rows=[[str(c) for c in row] for row in rows];TABLES.append(dict(name=name,caption=caption,headers=headers,rows=rows,note=note))
 with (OUT/'tables'/f'{name}.csv').open('w',newline='') as f:csv.writer(f,lineterminator="\n").writerows([headers]+rows)
 # Fixed-width paragraph columns keep long captions/intervals legible in a paper.
 weights=[1.0]*len(headers)
 if len(headers)>=5:weights[0]=1.25
 if 'Self-BLEU [95% CI]' in headers:weights[headers.index('Self-BLEU [95% CI]')]=1.65
 columns=''.join(r'>{\raggedright\arraybackslash}p{\dimexpr'+f'{w/sum(weights):.6f}'+r'\textwidth-2\tabcolsep\relax}' for w in weights)
 lines=['% Generated from hash-pinned saved results. Requires booktabs, graphicx and array.',r'\begin{table*}[t]',r'\centering',r'\caption{'+tex(caption)+'}',r'\label{tab:'+name+'}',r'\small',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{'+columns+'}',r'\toprule',' & '.join(map(tex,headers))+r' \\',r'\midrule']
 lines += [' & '.join(map(tex,row))+r' \\' for row in rows]
 lines += [r'\bottomrule',r'\end{tabular}}']
 if note:lines += [r'\par\smallskip',r'\begin{minipage}{\textwidth}\footnotesize '+tex(note)+r'\end{minipage}']
 lines += [r'\end{table*}'];(OUT/'tables'/f'{name}.tex').write_text('\n'.join(lines)+'\n')
 BLOCKS.append(dict(type='table',name=name,caption=caption,headers=headers,rows=rows,note=note))
def fig(name,caption):
 FIGURES.append(dict(name=name,caption=caption));BLOCKS.append(dict(type='figure',name=name,caption=caption))

def normalize(data):
 rows={};contrasts=[];nulls=[];checks=[]
 def add(reg,r,source):
  r=copy.deepcopy(r);setting=ALIASES.get(r['setting'],r['setting']);k=(reg,setting,r['length'])
  new=dict(regime=reg,setting=setting,length=r['length'],metrics=r['metrics'],detected=r.get('detected'),responses=100,prompts=50,sources=[FILES[source]])
  if k in rows:
   for m,v in new['metrics'].items():
    if m in rows[k]['metrics']:
     assert abs(rows[k]['metrics'][m]['mean']-v['mean'])<1e-12,(k,m)
     assert max(abs(a-b) for a,b in zip(rows[k]['metrics'][m]['ci95'],v['ci95']))<1e-12,(k,m,'CI')
     checks.append([*k,m])
    else:rows[k]['metrics'][m]=v
   if new['detected'] is not None:
    assert rows[k]['detected'] in (None,new['detected']);rows[k]['detected']=new['detected']
   rows[k]['sources'].append(FILES[source])
  else:rows[k]=new
 for r in data['paired']['results']:add('8b_full',r,'paired')
 for r in data['repetition']['results']:add('8b_full',r,'repetition')
 for r in data['repeat']['results']:
  if r['arm']=='synthid_off':add('8b_full',dict(setting=r['arm'],length=r['length'],detected=r['detected'],metrics={'self_bleu':r['self_bleu'],'tpr':r['tpr']}),'repeat')
 for r in data['depth']['results']:add('8b_full',r,'depth')
 for r in data['temperature']['results']:add('8b_full' if r['temperature']==1 else '8b_t07',r,'temperature')
 for src,reg in [('topk','8b_topk'),('small','0p6b_full')]:
  for r in data[src]['results']:add(reg,r,src)
 for src,reg in [('depth','8b_full'),('topk','8b_topk'),('small','0p6b_full'),('temperature','8b_t07')]:
  for r in data[src]['contrasts']:
   if src=='temperature' and r['temperature']!=.7:continue
   contrasts.append(dict(regime=reg,left=ALIASES.get(r['left'],r['left']),right=ALIASES.get(r['right'],r['right']),length=r['length'],metrics=r['metrics'],source=FILES[src]))
 for r in data['paired']['contrasts']:
  # SynthID is the same native fallback-on arm in both policy views.
  if r['baseline']=='synthid_on':continue
  contrasts.append(dict(regime='8b_full',left='prc',right=r['baseline'],length=r['length'],metrics=r['prc_minus_baseline'],source=FILES['paired']))
 for r in data['repetition']['contrasts']:
  if r.get('kind')=='prc_minus_baseline' or r.get('left')=='prc':
   match=[c for c in contrasts if c['regime']=='8b_full' and c['left']=='prc' and c['right']==ALIASES.get(r['right'],r['right']) and c['length']==r['length']]
   if match:match[0]['metrics'].update(r['metrics'])
 for r in data['paired']['null_counts']:
  setting=ALIASES.get(r['method'],r['method']);setting={'textseal_on':'textseal','gumbel_off':'gumbel'}.get(setting,setting)
  for cohort,key in [('pilot','fresh_pilot'),('historical','historical_shared')]:nulls.append(dict(regime='8b_full',setting=setting,length=r['length'],cohort=cohort,detected=r[key]['false_positives'],responses=r[key]['responses'],source=FILES['paired']))
 for src,reg in [('depth','8b_full'),('topk','8b_topk'),('small','0p6b_full'),('temperature','8b_t07')]:
  for r in data[src]['null_counts']:
   if src=='temperature' and r['temperature']!=.7:continue
   setting=f"synthid_depth{r['depth']}" if src=='depth' else ALIASES.get(r['setting'],r['setting'])
   cohort='historical' if r.get('cohort')=='historical_null' else 'pilot'
   new=dict(regime=reg,setting=setting,length=r['length'],cohort=cohort,detected=r['false_positives'],responses=r['responses'],source=FILES[src])
   existing=[x for x in nulls if all(x[k]==new[k] for k in ('regime','setting','length','cohort'))]
   if existing:assert existing[0]['detected']==new['detected']
   else:nulls.append(new)
 assert len(rows)==50,len(rows)
 assert sum(r['responses'] for r in rows.values() if r['length']==1024)==2500
 for r in rows.values():
  if r['detected'] is not None:assert abs(r['metrics']['tpr']['mean']-r['detected']/100)<1e-12
 return list(rows.values()),contrasts,nulls,checks


def figures(data,rows,contrasts):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.ticker import ScalarFormatter,PercentFormatter
 import numpy as np
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True,'savefig.dpi':300})
 colors={'prc':'#0072B2','null':'#777777','synthid_depth2':'#009E73','synthid_depth10':'#E69F00','synthid_depth30':'#CC79A7','textseal_on':'#D55E00','textseal_off':'#D55E00','gumbel_on':'#8C564B','gumbel_off':'#8C564B','synthid_off':'#E69F00'}
 markers={'prc':'o','null':'x','synthid_depth2':'s','synthid_depth10':'^','synthid_depth30':'D','textseal_on':'v','textseal_off':'v','gumbel_on':'P','gumbel_off':'P','synthid_off':'^'}
 lookup={(r['regime'],r['setting'],r['length']):r for r in rows}
 def finish(f,name):
  for ext in ('pdf','svg','png'):f.savefig(OUT/'figures'/f'{name}.{ext}',bbox_inches='tight',metadata={'Creator':'Saved-results comparison report'} if ext=='pdf' else None)
  plt.close(f)
 def error(ax,x,y,ci,orientation='y',**kw):
  e=[[max(0,y-ci[0])],[max(0,ci[1]-y)]] if orientation=='y' else [[max(0,x-ci[0])],[max(0,ci[1]-x)]]
  ax.errorbar([x],[y],**{'yerr' if orientation=='y' else 'xerr':e},capsize=2.3,elinewidth=1,**kw)
 # 1: do not visually hide coincident high-detection curves.
 f,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
 styles={'prc':('-',0),'textseal_on':('--',2),'synthid_depth10':(':',4),'gumbel_off':('-.',6)}
 for raw,setting in [('online_prc','prc'),('textseal','textseal_on'),('synthid_text','synthid_depth10'),('gumbel_max','gumbel_off')]:
  rr=[r for r in data['large'] if r['Method']==raw];style,z=styles[setting]
  axs[0].plot([int(r['n']) for r in rr],[number(r['TPR'])/5 for r in rr],ls=style,marker=markers[setting],ms=5 if setting=='prc' else 6+z/2,mfc='none',lw=1.2,label=NAMES[setting].replace(' (on)','').replace(' (off)',''),color=colors[setting],zorder=10-z)
 axs[0].set(xlabel='Completion tokens',ylabel='Detected (%)',ylim=(0,104),xticks=[128,400,768,1024],title='A  Completion-only detection')
 axs[0].legend(loc='lower right',frameon=False)
 rr=[r for r in data['large_repetition']['results'] if r['n']==1024]
 order=['null','online_prc','synthid_text','textseal','gumbel_max']
 # The historical audit may call ordinary sampling null.
 for i,s in enumerate(order):
  r=next(r for r in rr if r['method']==s);name=ALIASES.get(s,s);name='textseal_on' if s=='textseal' else name
  axs[1].barh(i,r['mean_repetition_rate']*100,color=colors[name],height=.58)
  axs[1].text(r['mean_repetition_rate']*100+.7,i,f"{r['mean_repetition_rate']*100:.2f}%",va='center',fontsize=8)
 axs[1].set(yticks=range(5),yticklabels=['Ordinary','PRC','SynthID d=10','TextSeal (off)','Gumbel-Max (off)'],xlabel='Repeated token 4-grams (%)',xlim=(0,68),title='B  Native repetition, 1,024 tokens');axs[1].invert_yaxis()
 finish(f,'01_redetection')
 # 2: two primary lengths. The ordinary control is a vertical band, not a TPR point.
 f,axs=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
 for ax,n in zip(axs,[400,1024]):
  null=lookup['8b_full','null',n]['metrics']['self_bleu'];ax.axvspan(*null['ci95'],color=colors['null'],alpha=.12);ax.axvline(null['mean'],color=colors['null'],ls='--',lw=1)
  for s in ['prc','synthid_depth2','synthid_depth10','synthid_depth30','textseal_on','gumbel_on']:
   r=lookup['8b_full',s,n];x=r['metrics']['self_bleu'];y=r['metrics']['tpr']
   ax.errorbar(x['mean'],100*y['mean'],xerr=[[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]],yerr=[[100*(y['mean']-y['ci95'][0])],[100*(y['ci95'][1]-y['mean'])]],fmt=markers[s],ms=5,color=colors[s],capsize=2,label=NAMES[s])
  ax.set(xscale='log',xlim=(.012,.55),ylim=(35,105),title=f'{n:,} tokens',xlabel='Self-BLEU (log scale; lower is more diverse)',ylabel='Detected (%)')
 for ax in axs:ax.set_xticks([.02,.05,.1,.2,.5],labels=['.02','.05','.1','.2','.5'])
 handles,labels=axs[1].get_legend_handles_labels();f.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False)
 finish(f,'02_matched_policy_tradeoff')
 # 3: all predeclared/motivating depth-2 comparisons, with their detection tradeoff.
 regs=list(REGIMES);f,axs=plt.subplots(1,2,figsize=(7.2,3.6),layout='constrained')
 for i,reg in enumerate(regs):
  r=next(c for c in contrasts if c['regime']==reg and c['left']=='prc' and c['right']=='synthid_depth2' and c['length']==1024)
  for ax,m,scale in [(axs[0],'self_bleu',1),(axs[1],'tpr',100)]:
   v=r['metrics'][m];error(ax,v['mean']*scale,i,[x*scale for x in v['ci95']],orientation='x',fmt='o',color=colors['prc'],ms=5)
 for ax in axs:
  ax.axvline(0,color='#444',ls='--',lw=1);ax.set(yticks=range(4),yticklabels=[REGIMES[r] for r in regs]);ax.invert_yaxis()
 axs[0].set(xlabel='PRC minus SynthID d=2 Self-BLEU',title='A  Paired Self-BLEU difference\nNegative favors PRC')
 axs[0].set_xticks([-.02,-.01,0,.01])
 axs[1].set(yticklabels=[],xlabel='PRC minus SynthID d=2 detection (pp)',title='B  Paired detection difference\nPositive favors PRC',xlim=(-102,5))
 finish(f,'03_paired_depth2')
 # 4: repeated contexts and between-response overlap are different.
 f,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
 for i,(off,on,label) in enumerate([('textseal_off','textseal_on','TextSeal'),('gumbel_off','gumbel_on','Gumbel-Max'),('synthid_off','synthid_depth10','SynthID d=10')]):
  for ax,m,scale in [(axs[0],'self_bleu',1),(axs[1],'repeated_4gram_fraction',100)]:
   a,b=[lookup['8b_full',s,1024]['metrics'][m] for s in (off,on)]
   ax.plot([a['mean']*scale,b['mean']*scale],[i,i],color=colors[on],lw=1.5)
   for v,filled in [(a,False),(b,True)]:error(ax,v['mean']*scale,i,[x*scale for x in v['ci95']],orientation='x',fmt='o',color=colors[on],mfc=colors[on] if filled else 'white',ms=6)
 for ax in axs:ax.set(yticks=range(3),yticklabels=['TextSeal','Gumbel-Max','SynthID d=10']);ax.invert_yaxis()
 axs[0].set(xscale='log',xlim=(.013,1.2),xlabel='Self-BLEU (log scale)',title='A  Between-response overlap')
 axs[1].set(yticklabels=[],xlabel='Repeated token 4-grams (%)',title='B  Within-response repetition',xlim=(-3,62))
 axs[0].set_xticks([.02,.05,.1,.2,.5,1],labels=['.02','.05','.1','.2','.5','1'])
 from matplotlib.lines import Line2D
 f.legend([Line2D([],[],marker='o',color='#555',mfc='white',ls='none'),Line2D([],[],marker='o',color='#555',ls='none')],['Fallback OFF','Fallback ON'],loc='outside lower center',ncol=2,frameon=False)
 finish(f,'04_repeat_policy')
 # 5: depths at short lengths with correct uncertainty and no jitter of x coordinates.
 f,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
 for depth in [2,10,30]:
  s=f'synthid_depth{depth}';rr=[r for r in data['short']['results'] if r['depth']==depth]
  axs[0].plot([r['length'] for r in rr],[r['watermarked']['detected'] for r in rr],marker=markers[s],color=colors[s],mfc='none' if depth==30 else colors[s],ms=8 if depth==30 else 5,ls='--' if depth==30 else '-',lw=1,label=f'd={depth}')
  for r in rr:error(axs[0],r['length'],r['watermarked']['detected'],[100*x for x in r['watermarked']['detection_rate']['ci95']],fmt='none',color=colors[s])
  for j,n in enumerate([400,1024]):
   r=lookup['8b_full',s,n]['metrics']['self_bleu'];x=[2,10,30].index(depth)+(j-.5)*.14
   error(axs[1],x,r['mean'],r['ci95'],fmt='o' if j==0 else 's',color=colors[s],mfc='white' if j==0 else colors[s],ms=5)
 axs[0].set(xlabel='Completion tokens',ylabel='Detected (%)',xticks=[64,128,256],ylim=(25,104),title='A  Short-prefix detection');axs[0].legend(frameon=False,loc='lower right')
 axs[1].set(xticks=[0,1,2],xticklabels=['2','10','30'],xlabel='SynthID depth',ylabel='Self-BLEU',title='B  Diversity at 400 and 1,024 tokens',ylim=(.01,.042))
 axs[1].legend([Line2D([],[],marker='o',color='#555',mfc='white',ls='none'),Line2D([],[],marker='s',color='#555',ls='none')],['400 tokens','1,024 tokens'],frameon=False,loc='upper left')
 finish(f,'05_synthid_depth')
 # 6: repetition with matched ordinary controls, by regime (different precision recorded in captions).
 f,axs=plt.subplots(1,2,figsize=(7.2,4.1),layout='constrained')
 for j,s in enumerate(['null','prc','synthid_depth2','synthid_depth10']):
  for i,reg in enumerate(regs):
   r=lookup[reg,s,1024]
   for ax,m in zip(axs,['repeated_4gram_fraction','distinct_3']):
    v=r['metrics'][m];error(ax,v['mean']*100,i+(j-1.5)*.16,[x*100 for x in v['ci95']],orientation='x',fmt=markers[s],color=colors[s],ms=4,label=NAMES[s] if i==0 else None)
 for ax in axs:ax.set(yticks=range(4),yticklabels=[REGIMES[r] for r in regs]);ax.invert_yaxis()
 axs[0].set(xlabel='Repeated token 4-grams (%)',title='A  Lower is less repetitive')
 axs[1].set(yticklabels=[],xlabel='Distinct-3 (%)',title='B  Higher is less repetitive')
 handles,labels=axs[1].get_legend_handles_labels();f.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
 f.legends[0].remove();f.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
 finish(f,'06_repetition_sensitivity')
 # 7: fixed histories, descriptive only; no inferential error bars.
 f,axs=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
 methods=['ordinary','synthid_depth2','synthid_depth10','synthid_depth30']
 for j,(dec,lab) in enumerate([('full_vocab_fp32_reference','Full vocab (FP32 reference)'),('top100_fp32','Top-100 (FP32)')]):
  for ax,m,title in zip(axs,['collision_probability','maximum_probability','top100_retained_mass'],['Collision probability','Maximum token probability','Top-100 retained mass']):
   vals=[next(r for r in data['topk']['common_histories'] if r['decoder']==dec and r['method']==s)['metrics'][m]['mean'] for s in methods]
   ax.bar(np.arange(4)+(j-.5)*.32,vals,width=.3,color=['#899BA8','#0072B2'][j],label=lab)
   ax.set(xticks=range(4),xticklabels=['Ord.','d=2','d=10','d=30'],ylim=(0,1.08),title=title)
 handles,labels=axs[0].get_legend_handles_labels();f.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
 finish(f,'07_common_histories')
 # 8: the temperature-only study is the precision-faithful comparison.
 f,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
 for s in ['null','prc','synthid_depth2','synthid_depth10']:
  for ax,m,scale in [(axs[0],'self_bleu',1),(axs[1],'tpr',100)]:
   if m=='tpr' and s=='null':continue
   vals=[lookup[reg,s,1024]['metrics'][m] for reg in ['8b_full','8b_t07']]
   ax.plot([0,1],[v['mean']*scale for v in vals],color=colors[s],marker=markers[s],lw=1.2,label=NAMES[s])
   for x,v in enumerate(vals):error(ax,x,v['mean']*scale,[y*scale for y in v['ci95']],fmt='none',color=colors[s])
 for ax in axs:ax.set(xticks=[0,1],xticklabels=['T=1','T=0.7'],xlim=(-.15,1.15))
 axs[0].set(ylabel='Self-BLEU',title='A  Between-response overlap at 1,024 tokens',ylim=(0,.08));axs[1].set(ylabel='Detected (%)',title='B  Completion-only detection',ylim=(-3,104))
 handles,labels=axs[0].get_legend_handles_labels();f.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
 finish(f,'08_temperature')

NOTE='Self-BLEU is on a 0-1 scale. Intervals: 2,000 paired resamples of 50 prompts, retaining both seeds and all arms; marginal 95% percentile intervals. Detection uses nominal p<.001, without empirical FPR matching. Repetition is measured within responses; Self-BLEU is measured between responses.'

def report(data,rows,contrasts,nulls):
 lookup={(r['regime'],r['setting'],r['length']):r for r in rows}
 heading('Watermark comparisons: final consolidated report',1)
 para('Closed 19 September 2026. This report consolidates the completed baseline redetection, paired diversity studies and bounded sensitivity experiments. It uses saved results only. No generation, model inference, threshold tuning or further experiment was performed for this report.')
 heading('Highlighted takeaways')
 for text in [
  'No general PRC advantage over SynthID is established. At T=1 and 1,024 tokens, PRC-minus-depth-2 Self-BLEU is -0.00097 [-0.00459, +0.00265] for 8B full vocabulary, +0.00095 [-0.00503, +0.00734] for 8B top-100, and +0.00051 [-0.00141, +0.00248] for 0.6B full vocabulary. These intervals include zero; that is uncertainty, not equivalence. PRC detection is respectively 97/100, 85/100 and 99/100, versus 100/100 for depth 2.',
  'PRC does have lower between-response Self-BLEU than the tested TextSeal alpha=.1 and Gumbel-Max configurations. In the 8B fallback-on comparison at 1,024 tokens, paired differences are -0.02837 [-0.03531, -0.02201] against TextSeal and -0.18829 [-0.21248, -0.16445] against Gumbel. Both baselines detect 100/100 versus PRC 97/100. This is a configuration-specific diversity/detection tradeoff, not a parameter-frontier result.',
  'Repeat handling explains much of the native-policy repetition gap. With fallback on, repeated-four-gram fractions at 1,024 tokens are 2.38% PRC, 2.42% SynthID, 2.74% TextSeal and 2.81% Gumbel. All paired PRC-versus-baseline repetition intervals include zero. Between-response Self-BLEU remains different: reducing loops does not necessarily reduce overlap between two responses.',
  'SynthID depth 2 is an essential comparator. Depth 30 has higher Self-BLEU than depth 2 at the long lengths, while both detect 100/100 at T=1. At 64/128 tokens depth 10 detects 100/100 versus depth 2 at 45/100 and 82/100; depth 30 adds no observed detection benefit over depth 10 on these cutoffs. Neither detector equivalence nor perfect population detection follows.',
  'The final temperature result is unfavorable for PRC detection. At T=.7, PRC-minus-depth-2 Self-BLEU is -0.01098 [-0.01976, -0.00227], but PRC detects only 3/100 at 1,024 tokens versus 93/100 and 100/100 for SynthID depths 2 and 10. Its T=1 detection was 97/100. At 400 tokens the counts are 1/100, 87/100 and 100/100. This does not demonstrate an improved overall tradeoff.',
  'The larger, separate corrected comparison also shows weaker PRC detection at shorter lengths: 274/500 at 400 tokens and 466/500 at 1,024, versus near-ceiling baseline counts. Its native TextSeal/Gumbel repetition results are policy-sensitive and should not be presented as a matched-fallback advantage.',
  'All conclusions are exploratory: fixed keys, a reused 50-prompt paired cohort, nominal thresholds, and no empirical FPR calibration. The two response slots are not 100 independent prompts. Historical and pilot nulls overlap and must not be pooled. Self-BLEU and repetition do not measure semantic quality.',
  'The campaign is closed. Depth 20, a full TextSeal alpha sweep, PRC eta sweep, Bayesian SynthID, attacks, unseen-prompt confirmation and a large null-calibration campaign were not completed. Cumulative Self-BLEU planning charge is $12.51222 of the $200 ceiling, including allowances; this is not an account-wide settled invoice.'
 ]:bullet(text)
 heading('1. Cohorts, scope and experiment ledger')
 para('Two different cohorts are kept separate throughout. The historical baseline campaign has 500 completions per watermark and 500 shared ordinary completions (2,500 source responses). The paired diversity campaign uses the same 50 canonical prompts with two sampling seeds per configuration. Its completed evaluation configurations contain 2,500 response slots across four decoder/model regimes; these are not 2,500 independent prompts or necessarily unique texts. Native Gumbel seed slots duplicate one another. Some historical first-seed TextSeal/SynthID/Gumbel responses also match the later pilot; do not add cohorts as independent evidence.')
 ledger=[
 ['Historical controlled baseline / proxy campaign','500 prompts per setting','Saved generation and historical detection','Completed; prompted native/proxy scores superseded for paper detection'],
 ['Completion-only baseline redetection','500 per watermark + 500 shared nulls','PRC prefix rescoring; shared-null replay; TextSeal direct-prefix replay; SynthID/Gumbel saved-score audit','Completed; 24 method/length cells'],
 ['Two-response implementation validation','50 prompts; seeds 12345/67890','600 saved full-length slots, 200 short slots; same-seed controls and repairs','Completed; only five settings enter Stage A'],
 ['Stage A native-policy pilot','5 settings x 100 = 500 slots','Reuse validation outputs; completion-only scoring','Completed; no extra generation'],
 ['Repeat-handling intervention','3 modified settings x 100 = 300','SynthID d=10 OFF; TextSeal/Gumbel ON','Completed; native controls reused'],
 ['Paired Self-BLEU / matched repetition','Saved Stage A + repeat outputs','Paired analysis only','Completed; zero GPU cost'],
 ['SynthID depths 2 and 30','2 settings x 100 = 200','New responses; d=10/PRC/ordinary reused','Completed; depth 20 cancelled'],
 ['SynthID 64/128/256-token scoring','Saved d=2/10/30 and nulls','Detection-only CPU analysis; no new text','Completed; 6,300 prefix scores'],
 ['8B top-100, T=1','5 settings x 100 = 500','Ordinary, PRC, SynthID d=2/10/30','Completed; matched ordinary controls'],
 ['0.6B full vocabulary, T=1','6 settings x 100 = 600','Ordinary, PRC, SynthID d=2/10, TextSeal, Gumbel','Completed; all contextual fallbacks ON'],
 ['8B full vocabulary, T=.7','4 settings x 100 = 400','Ordinary, PRC, SynthID d=2/10','Completed; temperature-faithful precision; stopped'],
 ]
 table('01_experiment_ledger','Completed experiments and their non-overlapping roles',['Experiment','Evaluation cohort','Work performed','Status'],ledger,'Counts are response slots. Validation controls, short checks and reused rows are not added to evaluation sample sizes.')
 para('Validation-only configurations include TextSeal alpha=0 full-length pairs and alpha=.5 / SynthID depths 2/20/30 short checks. Alpha=0 was deterministic across seeds. The alpha=0 pairs were not promoted to a scored comparison row. Depth 20 has a short validation check only, not a full-length evaluation arm. The initial setup had image/import failures and a PRC assertion repair, and Stage A had a detector-verification argument repair; all are preserved in the historical reports. The top-100 v1 run completed validation only; v2 supplies the evaluation cohort. These records are not hidden or counted as extra evidence.')
 heading('2. What was held fixed, and what was not')
 table('02_protocols','Numerical and policy differences that must accompany cross-study comparisons',['Regime','Model / temperature','Sampling','Probability arithmetic','Repeat policy'],[
 ['8B original + depths','Qwen3-8B-Base / 1','Full vocabulary','Ordinary FP32; PRC bucket BF16; SynthID update BF16','Native plus separate on/off ablation'],
 ['8B top-100','Qwen3-8B-Base / 1','Top-100 before watermark','Common FP32 probabilities; BF16 model','Native SynthID ON'],
 ['0.6B full vocabulary','Qwen3-0.6B-Base / 1','Full vocabulary','Common FP32 probabilities; BF16 model','All contextual baselines ON'],
 ['8B temperature follow-up','Qwen3-8B-Base / .7','Full vocabulary','Original method paths; single BF16 temperature division','Native SynthID ON'],
 ],'Consequently top-100 versus original full vocabulary is not a pure truncation-only comparison, and 0.6B versus original 8B is not a pure model-size comparison. The T=.7 follow-up explicitly preserves the original numerical paths.')
 para('Paired studies preserve prompts 0-49, their original 50-token formatting, seeds 12345 and 67890, fixed secret keys independent of sampling seeds, H100 BF16 model execution, and exactly 1,024 completion-generation steps with the original forced-length/EOS policy. PRC uses eta=.05, t=3 and row rate 99/100 with the fixed artifact and position-addressed randomness. TextSeal uses alpha=.1 and original keys. SynthID uses its fixed nested key bank, ngram length 4, two leaves, history 1,024, native context initialization and fresh repeat state per response. No key sweep was run.')
 para('Detection takes raw completion IDs only, without the original prompt, BOS/chat template or generation-time traces. PRC reconstructs bucket probabilities and abstains at coordinate 1, preserving coordinates and its MAP/Hoeffding detector. TextSeal uses pinned upstream entropy-weighted scoring, with a separate direct forward at each prefix. Its BF16 shape-dependent entropy check showed that slicing a longest-prefix trace is not numerically identical at every length. SynthID receives the exact key list for each depth and the official context-repetition mask in the paired studies; the legacy 500-prompt comparison retains its audited historical tuple mask. Its detector is the existing layer-weighted normal test, not the Bayesian detector. Gumbel retains its Gamma test. All comparisons use nominal p<.001; none tunes thresholds on the reported nulls.')
 para('Self-BLEU is the average of both directions of sentence BLEU between a prompt\'s two responses, divided by 100: SacreBLEU 2.4.3, 13a tokenization, exponential smoothing, effective order, case-sensitive, decoded with the pinned model tokenizer and special tokens skipped. Repeated token four-gram fraction is 1-unique4/(T-3); distinct-3 is unique3/(T-2), calculated on raw completion IDs including special tokens. Average within each prompt before summarizing. The bootstrap resamples 50 prompt clusters jointly 2,000 times, seed 20260918, keeping both seed slots and all compared arms; differences are bootstrapped directly, not formed from marginal interval endpoints.')
 heading('3. Corrected 500-prompt baseline comparison')
 para('This is the main larger-cohort completion-only comparison. PRC watermarked traces were reused only after score agreement, and all 500 comparison shared nulls were replayed because they differ from the older PRC null cohort. TextSeal replayed all 500 marked and 500 null responses at each actual length. Token-only SynthID/Gumbel scores were audited against saved text and reused. Repetition was recomputed on actual prefixes, correcting older quality columns that referred to the full response even on shorter detection rows.')
 table('03_large_detection','Completion-only detection and shared-null counts (500 responses per cell)',['Tokens','PRC TP / FP','TextSeal TP / FP','SynthID d=10 TP / FP','Gumbel TP / FP'],[[n]+[f"{number(next(r for r in data['large'] if int(r['n'])==n and r['Method']==m)['TPR'])} / {number(next(r for r in data['large'] if int(r['n'])==n and r['Method']==m)['FPR'])}" for m in ['online_prc','textseal','synthid_text','gumbel_max']] for n in [128,256,400,512,768,1024]],'Entries are true positives out of 500 / false positives out of 500, not percentages. SynthID uses the legacy audited tuple-mask scorer here. Native TextSeal/Gumbel generation fallback is OFF.')
 table('03b_prc_redetection_change','PRC detection correction on the same saved historical watermarked responses',['Tokens','Old prompted TP /500','Completion-only TP /500','Change (pp)'],[[r['n'],number(r['Old Posterior TPR']),number(r['Posterior TPR']),r['Posterior TPR change (pp)']] for r in data['large'] if r['Method']=='online_prc'],'Old values are historical prompt-conditioned scores, shown only to document the correction. They are not valid completion-only comparator results and are excluded from all paper tradeoff figures.')
 large_rep=sorted(data['large_repetition']['results'],key=lambda r:(r['n'],r['method']))
 table('04_large_repetition','Within-response repetition on the historical cohort at 400 and 1,024 tokens',['Tokens','Method','Repeated 4-grams (%)','Distinct-3 (%)'],[[r['n'],r['method'],f"{100*r['mean_repetition_rate']:.2f}",f"{100*r['mean_distinct_3']:.2f}"] for r in large_rep if r['n'] in [400,1024]],'Means over 500 responses; no paired Self-BLEU can be inferred from a single response per prompt. Native generation policies differ.')
 fig('01_redetection','Historical 500-response-per-setting comparison after completion-only correction. Left: each cutoff is a separate one-shot detection test; coincident baseline curves are retained. Right: 1,024-token repetition under the native mixed policies, with TextSeal/Gumbel fallback OFF. These repetition gaps are not the matched-policy result.')
 heading('4. Original 8B paired comparison, native and matched policies')
 para('The original native view has SynthID fallback ON and TextSeal/Gumbel OFF. The matched-on view enables fallback for every contextual baseline; PRC is position based and unchanged, and ordinary sampling is unchanged. Native context initialization and RNG conventions are retained, so this matches the fallback rule, not every implementation detail. The same PRC, ordinary and native SynthID responses appear in both views and count only once.')
 for n in [1024,400]:
  selected=[lookup['8b_full',s,n] for s in ORDER if s!='synthid_off']
  table(f'05_8b_full_{n}',f'Original 8B full-vocabulary paired comparison at {n:,} tokens, T=1',['Setting / fallback','Self-BLEU [95% CI]','Detected /100','Repeated 4-grams (%)','Distinct-3 (%)'],[[NAMES[r['setting']],fmt(r['metrics']['self_bleu']),r['detected'] if r['detected'] is not None else '--',f"{r['metrics']['repeated_4gram_fraction']['mean']*100:.2f}" if 'repeated_4gram_fraction' in r['metrics'] else 'NR',f"{r['metrics']['distinct_3']['mean']*100:.2f}" if 'distinct_3' in r['metrics'] else 'NR'] for r in selected],NOTE+' NR: metric not reported for depth 30 in the saved summaries; no value is imputed. Native SynthID fallback is ON.')
  cs=[r for r in contrasts if r['regime']=='8b_full' and r['left']=='prc' and r['length']==n and r['right']!='null']
  table(f'06_8b_contrasts_{n}',f'Direct paired PRC-minus-baseline differences at {n:,} tokens',['Baseline','Self-BLEU difference [95% CI]','Detection difference (pp) [95% CI]'],[[NAMES[r['right']],delta(r['metrics']['self_bleu']),delta(r['metrics']['tpr'],100,1)] for r in cs],NOTE+' Negative Self-BLEU favors PRC; positive detection difference favors PRC. Fallback labels identify separately generated arms.')
 fig('02_matched_policy_tradeoff','8B T=1 comparison with fallback ON for all contextual baselines, at 400 and 1,024 tokens. Points and error bars show means and marginal 95% prompt-bootstrap intervals. Ordinary Self-BLEU is a gray vertical mean/band because ordinary text has no watermarked TPR. The logarithmic Self-BLEU axis retains Gumbel without hiding the near-ordinary settings. This is a set of tested configurations, not a fitted frontier or equal-FPR comparison.')
 heading('5. Repeat-handling intervention and trajectory checks')
 para('One hundred new responses per modified arm used the same prompts, sampling seeds, fixed keys and decoder. All 300 original/modified response pairs passed the full-trajectory check that divergence never precedes the first repeated context, with native-prefix and forced-repeat controls. Original and modified repeat/fallback counts and first-repeat/divergence positions were retained. This supports a specific implementation intervention; it does not establish why the TextSeal paper selected its protocol or any intent by its authors.')
 policy_rows=[]
 for n in [400,1024]:
  for off,on,label in [('textseal_off','textseal_on','TextSeal'),('gumbel_off','gumbel_on','Gumbel-Max'),('synthid_off','synthid_depth10','SynthID d=10')]:
   a,b=[lookup['8b_full',s,n] for s in (off,on)]
   saved=next(r for r in data['repeat']['results'] if r['length']==n and r['arm']==('synthid_off' if off=='synthid_off' else on))['new_minus_original']['self_bleu_difference']
   effect=saved if off!='synthid_off' else dict(mean=-saved['mean'],ci95=[-saved['ci95'][1],-saved['ci95'][0]])
   policy_rows.append([n,label,f"{a['metrics']['self_bleu']['mean']:.5f} -> {b['metrics']['self_bleu']['mean']:.5f}",delta(effect),f"{a['metrics']['repeated_4gram_fraction']['mean']*100:.2f} -> {b['metrics']['repeated_4gram_fraction']['mean']*100:.2f}"])
 table('07_repeat_policy','Effect of enabling repeated-context fallback',['Tokens','Method','Self-BLEU OFF -> ON','Paired ON-OFF Self-BLEU [95% CI]','Repeat-4 (%) OFF -> ON'],policy_rows,NOTE+' All these baseline arms detected 100/100 at both lengths. SynthID OFF is diagnostic and never replaces native ON in the main comparison.')
 para('For SynthID, removing fallback increased mean repeated contexts from 45.53 to 114.21 per 1,024-token response, but changed Self-BLEU by only -0.00067 [-0.00421, +0.00309]. The ablation therefore does not support fallback as the main explanation for SynthID\'s between-response diversity on this cohort. TextSeal fallback reduced repeated contexts but increased long-prefix Self-BLEU. Gumbel OFF is deterministic at fixed key; fallback ON breaks that determinism and substantially lowers Self-BLEU, while still leaving considerable overlap between the two responses.')
 trajectory=[]
 for method,obj in [('SynthID',data['trajectory'])]+[(k,data['followup_trajectory']['arms'][k]) for k in ['textseal_on','gumbel_on']]:
  assert obj['passed'] and not obj['failed_pairs'] and obj['response_pairs']==100
  for r in obj['summaries']:
   trajectory.append([method,r['length'],f"{r['original']['responses_with_repeat']}/100 -> {r['modified']['responses_with_repeat']}/100",f"{r['original']['repeat_count_mean']:.2f} -> {r['modified']['repeat_count_mean']:.2f}",f"{r['original']['fallback_count_mean']:.2f} -> {r['modified']['fallback_count_mean']:.2f}",f"{r['original']['first_repeat_position_median_among_affected']} -> {r['modified']['first_repeat_position_median_among_affected']}",r['pairs_with_divergence'],r['first_token_divergence_median_among_diverged']])
 table('08_trajectories','Original-to-modified repeat trajectories',['Arm','Tokens','Responses with repeat','Mean repeats','Mean fallbacks','Median first repeat','Diverged /100','Median first divergence'],trajectory,'SynthID changes ON to OFF; TextSeal/Gumbel change OFF to ON. First-repeat positions are medians among affected responses; divergence positions are medians among diverged pairs. Both use zero-based completion positions. Full response-level first-repeat positions remain in the source diagnostic files.')
 fig('04_repeat_policy','Repeat fallback at 1,024 tokens: open circles are OFF, filled circles ON; each line connects the same method under the two policies. Left: between-response Self-BLEU on a log scale. Right: within-response repeated-four-gram fraction. Marginal 95% intervals are shown; paired policy effects are reported in the tables. Reduced within-response repetition need not reduce between-response overlap.')
 heading('6. SynthID depth and short-prefix detection')
 para('Depths 2 and 30 each added 100 full-length responses with native fallback ON; saved depth-10, PRC and ordinary pairs were reused. Every depth detected 100/100 at 400 and 1,024 tokens, but depth 30 had higher Self-BLEU than depths 2 and 10. This makes a comparison to depth 30 alone insufficient to establish superiority over SynthID. Later scoring of saved prefixes at 64, 128 and 256 tokens exposed a detection advantage of depth 10 over depth 2 at shorter lengths, with no additional observed gain for depth 30. No short-prefix Self-BLEU endpoint was substituted for the main long-prefix results.')
 table('09_short_prefixes','Saved SynthID generations scored at short prefixes',['Depth','Tokens','Detected /100','TPR (%) [95% CI]','Pilot FP /100','Historical FP /500'],[[r['depth'],r['length'],r['watermarked']['detected'],fmt(r['watermarked']['detection_rate'],100,1),r['pilot_null']['detected'],r['historical_null']['detected']] for r in data['short']['results']],NOTE+' Official context mask and exact per-depth keys. Historical nulls were rescored for each depth, so these counts need not equal the legacy 500-prompt tuple-mask counts.')
 fig('05_synthid_depth','SynthID depth comparison on saved 8B full-vocabulary T=1 responses. Left: detection at 64/128/256 tokens with prompt-bootstrap intervals; depth-10 and depth-30 curves coincide. Right: Self-BLEU at the original 400/1,024-token endpoints. Depth 30 adds observed overlap without an observed detection gain over depth 10 on these measured cutoffs; no population equivalence or Bayesian-detector claim follows.')
 heading('7. Matched top-100, 0.6B and temperature follow-ups')
 para('Each follow-up generated its own temperature/model/decoder-matched ordinary responses. Incompatible old responses were not reused as controls. The top-100 pipeline truncates before watermarking with deterministic tie handling and common FP32 probability arithmetic. The 0.6B batch retains that FP32 probability path but uses full vocabulary, with TextSeal/Gumbel/SynthID fallback ON. Both differ numerically from the original 8B BF16 bucket/update paths. The final T=.7 experiment was separately audited to preserve the original 8B arithmetic and adds one BF16 logit division before the unchanged samplers; SynthID internal temperature remains 1, avoiding double scaling.')
 for reg,src in [('8b_topk','topk'),('0p6b_full','small'),('8b_t07','temperature')]:
  for n in [1024,400]:
   rr=sorted([r for r in rows if r['regime']==reg and r['length']==n],key=lambda r:ORDER.index(r['setting']))
   table(f'10_{reg}_{n}',f'{REGIMES[reg]} at {n:,} tokens',['Setting','Self-BLEU [95% CI]','Detected /100','TPR (%) [95% CI]','Repeat-4 (%)','Distinct-3 (%)'],[[NAMES[r['setting']],fmt(r['metrics']['self_bleu']),r['detected'] if r['detected'] is not None else '--',fmt(r['metrics'].get('tpr'),100,1),f"{r['metrics']['repeated_4gram_fraction']['mean']*100:.2f}",f"{r['metrics']['distinct_3']['mean']*100:.2f}"] for r in rr],NOTE)
 primary=[]
 for reg in REGIMES:
  for n in [1024,400]:
   r=next(c for c in contrasts if c['regime']==reg and c['left']=='prc' and c['right']=='synthid_depth2' and c['length']==n)
   primary.append([REGIMES[reg],n,delta(r['metrics']['self_bleu']),delta(r['metrics']['tpr'],100,1)])
 table('11_primary_contrasts','PRC-minus-SynthID-depth-2 results across completed regimes',['Regime','Tokens','Self-BLEU difference [95% CI]','Detection difference (pp) [95% CI]'],primary,NOTE+' The 1,024-token endpoint was predeclared primary for the top-100, 0.6B and T=.7 follow-ups. The original depth study reported both lengths; all are exploratory sensitivity results on the reused cohort.')
 fig('03_paired_depth2','Direct paired PRC-minus-SynthID-depth-2 contrasts at 1,024 tokens. Left: Self-BLEU differences, where negative favors PRC. Right: detection differences in percentage points, where positive favors PRC. Error bars are paired prompt-bootstrap 95% intervals. Rows are separate matched-control experiments, not a pure one-factor model-size/truncation sweep; probability arithmetic differs for top-100 and 0.6B. The T=.7 study preserves the original 8B paths.')
 fig('06_repetition_sensitivity','Within-response repetition at 1,024 tokens across the four completed regimes, showing ordinary, PRC, and SynthID depths 2/10. Error bars are marginal 95% prompt-bootstrap intervals. Cross-regime numerical paths differ as documented. These metrics concern individual responses and should not be substituted for between-response Self-BLEU.')
 fig('08_temperature','The precision-faithful 8B full-vocabulary temperature sensitivity at 1,024 tokens. T=1 uses saved original outputs; T=.7 uses new matched controls. Left: Self-BLEU rises for every arm at the lower temperature. Right: PRC detection falls from 97/100 to 3/100, depth 2 from 100/100 to 93/100, and depth 10 remains 100/100. Intervals use prompt resampling; lines connect the two evaluated temperatures and do not imply an unmeasured sweep.')
 heading('8. Null counts and the limits of calibration')
 para('All cutoffs use the same nominal .001 threshold definitions within each detector. No method was tuned to the pilot nulls, no FPR matching was performed, and 0/100 does not establish a calibrated 0.1% tail. The historical 500-null cohort overlaps pilot prompts. Do not pool 100 pilot slots with 500 historical responses into 600 independent observations. Degenerate [0,0] or [100,100] bootstrap intervals describe the observed cluster sample, not zero population error or perfect detection.')
 for reg in REGIMES:
  rr=sorted([r for r in nulls if r['regime']==reg],key=lambda r:(r['setting'],r['cohort'],r['length']))
  table('12_nulls_'+reg,'Null counts: '+REGIMES[reg],['Detector','Cohort','Tokens','False positives','Responses'],[[r['setting'],r['cohort'],r['length'],r['detected'],r['responses']] for r in rr],'Historical and pilot cohorts are separate. Counts may be reused across generation-policy views because the detector is unchanged; they are not new evidence in each view.')
 para('The consolidated count correction is Gumbel 2/500 historical false positives at 400 tokens, not zero as stated in early Stage A prose. The paired SynthID depth-10 pilot count is 1/100 at 400. The original legacy SynthID historical detector gives 1/500 at 128 and 0/500 at 256, whereas the official-context-mask short-prefix reanalysis gives 1/500 at 128 and 256. These are different documented detector masks, not an unexplained contradiction. The tables retain this distinction.')
 heading('9. Replay, precision and common-history diagnostics')
 diagnostics=[]
 for src in ['topk','small','temperature']:
  for r in data[src]['replay_diagnostics']:
   key='outside_top100' if src=='topk' else 'zero_token_probability'
   diagnostics.append([src,r['source'],r['length'],r['window'],r['positions'],r['metrics'][key]['count'],r['metrics']['endpoint_contradiction']['count']])
 for src in ['topk','small','temperature']:
  table('13_replay_'+src,'Prefix-specific replay observations: '+{'topk':'8B top-100','small':'0.6B full vocabulary','temperature':'8B T=.7'}[src],['Study','Response source','Tokens','Window','Positions','Support / zero-token events','Endpoint contradictions'],[r for r in diagnostics if r[0]==src],'Top-100 events mean the observed token is outside the prompt-free replay top-100 set; full-vocabulary events mean zero observed-token probability. Early positions are 2-64; later positions 65-n. These are different diagnostics and must not be conflated. The primary detector is unchanged.')
 para('Top-100 generation had zero support violations in all 512,000 generated tokens and zero contradictory replay endpoints. Prompt-free replay nevertheless places 736/102,300 PRC tokens and 788/102,300 ordinary tokens outside its top-100 support at 1,024 tokens, concentrated early. Generation saw the prompt and replay does not, so these are not automatically generation violations. The observed token can be absent while its binary bucket still has positive probability. No tokens were dropped and no coordinates shifted.')
 para('The 0.6B full-vocabulary study recorded no zero-probability observations or endpoint contradictions. At T=.7, both the PRC and ordinary 1,024-token cohorts have eight contradictory saved bucket scalars, all after position 64: BF16-aggregated p1=1 with an observed bucket-0 token. All observed tokens retain positive replay probability. This indicates rounding of the scalar, not that the entire bucket or observed token is actually impossible. The existing clipped-endpoint score was retained. These diagnostic counts do not establish the cause of the PRC detection loss.')
 history_rows=[]
 for r in data['topk']['common_histories']:
  history_rows.append([r['decoder'],r['method']]+[f"{r['metrics'][m]['mean']:.6f}" for m in ['collision_probability','maximum_probability','top100_retained_mass','base_top100_retained_mass']])
 table('14_common_histories','Distribution measurements on 25 preselected common histories',['Decoder','Method','Collision probability','Max token probability','Top-100 retained mass','Ordinary base mass'],history_rows,'Five saved ordinary histories (prompts 0,7,19,31,49) at positions 0,32,128,400,1023. Means are descriptive, with no bootstrap inference. Both decoders use FP32 probability arithmetic; the full-vocabulary reference is not the original BF16 SynthID path. Native repeat fallback remains enabled.')
 fig('07_common_histories','Means on 25 preselected shared histories. Collision probability is the sum of squared token probabilities; maximum probability is the largest token probability. Retained mass is on the ordinary top-100 support after watermarking. Both decoders use FP32 arithmetic, so this controlled distribution probe is distinct from reproducing the historical BF16 full-vocabulary pipeline. No causal claim about whole-response Self-BLEU is inferred from these descriptive means.')
 heading('10. Paper-ready interpretation and suggested wording')
 para('Suggested results wording: "Under completion-only detection and fixed keys, PRC maintained low between-response lexical overlap relative to the evaluated TextSeal and Gumbel-Max settings. This did not establish a diversity advantage over shallow SynthID at temperature 1, and PRC detection was lower, especially at shorter prefixes. Matching repeated-context fallback substantially reduced the apparent within-response repetition gaps. In the precision-preserving temperature-0.7 sensitivity, PRC had lower Self-BLEU than SynthID depth 2 but suffered a large loss in detection. These exploratory results characterize specific configurations and nominal thresholds, rather than a matched-FPR frontier or general superiority."')
 para('For a main paper, use the corrected 500-prompt detection table together with its policy caveat, the matched-policy paired comparison, and the direct depth-2 contrast figure. Use the repeat-policy intervention, depth/short-prefix results, model/decoder/temperature sensitivities, and numerical diagnostics as transparent supporting material. Do not omit the temperature detection loss or the shallow SynthID comparator. Every asset includes its cohort and detector assumptions.')
 heading('11. Completed, cancelled, and outside the evidence')
 for text in [
  'Completed: corrected native 500-prompt comparison; fixed-key/fresh-response validation; Stage A pilot; three repeat-policy interventions; paired repetition/Self-BLEU synthesis; SynthID depths 2/30; short-prefix detection; matched top-100; 0.6B full-vocabulary; final 8B T=.7 sensitivity.',
  'Validation only: TextSeal alpha=0 pairs, alpha=.5 short checks, and SynthID depth=20 short checks. These are not a completed TextSeal/SynthID parameter frontier.',
  'Cancelled or not run: full depth-20 generation, broad alpha/depth/eta sweeps, more temperatures, more prompts, more than two response slots per prompt, Bayesian SynthID, attacks, multi-key replication, a held-out confirmation cohort, and large-scale empirical null calibration.',
  'Historical proxy/native results were produced before the detection correction and remain archived as historical evidence. Prompt-conditioned PRC/TextSeal score columns and old proxy figures are superseded for the current completion-only paper comparison. No corrected common-method proxy panel was completed in this comparison branch; separate PRC redetection results do not by themselves repair the whole proxy panel.',
  'No claims about authors\' intent, semantic quality, broad superiority, equal FPR, or detector equivalence are supported by these experiments. Repeated use of the same cohort and post-pilot choices make the campaign exploratory even where an individual follow-up endpoint was fixed in advance.'
 ]:bullet(text)
 heading('12. Cost, validation and reproducibility')
 costs=[['Implementation validation and setup',4.05505],['Stage A scoring',5.51880],['SynthID OFF',6.24467],['TextSeal/Gumbel ON',6.70584],['Depths 2/30 + short-prefix analysis',7.71132],['Matched top-100',9.77418],['0.6B full vocabulary',11.00279339948393],['Final 8B T=.7',12.51222201949765]]
 table('15_cost','Self-BLEU campaign cumulative planning charges',['Completed stage','Cumulative USD'],[[a,f'{b:.5f}'] for a,b in costs],'Cumulative entries must not be summed. Charges include recorded resource-time estimates and conservative overhead/failure allowances, not settled Modal billing. This ledger does not claim to audit spending by other sessions or the earlier baseline/proxy campaigns. This consolidation launches no Modal workers.')
 para('The final temperature worker used $1.00943 of measured resource time plus a $0.50 allowance; its setup was pushed before full dispatch. Validation verified native temperature placement, ordinary-equivalent SynthID fallback, fixed PRC codewords, original arithmetic paths, completion-only replay inputs, first-coordinate abstention and exact smoke-prefix reproduction. The larger campaign additionally checks upstream TextSeal parity, per-depth SynthID keys, common top-100 support, paired no-divergence-before-repeat trajectories, and source/artifact hashes. Individual run reports retain all repairs and exceptions. The report builder checks repeated values across studies for exact numerical consistency and verifies every exported contrast mean against its source arms.')
 para('Raw completions and model traces remain at their existing local ignored paths and the existing prc-completion-only Modal volume. They are not deleted, renamed or re-uploaded during cleanup. Versioned per-study manifests, summaries and archive indices remain authoritative. Source hashes for this report are listed in data/source_manifest.json; normalized rows and all contrasts are in data/results.json, data/contrasts.json and CSV exports. The one offline builder only reads saved aggregates and creates tables/figures; it cannot dispatch experiments.')
 heading('Source register')
 for name,path in FILES.items():para(f'{name}: ../../../{path}')
 for path in ['outputs/self_bleu_validation/step3-v4/REPORT.md','outputs/self_bleu_pilot/stage_a_v2/REPORT.md','outputs/self_bleu_repeat/setup_v4/REPORT.md','outputs/self_bleu_repeat/setup_v4/FOLLOWUP_REPORT.md','outputs/self_bleu_temperature/t07_v1/REPORT.md','baseline_comparison/README.md','proxy_8b_detector_report.md']:
  SOURCES[path]=sha(ROOT/path);para('../../../'+path)

def exports(rows,contrasts,nulls,checks):
 # Merge the original-T=1 repetition contrasts already retained in the later
 # temperature analysis. This adds metrics, never substitutes a different cohort.
 temp=read('temperature')
 for r in temp['contrasts']:
  if r['temperature']!=1:continue
  found=[c for c in contrasts if c['regime']=='8b_full' and c['left']==r['left'] and c['right']==r['right'] and c['length']==r['length']]
  if found:
   for k,v in r['metrics'].items():
    if k in found[0]['metrics']:assert abs(found[0]['metrics'][k]['mean']-v['mean'])<1e-12
    else:found[0]['metrics'][k]=v
 lookup={(r['regime'],r['setting'],r['length']):r for r in rows}
 verified=0
 for c in contrasts:
  a,b=lookup[c['regime'],c['left'],c['length']],lookup[c['regime'],c['right'],c['length']]
  for m,v in c['metrics'].items():
   if m in a['metrics'] and m in b['metrics']:
    assert abs(v['mean']-(a['metrics'][m]['mean']-b['metrics'][m]['mean']))<1e-12,(c,m)
    verified+=1
 for name,items in [('results',rows),('contrasts',contrasts),('null_counts',nulls)]:save(OUT/'data'/f'{name}.json',items)
 flattened=[]
 for r in rows:
  for metric,value in r['metrics'].items():
   flattened.append({k:r[k] for k in ['regime','setting','length','prompts','responses','detected']}|dict(metric=metric,mean=value['mean'],ci95_low=value['ci95'][0],ci95_high=value['ci95'][1],source=';'.join(r['sources'])))
 with (OUT/'data/absolute_results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flattened[0]),lineterminator="\n");w.writeheader();w.writerows(flattened)
 flat=[]
 for r in contrasts:
  for m,v in r['metrics'].items():flat.append({k:r[k] for k in ['regime','left','right','length','source']}|dict(metric=m,mean=v['mean'],ci95_low=v['ci95'][0],ci95_high=v['ci95'][1]))
 with (OUT/'data/paired_contrasts.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");w.writeheader();w.writerows(flat)
 return dict(duplicate_metric_checks=len(checks),contrast_mean_checks=verified,absolute_rows=len(rows),contrast_rows=len(contrasts),null_rows=len(nulls),evaluation_response_slots=2500,cohorts_not_pooled=True,new_generation=0,new_inference=0)

def write_documents():
 # Keep the edited paper summary separate from the exhaustive generated assets.
 save(OUT/'data/report_blocks.json',BLOCKS)
 lines=[r'\documentclass[10pt]{article}',r'\usepackage[margin=0.75in]{geometry}',r'\usepackage{graphicx,booktabs,array}',r'\usepackage[T1]{fontenc}',r'\usepackage[hidelinks]{hyperref}',r'\title{Watermark comparisons: paper assets}',r'\author{}',r'\date{19 September 2026}',r'\begin{document}',r'\maketitle',r'\noindent All results are completion-only at nominal $p<.001$. See REPORT.md for complete protocols, caveats, numerical paths, provenance, and unfavorable results. No matched empirical FPR claim is made. The paired studies use 50 prompts with two responses each; the historical comparison uses 500 responses per setting.']
 for r in FIGURES:
  lines += [r'\begin{figure}[p]',r'\centering',r'\includegraphics[width=\textwidth]{figures/'+r['name']+'.pdf}',r'\caption{'+tex(r['caption'])+'}',r'\label{fig:'+r['name']+'}',r'\end{figure}']
 lines+=[r'\clearpage']
 for t in TABLES:lines += [r'\input{tables/'+t['name']+'.tex}']
 lines += [r'\end{document}'];(OUT/'paper_assets.tex').write_text('\n'.join(lines)+'\n')
 catalog=['# Paper assets and reproduction','', 'The campaign is closed. This package is generated offline from saved results.','', '- [Concise comparison report](REPORT.md): principal findings, essential tables and methodological limitations.','- [Detailed results PDF](comparison_report.pdf): the complete September 19 results and diagnostics, retained as supporting material.','- [LaTeX assembly](paper_assets.tex): figure/table fragments can also be included independently.','- `figures/`: vector PDF/SVG and 300-dpi PNG exports.','- `tables/`: matching LaTeX fragments and human-readable CSVs.','- `data/`: full-precision normalized values, all paired contrasts, source hashes and checks.','','REPORT.md is maintained separately. The asset builder preserves it and regenerates only the detailed supporting material.','','## Figure catalogue','']
 for i,r in enumerate(FIGURES,1):catalog += [f"{i}. **{r['name']}**: {r['caption']}",'']
 catalog += ['## Table catalogue','']
 for t in TABLES:catalog += [f"- [{t['name']}](tables/{t['name']}.tex): {t['caption']}"]
 catalog += ['', 'Use figures 01-03 and the 500-prompt/matched-policy tables for the central comparison; the remaining figures document interventions and sensitivity. Keep the cohort/mask/precision captions when moving assets into a manuscript. Figure numbering in this package is an asset identifier, not a proposed final manuscript numbering.','','## Offline build','', '```sh','MPLCONFIGDIR=/tmp/prc-comparison-matplotlib python reports/comparisons/build.py','python reports/comparisons/build.py --pdf','```','','First command: Python, NumPy and Matplotlib. Second: ReportLab. It reads only saved aggregate data; there are no Modal imports or dispatch calls. Rebuild from the repository root. Run the two commands in environments providing those dependencies. The numerical tables are not rounded until formatting; CSV data exports retain full precision.','','Every cited source is hashed in `data/source_manifest.json`. Repeated source cells and contrast means are checked. PDFs are rendered and visually inspected before delivery. Source reports remain untouched; historical prompted/proxy results are excluded from paper-ready numerical panels.','','The export archive contains this entire report package, without raw completions, keys or model traces. It is built after visual verification.']
 (OUT/'README.md').write_text('\n'.join(catalog)+'\n')

def render_pdf():
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak
 from reportlab.lib import colors
 from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
 from reportlab.lib.enums import TA_LEFT
 from reportlab.lib.utils import ImageReader
 from xml.sax.saxutils import escape
 blocks=json.loads((OUT/'data/report_blocks.json').read_text())
 styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyReport',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=7))
 styles.add(ParagraphStyle(name='NoteReport',fontName='Helvetica',fontSize=7.4,leading=10,textColor=colors.HexColor('#45566A'),spaceAfter=7))
 styles.add(ParagraphStyle(name='CellReport',fontName='Helvetica',fontSize=7,leading=9,wordWrap='LTR'))
 styles.add(ParagraphStyle(name='HeadCellReport',fontName='Helvetica-Bold',fontSize=7,leading=9,textColor=colors.white))
 styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=23;styles['Title'].leading=27;styles['Title'].alignment=TA_LEFT
 styles['Heading2'].fontName='Helvetica-Bold';styles['Heading2'].fontSize=13;styles['Heading2'].leading=17;styles['Heading2'].textColor=colors.HexColor('#163C56')
 def p(text,style='BodyReport'):return Paragraph(escape(str(text)),styles[style])
 class ReportDoc(SimpleDocTemplate):
  def afterFlowable(self,flowable):
   if isinstance(flowable,Paragraph) and flowable.style.name=='Heading2':
    title=flowable.getPlainText();key=hashlib.md5(title.encode()).hexdigest();self.canv.bookmarkPage(key);self.canv.addOutlineEntry(title,key,level=0,closed=False)
 def footer(canvas,doc):
  canvas.setStrokeColor(colors.HexColor('#DBE3E8'));canvas.line(36,32,576,32);canvas.setFont('Helvetica',7);canvas.setFillColor(colors.HexColor('#45566A'))
  canvas.drawString(36,21,'PRC watermark comparisons | Closed 19 September 2026 | Saved results only');canvas.drawRightString(576,21,str(doc.page))
 story=[];skip=False
 for idx,b in enumerate(blocks):
  if skip:skip=False;continue
  typ=b['type']
  if typ=='heading':
   if b['level']==1:story += [p(b['text'],'Title'),Spacer(1,8)]
   else:
    head=p(b['text'],'Heading2')
    if idx+1<len(blocks) and blocks[idx+1]['type']=='paragraph':
     story.append(KeepTogether([head,p(blocks[idx+1]['text'])]));skip=True
    else:story.append(head)
  elif typ=='paragraph':story.append(p(b['text']))
  elif typ=='bullet':story.append(Paragraph('&#8226; '+escape(b['text']),styles['BodyReport']))
  elif typ=='table':
   n=len(b['headers']);weights=[1.0]*n
   if n>=5:weights[0]=1.25
   if 'Self-BLEU [95% CI]' in b['headers']:weights[b['headers'].index('Self-BLEU [95% CI]')]=1.65
   if n==4 and len(b['rows'])>0 and 'Status' in b['headers']:weights=[1.0,.85,1.3,1.2]
   widths=[540*w/sum(weights) for w in weights]
   cells=[[p(x,'HeadCellReport') for x in b['headers']]]+[[p(x,'CellReport') for x in row] for row in b['rows']]
   t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
   t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#163C56')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F4F7')]),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#B9C9D2'))]))
   cap=p(b['caption'],'Heading3');story.append(KeepTogether([cap,t,Spacer(1,5),p(b['note'],'NoteReport'),Spacer(1,5)]))
  elif typ=='figure':
   path=OUT/'figures'/f"{b['name']}.png";iw,ih=ImageReader(str(path)).getSize();im=Image(str(path),width=540,height=540*ih/iw)
   story += [KeepTogether([im,p(b['caption'],'NoteReport'),Spacer(1,8)])]
 doc=ReportDoc(str(OUT/'comparison_report.pdf'),pagesize=(612,792),rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=44,title='Watermark comparisons: final consolidated report',author='PRC watermark comparison study')
 doc.build(story,onFirstPage=footer,onLaterPages=footer)
 print('Created',OUT/'comparison_report.pdf')


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pdf',action='store_true');args=p.parse_args()
 if args.pdf:render_pdf();return
 for folder in ['data','figures','tables']:(OUT/folder).mkdir(parents=True,exist_ok=True)
 data={k:read(k) for k in FILES};save(OUT/'data/historical_detection.json',data['large']);save(OUT/'data/historical_repetition.json',data['large_repetition']);rows,contrasts,nulls,checks=normalize(data)
 verification=exports(rows,contrasts,nulls,checks)
 figures(data,rows,contrasts)
 report(data,rows,contrasts,nulls)
 heading('Appendix: all saved direct paired contrasts')
 para('This appendix retains ordinary-control, depth-to-depth and PRC-to-baseline contrasts, including unfavorable and inconclusive results. A dash denotes an unavailable or inapplicable metric; no metric is imputed. Percentage-point units apply only to detection/repetition/distinct-3. All differences are left minus right. The complete full-precision records accompany the formatted tables.')
 for reg in REGIMES:
  for n in [1024,400]:
   rr=[r for r in contrasts if r['regime']==reg and r['length']==n]
   def d(r,m,scale,digits):return delta(r['metrics'][m],scale,digits) if m in r['metrics'] else '--'
   table(f'16_all_contrasts_{reg}_{n}',f'All contrasts: {REGIMES[reg]}, {n:,} tokens',['Left minus right','Self-BLEU [95% CI]','Detection pp [95% CI]','Repeat-4 pp [95% CI]','Distinct-3 pp [95% CI]'],[[NAMES[r['left']]+' - '+NAMES[r['right']],d(r,'self_bleu',1,5),d(r,'tpr',100,1),d(r,'repeated_4gram_fraction',100,2),d(r,'distinct_3',100,2)] for r in rr],NOTE)
 for b in BLOCKS:
  if 'text' in b:b['text']=b['text'].replace('../../../','../../')
 write_documents()
 # All aggregate inputs and copied historical references are content-hashed.
 save(OUT/'data/source_manifest.json',dict(source_files=SOURCES,builder_sha256=sha(__file__),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),verification=verification,figures=FIGURES,tables=[dict(name=t['name'],caption=t['caption']) for t in TABLES]))
 print(json.dumps(verification,indent=2));print('Figures',len(FIGURES),'Tables',len(TABLES))

if __name__=='__main__':main()
