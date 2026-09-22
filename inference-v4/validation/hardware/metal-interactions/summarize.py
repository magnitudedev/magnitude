"""Rebuild study statistics and a static figure from retained observations."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent/'results'
original=json.loads((R/'sources/manifest.json').read_text())
followup=json.loads((R/'followup/manifest.json').read_text())
def load(name,cases):
    samples=defaultdict(list);host=defaultdict(list);rows=[json.loads(x) for x in (R/name).read_text().splitlines()]
    for r in rows:
        if r['kind']=='sample':samples[r['id']].append(r['gpu_s']*1e6);host[r['id']].append(r['host_s']*1e6)
    return {c['id']:{'case':c,'median':float(np.median(samples[c['id']])),'min':min(samples[c['id']]),'max':max(samples[c['id']]),'host_median':float(np.median(host[c['id']]))} for c in cases if c['id'] in samples}

first={}
for name in ['cal-warm-first.jsonl','test-warm.jsonl','diagnostic-warm.jsonl']:first.update(load(name,original))
repeat=load('all-warm-repeat.jsonl',original)
interventions=load('followup.jsonl',followup)
cold=load('cal-first.jsonl',original)
warm=load('cal-warm-first.jsonl',original)
def get(data,**filters):return next(r for r in data.values() if all(r['case'].get(k)==v for k,v in filters.items()))
def quant(v):return dict(zip(['median','p90','maximum'],map(float,np.quantile(v,[.5,.9,1]))))
summary={'cold_calibration_range_over_median':quant([(r['max']-r['min'])/r['median'] for r in cold.values()]),'warm_calibration_range_over_median':quant([(r['max']-r['min'])/r['median'] for r in warm.values()]),'repeat_relative_change':quant([abs(repeat[k]['median']/r['median']-1) for k,r in first.items()]),'followups':interventions}
slow=get(interventions,kernel='strict_fma',mode=7);fast=get(interventions,kernel='strict_fma',mode=8)
summary['same_multiset_fma']={'mixed_us':slow['median'],'coherent_us':fast['median'],'ratio':slow['median']/fast['median'],'minimum_worst_relative_error_single_prediction':(slow['median']-fast['median'])/(slow['median']+fast['median']),'using_nearest_sample_endpoints':(slow['min']-fast['max'])/(slow['min']+fast['max'])}
(R/'summary.json').write_text(json.dumps(summary,indent=2))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
a=json.loads((R/'assessment-test-warm.json').read_text())['models']['local_issue']['errors']
ax=axs[0,0]
for c,label,color in [(False,'1–64 live chains','#267f9b'),(True,'128 live chains','#d96440')]:
    pts=[r for r in a if ('c128' in r['id'])==c]
    ax.scatter([p['actual_us'] for p in pts],[p['predicted_us'] for p in pts],label=label,c=color,s=25)
ax.plot([1,10000],[1,10000],color='#777',lw=1,ls='--');ax.set(xscale='log',yscale='log',xlabel='Measured duration (µs)',ylabel='Frozen prediction (µs)',title='Held-out compute: local issue model');ax.legend(frameon=False)

ax=axs[0,1];ops=['add','mul','div','fma','cmp'];x=np.arange(len(ops))
for mode,offset,label,color in [(8,-.18,'Coherent SIMD groups','#267f9b'),(7,.18,'Mixed lanes','#d96440')]:
    rows=[get(interventions,kernel='strict_'+op,mode=mode) for op in ops]
    ax.bar(x+offset,[r['median'] for r in rows],.36,color=color,label=label,yerr=np.array([[r['median']-r['min'] for r in rows],[r['max']-r['median'] for r in rows]]),capsize=2)
ax.set(xticks=x,xticklabels=ops,ylabel='Duration (µs)',title='Same strict-helper input multiset, different arrangement');ax.legend(frameon=False)

ax=axs[1,0];names=['fma_c128','blocked128_b4','blocked128_b16','blocked128_b32']
rows=[get(interventions,kernel=n,threads=512,compile_max_threads=None) for n in names]
ax.bar(['128 live','4 live','16 live','32 live'],[r['median'] for r in rows],color=['#d96440']+['#267f9b']*3,yerr=np.array([[r['median']-r['min'] for r in rows],[r['max']-r['median'] for r in rows]]),capsize=3)
ax.set(ylabel='Duration (µs)',title='Same 128-chain work and output, changed lifetimes')

ax=axs[1,1];ns=[1,32,1024,8192]
for op,label,color in [('integer','Native integer add','#267f9b'),('cas','Strict F32 CAS add','#d96440')]:
    rows=[get(first,kernel='atomic_'+op,n=n) for n in ns]
    ax.plot(ns,[r['median'] for r in rows],marker='o',label=label,c=color)
ax.set(xscale='log',yscale='log',xlabel='Distinct destination addresses',ylabel='Duration (µs)',title='Same 32,768 logical atomic updates');ax.legend(frameon=False)
fig.suptitle('Metal interaction study · m4-pro-02 · warmed device\nObserved results; no universal error bound established',fontsize=14)
fig.savefig(R/'study.png',dpi=180)
print(json.dumps({k:v for k,v in summary.items() if k!='followups'},indent=2))
