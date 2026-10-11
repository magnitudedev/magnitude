"""Post-freeze diagnostics and presentation. Never changes model predictions."""
from pathlib import Path
from collections import defaultdict
import json,sys,shutil,hashlib
import numpy as np

ROOT=Path(__file__).parent/'results'
def read(path):
    raw=[json.loads(s) for s in Path(path).read_text().splitlines()];samples=defaultdict(list)
    for r in raw:
        if r['kind']=='sample':samples[r['id']].append(r['gpu_s']*1e9)
    return raw,{k:float(np.median(v)) for k,v in samples.items()}
def prepare_counters():
    target=ROOT/'counters';target.mkdir(exist_ok=True)
    source=(Path(__file__).parent/'runner.swift').read_text()
    old='do {let value=try trial(c,reps);log(["kind":"sample","id":id,"round":round,"reps":reps,"gpu_s":value.0,"host_s":value.1,"thermal_state":ProcessInfo.processInfo.thermalState.rawValue])}'
    new='''do {let value=try trial(c,reps)
            let n=num(c,"n"),threads=num(c,"threads")
            let attempts=(0..<threads).map{UInt64(yu[n+$0])}.reduce(0,+)
            log(["kind":"sample","id":id,"round":round,"reps":reps,"gpu_s":value.0,"host_s":value.1,"attempts":attempts,"thermal_state":ProcessInfo.processInfo.thermalState.rawValue])}'''
    assert old in source
    (target/'runner.swift').write_text(source.replace(old,new))
    shutil.copy(ROOT/'calibration-sources/probes.metal',target/'probes.metal')
    cases=json.loads((ROOT/'sources/manifest.json').read_text());(target/'manifest.json').write_text(json.dumps([c for c in cases if c.get('counted')]))
def category(c):
    if c['constructor']=='atomic':return 'CAS retry loop' if c['parameters']['cas'] else 'Native atomics'
    if c['constructor']=='chains':return 'Large live sets / unroll' if c['chains']>=64 else 'Arithmetic chains'
    return dict(blocked='Fixed compiled blocks',chase='Dependent loads',matrix='Matrix staging',mixed='FP + integer mixtures',noop='Dispatch sequences',stream='Streaming memory',strict='Strict helper paths',sync='Group synchronization')[c['constructor']]
def finish():
    report=json.loads((ROOT/'assessment.json').read_text());frozen=json.loads((ROOT/'frozen-predictions.json').read_text())
    raw,a=read(ROOT/'qualification.jsonl');rr,b=read(ROOT/'qualification-repeat.jsonl')
    assert set(a)==set(b)
    transfer=np.array([abs(b[k]/a[k]-1) for k in a]);groups=defaultdict(list)
    for row in report['rows']:groups[category(row['case'])].append(row)
    out=dict(repeat=dict(cases=len(a),median_pct=float(np.median(transfer)*100),p90_pct=float(np.quantile(transfer,.9)*100),max_pct=float(max(transfer)*100)),
             numerical_checks=dict(checked=sum(r['check']['checked'] for r in raw+rr if r['kind']=='pilot'),failures=sum(r['check']['failures'] for r in raw+rr if r['kind']=='pilot')),
             source_integrity={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()==digest for n,digest in frozen['model_sources'].items()},
             categories={},worst=sorted(report['rows'],key=lambda r:-r['relative_error'])[:12])
    for name,rows in groups.items():
        err=np.array([r['relative_error'] for r in rows])*100
        out['categories'][name]=dict(cases=len(rows),median_pct=float(np.median(err)),p90_pct=float(np.quantile(err,.9)),max_pct=float(max(err)))
    times=[r['evaluation_s'] for r in frozen['predictions'].values()]
    out['evaluation']=dict(cases=len(times),total_s=sum(times),median_ms=float(np.median(times)*1000),p90_ms=float(np.quantile(times,.9)*1000),max_ms=max(times)*1000,freeze_with_sensitivity_s=frozen['evaluation_s'])
    if (ROOT/'counters.jsonl').exists():
        cr,ct=read(ROOT/'counters.jsonl');counts=defaultdict(list)
        for r in cr:
            if r['kind']=='sample':counts[r['id']].append(r['attempts'])
        cases=json.loads((ROOT/'sources/manifest.json').read_text());calraw,cal=read(ROOT/'cal.jsonl');out['counter_evidence']=[]
        for c in cases:
            if not c.get('counted'):continue
            base=next(x for x in cases if x['split']=='cal' and x['constructor']=='atomic' and x['parameters']==dict(cas=True) and x['n']==c['n'] and x['threads']==c['threads'] and x['it']==c['it'])
            v=counts[c['id']];pred=frozen['predictions'][c['id']]
            out['counter_evidence'].append(dict(destinations=c['n'],observed_attempts=float(np.median(v)),attempt_range=[min(v),max(v)],predicted_attempts=pred['attempts'],counted_ns=ct[c['id']],uncounted_ns=cal[base['id']],timing_change_pct=(ct[c['id']]/cal[base['id']]-1)*100))
    (ROOT/'diagnostics.json').write_text(json.dumps(out,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(15,6),gridspec_kw={'width_ratios':[1,1.2]},layout='constrained')
    palette=plt.get_cmap('tab20');names=sorted(groups)
    for idx,name in enumerate(names):
        rows=groups[name];axes[0].scatter([r['actual_ns']/1000 for r in rows],[r['predicted_ns']/1000 for r in rows],s=19,alpha=.7,color=palette(idx),label=name)
    limits=[.5,100000];axes[0].plot(limits,limits,color='#222222',linewidth=1);axes[0].fill_between(limits,np.array(limits)*.8,np.array(limits)*1.2,color='grey',alpha=.12)
    axes[0].set(xscale='log',yscale='log',xlabel='Measured GPU time (µs)',ylabel='Predicted GPU time (µs)',title='398 fresh held-out configurations\nDiagonal = exact; shading = ±20% reference')
    ordered=sorted(names,key=lambda n:out['categories'][n]['median_pct'])
    for y,name in enumerate(ordered):
        s=out['categories'][name];axes[1].plot([s['median_pct'],s['p90_pct']],[y,y],color='#286796',linewidth=4)
        axes[1].scatter([s['median_pct']],[y],color='#123953',s=40,zorder=3)
        axes[1].scatter([s['max_pct']],[y],color='#b34337',marker='x',s=36,zorder=3)
    axes[1].set_yticks(range(len(ordered)),[f'{n} (n={out["categories"][n]["cases"]})' for n in ordered]);axes[1].invert_yaxis()
    axes[1].set(xscale='log',xlabel='Absolute relative prediction error (%)',title='Dot → bar end: median → 90th percentile\nRed ×: worst case, including all outliers')
    for ax in axes:ax.grid(True,alpha=.15)
    fig.suptitle('Metal execution-model prototype · M4 Pro · '+frozen['profile']['version'],fontsize=16)
    fig.savefig(ROOT/'qualification.png',dpi=170);fig.savefig(ROOT/'qualification.svg')
    print(json.dumps({k:v for k,v in out.items() if k not in ['worst','source_integrity']},indent=2))
if __name__=='__main__':prepare_counters() if sys.argv[1]=='prepare-counters' else finish()
