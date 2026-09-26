"""Freeze on calibration ONLY, then assess unchanged predictions on separate runs."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
cases=json.loads((R/'sources/manifest.json').read_text())
byid={c['id']:c for c in cases}
def read(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    samples=defaultdict(list)
    for row in rows:
        if row['kind']=='sample':samples[row['id']].append(row['gpu_s']*1e6)
    return rows,samples,{k:float(np.median(v)) for k,v in samples.items()}

def compute(c,p,kind):
    capacity=p[1]*c['ops']
    if kind=='additive':return p[0]+capacity
    terms=[capacity,p[2]*c['it']*8]
    if kind=='local_issue':terms.append(p[3]*c['it']*8*c['chains'])
    return p[0]+max(terms)

def freeze():
    calpath=R/'cal-warm-first.jsonl'
    rows,samples,med=read(calpath)
    cal=[c for c in cases if c['family']=='compute' and c['split']=='cal']
    models={}
    for kind in ['additive','envelope','local_issue']:
        init=[1,2.5e-7]+([] if kind=='additive' else [0.003])+([0.001] if kind=='local_issue' else [])
        # Positive physically named time parameters, squared log-relative residual objective.
        fit=least_squares(lambda z:np.log([compute(c,np.exp(z),kind)/med[c['id']] for c in cal]),np.log(init),max_nfev=10000)
        models[kind]={'parameters_us':np.exp(fit.x).tolist(),'calibration_ids':[c['id'] for c in cal],'loss':'unweighted squared log-relative time residual','predictions':{c['id']:compute(c,np.exp(fit.x),kind) for c in cases if c['family']=='compute' and c['split']=='test'}}
    alpha,unit=models['local_issue']['parameters_us'][:2]
    for kind in ['mixed_sum','mixed_max','stream_sum','stream_max']:
        preds={}
        for c in cases:
            if c['split']!='test':continue
            if c['family']=='mixed' and kind.startswith('mixed'):
                def pure(f,u):return next(med[q['id']] for q in cases if q['family']=='mixed' and q['threads']==c['threads'] and q['f']==f and q['u']==u)
                f=max(0,pure(16,0)-alpha)*c['f']/16
                u=max(0,pure(0,16)-alpha)*c['u']/16
                preds[c['id']]=alpha+(f+u if kind.endswith('sum') else max(f,u))
            if c['family']=='stream' and kind.startswith('stream'):
                mem=next(med[q['id']] for q in cases if q['family']=='stream' and q['intensity']==0 and q['n']==c['n'] and q['stride']==c['stride'])
                comp=unit*c['n']*c['intensity']
                preds[c['id']]=alpha+(max(0,mem-alpha)+comp if kind.endswith('sum') else max(max(0,mem-alpha),comp))
        models[kind]={'predictions':preds}
    payload={'calibration_sha256':hashlib.sha256(calpath.read_bytes()).hexdigest(),'source_identity':json.loads((R/'sources/identity.json').read_text()),'units':'microseconds','models':models}
    out=R/'frozen-predictions.json'
    if out.exists():raise RuntimeError('Refusing to overwrite frozen predictions')
    out.write_text(json.dumps(payload,indent=2))
    print(json.dumps({k:{'parameters_us':v.get('parameters_us'),'predictions':len(v['predictions'])} for k,v in models.items()},indent=2))

def assess():
    frozen=json.loads((R/'frozen-predictions.json').read_text())
    allrows=[];samples={};med={}
    for filename in sys.argv[2:]:
        rows,s,m=read(R/filename);allrows.extend(rows);samples.update(s);med.update(m)
    result={'bundles':sys.argv[2:],'models':{},'failures':[r for r in allrows if r['kind']=='failure'],'failed_checks':[r for r in allrows if r['kind']=='pilot' and r['check']['failures']],'timings':[r for r in allrows if r['kind'] in ['suite','library']],'cases':{}}
    for name,model in frozen['models'].items():
        errors=[]
        for id,pred in model['predictions'].items():
            if id not in med:continue
            actual=med[id];errors.append({'id':id,'predicted_us':pred,'actual_us':actual,'relative_error':abs(pred/actual-1),'signed_relative_error':pred/actual-1,'absolute_error_us':abs(pred-actual)})
        if errors:
            rel=[x['relative_error'] for x in errors]
            result['models'][name]={'n':len(errors),'median_relative_error':float(np.median(rel)),'p90_relative_error':float(np.quantile(rel,.9)),'max_relative_error':max(rel),'worst':sorted(errors,key=lambda x:x['relative_error'],reverse=True)[:8],'errors':errors}
    for id,values in samples.items():
        result['cases'][id]={'case':byid[id],'median_us':med[id],'min_us':min(values),'max_us':max(values),'range_over_median':(max(values)-min(values))/med[id]}
    output=R/('assessment-'+Path(sys.argv[2]).stem+'.json')
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({'models':{k:{kk:vv for kk,vv in v.items() if kk not in ['errors','worst']} for k,v in result['models'].items()},'failures':result['failures'],'failed_checks':result['failed_checks'],'timings':result['timings']},indent=2))

if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    else:assess()
