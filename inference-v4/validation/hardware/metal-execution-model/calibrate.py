"""Identify shared resource parameters using only predeclared fixed probes."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,time
import numpy as np
from scipy.optimize import least_squares
from corpus import corpus
from machine import Machine,NAMES,DEFAULT
from helpers import FrozenHelpers

ROOT=Path(__file__).parent/'results'
def observations(path):
    raw=[json.loads(s) for s in Path(path).read_text().splitlines()];samples=defaultdict(list)
    failures=[r for r in raw if r['kind']=='failure' or r['kind']=='pilot' and (r['check']['failures'] or not r['check']['checked'])]
    if failures:raise RuntimeError(f'Acquisition failures: {failures}')
    for r in raw:
        if r['kind']=='sample':samples[r['id']].append(r['gpu_s']*1e9)
    if any(len(v)!=7 for v in samples.values()):raise RuntimeError('Incomplete randomized blocks')
    return {k:float(np.median(v)) for k,v in samples.items()},samples,raw

def fit():
    obs,samples,_=observations(ROOT/'cal.jsonl');cases=[c for c in corpus() if c['split']=='cal'];m=Machine(FrozenHelpers(ROOT/'helpers.json'));p=np.array(DEFAULT,float);log=[]
    p[NAMES.index('int_issue')]=.5
    p[NAMES.index('l2_size')]=16777216
    p[0]=obs['cal-0103']/32-70
    def stage(name,chosen,parameters,limit=60):
        indices=[NAMES.index(n) for n in parameters];start=time.monotonic()
        def objective(x):
            q=p.copy();q[indices]=np.exp(x)
            return np.log([m.predict(c,q)['ns']/obs[c['id']] for c in chosen])
        result=least_squares(objective,np.log(p[indices]),diff_step=.015,max_nfev=limit,bounds=(np.log(np.maximum(p[indices]/20,.01)),np.log(p[indices]*20)),ftol=.002,xtol=.002,gtol=.002)
        p[indices]=np.exp(result.x)
        row=dict(stage=name,parameters={NAMES[i]:float(p[i]) for i in indices},cases=[c['id'] for c in chosen],median_error=float(np.median(abs(np.exp(result.fun)-1))),seconds=time.monotonic()-start,jacobian_rank=int(np.linalg.matrix_rank(result.jac)),parameter_count=len(indices),locally_unidentified=[NAMES[indices[i]] for i in range(len(indices)) if np.linalg.norm(result.jac[:,i])<1e-7])
        log.append(row);print(json.dumps(row),flush=True)
    fp_cases=[c for c in cases if c['constructor']=='chains' and c['parameters'].get('operation')=='fma' or c['constructor']=='blocked' or c['constructor']=='mixed' and c['parameters']['u']==0]
    stage('FP scheduling',fp_cases,['warp_issue','front_issue','fp_issue','fp_latency','loop_latency'])
    for op,names in [('alu',['int_issue','int_latency']),('mul',['mul_issue','mul_latency']),('wide',['wide_issue','wide_latency']),('sfu',['sfu_issue','sfu_latency'])]:
        stage(op,[c for c in cases if c['constructor']=='chains' and c['parameters'].get('operation')==op],names)
    stage('constant multiply',[c for c in cases if c['constructor']=='chains' and c['parameters'].get('operation')=='cmul'],['cmul_issue','cmul_latency'])
    stage('constant multiply-add',[c for c in cases if c['constructor']=='mixed' and c['parameters']['f']==0],['cmad_issue','cmad_latency'])
    chase=[c for c in cases if c['constructor']=='chase'];sector_trials=[]
    for sector in [16,32,64,128]:
        p[NAMES.index('random_sector_bytes')]=sector
        stage('dependent memory sector '+str(sector),chase,['l1_latency','l2_latency','dram_latency'])
        loss=float(np.mean(np.log([m.predict(c,p)['ns']/obs[c['id']] for c in chase])**2))
        sector_trials.append((loss,sector,p.copy()))
    _,_,p=min(sector_trials,key=lambda t:t[0])
    log.append(dict(stage='memory transaction hypotheses',choices=[dict(sector=s,loss=e) for e,s,_ in sector_trials],chosen=p[NAMES.index('random_sector_bytes')]))
    stage('memory service',[c for c in cases if c['constructor']=='stream'],['l1_bw','l2_bw','dram_bw','memory_issue'])
    stage('barrier',[c for c in cases if c['constructor']=='sync' and c['op']=='barrier' and c['tg']>32],['barrier_latency','shared_latency'])
    stage('single SIMD fence',[c for c in cases if c['constructor']=='sync' and c['op']=='barrier' and c['tg']==32],['fence_latency'])
    stage('SIMD exchange',[c for c in cases if c['constructor']=='sync'],['shuffle_issue','shuffle_latency'])
    stage('matrix',[c for c in cases if c['constructor']=='matrix'],['mma_issue','mma_latency','shared_issue'])
    stage('shared-memory and cooperative execution',[c for c in cases if c['constructor']=='matrix' or c['constructor']=='sync' and c['op']=='barrier'],['shared_issue','shared_latency','fence_latency','barrier_latency','mma_issue','mma_latency'],limit=40)
    stage('data-dependent control and calls',[c for c in cases if c['constructor']=='strict'],['branch_latency','call_latency','convert_latency','convert_issue'])
    # Register allocation threshold is a discrete property of this fixed
    # compiler policy, selected from a declared finite hypothesis set.
    pressure=[c for c in cases if c['constructor']=='chains' and c['parameters'].get('chains',0)>=64]
    trials=[]
    for regs in [48,56,64,72,80,88,96,104,112,120,124,128]:
        q=p.copy();q[NAMES.index('usable_regs')]=regs
        loss=float(np.mean(np.log([m.predict(c,q)['ns']/obs[c['id']] for c in pressure])**2));trials.append((loss,regs))
    p[NAMES.index('usable_regs')]=min(trials)[1]
    log.append(dict(stage='register ceiling',hypotheses=trials,chosen=int(p[NAMES.index('usable_regs')])))
    names=['address_service','line_service','global_service','latency'];initial=np.array([.75,.125,.05,250])
    atomic={}
    for operation in [None,'failure','success']:
        atomic_cases=[c for c in cases if c['constructor']=='atomic' and not c['parameters']['cas'] and c['parameters'].get('operation')==operation]
        def atom_loss(z):
            a=dict(zip(names,np.exp(z)))
            return np.log([m.predict(c,p,atomic=a)['ns']/obs[c['id']] for c in atomic_cases])
        atom_fit=least_squares(atom_loss,np.log(initial),diff_step=.02,max_nfev=100)
        fitted=dict(zip(names,map(float,np.exp(atom_fit.x))))
        if operation:atomic[operation]=fitted
        else:atomic.update(fitted)
    for operation in ['dependent_failure','dependent_success','dependent_load']:
        selected=[c for c in cases if c['constructor']=='atomic' and c['parameters'].get('operation')==operation]
        x=np.array([c['it'] for c in selected]);y=np.array([obs[c['id']] for c in selected]);slope,intercept=np.polyfit(x,y,1)
        if slope<=0:raise RuntimeError('Dependent returning atomic recurrence not observable')
        slopes=[float(np.polyfit(x,[samples[c['id']][r] for c in selected],1)[0]) for r in range(7)]
        atomic[operation]=dict(recurrence_ns=float(slope),intercept_ns=float(intercept),cases=[c['id'] for c in selected],max_fit_residual_ns=float(max(abs(y-(intercept+x*slope)))),round_slope_range_ns=[min(slopes),max(slopes)],definition='Returned atomic value determines next address. Slope includes one address mask and uniform loop scaffolding; aggregate completion intercept is not reused as instruction latency.')
    payload=dict(version='event-v3',status='experimental hypothesis, not a CertifiedMetalProfile',parameters=dict(zip(NAMES,map(float,p))),atomic=atomic,hardware=dict(cores=20,partitions=4,max_waves=32,register_file=65536),assumptions=['uniform workgroup placement','AIR operation realization matches fixed compiler policy','32 SIMD lanes','core count from machine inventory; partitions/wave/register capacity are hypotheses','working-set cache tiers; controlled warm GPU state','native atomics identify service; CAS-loop times and attempt counts excluded from parameter fitting','dependent returning-CAS recurrence is a conservative latency proxy with explicit scaffolding'],calibration_sha256=hashlib.sha256((ROOT/'cal.jsonl').read_bytes()).hexdigest(),stages=log)
    (ROOT/'profile-development.json').write_text(json.dumps(payload,indent=2))
    return payload

if __name__=='__main__':fit()
