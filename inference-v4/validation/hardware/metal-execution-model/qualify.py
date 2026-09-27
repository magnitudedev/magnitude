"""Freeze every prediction before qualification acquisition; report all cases."""
from pathlib import Path
from collections import defaultdict
import sys,json,hashlib,time,datetime
import numpy as np
from corpus import corpus
from machine import Machine,NAMES
from helpers import FrozenHelpers
from calibrate import observations

ROOT=Path(__file__).parent/'results'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze():
    destination=ROOT/'frozen-predictions.json'
    if destination.exists():raise RuntimeError('Refusing to overwrite frozen predictions')
    profile=json.loads((ROOT/'profile-development.json').read_text());p=[profile['parameters'][n] for n in NAMES]
    m=Machine(FrozenHelpers(ROOT/'helpers.json'));start=time.monotonic();predictions={}
    for c in corpus():
        t=time.monotonic();r=m.predict(c,p,profile['hardware'],profile['atomic']);r['evaluation_s']=time.monotonic()-t
        if c['split']!='cal':
            alternatives=[]
            # Explicit sensitivity hypotheses, not statistical confidence or
            # certified physical bounds. A failed correspondence cannot be
            # certified by making this envelope wider.
            for waves,register_file,register_limit in [(8,32768,64),(16,32768,96),(32,65536,128)]:
                q=list(p);q[NAMES.index('usable_regs')]=register_limit
                hw=dict(profile['hardware'],max_waves=waves,register_file=register_file)
                alternatives.append(m.predict(c,q,hw,profile['atomic'])['ns'])
            r['sensitivity_ns']=[min([r['ns']]+alternatives),max([r['ns']]+alternatives)]
        if not np.isfinite(r['ns']) or r['ns']<=0:raise RuntimeError(f'Invalid prediction {c["id"]}')
        predictions[c['id']]=r
    payload=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),profile=profile,
                 model_sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}|{'scheduler.cpp':sha(Path(__file__).parent/'scheduler.cpp')},
                 helpers_sha256=sha(ROOT/'helpers.json'),manifest_sha256=sha(ROOT/'sources/manifest.json'),
                 source_sha256=sha(ROOT/'sources/probes.metal'),predictions=predictions,evaluation_s=time.monotonic()-start,
                 qualification_policy='All declared cases included. Report absolute relative time error, percentile distribution, decision regret, and repeatability. No universal bound or residual-based correction. Qualification data never alter predictions.')
    destination.write_text(json.dumps(payload,indent=2));print(json.dumps({'predictions':len(predictions),'seconds':payload['evaluation_s'],'sha256':sha(destination)}))
def metrics(rows):
    e=np.array([r['relative_error'] for r in rows]);ratio=np.array([max(r['predicted_ns']/r['actual_ns'],r['actual_ns']/r['predicted_ns']) for r in rows])
    return dict(cases=len(rows),median_pct=float(np.median(e)*100),p90_pct=float(np.quantile(e,.9)*100),p95_pct=float(np.quantile(e,.95)*100),max_pct=float(max(e)*100),worst_factor=float(max(ratio)),within_10_pct=float(np.mean(e<=.1)*100),within_20_pct=float(np.mean(e<=.2)*100),within_50_pct=float(np.mean(e<=.5)*100))
def assess():
    frozen=json.loads((ROOT/'frozen-predictions.json').read_text());obs={};samples={};raw=[]
    for path in sys.argv[2:]:
        o,s,r=observations(ROOT/path);obs.update(o);samples.update(s);raw.extend(r)
    cases=[c for c in corpus() if c['split']!='cal'];missing=[c['id'] for c in cases if c['id'] not in obs]
    if missing:raise RuntimeError(f'Missing qualification cases: {missing}')
    rows=[];groups=defaultdict(list)
    for c in cases:
        actual=obs[c['id']];pred=frozen['predictions'][c['id']]['ns'];s=samples[c['id']]
        bounds=frozen['predictions'][c['id']]['sensitivity_ns']
        row=dict(case=c,actual_ns=actual,predicted_ns=pred,relative_error=abs(pred/actual-1),sample_range_relative=(max(s)-min(s))/actual,sensitivity_ns=bounds,sensitivity_contains=bounds[0]<=actual<=bounds[1],sensitivity_width_factor=bounds[1]/bounds[0])
        rows.append(row)
        if c.get('equivalence'):groups[c['equivalence']].append(row)
    ranking=[]
    for group,rs in groups.items():
        chosen=min(rs,key=lambda r:r['predicted_ns']);best=min(rs,key=lambda r:r['actual_ns']);regret=chosen['actual_ns']/best['actual_ns']-1
        ranking.append(dict(group=group,candidates=len(rs),chosen=chosen['case']['id'],best=best['case']['id'],regret_pct=regret*100))
    families=sorted(set(c['constructor'] for c in cases))
    report=dict(all=metrics(rows),by_constructor={f:metrics([r for r in rows if r['case']['constructor']==f]) for f in families},
                by_partition={s:metrics([r for r in rows if r['case']['split']==s]) for s in ['interval','test']},
                ranking=dict(groups=len(ranking),median_regret_pct=float(np.median([r['regret_pct'] for r in ranking])),max_regret_pct=max(r['regret_pct'] for r in ranking),within_5_pct=sum(r['regret_pct']<=5 for r in ranking)/len(ranking)*100,rows=ranking),
                noise=dict(median_range_pct=float(np.median([r['sample_range_relative'] for r in rows])*100),p90_range_pct=float(np.quantile([r['sample_range_relative'] for r in rows],.9)*100),max_range_pct=max(r['sample_range_relative'] for r in rows)*100),
                sensitivity=dict(coverage_pct=float(np.mean([r['sensitivity_contains'] for r in rows])*100),median_width_factor=float(np.median([r['sensitivity_width_factor'] for r in rows])),max_width_factor=max(r['sensitivity_width_factor'] for r in rows)),
                acquisition=[r for r in raw if r['kind'] in ['suite','library','environment']],frozen_sha256=sha(ROOT/'frozen-predictions.json'),rows=rows)
    (ROOT/'assessment.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['rows','acquisition','ranking']},indent=2));print(json.dumps({k:v for k,v in report['ranking'].items() if k!='rows'}))
if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else assess()
