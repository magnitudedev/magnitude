"""Budgeted searches plus an offline reference and matched MLX comparison."""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


def campaigns(args):
    root=Path(__file__).resolve().parent
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    wrapper=output/'qwen-observer'
    # No shell interpolation of paths: the executable wrapper launches Python.
    wrapper.write_text('#!'+sys.executable+'\nimport os\nos.execv('+repr(sys.executable)+', '+
                       repr([sys.executable,str(root/'qwen_worker.py')])+ ' + __import__("sys").argv[1:])\n')
    wrapper.chmod(0o755)
    for seed in range(args.seeds):
        for algorithm in (['evolution','random'] if seed%2==0 else ['random','evolution']):
            path=output/f'{args.regime}-{algorithm}-{seed}.jsonl'
            print('START',path.name,flush=True)
            subprocess.run([sys.executable,str(root/'tune.py'),'--worker',str(wrapper.resolve()),
                '--fixture',args.regime,'--algorithm',algorithm,'--seed',str(seed),'--budget','60',
                '--trials','3','--target-ms','2','--output',str(path)],check=True)


def reference(args):
    from qwen_worker import Observer
    observer=Observer(args.regime)
    rng=random.Random(1729)
    # Explicit offline comparison budget, not charged to the online campaigns.
    points=[(fusion,r,g,c,dr,dg,dc) for fusion in range(3) for r in range(4)
            for g in range(4) for c in range(3) for dr in range(4) for dg in range(4) for dc in range(3)]
    rng.shuffle(points)
    with (args.output/f'{args.regime}-offline.jsonl').open('w') as f:
        for i,point in enumerate(points):
            record=observer.observe(list(point),trials=1)
            f.write(json.dumps(record)+'\n');f.flush()
            if i%128==0:print('SCREEN',i,'of',len(points),flush=True)


def confirm(args):
    from qwen_worker import Observer
    rng=random.Random(8457)
    runs=[]
    for p in args.output.glob(f'{args.regime}-*.jsonl'):
        if any(word in p.name for word in ['offline','confirmation']):continue
        for r in map(json.loads,p.read_text().splitlines()):
            if r.get('kind')=='result':runs.append(dict(r,file=p.name))
    offline=[json.loads(line) for line in (args.output/f'{args.regime}-offline.jsonl').read_text().splitlines()]
    ranked=sorted(offline,key=lambda r:statistics.mean(r['samples_ms']))
    points=sorted({tuple(r['selected']) for r in runs}|{tuple(r['point']) for r in ranked[:24]})
    observer=Observer(args.regime)
    samples={point:[] for point in points}
    native=[]
    raw=args.output/f'{args.regime}-confirmation.jsonl'
    with raw.open('w') as f:
        for point in points:
            r=observer.observe(list(point),trials=1)
            f.write(json.dumps(dict(kind='warmup',**r))+'\n')
        r=observer.observe([0]*7,trials=1,native=True)
        f.write(json.dumps(dict(kind='native_warmup',**r))+'\n')
        for block in range(8):
            order=list(points)+[None];rng.shuffle(order)
            for point in order:
                r=observer.observe(list(point) if point is not None else [0]*7,trials=5,native=point is None)
                f.write(json.dumps(dict(kind='comparison',native=point is None,block=block,**r))+'\n');f.flush()
                (native if point is None else samples[point]).append(statistics.mean(r['samples_ms']))
            print('CONFIRM',block,flush=True)
    means={point:statistics.mean(values) for point,values in samples.items()}
    best=min(means,key=means.get)
    summary=dict(regime=args.regime,offline_points=len(offline),domain_size=6912,
        best_confirmed_point=best,best_confirmed_ms=means[best],native_ms=statistics.mean(native),
        native_samples=native,runs=[dict(r,confirmed_ms=means[tuple(r['selected'])],
            regret_percent=100*(means[tuple(r['selected'])]/means[best]-1)) for r in runs])
    (args.output/f'{args.regime}-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['campaigns','reference','confirm'],required=True)
    p.add_argument('--regime',choices=['decode','prefill32'],default='decode')
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--output',type=Path,default=Path('qwen-results'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    globals()[a.mode](a)
