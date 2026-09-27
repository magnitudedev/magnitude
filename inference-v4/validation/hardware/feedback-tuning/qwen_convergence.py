"""Revalidate checkpoint shortlists, then plot a wall-budget convergence curve."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


def validate(args):
    from qwen_worker import Observer
    rows=[json.loads(line) for line in args.input.read_text().splitlines()]
    observations=[r for r in rows if r.get('kind')=='observation' and not r.get('fresh')]
    result=next(r for r in reversed(rows) if r.get('kind')=='result')
    final=tuple(result['selected'])
    baseline=tuple(observations[0]['point'])
    end=result['elapsed_s']
    checkpoints=[observations[0]['elapsed_s']]+[t for t in [5,10,15,30,45,60,90,120,180,240] if t<end]+[end]
    nominees=[]
    for cutoff in checkpoints:
        prefix=[r for r in observations if r['elapsed_s']<=cutoff]
        shortlist=sorted(prefix,key=lambda r:statistics.mean(r['samples_ms']))[:6]
        nominees.append(dict(elapsed_s=cutoff,observed=len(prefix),points=[r['point'] for r in shortlist]))
    points=sorted({tuple(p) for checkpoint in nominees for p in checkpoint['points']}|{baseline,final})
    observer=Observer('decode')
    rng=random.Random(61423)
    samples={p:[] for p in points}
    native=[]
    with args.output.with_suffix('.jsonl').open('w') as raw:
        for p in points:
            r=observer.observe(list(p),trials=1)
            raw.write(json.dumps(dict(kind='warmup',**r))+'\n')
        r=observer.observe(list(baseline),trials=1,native=True)
        raw.write(json.dumps(dict(kind='native_warmup',**r))+'\n')
        for block in range(10):
            order=list(points)+[None];rng.shuffle(order)
            for p in order:
                r=observer.observe(list(p or baseline),trials=5,native=p is None)
                raw.write(json.dumps(dict(kind='confirmation',block=block,native=p is None,**r))+'\n');raw.flush()
                (native if p is None else samples[p]).append(statistics.mean(r['samples_ms']))
            print(json.dumps(dict(confirmation_block=block,points=len(points))),flush=True)
    means={p:statistics.mean(v) for p,v in samples.items()}
    best=min(points,key=means.get)
    curve=[]
    for checkpoint in nominees:
        selected=min(map(tuple,checkpoint['points']),key=means.get)
        values=samples[selected]
        # Descriptive block variability; selection among finalists is not adjusted.
        error=1.96*statistics.stdev(values)/math.sqrt(len(values))
        ratios=[a/b-1 for a,b in zip(values,samples[best])]
        curve.append(dict(**checkpoint,selected=selected,confirmed_ms=means[selected],
                          error_95_ms=error,excess_over_best_percent=100*(means[selected]/means[best]-1),
                          paired_excess_percent=[100*x for x in ratios]))
    running=[];incumbent=math.inf
    for r in observations:
        incumbent=min(incumbent,statistics.mean(r['samples_ms']))
        running.append(dict(elapsed_s=r['elapsed_s'],raw_best_ms=incumbent))
    summary=dict(result=result,method='Retrospective fresh comparison of each checkpoint top-six shortlist; validation outside tuning budget',
                 confirmation_blocks=10,confirmed_points=len(points),curve=curve,raw_curve=running,
                 final_point=final,final_confirmed_ms=means[final],baseline_ms=means[baseline],
                 best_confirmed_point=best,best_confirmed_ms=means[best],native_matched_ms=statistics.mean(native),
                 samples=[dict(point=p,block_means_ms=v) for p,v in samples.items()],native_block_means_ms=native)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['raw_curve','samples']},indent=2),flush=True)


def plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=json.loads(args.input.read_text());curve=d['curve'];raw=d['raw_curve']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,detail)=plt.subplots(2,1,figsize=(12,8),gridspec_kw={'height_ratios':[1.2,1]},sharex=True)
    fig.patch.set_facecolor('#fbfcfe')
    x=[p['elapsed_s'] for p in curve]; y=[p['confirmed_ms'] for p in curve]
    for a in [ax,detail]:
        a.set_facecolor('#fbfcfe');a.grid(axis='y',alpha=.2);a.axvline(60,color='#9ca3af',lw=1,ls=':')
        a.set_xlim(0,300)
    ax.plot([p['elapsed_s'] for p in raw],[p['raw_best_ms'] for p in raw],color='#a1a1aa',ls='--',lw=1.5,label='Raw running minimum (selection noise)')
    ax.errorbar(x,y,yerr=[p['error_95_ms'] for p in curve],color='#176b87',marker='o',ms=4,lw=2,capsize=3,label='Checkpoint shortlist, independently remeasured')
    ax.axhline(d['native_matched_ms'],color='#c2762b',lw=1.5,ls='-.',label='MLX quantized matmul, matched F32 decode')
    ax.set_ylabel('32 FFNs · wall latency (ms)')
    ax.legend(loc='upper right',fontsize=9,frameon=False)
    ax.set_title('Qwen3.5-4B: five-minute Metal tuning run',loc='left',fontsize=19,fontweight='bold',pad=24)
    ax.text(0,1.035,'Actual weights + captured decode activations · all 32 feedforward blocks in layer order',transform=ax.transAxes,fontsize=10,color='#4b5563')
    improvements=[100*(d['baseline_ms']/v-1) for v in y]
    detail.plot(x,improvements,color='#176b87',marker='o',ms=4,lw=2)
    detail.set_ylabel('Speedup over initial policy (%)')
    detail.set_xlabel('Elapsed tuning time (seconds; compilation and checks included)')
    detail.text(62,.02,'1 minute',transform=detail.get_xaxis_transform(),fontsize=9,color='#6b7280')
    end=d['result']['elapsed_s']
    detail.axvline(end,color='#176b87',lw=1,alpha=.4)
    detail.text(end-2,.95,f'Selection: {end:.1f}s',transform=detail.get_xaxis_transform(),ha='right',va='top',fontsize=9,color='#176b87')
    fig.text(.08,.055,f"{d['result']['unique_points']:,} / {d['result']['total_points']:,} policies observed · {d['confirmed_points']} distinct checkpoint nominees remeasured in 10 randomized blocks",fontsize=10,color='#374151')
    fig.text(.08,.029,'Checkpoint verification is offline, outside the tuning budget. Error bars describe block variability, not a global-optimality guarantee.',fontsize=9,color='#6b7280')
    fig.tight_layout(rect=(.02,.085,.99,.99),h_pad=2)
    fig.savefig(args.output,dpi=180,facecolor=fig.get_facecolor())
    fig.savefig(args.output.with_suffix('.svg'),facecolor=fig.get_facecolor())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['validate','plot'],required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.mode](a)
