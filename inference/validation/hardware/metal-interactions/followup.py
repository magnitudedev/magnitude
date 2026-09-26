"""Controlled interventions after the initial held-out results; distinct corpus."""
from pathlib import Path
import json,hashlib,re

root=Path(__file__).resolve().parent
old=root/'results/sources';out=root/'results/followup';out.mkdir(exist_ok=True)
source=(old/'probes.metal').read_text();original=json.loads((old/'manifest.json').read_text());cases=[]
def add(c,**kw):
    c=dict(c);c.update(kw);c['split']='diagnostic';c['id']=f"followup_{len(cases):03d}_{c['kernel']}";cases.append(c)
for c in original:
    if c['family']=='strict' and c['mode']==0:
        add(c,mode=7);add(c,mode=8)
    if c['family']=='compute' and c['chains'] in [16,64,128] and c['threads'] in [512,65536] and c['it']==256:
        for cap in [128,256,512,1024]:add(c,compile_max_threads=cap)

signature=re.search(r'kernel void fma_c128\((.*?)\) \{',source,re.S).group(1)
for block in [4,16,32]:
    name=f'blocked128_b{block}'
    body='float sum=0;'
    for start in range(0,128,block):
        body+='{'+''.join(f'float v{k}=x[id+{start+k}*p.n];' for k in range(block))
        body+='for(uint j=0;j<p.it;j++){'+''.join(f'v{k}=fma(v{k},p.a,p.b);' for _ in range(8) for k in range(block))+'}'
        body+=''.join(f'sum+=v{k};' for k in range(block))+'}'
    body+='y[id]=sum;'
    source+=f'\nkernel void {name}({signature}) {{ {body} }}\n'
    for threads in [512,8192,65536]:
        base=next(c for c in original if c['family']=='compute' and c['chains']==128 and c['threads']==threads and c['it']==256)
        add(base,kernel=name,block=block)
for c in original:
    if c['family']=='compute' and c['chains']==128 and c['it']==256 and c['threads'] in [512,8192,65536]:add(c)

(out/'probes.metal').write_text(source)
(out/'manifest.json').write_text(json.dumps(cases,indent=2))
(out/'identity.json').write_text(json.dumps({'protocol':'metal-interventions-v1','cases':len(cases),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'purpose':'identical input-pair multiset reordered across lanes; identical source under compiler threadgroup caps; identical FMA work/output with blockwise value lifetimes'},indent=2))
print(len(cases))
