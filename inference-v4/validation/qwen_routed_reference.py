#!/usr/bin/env python3
"""Routed Qwen suffix and routing edge cases from V3 primitive references."""
import argparse, hashlib, json, sys
from pathlib import Path
from reference_source import activate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();source=a.source.resolve(strict=True);activate(source)
import numpy as np
from ops.tensor import ops
from ops.tensor.primitive import round_reference
from ops.tensor.types import DType
rng=np.random.default_rng(917352)
q=lambda x:round_reference(x,DType.BF16).astype(np.float32)
m,h,e,k,f,s=3,8,7,3,12,16
weights={}
def w(name,shape,scale=.35,offset=0,compact=True):
 x=(rng.standard_normal(shape)*scale+offset).astype(np.float32)
 weights[name]=q(x) if compact else x
 return weights[name]
w('norm',(h,),.1,1);w('router',(e,h));w('shared_router',(h,),compact=False)
w('expert_gate',(e,f,h));w('expert_up',(e,f,h));w('expert_down',(e,h,f))
w('shared_gate',(s,h));w('shared_up',(s,h));w('shared_down',(h,s))
residual=(rng.standard_normal((m,h))*1.5).astype(np.float32)
linear=lambda x,y:q(ops._linear_reference((x,y),{}))
x=q(ops._rms_reference((residual,weights['norm']),{'epsilon':1e-6}))
logits=ops._linear_reference((x,weights['router']),{})
cases=[]
for normalize in [False,True]:
 routes,scores=ops._route_reference(logits,{'scoring':'softmax','normalize':normalize,'k':k})
 selected=ops._experts_reference((x,routes,scores,weights['expert_gate'],weights['expert_up'],weights['expert_down']),{'storage_dtype':DType.BF16,'activation':'silu'})
 gate=linear(x,weights['shared_gate']);up=linear(x,weights['shared_up'])
 shared=linear(q(q(gate/(1+np.exp(-gate)))*up),weights['shared_down'])
 coefficient=q(1/(1+np.exp(-np.sum(x*weights['shared_router'],axis=-1,keepdims=True))))
 out=residual+(selected+q(shared*coefficient))
 cases.append({'normalize':normalize,'routes':routes.reshape(-1).tolist(),'scores':scores.reshape(-1).tolist(),'out':out.reshape(-1).tolist()})
routing=[]
for logits in [np.zeros((2,e),np.float32),np.array([[100,100,99,100,-100,100,100],[-1000,-1001,-999,-999,-999,-1000,-999]],np.float32)]:
 for count in [1,3,e]:
  for normalize in [False,True]:
   routes,scores=ops._route_reference(logits,{'scoring':'softmax','normalize':normalize,'k':count})
   routing.append({'logits':logits.reshape(-1).tolist(),'M':2,'E':e,'K':count,'normalize':normalize,'routes':routes.reshape(-1).tolist(),'scores':scores.reshape(-1).tolist()})
record={'reference':'V3 _route_reference and _experts_reference plus routed_feedforward equations; BF16 publications', 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__, 'source_sha256':{str(p):hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['src/ops/tensor/ops.py','src/ops/tensor/primitive.py','src/engine/models/qwen35/equations.py']}, 'shape':{'M':m,'H':h,'E':e,'K':k,'F':f,'S':s},'weights':{name:{'shape':list(v.shape),'values':v.reshape(-1).tolist()} for name,v in weights.items()},'residual':residual.reshape(-1).tolist(),'cases':cases,'routing':routing}
a.output.write_text(json.dumps(record,indent=2)+'\n')
