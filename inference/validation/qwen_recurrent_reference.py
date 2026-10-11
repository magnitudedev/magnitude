#!/usr/bin/env python3
"""Generate small independent V3 recurrent-step fixtures; validation only."""
import argparse
import hashlib
import json
from pathlib import Path
from reference_source import activate
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=args.source.resolve(strict=True)
activate(source)
import numpy as np
from ops.tensor import ops
from ops.tensor.primitive import round_reference
from ops.tensor.types import DType

rng=np.random.default_rng(7183)
h,nk,gv,w,c=8,2,2,4,4
nv=nk*gv
channels=(2*nk+nv)*w
q=lambda x:round_reference(x,DType.BF16)
f=lambda shape,scale:q((rng.standard_normal(shape)*scale).astype(np.float32))
weights={
 'input_norm':f((h,),0.15)+np.float32(1),
 'qkv_weight':f((channels,h),0.2), 'gate_weight':f((nv*w,h),0.2),
 'alpha_weight':f((nv,h),0.1), 'beta_weight':f((nv,h),0.1),
 'convolution':(rng.standard_normal((channels,c))*0.2).astype(np.float32),
 'rate':-np.exp(rng.uniform(-2,0,nv).astype(np.float32)),
 'time_bias':rng.uniform(-1,1,nv).astype(np.float32),
 'recurrent_norm':f((w,),0.1)+np.float32(1), 'output_weight':f((h,nv*w),0.15),
}
weights['input_norm']=q(weights['input_norm']);weights['recurrent_norm']=q(weights['recurrent_norm'])
initial_window=f((1,channels,c-1),0.3)
initial_delta=(rng.standard_normal((1,nv,w,w))*0.1).astype(np.float32)
inputs=[(rng.standard_normal((1,h))*0.8).astype(np.float32) for _ in range(3)]
linear=lambda x,weight:q(ops._linear_reference((x,weight),{}))
norm=lambda x,weight:q(ops._rms_reference((x,weight),{'epsilon':1e-6}))
serialize=lambda x:np.asarray(x).reshape(-1).tolist()
cases=[]
for mapping in ['grouped','tiled']:
 window=initial_window.copy();delta=initial_delta.copy();steps=[]
 for hidden in inputs:
  normalized=norm(hidden,weights['input_norm'])
  projected=linear(normalized,weights['qkv_weight'])
  gate=linear(normalized,weights['gate_weight'])
  alpha=linear(normalized,weights['alpha_weight'])
  beta_input=linear(normalized,weights['beta_weight'])
  query,key,value,beta,decay,window=ops._recurrent_prepare_reference(
   (projected,weights['convolution'],window,alpha,beta_input,weights['rate'],weights['time_bias'],np.array([0,1],np.int32)),
   {'key_heads':nk,'value_heads':nv,'width':w,'convolution_width':c,'epsilon':1e-6*w})
  query,key,value,beta=map(q,(query,key,value,beta))
  mixed,delta=ops._gated_delta_reference((query,key,value,decay,beta,delta,np.array([0,1],np.int32)),{'mapping':mapping})
  mixed=q(mixed)
  mixed_norm=norm(mixed,weights['recurrent_norm'])
  activated=q(gate/(1+np.exp(-gate)))
  gated=q(mixed_norm.reshape(1,nv*w)*activated)
  output_projection=linear(gated,weights['output_weight'])
  out=hidden+output_projection
  fields={name:serialize(value) for name,value in locals().copy().items() if name in
   ['hidden','normalized','projected','gate','alpha','beta_input','beta','decay','mixed','mixed_norm','activated','gated','output_projection','out','delta']}
  fields['qkv']=serialize(np.concatenate((query,key,value),axis=1))
  fields['window']=serialize(window[0].T)
  steps.append(fields)
 cases.append({'mapping':mapping,'steps':steps})
record={'reference':'V3 primitive reference functions with explicit BF16 output publication; three sequential rows per mapping',
 'numpy_version':np.__version__,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'source_sha256':{str(p):hashlib.sha256((source/p).read_bytes()).hexdigest() for p in
  ['src/ops/tensor/ops.py','src/ops/tensor/primitive.py','src/engine/models/qwen35/equations.py']},
 'shapes':{'H':h,'NK':nk,'GV':gv,'W':w,'C':c},'weights':{k:serialize(v) for k,v in weights.items()},
 'initial_window':serialize(initial_window[0].T),'initial_delta':serialize(initial_delta),'cases':cases}
args.output.write_text(json.dumps(record,indent=2)+'\n')
