#!/usr/bin/env python3
"""V3 primitive-reference composition for a small four-block dense decoder."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--routed',action='store_true');a=p.parse_args()
source=a.source.resolve(strict=True);sys.path.insert(0,str(source/'src'))
import numpy as np
from ops.tensor import ops
from ops.tensor.primitive import round_reference
from ops.tensor.types import DType
rng=np.random.default_rng(92714);q=lambda x:round_reference(x,DType.BF16)
d,vocab,ff,h,kv,pairs,tail,nk,nv,rw,c=8,32,12,4,2,6,4,2,4,4,4;aw=2*pairs+tail;channels=(2*nk+nv)*rw
weights={};kinds=['recurrent','attention','recurrent','attention']
def weight(name,shape,scale=.2,offset=0,f32=False):
 x=(rng.standard_normal(shape)*scale+offset).astype(np.float32)
 weights[name]=x if f32 else q(x);return weights[name]
weight('embedding',(vocab,d));weight('output_norm',(d,),.1,1)
for i,kind in enumerate(kinds):
 pre=f'b{i}.';weight(pre+'input_norm',(d,),.1,1);weight(pre+'ff_norm',(d,),.1,1)
 if a.routed:
  for name,shape in [('router',(7,d)),('expert_gate',(7,ff,d)),('expert_up',(7,ff,d)),('expert_down',(7,d,ff)),('shared_gate',(16,d)),('shared_up',(16,d)),('shared_down',(d,16))]:weight(pre+name,shape)
  weight(pre+'shared_router',(d,),f32=True)
 else:
  for name,shape in [('ff_gate',(ff,d)),('ff_up',(ff,d)),('ff_down',(d,ff))]:weight(pre+name,shape)
 if kind=='attention':
  for name,shape in [('query_gate',(h*2*aw,d)),('key',(kv*aw,d)),('value',(kv*aw,d)),('output',(d,h*aw))]:weight(pre+name,shape)
  weight(pre+'query_norm',(aw,),.1,1,True);weight(pre+'key_norm',(aw,),.1,1,True)
 else:
  for name,shape in [('qkv',(channels,d)),('gate',(nv*rw,d)),('alpha',(nv,d)),('beta',(nv,d)),('output',(d,nv*rw))]:weight(pre+name,shape)
  weight(pre+'convolution',(channels,c),.2,0,True);weight(pre+'rate',(nv,),.1,-.5,True);weight(pre+'time_bias',(nv,),.1,0,True);weight(pre+'recurrent_norm',(rw,),.1,1)
linear=lambda x,w:q(ops._linear_reference((x,w),{}));norm=lambda x,w:q(ops._rms_reference((x,w),{'epsilon':1e-6}))
states={i:(([],[]) if kind=='attention' else (np.zeros((1,channels,c-1),np.float32),np.zeros((1,nv,rw,rw),np.float32))) for i,kind in enumerate(kinds)}
steps=[]
for position,token in enumerate([1,7,3]):
 hidden=weights['embedding'][token:token+1].copy();boundaries=[]
 for i,kind in enumerate(kinds):
  w=lambda name:weights[f'b{i}.{name}'];x=norm(hidden,w('input_norm'))
  if kind=='attention':
   qg=linear(x,w('query_gate'));key=linear(x,w('key'));value=linear(x,w('value')).reshape(1,kv,aw)
   query,key,gate=map(q,ops._attention_prepare_reference((qg,key,w('query_norm'),w('key_norm'),np.asarray([[position]*4],np.int32)),{'query_heads':h,'kv_heads':kv,'width':aw,'rotary_width':pairs*2,'base':1e6,'sections':(3,2,1,0),'epsilon':1e-6}))
   keys,values=states[i];oldk=np.asarray(keys,np.float32).reshape(position,kv,aw);oldv=np.asarray(values,np.float32).reshape(position,kv,aw)
   history=np.concatenate((oldk,oldv),axis=-1)
   mixed=q(ops._persistent_attention_reference((query,history,key,value,np.asarray([[0,position,0,1]],np.int32)),{'scale':1/np.sqrt(aw)}))
   keys.append(key[0].copy());values.append(value[0].copy())
   activated=q(1/(1+np.exp(-gate.reshape(1,h*aw))));gated=q(mixed.reshape(1,h*aw)*activated)
  else:
   projected=linear(x,w('qkv'));gate=linear(x,w('gate'));alpha=linear(x,w('alpha'));beta=linear(x,w('beta'));window,delta=states[i]
   query,key,value,beta,decay,window=ops._recurrent_prepare_reference((projected,w('convolution'),window,alpha,beta,w('rate'),w('time_bias'),np.asarray([0,1],np.int32)),{'key_heads':nk,'value_heads':nv,'width':rw,'convolution_width':c,'epsilon':1e-6*rw})
   query,key,value,beta=map(q,(query,key,value,beta));mixed,delta=ops._gated_delta_reference((query,key,value,decay,beta,delta,np.asarray([0,1],np.int32)),{'mapping':'grouped'})
   states[i]=(window,delta);mixed_norm=norm(q(mixed),w('recurrent_norm'));activated=q(gate/(1+np.exp(-gate)));gated=q(mixed_norm.reshape(1,nv*rw)*activated)
  hidden=hidden+linear(gated,w('output'))
  x=norm(hidden,w('ff_norm'))
  if a.routed:
   routes,scores=ops._route_reference(ops._linear_reference((x,w('router')),{}),{'scoring':'softmax','normalize':True,'k':3})
   selected=ops._experts_reference((x,routes,scores,w('expert_gate'),w('expert_up'),w('expert_down')),{'storage_dtype':DType.BF16,'activation':'silu'})
   gate=linear(x,w('shared_gate'));up=linear(x,w('shared_up'));activated=q(gate/(1+np.exp(-gate)));product=q(activated*up)
   shared=linear(product,w('shared_down'));coefficient=q(1/(1+np.exp(-np.sum(x*w('shared_router'),axis=-1,keepdims=True))))
   hidden=hidden+(selected+q(shared*coefficient))
  else:
   gate=linear(x,w('ff_gate'));up=linear(x,w('ff_up'));activated=q(gate/(1+np.exp(-gate)));product=q(activated*up)
   hidden=hidden+linear(product,w('ff_down'))
  boundaries.append(hidden.reshape(-1).tolist())
 normalized=norm(hidden,weights['output_norm']);logits=ops._linear_reference((normalized,weights['embedding']),{})
 steps.append({'token':token,'block_outputs':boundaries,'logits':logits.reshape(-1).tolist()})
record={'reference':'V3 primitive reference equations, four alternating recurrent/attention blocks, dense BF16 history, tied readout; independent NumPy arithmetic',
'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,
'source_sha256':{str(p):hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['src/ops/tensor/ops.py','src/ops/tensor/primitive.py','src/engine/models/qwen35/equations.py']},
'weights':{n:{'shape':list(x.shape),'values':x.reshape(-1).tolist()} for n,x in weights.items()},'kinds':kinds,'routed':a.routed,'steps':steps}
a.output.write_text(json.dumps(record,indent=2)+'\n')
