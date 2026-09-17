#!/usr/bin/env python3
"""Generate V3 dense-history attention decode fixtures; validation only."""
import argparse,hashlib,json,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=args.source.resolve(strict=True);sys.path.insert(0,str(source/'src'))
import numpy as np
from ops.tensor import ops
from ops.tensor.primitive import round_reference
from ops.tensor.types import DType
rng=np.random.default_rng(17012);q=lambda x:round_reference(x,DType.BF16)
d,t,h,kv,p,s,sh,sw=8,8,4,2,6,4,2,1;w=2*p+s
random=lambda shape,scale:q((rng.standard_normal(shape)*scale).astype(np.float32))
weights={'input_norm':q(1+random((d,),0.1)),'query_gate_weight':random((h*2*w,d),0.2),
 'key_weight':random((kv*w,d),0.2),'value_weight':random((kv*w,d),0.2),
 'query_norm':rng.uniform(0.8,1.2,w).astype(np.float32),'key_norm':rng.uniform(0.8,1.2,w).astype(np.float32),
 'output_weight':random((d,h*w),0.1)}
initial_key=random((t,kv,w),0.6);initial_value=random((t,kv,w),0.6)
base=1000000.;epsilon=1e-6;scale=1/np.sqrt(w)
linear=lambda x,weight:q(ops._linear_reference((x,weight),{}))
serialize=lambda x:np.asarray(x).reshape(-1).tolist()
cases=[]
for start,end,destination,coordinate in [(0,0,3,[0,0,0,0]),(1,4,6,[13,7,4,13]),(0,7,7,[131071,17476,1001,131071])]:
 hidden=(rng.standard_normal((1,d))*0.8).astype(np.float32)
 coordinates=np.asarray([coordinate],np.int32);visible=np.asarray([[start,end]],np.int32)
 normalized=q(ops._rms_reference((hidden,weights['input_norm']),{'epsilon':epsilon}))
 query_gate=linear(normalized,weights['query_gate_weight']);key=linear(normalized,weights['key_weight']);value=linear(normalized,weights['value_weight'])
 query,prepared_key,gate=map(q,ops._attention_prepare_reference((query_gate,key,weights['query_norm'],weights['key_norm'],coordinates),{
  'query_heads':h,'kv_heads':kv,'width':w,'rotary_width':p*2,'base':base,'sections':(3,sh,sw,0),'epsilon':epsilon}))
 history=np.concatenate((initial_key,initial_value),axis=-1)
 attended=q(ops._persistent_attention_reference((query,history,prepared_key,value.reshape(1,kv,w),np.asarray([[start,end-start,0,1]],np.int32)),{'scale':scale}))
 activated=q(1/(1+np.exp(-gate.reshape(1,h*w))));gated=q(attended.reshape(1,h*w)*activated)
 projected=linear(gated,weights['output_weight']);out=hidden+projected
 history_key=initial_key.copy();history_value=initial_value.copy();history_key[destination]=prepared_key[0];history_value[destination]=value.reshape(1,kv,w)[0]
 outputs={name:serialize(v) for name,v in locals().copy().items() if name in
  ['normalized','query_gate','key','value','query','prepared_key','gate','attended','activated','gated','projected','out','history_key','history_value']}
 cases.append({'inputs':{'hidden':serialize(hidden),'coordinates':serialize(coordinates),'visible':serialize(visible)},'destination':destination,'outputs':outputs})
record={'reference':'V3 attention preparation and persistent attention primitive references, explicit BF16 publications; fresh KV remains separate until persistence',
 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,
 'source_sha256':{str(p):hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['src/ops/tensor/ops.py','src/ops/tensor/primitive.py']},
 'shapes':{'D':d,'T':t,'H':h,'KV':kv,'P':p,'S':s,'SH':sh,'SW':sw},
 'scalars':{'base':base,'epsilon':epsilon,'scale':float(scale)},'weights':{k:serialize(v) for k,v in weights.items()},
 'initial_history_key':serialize(initial_key),'initial_history_value':serialize(initial_value),'cases':cases}
args.output.write_text(json.dumps(record,indent=2)+'\n')
