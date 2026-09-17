#!/usr/bin/env python3
"""Generate V3 Qwen multimodal rotary fixtures; validation only."""
import argparse, hashlib, json, sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();source=args.source.resolve(strict=True);sys.path.insert(0,str(source/'src'))
import numpy as np
from ops.tensor import ops
from ops.tensor.primitive import round_reference
from ops.tensor.types import DType
rng=np.random.default_rng(773)
q=lambda x:round_reference(x,DType.BF16)
rows,heads,kv,width,rotary=4,4,2,16,12
query_gate=q(rng.standard_normal((rows,heads*2*width)).astype(np.float32))
key=q(rng.standard_normal((rows,kv*width)).astype(np.float32))
query_norm=rng.uniform(0.8,1.2,width).astype(np.float32)
key_norm=rng.uniform(0.8,1.2,width).astype(np.float32)
coordinates=np.array([[0,0,0,0],[13,7,4,13],[2048,1023,511,2048],[131071,17476,1001,131071]],np.int32)
base=1000000.;epsilon=1e-6;sections=(3,2,1,0)
query,prepared_key,gate=map(q,ops._attention_prepare_reference((query_gate,key,query_norm,key_norm,coordinates),{
 'query_heads':heads,'kv_heads':kv,'width':width,'rotary_width':rotary,'base':base,'sections':sections,'epsilon':epsilon}))
serialize=lambda x:np.asarray(x).reshape(-1).tolist()
record={'reference':'V3 attention preparation primitive, explicit BF16 publication; text and distinct multimodal coordinates including long context',
 'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,
 'source_sha256':{str(p):hashlib.sha256((source/p).read_bytes()).hexdigest() for p in ['src/ops/tensor/ops.py','src/ops/tensor/primitive.py']},
 'shapes':{'Q':rows,'H':heads,'KV':kv,'P':rotary//2,'S':width-rotary,'SH':sections[1],'SW':sections[2]},
 'scalars':{'base':base,'epsilon':epsilon},
 'inputs':{name:serialize(value) for name,value in {'query_gate':query_gate,'key':key,'query_norm':query_norm,'key_norm':key_norm,'coordinates':coordinates}.items()},
 'outputs':{name:serialize(value) for name,value in {'query':query,'prepared_key':prepared_key,'gate':gate}.items()}}
args.output.write_text(json.dumps(record,indent=2)+'\n')
