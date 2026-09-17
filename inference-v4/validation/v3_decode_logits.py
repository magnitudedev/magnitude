#!/usr/bin/env python3
"""Inspect pinned V3 logits for explicit token IDs and a declared KV format."""
import argparse,hashlib,json,sys,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--backend',required=True);p.add_argument('--tokens',default='1,2,3');p.add_argument('--context',type=int,default=128);p.add_argument('--dense-kv',action='store_true');p.add_argument('--memory-gib',type=int,default=16);a=p.parse_args();source=a.source.resolve(strict=True);sys.path.insert(0,str(source/'src'))
import numpy as np
import ops
from engine import DevicePlan
from engine.data import TokenId
from engine.models.qwen35.formats.mlx import describe
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.qwen35.inputs import InputPlan
from engine.models.sequence import ModelRequest,LogitsSelection
from engine.weights.formats.mlx_safetensors import MLXFormat
from engine.weights.tensor_residency import TensorWeights
if a.dense_kv:ops.default_kv_representation=lambda key,value:ops.dense_kv(key,value,ops.DType.BF16)
tokens=tuple(TokenId(int(t)) for t in a.tokens.split(','))
if a.artifact.is_dir():
 artifact=MLXFormat(str(a.artifact));description=describe(artifact)
else:
 from engine.weights.formats.gguf import GGUFFormat
 from engine.models.qwen35.formats.gguf import describe as describe_gguf
 artifact=GGUFFormat(str(a.artifact));description=describe_gguf(artifact)
print(json.dumps({'kind':'reference','artifact':str(artifact.identity),'kv':'dense_bf16' if a.dense_kv else 'v3_default','source':str(source),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}),flush=True)
with ops.DeviceRuntime.open(DevicePlan.discover(backend=a.backend,maximum_bytes=a.memory_gib<<30)) as device:
 weights=TensorWeights(artifact,device);model=DenseRuntime(description,device,weights,max_sequences=1,context_capacity=a.context)
 sequence=model.create(InputPlan.text(tokens))
 try:
  for token in tokens:
   start=time.monotonic();batch=model.prepare((ModelRequest(sequence,(token,),LogitsSelection.LAST),))
   try:
    batch.completion.wait();data=batch.advances[0].forward.read_logits();logits=np.frombuffer(data,np.float32);top=np.argsort(-logits,kind='stable')[:8]
    batch.advances[0].commit()
    print(json.dumps({'kind':'logits','input_token':int(token),'seconds':time.monotonic()-start,'finite':bool(np.isfinite(logits).all()),'logits_sha256':hashlib.sha256(data).hexdigest(),'top':[{'token':int(i),'logit':float(logits[i])} for i in top]}),flush=True)
   finally:batch.close()
 finally:sequence.close();model.close();weights.close();artifact.close()
