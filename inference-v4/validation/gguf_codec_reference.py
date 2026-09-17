#!/usr/bin/env python3
"""Fixture from the exact V3 GGUFCodec methods, executed with NumPy scalars."""
import argparse,ast,hashlib,json,struct
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
path=args.source/'src/engine/weights/formats/gguf.py'
module=ast.parse(path.read_text());selected=[node for node in module.body if isinstance(node,ast.ClassDef) and node.name in ('Encoding','GGUFCodec')]
code=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
env={'dataclass':dataclass,'IntEnum':IntEnum};exec(compile(ast.fix_missing_locations(code),str(path),'exec'),env)
Encoding,Codec=env['Encoding'],env['GGUFCodec'];rng=np.random.default_rng(718732)
records=[]
for encoding in [Encoding.Q4_K,Encoding.Q5_K,Encoding.Q6_K,Encoding.Q8_0,Encoding.IQ4_XS]:
 codec=Codec(encoding);blocks=3;raw=rng.integers(0,256,blocks*encoding.block_bytes,dtype=np.uint8)
 for b in range(blocks):
  offset=b*encoding.block_bytes+(208 if encoding==Encoding.Q6_K else 0)
  raw[offset:offset+2]=np.frombuffer(struct.pack('<e',[0.03125,-0.017578125,2**-20][b]),np.uint8)
  if encoding in (Encoding.Q4_K,Encoding.Q5_K):raw[offset+2:offset+4]=np.frombuffer(struct.pack('<e',[0.0625,0.001953125,-0.125][b]),np.uint8)
 group=16 if encoding==Encoding.Q6_K else 32;bits=4 if encoding in (Encoding.Q4_K,Encoding.IQ4_XS) else 8;cpw=32//bits
 words=[];scales=[];biases=[];values=[]
 reinterpret=lambda value,dtype:np.asarray(value).view(dtype)
 for b in range(blocks):
  base=b*encoding.block_bytes;codes=[int(codec.code(raw,base,i)) for i in range(encoding.block_elements)]
  words += [sum(codes[w+c] << (bits*c) for c in range(cpw)) for w in range(0,len(codes),cpw)]
  for g in range(encoding.block_elements//group):
   if encoding in (Encoding.Q4_K,Encoding.Q5_K,Encoding.Q6_K):scale=float(codec.direct_scale(raw,base,g,np.where,reinterpret))
   elif encoding==Encoding.Q8_0:scale=struct.unpack('<e',bytes(raw[base:base+2]))[0]
   else:scale=float(np.float32(codec.super_scale(raw,base,reinterpret)*np.float32(int(codec.local_scale(raw,base,g,np.where))-32)))
   bias=float(codec.direct_bias(raw,base,g,np.where,reinterpret)) if encoding in (Encoding.Q4_K,Encoding.Q5_K) else 0.
   scales.append(scale)
   if encoding in (Encoding.Q4_K,Encoding.Q5_K):biases.append(bias)
   for c in codes[g*group:(g+1)*group]:
    if encoding==Encoding.Q6_K:c-=32
    elif encoding==Encoding.Q8_0:c=c-256 if c>=128 else c
    elif encoding==Encoding.IQ4_XS:c=(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)[c]
    values.append(float(np.float32(c*scale+bias)))
 records.append({'encoding':int(encoding),'name':encoding.name,'shape':[blocks,encoding.block_elements],'source_hex':bytes(raw).hex(),'words':words,'scales':scales,'biases':biases,'values':values})
args.output.write_text(json.dumps({'reference':'Exact V3 GGUFCodec class methods with NumPy values; decoded result rounded once to f32','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,'cases':records},indent=2)+'\n')
