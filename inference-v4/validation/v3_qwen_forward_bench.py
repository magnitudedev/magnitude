import argparse, dataclasses, hashlib, json, pathlib, sys, time, traceback
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--artifact',required=True);p.add_argument('--output',required=True);p.add_argument('--profiles-only',action='store_true');a=p.parse_args()
sys.path.insert(0,str(pathlib.Path(a.source)/'src'));sys.path.insert(0,a.source)
import numpy as np
import ops
from engine import DevicePlan
from engine.data import TokenId
from engine.models.qwen35.formats.mlx import describe
from engine.models.qwen35.runtime import DenseRuntime
from engine.models.qwen35.inspection import inspect_forwards
from engine.models.qwen35.inputs import InputPlan
from engine.models.sequence import ModelRequest,LogitsSelection
from engine.weights.formats.mlx_safetensors import MLXFormat
from engine.weights.tensor_residency import TensorWeights
ops.default_kv_representation=lambda key,value:ops.dense_kv(key,value,ops.DType.BF16)
out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
report={'protocol':'forced-token model forward; no sampling/tokenization; dense BF16 KV; batch 1; cold compile separated; profiles separate from throughput','records':[],'profiles':[],'errors':[]}
def save(): (out/'result.json').write_text(json.dumps(report,indent=2,default=str))
def observe(inv,obs):
 graph=inv.compiled.graph
 calls={c.occurrence:c for c in graph.formulas}
 nodes={n.id:n for n in graph.nodes}
 profile={'graph_fingerprint':graph.fingerprint,'mode':inv.mode,'positions':inv.positions,'lengths':inv.lengths,'physical_rows':inv.physical_rows,'observation':dataclasses.asdict(obs),'formulas':{str(i):{'id':c.formula.id,'parent':c.parent,'nodes':c.nodes} for i,c in calls.items()},'nodes':{str(i):str(getattr(n,'op',getattr(n,'operation',type(n).__name__))) for i,n in nodes.items()}}
 report['profiles'].append(profile);save()
artifact=MLXFormat(a.artifact);desc=describe(artifact);report['artifact']=str(artifact.identity);report['geometry']=str(desc.geometry);save()
with ops.DeviceRuntime.open(DevicePlan.discover(backend='metal',maximum_bytes=24<<30)) as device:
 weights=TensorWeights(artifact,device);model=DenseRuntime(desc,device,weights,max_sequences=1,context_capacity=256)
 def submit(seq,ids,label):
  start=time.perf_counter();batch=model.prepare((ModelRequest(seq,tuple(map(TokenId,ids)),LogitsSelection.LAST),))
  try:
   batch.completion.wait();data=batch.advances[0].forward.read_logits();batch.advances[0].commit(); elapsed=time.perf_counter()-start
   logits=np.frombuffer(data,np.float32);report['records'].append({'label':label,'tokens':len(ids),'seconds':elapsed,'finite':bool(np.isfinite(logits).all()),'top1':int(np.argmax(logits))});save();print(label,elapsed,flush=True)
  finally:batch.close()
 try:
  seq=model.create(InputPlan.text(tuple(map(TokenId,range(1,49)))))
  try:
   for token in (range(1,1) if a.profiles_only else range(1,49)): submit(seq,[token],'decode-warmup' if token<=32 else 'decode')
  finally:seq.close()
  seq=model.create(InputPlan.text(tuple(map(TokenId,range(1,37)))))
  try:
   for token in range(1,33):submit(seq,[token],'profile-history')
   with inspect_forwards(model,observed=observe,kernel_limit=2048):
    for token in range(33,37):submit(seq,[token],'decode-profile')
  finally:seq.close()
  for length in (32,128):
   try:
    for repeat in range(1 if a.profiles_only else 4):
     seq=model.create(InputPlan.text(tuple(map(TokenId,range(1,length+1)))))
     try:submit(seq,list(range(1,length+1)),f'prefill-{length}-'+('cold' if repeat==0 else 'warm'))
     finally:seq.close()
    seq=model.create(InputPlan.text(tuple(map(TokenId,range(1,length+1)))))
    try:
     with inspect_forwards(model,observed=observe,kernel_limit=2048):submit(seq,list(range(1,length+1)),f'prefill-{length}-profile')
    finally:seq.close()
   except Exception:
    report['errors'].append({'mode':f'prefill-{length}','traceback':traceback.format_exc()});save();print(traceback.format_exc(),flush=True)
 finally:model.close();weights.close();artifact.close();save()
