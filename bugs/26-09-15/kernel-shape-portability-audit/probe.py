import argparse, json, sys, warnings, time, statistics
from pathlib import Path
parser=argparse.ArgumentParser(description="Read-only kernel shape audit; numerical mismatches are reported as data.")
parser.add_argument("--native",action="store_true")
parser.add_argument("--samples",type=int,default=0)
parser.add_argument("--output",type=Path,default=Path("/tmp/kernel-shape-audit.json"))
args=parser.parse_args()
from dataclasses import replace
import numpy as np
import ops
from ops.kv import KVRepresentation
from engine import DevicePlan
from ops.lab.ownership import exclusive_measurement

def sanitize(value):
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    if isinstance(value, dict): return {k:sanitize(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [sanitize(v) for v in value]
    return value

records=[]
def emit(label, **fields):
    record=dict(case=label, **fields)
    records.append(record)
    print(json.dumps(sanitize(record), allow_nan=False), flush=True)
    with args.output.open('w') as f: json.dump(sanitize(records),f,indent=2,allow_nan=False)

def signature(specs):
    return ops.Signature(tuple(ops.Argument(s,f'v{i}',ops.ValueKind.RESOURCE if isinstance(s.representation,KVRepresentation) else ops.ValueKind.INPUT) for i,s in enumerate(specs)))

def plan(label, function, specs, mode='prefill', target=None):
    try:
        graph=ops.trace(function, signature(specs))
        p=ops.analyze(function, signature=signature(specs), compiler_target=target or ops.CompilerTarget(32,1024,32768,identity='audit-metal-resources'), available_bytes=1<<30, options=ops.CompileOptions(mode=mode))
        emit(label, trace='accepted', plan='accepted', operations=[x.name for x in p.operations])
        return p
    except Exception as e:
        emit(label, plan='rejected', error=f'{type(e).__name__}: {e}')

def q8(shape):
    return ops.TensorSpec(shape,ops.DType.F16).with_representation(ops.Affine(ops.Code(8,interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),32,ops.DirectCoefficients(ops.DType.F16)))

for group in (1,2,3,4,8,12,16):
    q=ops.TensorSpec((1,group*2,256),ops.DType.F16)
    h=ops.kv_state_spec(17,2,ops.DType.F16,ops.affine_k8_uniform_v4(256,256))
    kv=ops.TensorSpec((1,2,256),ops.DType.F16)
    plan(f'persistent-decode-group-{group}',lambda q,h,k,v,r:ops.persistent_attention(q,h,k,v,r,sequence_count=1),(q,h,kv,kv,ops.TensorSpec((1,4),ops.DType.I32)),mode='decode')
for rows,experts,selected in ((1,256,8),(16,256,8),(32,256,8),(64,512,8)):
    width=intermediate=256
    specs=(ops.TensorSpec((rows,width),ops.DType.F16),ops.TensorSpec((rows,selected),ops.DType.I32),ops.TensorSpec((rows,selected),ops.DType.F32),q8((experts,intermediate,width)),q8((experts,intermediate,width)),q8((experts,width,intermediate)))
    for mode in ('prefill','decode'):
        plan(f'experts-{mode}-rows{rows}-experts{experts}',lambda *x:ops.routed_experts(*x),specs,mode)
for width in (32,64,128,256,288,512):
    for rows in (1,8):
        plan(f'q8-linear-{rows}x{width}',lambda x,w:ops.linear(x,w),(ops.TensorSpec((rows,width),ops.DType.F16),q8((17,width))))
for shape in ((17,),(2,17),(2,3,17)):
    plan(f'rms-rank{len(shape)}',lambda x:ops.rms_norm(x),(ops.TensorSpec(shape,ops.DType.F32),))
    plan(f'linear-rank{len(shape)}',lambda x,w:ops.linear(x,w),(ops.TensorSpec(shape,ops.DType.F32),ops.TensorSpec((9,17),ops.DType.F32)))

from ops.kv import dense_kv
for width in (16,32,64,96,128,192,256,512):
    for representation in ('dense','affine'):
        rep=dense_kv(width,width,ops.DType.F16) if representation=='dense' else ops.affine_k8_uniform_v4(width,width)
        q=ops.TensorSpec((2,8,width),ops.DType.F16)
        h=ops.kv_state_spec(17,1,ops.DType.F16,rep)
        kv=ops.TensorSpec((2,1,width),ops.DType.F16)
        plan(f'persistent-prefill-{representation}-width{width}',lambda q,h,k,v,r:ops.persistent_attention(q,h,k,v,r,sequence_count=1),(q,h,kv,kv,ops.TensorSpec((2,4),ops.DType.I32)))
for mode in ('prefill','decode'):
    rows,experts,width,selected=32,8,256,2
    dense=lambda shape:ops.TensorSpec(shape,ops.DType.F16)
    specs=(dense((rows,width)),ops.TensorSpec((rows,selected),ops.DType.I32),ops.TensorSpec((rows,selected),ops.DType.F32),dense((experts,width,width)),dense((experts,width,width)),dense((experts,width,width)))
    plan(f'dense-expert-banks-{mode}',lambda *x:ops.routed_experts(*x),specs,mode)
for experts in (256,512,1024,1025):
    plan(f'route-topk-experts{experts}',lambda x:ops.route_topk(x,k=2), (ops.TensorSpec((2,experts),ops.DType.F32),))

if not args.native: sys.exit()

def timing(compiled, inputs, resources=None):
    if not args.samples:
        return {}
    samples=[]
    for index in range(5 + args.samples):
        start=time.perf_counter_ns()
        run=compiled.submit(*inputs, resources=resources)
        run.completion.wait()
        elapsed=time.perf_counter_ns()-start
        for output in run.outputs: output.close()
        if index >= 5: samples.append(elapsed)
    return dict(submit_wait_ns=samples, median_submit_wait_ns=statistics.median(samples))

def native(label, function, arrays, device, mode='prefill', atol=3e-3, rtol=3e-3):
    kinds={np.dtype('float32'):ops.DType.F32,np.dtype('float16'):ops.DType.F16,np.dtype('int32'):ops.DType.I32}
    specs=tuple(ops.TensorSpec(a.shape,kinds[a.dtype]) for a in arrays)
    sig=signature(specs)
    resources=[]; compiled=execution=None
    try:
        graph=ops.trace(function,sig)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            expected=ops.evaluate_reference(graph,{f'v{i}':a for i,a in enumerate(arrays)}).outputs
        resources=[device.upload(s,a.tobytes()) for s,a in zip(specs,arrays)]
        compiled=ops.compile(function,signature=sig,device=device,constants={},options=ops.CompileOptions(mode=mode))
        execution=compiled.submit(*resources)
        execution.completion.wait()
        observed=[x.native.cpu().numpy() for x in execution.outputs]
        matches=[bool(np.allclose(a,b,atol=atol,rtol=rtol,equal_nan=True)) for a,b in zip(observed,expected)]
        emit(label,native='passed' if all(matches) else 'mismatch',matches=matches, max_abs=[float(np.max(np.abs(a-b))) if a.size else 0 for a,b in zip(observed,expected)],nan_counts=[int(np.isnan(a).sum()) for a in observed], **timing(compiled, resources))
    except Exception as e:
        emit(label,native='failed',error=f'{type(e).__name__}: {e}')
    finally:
        if execution:
            for x in execution.outputs:x.close()
        if compiled:compiled.close()
        for x in resources:x.close()

rng=np.random.default_rng(8123)
with exclusive_measurement(), ops.DeviceRuntime.open(DevicePlan.discover(backend='metal',maximum_bytes=1<<29)) as device:
    for width,offsets in ((6,[0,1,4]),(6,[0,0,4]),(1025,[0,1])):
        rows=offsets[-1]; batch=len(offsets)-1; channels=4*width
        shapes=((rows,channels),(channels,3),(batch,channels,2),(rows,2),(rows,2),(2,),(2,))
        arrays=tuple(rng.normal(0,.1,s).astype(np.float32) for s in shapes)+(np.array(offsets,np.int32),)
        def recurrent(*x):return ops.recurrent_prepare(*x,key_heads=1,value_heads=2,width=width,convolution_width=3,epsilon=1e-6)
        native(f'recurrent-prepare-width{width}-offsets{offsets}',recurrent,arrays,device)
    for width,masked in ((4095,True),(4097,False),(4097,True)):
        x=np.zeros((1,width),np.float32)
        if masked:x[0,:min(width-1,4096)]=-np.inf
        native(f'softmax-width{width}-masked{masked}',lambda x:ops.softmax(x),(x,),device,atol=1e-6)
    for shape in ((3,17),(2,3,17)):
        x=rng.normal(0,.1,shape).astype(np.float32)
        native(f'rms-native-{shape}',lambda x:ops.rms_norm(x),(x,),device)
    for m,n,k in ((1,17,31),(8,17,31),(9,33,65)):
        arrays=(rng.normal(0,.1,(m,k)).astype(np.float32),rng.normal(0,.1,(n,k)).astype(np.float32))
        native(f'dense-tail-{m}-{n}-{k}',lambda x,w:ops.linear(x,w),arrays,device)

with exclusive_measurement(), ops.DeviceRuntime.open(DevicePlan.discover(backend='metal',maximum_bytes=1<<29)) as device:
    for width in (256,512):
        rep=dense_kv(width,width,ops.DType.F16)
        state=ops.kv_state_spec(4,1,ops.DType.F16,rep)
        history=np.zeros(state.shape,np.float32)
        arrays=[rng.normal(0,.2,(1,8,width)).astype(np.float16),rng.normal(0,.2,(1,1,width)).astype(np.float16),np.ones((1,1,width),np.float16),np.array([[0,0,0,1]],np.int32)]
        specs=tuple(ops.TensorSpec(x.shape,ops.DType.I32 if x.dtype==np.int32 else ops.DType.F16) for x in arrays)
        sig=ops.Signature(tuple(ops.Argument(s,f'v{i}') for i,s in enumerate(specs))+(ops.Argument(state,'history',ops.ValueKind.RESOURCE),))
        def function(q,k,v,r,h): return ops.persistent_attention(q,h,k,v,r,sequence_count=1)
        resources=[];compiled=execution=None
        try:
            resources=[device.upload(s,x.tobytes()) for s,x in zip(specs,arrays)]
            resources.append(device.upload(state,bytes(state.storage_nbytes)))
            compiled=ops.compile(function,signature=sig,device=device,constants={},options=ops.CompileOptions(mode='decode'))
            execution=compiled.submit(*resources[:-1],resources={'history':resources[-1]})
            execution.completion.wait()
            actual=execution.outputs[0].native.cpu().numpy()
            record=dict(width=width,expected='all ones: one visible value row',first_half_correct=bool(np.allclose(actual[:,:,:256],1)),tail_correct=bool(np.allclose(actual[:,:,256:],1)),incorrect=int(np.count_nonzero(~np.isclose(actual,1))), **timing(compiled, resources[:-1], {'history':resources[-1]}))
        except Exception as e:
            record=dict(width=width,error=f'{type(e).__name__}: {e}')
        finally:
            if execution:
                for x in execution.outputs:x.close()
            if compiled:compiled.close()
            for x in resources:x.close()
        emit("persistent-attention-width", **record)

with exclusive_measurement(), ops.DeviceRuntime.open(DevicePlan.discover(backend='metal',maximum_bytes=1<<26)) as device:
    for shape in ((2,),(1,2)):
        table=ops.TensorSpec((4,32),ops.DType.F16).with_representation(ops.Affine(ops.Code(8,interpretation=ops.CodeInterpretation.TWOS_COMPLEMENT),32,ops.DirectCoefficients(ops.DType.F16)))
        indices=ops.TensorSpec(shape,ops.DType.I32)
        sig=ops.Signature((ops.Argument(indices,'ids'),ops.Argument(table,'table')))
        resources=[];program=execution=None
        try:
            resources=[device.upload(indices,np.array([0,1],np.int32).tobytes()),device.upload(table,bytes(table.storage_nbytes))]
            def embed(ids,table):return ops.embedding(ids,table)
            program=ops.compile(embed,signature=sig,device=device,constants={},options=ops.CompileOptions(mode='prefill'))
            execution=program.submit(*resources)
            execution.completion.wait()
            record=dict(shape=shape,status='passed' if np.all(execution.outputs[0].native.cpu().numpy()==0) else 'mismatch')
        except Exception as e:record=dict(shape=shape,status='failed',error=f'{type(e).__name__}: {e}')
        finally:
            if execution:
                for x in execution.outputs:x.close()
            if program:program.close()
            for x in resources:x.close()
        emit("packed-embedding-rank", **record)
