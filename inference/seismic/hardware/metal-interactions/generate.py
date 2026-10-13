"""Generate a fixed, versioned Metal interaction corpus; no production candidates."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'sources'
OUT.mkdir(parents=True, exist_ok=True)
render = (ROOT.parents[2] / 'seismic/backends/metal/src/render.rs').read_text()
soft = (ROOT.parents[2] / 'seismic/backends/metal/src/softfloat.metal').read_text()
prelude = re.search(r'const LIBRARY_PRELUDE: &str = r#"(.*?)"#;', render, re.S).group(1)
src = [prelude, soft, '\n#include <metal_simdgroup_matrix>\nstruct P { uint n, it, stride, mode; float a,b; uint groups, pad; };\n']
cases = []
names = set()

def kernel(name, body, extra=''):
    if name in names:
        return
    names.add(name)
    src.append(f'''kernel void {name}(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
      constant P& p [[buffer(2)]], threadgroup float* sm [[threadgroup(0)]],
      uint id [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
      uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]],
      uint nt [[threads_per_threadgroup]]) {{ {body} }}\n''')

def case(family, name, split='diagnostic', **kw):
    row = dict(family=family, kernel=name, split=split, threads=8192, tg=128, n=8192,
               it=128, stride=1, mode=0, shared=16, a=0.99991, b=0.00001)
    row.update(kw)
    row['id'] = f'{len(cases):04d}_{family}_{name}'
    cases.append(row)

# Native FMA is a hardware mechanism probe, NOT Seismic strict-F32 arithmetic.
for c in [1, 2, 4, 8, 16, 32, 64, 128]:
    init = ''.join(f'float v{k}=x[id+{k}*p.n];' for k in range(c))
    step = ''.join(f'v{k}=fma(v{k},p.a,p.b);' for _ in range(8) for k in range(c))
    tail = 'y[id]=' + '+'.join(f'v{k}' for k in range(c)) + ';'
    name = f'fma_c{c}'
    kernel(name, init + 'for(uint j=0;j<p.it;j++) {' + step + '}' + tail)
    for threads in [32, 512, 8192, 65536]:
        for it in [64, 256]:
            split = 'cal' if c in [1, 4, 16] and threads in [512, 65536] else 'test'
            case('compute', name, split, chains=c, threads=threads, n=threads,
                 tg=min(128, threads), it=it, ops=threads*c*it*8)
    if c in [4, 16, 64, 128]:
        kernel(f'pressure_c{c}', f'sm[tid]=x[id]; threadgroup_barrier(mem_flags::mem_threadgroup);' + init +
               'for(uint j=0;j<p.it;j++) {' + step + '}' +
               'threadgroup_barrier(mem_flags::mem_threadgroup);' + tail.replace(';', '+sm[(tid+1)%nt];'))
        for tg in [64, 256, 512]:
            for shared in [2048, 16384, 32768]:
                case('pressure', f'pressure_c{c}', chains=c, tg=tg, shared=shared, it=128)

# Mixed instruction streams: changing resource demands without candidate fitting.
for f, u in [(16,0),(0,16),(16,4),(16,16),(4,16),(8,8)]:
    name=f'mix_f{f}_u{u}'
    body='float v0=x[id],v1=x[id+p.n],v2=x[id+2*p.n],v3=x[id+3*p.n]; uint q0=id+1,q1=id+2,q2=id+3,q3=id+4;'
    body+='for(uint j=0;j<p.it;j++){'
    for k in range(max(f,u)):
        if k<f: body+=f'v{k%4}=fma(v{k%4},p.a,p.b);'
        if k<u: body+=f'q{k%4}=(q{k%4}*1664525u+1013904223u)^(q{k%4}>>13);'
    body+='} y[id]=v0+v1+v2+v3; reinterpret_cast<device uint*>(y)[id+p.n]=q0^q1^q2^q3;'
    kernel(name,body)
    for threads in [512,8192,65536]:
        case('mixed',name,'cal' if f==0 or u==0 else 'test',threads=threads,n=threads,it=512,f=f,u=u)

for intensity in [0,1,4,16,64]:
    name=f'stream_k{intensity}'
    body='for(uint j=id;j<p.n;j+=p.groups){float v=x[j*p.stride];'
    body+='v=fma(v,p.a,p.b);'*intensity
    body+='y[j]=v;}'
    kernel(name,body)
    for n in [4096,262144,4194304,16777216]:
        for stride in [1,4]:
            case('stream',name,'cal' if intensity==0 else 'test',n=n,stride=stride,it=1,intensity=intensity)

# Dependent random-looking full-cycle permutation: odd multiplier 5, odd increment.
kernel('chase','uint q=id & (p.n-1); device const uint* z=reinterpret_cast<device const uint*>(x); for(uint j=0;j<p.it;j++) q=z[q]; reinterpret_cast<device uint*>(y)[id]=q;')
for n in [4096,65536,1048576,16777216]:
    for threads in [1,32,512,8192]:
        case('chase','chase',n=n,threads=threads,tg=min(128,threads),it=4096)

# Actual helper functions, with every iteration observing a new input; no hoistable pure call.
for op, expr in [('add','f32_add(a,b)'),('mul','f32_mul(a,b)'),('div','f32_div(a,b)'),
                 ('rem','f32_rem(a,b)'),('fma','f32_mulAdd(a,b,a)'),('cmp','float(f32_lt(a,b))'),
                 ('f16','float(f16_add(half(a),half(b)))'),('bf16','float(bf16_add(bfloat(a),bfloat(b)))'),
                 ('convert','f16_to_f32(f32_to_f16(a))')]:
    name=f'strict_{op}'
    kernel(name,'uint h=0; float last=0; for(uint j=0;j<p.it;j++){ uint k=(id+j*p.groups)&(p.n-1); float a=x[k], b=x[k+p.n]; float v='+expr+'; last=v; h=(h*1664525u)^as_type<uint>(v); } reinterpret_cast<device uint*>(y)[id]=h; y[id+p.groups]=last;')
    for mode in range(7):
        case('strict',name,op=op,mode=mode,n=262144,it=64)

for op in ['integer','cas']:
    statement='atomic_fetch_add_explicit(z,1u,memory_order_relaxed);' if op=='integer' else '''uint expected=atomic_load_explicit(z,memory_order_relaxed); while(true){uint desired=as_type<uint>(f32_add(as_type<float>(expected),1.0f)); if(atomic_compare_exchange_weak_explicit(z,&expected,desired,memory_order_relaxed,memory_order_relaxed)) break;}'''
    kernel('atomic_'+op,'device atomic_uint* z=reinterpret_cast<device atomic_uint*>(y)+(id%p.n); for(uint j=0;j<p.it;j++){'+statement+'}')
    for n in [1,32,1024,8192]:
        case('atomic','atomic_'+op,n=n,it=4,op=op)

kernel('barrier','float v=x[id]; for(uint j=0;j<p.it;j++){sm[tid]=v; threadgroup_barrier(mem_flags::mem_threadgroup); v=sm[(tid+1)%nt]; threadgroup_barrier(mem_flags::mem_threadgroup);} y[id]=v;')
kernel('shuffle','float v=x[id]; for(uint j=0;j<p.it;j++) v=simd_shuffle(v,(lane+1)%32); y[id]=v;')
kernel('reduce','float v=x[id]; for(uint j=0;j<p.it;j++) v=simd_sum(v)*0.03125f; y[id]=v;')
for name in ['barrier','shuffle','reduce']:
    for tg in [32,128,512]:
        for it in [32,128,512]:
            case('sync',name,tg=tg,it=it,shared=max(16,tg*4))

for staged in [False,True]:
    name='mma_staged' if staged else 'mma_resident'
    # One 8x8 tile per group; only 32 threads. Published full output tile.
    begin='simdgroup_float8x8 a,b,c; simdgroup_load(a,x+gid*128,8); simdgroup_load(b,x+gid*128+64,8); c=make_filled_simdgroup_matrix<float,8,8>(0.0f);'
    loop='simdgroup_multiply_accumulate(c,a,b,c);'
    if staged:
        loop='for(uint k=tid;k<128;k+=32) sm[k]=x[gid*128+k]; threadgroup_barrier(mem_flags::mem_threadgroup); simdgroup_load(a,sm,8); simdgroup_load(b,sm+64,8);'+loop+'threadgroup_barrier(mem_flags::mem_threadgroup);'
    kernel(name,begin+'for(uint j=0;j<p.it;j++){'+loop+'} simdgroup_store(c,y+gid*64,8);')
    for groups in [1,16,256]:
        for it in [16,64,256]:
            case('matrix',name,threads=groups*32,tg=32,n=groups*128,it=it,shared=512)

kernel('noop','y[id]=x[id];')
for commands in [1,4,16,64,256]:
    case('lifecycle','noop',threads=32,tg=32,n=32,it=1,commands=commands)

(OUT/'probes.metal').write_text('\n'.join(src))
(OUT/'manifest.json').write_text(json.dumps(cases,indent=2))
(OUT/'identity.json').write_text(json.dumps({'protocol':'metal-interactions-v1','source_sha256':hashlib.sha256((OUT/'probes.metal').read_bytes()).hexdigest(),'softfloat_sha256':hashlib.sha256(soft.encode()).hexdigest(),'cases':len(cases),'pipelines':len(names)},indent=2))
print(json.dumps({'cases':len(cases),'pipelines':len(names),'out':str(OUT)}))
