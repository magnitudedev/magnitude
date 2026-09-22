"""Shared executable description for laboratory emission and analytical execution.

These constructors lower to explicit instructions/phases. Characterization sees
the same lowered structure as emission; candidate names are never cost inputs.
"""
from dataclasses import dataclass,field,asdict
from enum import IntEnum
from pathlib import Path
import hashlib,json,math,re

class Kind(IntEnum):
    FMA=0; ALU=1; MUL=2; WIDE=3; SFU=4; LOAD=5; STORE=6; SHLOAD=7; SHSTORE=8
    CONTROL=9; BARRIER=10; SHUFFLE=11; REDUCE=12; MMA=13; ATOMIC=14; CAS=15; NOP=16; CONVERT=17
    BRANCH=18; CALL=19; CMUL=20; CMAD=21

@dataclass(frozen=True)
class Instruction:
    kind:Kind
    dst:int=-1
    a:int=-1
    b:int=-1
    c:int=-1
    bytes:int=4
    lanes:int=32
    flags:int=0

@dataclass(frozen=True)
class Phase:
    instructions:tuple[Instruction,...]
    iterations:int=1
    live:int=1
    footprint:int=0

@dataclass(frozen=True)
class Program:
    phases:tuple[Phase,...]
    threads:int
    group:int
    shared:int
    registers:int
    commands:int=1
    # Path alternatives are complete alternatives for declared input facts.
    alternatives:tuple=()

@dataclass(frozen=True)
class Kernel:
    name:str
    # Closed constructors: chains, mixed, stream, chase, sync, matrix, atomic,
    # strict, noop. Parameters describe execution structure, not observed costs.
    constructor:str
    parameters:dict

SIGNATURE='''device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
constant P& p [[buffer(2)]], threadgroup float* sm [[threadgroup(0)]],
uint id [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
uint lane [[thread_index_in_simdgroup]], uint gid [[threadgroup_position_in_grid]],
uint nt [[threads_per_threadgroup]]'''

class Lowering:
    def __init__(self,kernel,case,helpers=None):
        self.k=kernel;self.c=case;self.helpers=helpers;self.source=[];self.phases=[]
    def phase(self,code,instructions,it=1,live=1,footprint=0):
        self.source.append(code);self.phases.append(Phase(tuple(instructions),it,live,footprint))
    def lower(self):
        getattr(self,'lower_'+self.k.constructor)()
        registers=max(p.live for p in self.phases)
        # Fixed-probe reflection marks the unused binding inactive. The paired
        # reservation sweep also shows no allocation effect in that case.
        uses_shared=any(i.kind in [Kind.SHLOAD,Kind.SHSTORE] for ph in self.phases for i in ph.instructions)
        p=Program(tuple(self.phases),self.c['threads'],self.c['tg'],self.c['shared'] if uses_shared else 0,registers,self.c.get('commands',1))
        return 'kernel void '+self.k.name+'('+SIGNATURE+') {'+''.join(self.source)+'}\n',p
    def lower_chains(self):
        c=self.c;k=self.k.parameters;chains=k['chains'];block=k.get('block',chains);op=k.get('operation','fma')
        it=c['it'];footprint=c['threads']*chains*4
        isint=op in ['alu','mul','wide','cmul'];ty='ulong' if op=='wide' else ('uint' if isint else 'float')
        self.source.append(ty+' sum=0;')
        for start in range(0,chains,block):
            n=min(block,chains-start)
            body='{'+''.join(f'{ty} v{i}='+ (f'{ty}(id+{start+i}+1)' if isint else f'x[id+{start+i}*p.n]')+';' for i in range(n))
            initial=[] if isint else [Instruction(Kind.LOAD,i,bytes=4) for i in range(n)]
            self.phase(body,initial or [Instruction(Kind.NOP)],live=n,footprint=footprint)
            instructions=[];statements=[]
            for _ in range(k.get('unroll',8)):
                for i in range(n):
                    if op=='fma':statements.append(f'v{i}=fma(v{i},p.a,p.b);');instructions.append(Instruction(Kind.FMA,i,i))
                    elif op=='alu':
                        statements.append(f'v{i}=(v{i}+p.mode)^(v{i}>>13);')
                        instructions.extend([Instruction(Kind.ALU,n,i),Instruction(Kind.ALU,i,i),Instruction(Kind.ALU,i,i,n)])
                    elif op=='mul':
                        statements.append(f'v{i}=v{i}*p.mode;')
                        # Fixed calibration AIR proves invariant mode**unroll
                        # is hoisted; one multiply per chain remains per loop.
                        if _==0:instructions.append(Instruction(Kind.MUL,i,i))
                    elif op=='wide':statements.append(f'v{i}=(v{i}*ulong(p.mode))^(v{i}>>33);');instructions.extend([Instruction(Kind.WIDE,i,i),Instruction(Kind.WIDE,i,i),Instruction(Kind.WIDE,i,i)])
                    elif op=='cmul':
                        statements.append(f'v{i}=(v{i}*1664525u)^(v{i}>>13);')
                        instructions.extend([Instruction(Kind.CMUL,n,i),Instruction(Kind.ALU,i,i),Instruction(Kind.ALU,i,i,n)])
                    elif op=='sfu':statements.append(f'v{i}=metal::precise::rsqrt(v{i}+1.0f);');instructions.extend([Instruction(Kind.ALU,i,i),Instruction(Kind.SFU,i,i)])
                    else:raise ValueError(op)
            instructions.append(Instruction(Kind.CONTROL))
            self.phase('for(uint j=0;j<p.it;j++){'+''.join(statements)+'}',instructions,it,live=n+(op=='alu'),footprint=footprint)
            self.phase(''.join(f'sum+=v{i};' for i in range(n))+'}',[Instruction(Kind.ALU,n+1,n+1,i) for i in range(n)],live=n+2,footprint=footprint)
        out='reinterpret_cast<device uint*>(y)[id]=uint(sum);' if isint else 'y[id]=sum;'
        self.phase(out,[Instruction(Kind.STORE,a=block+1)],footprint=footprint)
    def lower_mixed(self):
        c=self.c;k=self.k.parameters;f=k['f'];u=k['u'];ch=k.get('chains',4)
        self.phase(''.join(f'float v{i}=x[id+{i}*p.n];uint q{i}=id+{i}+1;' for i in range(ch)),[Instruction(Kind.LOAD,i) for i in range(ch)],live=2*ch,footprint=ch*c['threads']*4)
        body=[];inst=[]
        for j in range(max(f,u)):
            i=j%ch
            if j<f:body.append(f'v{i}=fma(v{i},p.a,p.b);');inst.append(Instruction(Kind.FMA,i,i))
            if j<u:
                body.append(f'q{i}=(q{i}*1664525u+1013904223u)^(q{i}>>13);')
                inst.extend([Instruction(Kind.CMAD,2*ch,ch+i),Instruction(Kind.ALU,2*ch+1,ch+i),Instruction(Kind.ALU,ch+i,2*ch,2*ch+1)])
        inst.append(Instruction(Kind.CONTROL))
        self.phase('for(uint j=0;j<p.it;j++){'+''.join(body)+'}',inst,c['it'],live=2*ch+2)
        self.phase('y[id]='+'+'.join(f'v{i}' for i in range(ch))+';reinterpret_cast<device uint*>(y)[id+p.n]='+'^'.join(f'q{i}' for i in range(ch))+';',[Instruction(Kind.STORE,a=0),Instruction(Kind.STORE,a=ch)],live=2*ch+2)
    def lower_blocked(self):
        # One fixed compiled realization per block width. Problem width and
        # iteration count are runtime data, so candidates cannot change native
        # allocation merely by changing source size or unroll decisions.
        c=self.c;block=self.k.parameters['block'];count=c['chains'];parts=math.ceil(count/block)
        self.source.append('float sum=0;for(uint base=0;base<p.mode;base+='+str(block)+'){')
        initial=''.join(f'float v{i}=base+{i}<p.mode?x[id+(base+{i})*p.n]:0.0f;' for i in range(block))
        body='for(uint j=0;j<p.it;j++){'+''.join(f'v{i}=fma(v{i},p.a,p.b);' for _ in range(8) for i in range(block))+'}'
        ending=''.join(f'if(base+{i}<p.mode)sum+=v{i};' for i in range(block))+'}'
        footprint=c['threads']*count*4
        for part in range(parts):
            self.phase(initial if part==0 else '',[Instruction(Kind.LOAD,i) for i in range(block)],live=block+4,footprint=footprint)
            self.phase(body if part==0 else '',[Instruction(Kind.FMA,i,i) for _ in range(8) for i in range(block)]+[Instruction(Kind.CONTROL)],c['it'],live=block+4,footprint=footprint)
            self.phase(ending if part==0 else '',[Instruction(Kind.ALU,block+1,block+1,i) for i in range(block)]+[Instruction(Kind.CONTROL)],live=block+4,footprint=footprint)
        self.phase('y[id]=sum;',[Instruction(Kind.STORE,a=block+1)],live=block+4,footprint=footprint)
    def lower_stream(self):
        c=self.c;k=self.k.parameters;ops=k.get('operations',['fma']*k.get('intensity',0));unroll=k.get('unroll',1)
        stride=c['stride'];footprint=c['n']*4*(stride+1)
        body='for(uint j=id;j<p.n;j+=p.groups){float v=x[j*p.stride];'
        inst=[Instruction(Kind.LOAD,0,bytes=4*min(stride,32),lanes=32)]
        for op in ops:
            if op=='fma':body+='v=fma(v,p.a,p.b);';inst.append(Instruction(Kind.FMA,0,0))
            elif op=='mul':body+='v=v*p.a;';inst.append(Instruction(Kind.FMA,0,0))
            elif op=='sfu':body+='v=metal::precise::rsqrt(v+1.0f);';inst.extend([Instruction(Kind.ALU,0,0),Instruction(Kind.SFU,0,0)])
            else:raise ValueError(op)
        body+='y[j]=v;}'
        inst.extend([Instruction(Kind.STORE,a=0),Instruction(Kind.CONTROL)])
        self.phase(body,inst,math.ceil(c['n']/c['threads']),live=3,footprint=footprint)
    def lower_chase(self):
        c=self.c
        self.phase('uint q=id&(p.n-1);device const uint* z=reinterpret_cast<device const uint*>(x);',[Instruction(Kind.ALU,0)],live=2)
        self.phase('for(uint j=0;j<p.it;j++)q=z[q];',[Instruction(Kind.LOAD,0,0,lanes=32,flags=1),Instruction(Kind.CONTROL)],c['it'],live=2,footprint=c['n']*4)
        self.phase('reinterpret_cast<device uint*>(y)[id]=q;',[Instruction(Kind.STORE,a=0)],live=2)
    def lower_sync(self):
        c=self.c;op=self.k.parameters['operation']
        self.phase('float v=x[id];',[Instruction(Kind.LOAD,0)],live=2,footprint=c['threads']*4)
        if op=='barrier':body='sm[tid]=v;threadgroup_barrier(mem_flags::mem_threadgroup);v=sm[(tid+1)%nt];threadgroup_barrier(mem_flags::mem_threadgroup);';inst=[Instruction(Kind.SHSTORE,a=0),Instruction(Kind.BARRIER),Instruction(Kind.SHLOAD,0),Instruction(Kind.BARRIER)]
        elif op=='shuffle':body='v=simd_shuffle(v,(lane+1)%32);';inst=[Instruction(Kind.SHUFFLE,0,0)]
        elif op=='reduce':body='v=simd_sum(v)*0.03125f;';inst=[Instruction(Kind.REDUCE,0,0),Instruction(Kind.FMA,0,0)]
        else:raise ValueError(op)
        self.phase('for(uint j=0;j<p.it;j++){'+body+'}',inst+[Instruction(Kind.CONTROL)],c['it'],live=2)
        self.phase('y[id]=v;',[Instruction(Kind.STORE,a=0)],live=2)
    def lower_matrix(self):
        c=self.c;staged=self.k.parameters.get('staged',False)
        self.phase('simdgroup_float8x8 a,b,c;simdgroup_load(a,x+gid*128,8);simdgroup_load(b,x+gid*128+64,8);c=make_filled_simdgroup_matrix<float,8,8>(0.0f);',[Instruction(Kind.LOAD,0,bytes=8),Instruction(Kind.LOAD,1,bytes=8)],live=8,footprint=c['n']*4)
        body='';inst=[]
        if staged:
            body+='for(uint k=tid;k<128;k+=32)sm[k]=x[gid*128+k];threadgroup_barrier(mem_flags::mem_threadgroup);simdgroup_load(a,sm,8);simdgroup_load(b,sm+64,8);'
            inst.extend([Instruction(Kind.LOAD,3,bytes=16),Instruction(Kind.SHSTORE,a=3,bytes=16),Instruction(Kind.BARRIER),Instruction(Kind.SHLOAD,0,bytes=8),Instruction(Kind.SHLOAD,1,bytes=8)])
        body+='simdgroup_multiply_accumulate(c,a,b,c);';inst.append(Instruction(Kind.MMA,2,0,1,2))
        if staged:body+='threadgroup_barrier(mem_flags::mem_threadgroup);';inst.append(Instruction(Kind.BARRIER))
        self.phase('for(uint j=0;j<p.it;j++){'+body+'}',inst+[Instruction(Kind.CONTROL)],c['it'],live=8,footprint=c['n']*4)
        self.phase('simdgroup_store(c,y+gid*64,8);',[Instruction(Kind.STORE,a=2,bytes=8)],live=8,footprint=c['n']*4)
    def lower_atomic(self):
        c=self.c;cas=self.k.parameters.get('cas',False)
        operation=self.k.parameters.get('operation')
        if operation in ['dependent_failure','dependent_success','dependent_load']:
            expected='0xffffffffu' if operation=='dependent_failure' else '0u'
            body='device atomic_uint* z=reinterpret_cast<device atomic_uint*>(y);uint q=id&(p.n-1);for(uint j=0;j<p.it;j++){uint expected='+expected+';atomic_compare_exchange_weak_explicit(z+q,&expected,0u,memory_order_relaxed,memory_order_relaxed);q=expected&(p.n-1);}reinterpret_cast<device uint*>(y)[p.n+id]=q;'
            if operation=='dependent_load':body=body.replace('uint expected=0u;atomic_compare_exchange_weak_explicit(z+q,&expected,0u,memory_order_relaxed,memory_order_relaxed);','uint expected=atomic_load_explicit(z+q,memory_order_relaxed);')
            self.phase(body,[Instruction(Kind.CAS,0,0),Instruction(Kind.ALU,0,0),Instruction(Kind.CONTROL)],c['it'],live=4,footprint=c['n']*4)
            return
        counted=self.k.parameters.get('counted',False)
        body='device atomic_uint* z=reinterpret_cast<device atomic_uint*>(y)+(id%p.n);uint attempts=0;for(uint j=0;j<p.it;j++){'
        operation=self.k.parameters.get('operation')
        if operation in ['failure','success']:
            expected='0xffffffffu' if operation=='failure' else '0u'
            body+='uint expected='+expected+';atomic_compare_exchange_weak_explicit(z,&expected,0u,memory_order_relaxed,memory_order_relaxed);'
        elif cas:body+='uint expected=atomic_load_explicit(z,memory_order_relaxed);while(true){'+('attempts++;' if counted else '')+'uint desired=as_type<uint>(f32_add(as_type<float>(expected),1.0f));if(atomic_compare_exchange_weak_explicit(z,&expected,desired,memory_order_relaxed,memory_order_relaxed))break;}'
        else:body+='atomic_fetch_add_explicit(z,1u,memory_order_relaxed);'
        self.phase(body+'}'+('reinterpret_cast<device uint*>(y)[p.n+id]=attempts;' if counted else ''),[Instruction(Kind.CAS if cas else Kind.ATOMIC,0,0),Instruction(Kind.CONTROL)],c['it'],live=8,footprint=c['n']*4)
    def lower_strict(self):
        c=self.c;op=self.k.parameters['operation'];expr={'add':'f32_add(a,b)','mul':'f32_mul(a,b)','div':'f32_div(a,b)','rem':'f32_rem(a,b)','fma':'f32_mulAdd(a,b,a)','cmp':'float(f32_lt(a,b))','convert':'f16_to_f32(f32_to_f16(a))'}[op]
        body='uint h=0;float last=0;for(uint j=0;j<p.it;j++){uint k=(id+j*p.groups)&(p.n-1);float a=x[k],b=x[k+p.n];float v='+expr+';last=v;h=(h*1664525u)^as_type<uint>(v);}reinterpret_cast<device uint*>(y)[id]=h;y[id+p.groups]=last;'
        if self.helpers is None:
            # Source generation may precede helper characterization. Prediction
            # requires the closed helper expansion and never accepts this shell.
            self.source.append(body);self.phases.append(Phase((Instruction(Kind.NOP),)))
        else:
            instructions,live=self.helpers.expand(op,c)
            self.phase(body,instructions+[Instruction(Kind.CMUL,live,live),Instruction(Kind.ALU,live,live,0),Instruction(Kind.CONTROL)],c['it'],live=live+1,footprint=c['n']*8)
    def lower_noop(self):
        self.phase('y[id]=x[id];',[Instruction(Kind.LOAD,0),Instruction(Kind.STORE,a=0)],live=1,footprint=self.c['n']*4)

def describe(name,constructor,parameters,**case):
    defaults=dict(threads=8192,tg=128,n=8192,it=128,stride=1,mode=1664525,shared=16,a=.99991,b=.00001)
    defaults.update(case)
    defaults.update(kernel=name,constructor=constructor,parameters=parameters)
    return defaults

def lower(case,helpers=None):return Lowering(Kernel(case['kernel'],case['constructor'],case['parameters']),case,helpers).lower()

def library_prelude():
    root=Path(__file__).resolve().parents[3]
    render=(root/'seismic/backends/metal/src/render.rs').read_text()
    source=re.search(r'const LIBRARY_PRELUDE: &str = r#"(.*?)"#;',render,re.S).group(1)
    source+=(root/'seismic/backends/metal/src/softfloat.metal').read_text()
    for name in ['f32_add','f32_mul','f32_div','f32_rem','f32_mulAdd','f32_lt','f32_to_f16','f16_to_f32']:
        source=re.sub(r'((?:inline )?(?:float32_t|float16_t|float|bool)\s+)('+name+r'\s*\()',r'__attribute__((noinline)) \1\2',source)
    source+='\n#include <metal_simdgroup_matrix>\nstruct P{uint n,it,stride,mode;float a,b;uint groups,pad;};\n'
    return source

def emit_manifest(cases,path):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);names=set();source=library_prelude()
    warm=describe('fma_c16','chains',{'chains':16},threads=65536,n=65536,it=256)
    for c in [warm]+cases:
        if c['kernel'] not in names:source+=lower(c)[0];names.add(c['kernel'])
    (path/'probes.metal').write_text(source);(path/'manifest.json').write_text(json.dumps(cases,indent=2))
    (path/'identity.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'kernels':len(names),'cases':len(cases),'version':'event-model-v3'},indent=2))
