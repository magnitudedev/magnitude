"""Fixed AIR helper descriptors, scalar path tracing, SIMT merge and allocation.

Only the fixed helper library is compiled/introspected. Native host execution
selects AIR paths, never supplies elapsed time to the estimator. AIR is not the
final GPU ISA: its correspondence is explicitly a qualification assumption.
"""
from pathlib import Path
import ctypes as C
import re,struct,json,difflib,collections
import llvmlite.binding as llvm
from program import Instruction,Kind

ROOT=Path(__file__).parent/'results'
class Helpers:
    def __init__(self):
        llvm.initialize_native_target();llvm.initialize_native_asmprinter()
        self.trace=[];self.modules={};self.cache={}
        self.callback=C.CFUNCTYPE(None,C.c_int)(lambda bid:self.trace.append(bid))
        llvm.add_symbol('record_block',C.cast(self.callback,C.c_void_p).value)
        for op in ['add','mul','div','rem','fma','cmp','convert']:self.load(op)
    def load(self,op):
        path=ROOT/'archives'/('helper_'+op+'.bc')
        if not path.exists():
            data=path.with_suffix('.bin').read_bytes();i=data.index(bytes.fromhex('dec0170b'));_,_,off,size,_=struct.unpack_from('<5I',data,i);path.write_bytes(data[i+off:i+off+size])
        mod=llvm.parse_bitcode(path.read_bytes());blocks={};functions={f.name:f for f in mod.functions};s=re.sub(r" #\d+","",str(mod))
        # Build from parsed functions to instrument after the phi prefix.
        for f in mod.functions:
            if f.is_declaration:continue
            original=re.sub(r" #\d+","",str(f));changed=original
            for b in f.blocks:
                bid=len(blocks);blocks[bid]=(f,b)
                instructions=list(b.instructions);first=next(i for i in instructions if i.opcode!='phi')
                bs=re.sub(r' #\d+','',str(b));fs=re.sub(r' #\d+','',str(first))
                changed=changed.replace(bs,bs.replace(fs,'  call void @record_block(i32 '+str(bid)+')\n'+fs,1),1)
            s=s.replace(original,changed)
        s=re.sub(r'target datalayout = .*\n','',s)
        s=re.sub(r'target triple = .*\n','',s)
        s=re.sub(r' addrspace\(\d+\)','',s)
        s=re.sub(r' #\d+','',s)
        s=re.sub(r'^attributes .*\n','',s,flags=re.M)
        s=s.replace('@air.clz.i32','@llvm.ctlz.i32')
        s+='\ndeclare void @record_block(i32)\n'
        s=s.replace('declare float @air.convert.f.f32.u.i1(i1) local_unnamed_addr','define float @air.convert.f.f32.u.i1(i1 %a) { %b = uitofp i1 %a to float\nret float %b\n}')
        (ROOT/('instrumented_'+op+'.ll')).write_text(s)
        native=llvm.parse_assembly(s);native.triple=llvm.get_default_triple();native.verify()
        tm=llvm.Target.from_default_triple().create_target_machine(opt=0)
        engine=llvm.create_mcjit_compiler(native,tm);engine.finalize_object()
        fn=C.CFUNCTYPE(None,C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_uint)(engine.get_function_address('helper_'+op))
        self.modules[op]=(mod,blocks,functions,engine,fn)
    def scalar(self,op,a,b,c):
        mod,blocks,functions,engine,fn=self.modules[op]
        inp=(C.c_uint32*3)(a,b,c);out=(C.c_float*1)();self.trace=[]
        fn(C.cast(inp,C.POINTER(C.c_float)),out,0)
        trace=collections.deque(self.trace);records=[];serial=0
        # SSA operands compare equal across the instruction and operand views.
        def execute(name,args,context):
            nonlocal serial
            func=functions[name];values={v:r for v,r in zip(func.arguments,args)};previous=None;occ=collections.Counter()
            while trace:
                bid=trace.popleft();f,block=blocks[bid]
                assert f.name==name,(f.name,name)
                visit=occ[block.name];occ[block.name]+=1
                for index,inst in enumerate(block.instructions):
                    operands=list(inst.operands);opcode=inst.opcode
                    key=(*context,name,block.name,visit,index)
                    def reg(v):return values.get(v,None)
                    if opcode=='phi':
                        preds=list(inst.incoming_blocks);chosen=next((v for v,p in zip(operands,preds) if p==previous),operands[0]);values[inst]=reg(chosen);continue
                    if opcode in ['bitcast','zext','sext','trunc','getelementptr']:
                        values[inst]=next((reg(v) for v in operands if reg(v) is not None),None);continue
                    if opcode=='ret':
                        result=reg(operands[0]) if operands else None
                        if not name.startswith('helper_'):records.append((key,Kind.CALL,None,() if result is None else (result,)))
                        return result
                    if opcode=='call':
                        target=operands[-1].name
                        if target in functions and not functions[target].is_declaration:
                            records.append((key+('call',),Kind.CALL,None,tuple(r for v in operands[:-1] if (r:=reg(v)) is not None)))
                            values[inst]=execute(target,[reg(v) for v in operands[:-1]],key);continue
                        kind=Kind.ALU if ('clz' in target or 'abs' in target or 'max' in target) else Kind.CONVERT
                    elif opcode in ['br','switch','unreachable']:kind=Kind.BRANCH
                    elif opcode=='load':kind=Kind.LOAD
                    elif opcode=='store':kind=Kind.STORE
                    elif opcode in ['fadd','fmul','fsub']:kind=Kind.FMA
                    elif opcode=='mul':kind=Kind.WIDE if str(inst.type)=='i64' else Kind.MUL
                    elif opcode in ['uitofp','sitofp','fptoui','fptosi','fptrunc','fpext']:kind=Kind.CONVERT
                    else:kind=Kind.WIDE if str(inst.type)=='i64' else Kind.ALU
                    deps=tuple(dict.fromkeys(r for v in operands if (r:=reg(v)) is not None))
                    result=None if str(inst.type)=='void' else key
                    # Wrapper load/store/index work is supplied by the kernel
                    # lowering. Its SSA definitions still seed helper arguments.
                    if name.startswith('helper_'):
                        if opcode=='load':values[inst]=('input',len([v for v in values.values() if v and v[0]=='input']))
                        else:values[inst]=result
                        continue
                    records.append((key,kind,result,deps));values[inst]=result
                previous=block
            raise RuntimeError('Incomplete helper trace')
        execute('helper_'+op,[None,None,None],())
        return records,out[0]
    @staticmethod
    def inputs(mode,i):
        normal=0x3f000000|((i*7919)&0x7fffff)
        if mode>=7:
            k=i%4 if mode==7 else (i//32)%4
            return [0x3fa00000,0x123,0,0x7f800000][k],[0x3fe00000,0x456,0,0x3f800000][k]
        k=mode if mode<5 else (i%5 if mode==5 else (i//32)%5)
        return [(normal,normal+0x100000),(normal,normal|0x80000000),(i%1023+1,(i*3)%1023+1),(0,0),(0x7fc00001 if i%2==0 else 0x7f800000,0x3f800000)][k]
    def expand(self,op,case):
        key=(op,case['mode'],case.get('_cohort',0))
        if key in self.cache:return self.cache[key]
        traces=[]
        for lane in range(32):
            a,b=self.inputs(case['mode'],case.get('_cohort',0)*32+lane)
            traces.append(self.scalar(op,a,b,a)[0])
        merged=traces[0]
        for trace in traces[1:]:
            matcher=difflib.SequenceMatcher(a=[r[0] for r in merged],b=[r[0] for r in trace],autojunk=False);new=[]
            for tag,a,b,c,d in matcher.get_opcodes():
                if tag=='equal':
                    for x,y in zip(merged[a:b],trace[c:d]):new.append((x[0],x[1],x[2],tuple(dict.fromkeys(x[3]+y[3]))))
                else:new.extend(merged[a:b]);new.extend(trace[c:d])
            merged=new
        # Linear-scan allocation after reconvergence merge. No native candidate
        # reflection is used. Phi-selected lane dependencies are conservative.
        last={}
        for i,(_,_,dst,deps) in enumerate(merged):
            for d in deps:last[d]=i
            if dst is not None:last.setdefault(dst,i)
        mapping={};free=list(range(3,512));maxlive=3;insts=[Instruction(Kind.LOAD,0),Instruction(Kind.LOAD,1)]
        for i,(_,kind,dst,deps) in enumerate(merged):
            for d in deps:
                if d not in mapping:
                    if d[0]=='input':mapping[d]=min(1,int(d[1]))
                    else:mapping[d]=free.pop(0)
            if dst is not None and dst not in mapping:mapping[dst]=free.pop(0)
            regs=[mapping[d] for d in deps]
            # More than three incoming phi alternatives are expressed as joins.
            while len(regs)>3:
                tmp=free[0];insts.append(Instruction(Kind.NOP,tmp,*regs[:3]));regs=[tmp]+regs[3:]
            insts.append(Instruction(kind,mapping.get(dst,-1),*(regs+[-1]*3)[:3],flags=2 if kind==Kind.LOAD else 0))
            maxlive=max(maxlive,max(mapping.values(),default=0)+1)
            for d in list(mapping):
                if last.get(d,-1)<=i and mapping[d]>=3:free.append(mapping.pop(d))
            free.sort()
        self.cache[key]=(insts,maxlive)
        return self.cache[key]
    def export(self,path):
        payload={}
        for op in self.modules:
            for mode in range(9):
                for cohort in range(5):
                    ins,live=self.expand(op,dict(mode=mode,_cohort=cohort))
                    payload[f'{op}/{mode}/{cohort}']={'instructions':[[int(i.kind),i.dst,i.a,i.b,i.c,i.bytes,i.lanes,i.flags] for i in ins],'live':live}
        Path(path).write_text(json.dumps(payload))

class FrozenHelpers:
    def __init__(self,path):self.data=json.loads(Path(path).read_text())
    def expand(self,op,case):
        d=self.data[f"{op}/{case['mode']}/{case.get('_cohort',0)}"]
        return [Instruction(Kind(i[0]),*i[1:]) for i in d['instructions']],d['live']

if __name__=='__main__':
    h=Helpers();h.export(ROOT/'helpers.json');print('Exported fixed helper instruction paths')
