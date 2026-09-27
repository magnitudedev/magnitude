"""Pure evaluator: executable structure and a frozen device profile only."""
from pathlib import Path
import ctypes as C
import json
from program import lower

NAMES='dispatch warp_issue front_issue fp_issue fp_latency int_issue int_latency mul_issue mul_latency wide_issue wide_latency sfu_issue sfu_latency memory_issue l1_latency l2_latency dram_latency l1_bw l2_bw dram_bw l1_size l2_size shared_issue shared_latency barrier_latency shuffle_issue shuffle_latency mma_issue mma_latency atomic_issue atomic_latency usable_regs loop_latency convert_issue convert_latency'.split()
DEFAULT=[650,.9,.16,.16,1.8,.16,1.2,.32,2.4,.64,3.6,1.28,8,.4,35,100,230,128,64,250,32768,4194304,.5,4,12,.4,2,1.2,8,.5,20,124,1.2,.32,2.4]
NAMES+='branch_latency call_latency cmul_issue cmul_latency cmad_issue cmad_latency'.split()
DEFAULT += [8,8,.32,2.4,.32,2.4]
NAMES+=['random_sector_bytes'];DEFAULT+=[32]
NAMES+=['fence_latency'];DEFAULT+=[4]
class Ins(C.Structure):_fields_=[(n,C.c_int32) for n in ['kind','dst','a','b','c']]+[(n,C.c_uint32) for n in ['bytes','lanes','flags']]
class Phase(C.Structure):_fields_=[(n,C.c_int32) for n in ['offset','count','iterations','live']]+[('footprint',C.c_uint64)]
class Launch(C.Structure):_fields_=[(n,C.c_int32) for n in ['threads','group','shared','registers','cores','partitions','max_waves','register_file']]
class Result(C.Structure):_fields_=[('ns',C.c_double)]+[(n,C.c_uint64) for n in ['issued','spill_loads','spill_stores','barriers']]+[('resident_groups',C.c_int32)]
class AtomicResult(C.Structure):_fields_=[('ns',C.c_double)]+[(n,C.c_uint64) for n in ['attempts','successes','rounds']]

class Machine:
    def __init__(self,helpers=None):
        self.helpers=helpers;self.cache={};self.atomic_cache={}
        self.lib=C.CDLL(str(Path(__file__).parent/'results/scheduler.dylib'))
        self.lib.simulate.argtypes=[C.POINTER(Ins),C.POINTER(Phase),C.c_int,C.POINTER(Launch),C.POINTER(C.c_double)]
        self.lib.simulate.restype=Result
        self.lib.simulate_cas.argtypes=[C.c_int]*4+[C.c_double]*10
        self.lib.simulate_cas.restype=AtomicResult
    def compile(self,case):
        key=json.dumps(case,sort_keys=True)
        if key not in self.cache:
            _,p=lower(case,self.helpers)
            if case['constructor']=='strict' and self.helpers is None:raise ValueError('Missing fixed helper expansion')
            instructions=[];phases=[]
            for ph in p.phases:
                phases.append(Phase(len(instructions),len(ph.instructions),ph.iterations,ph.live,ph.footprint))
                instructions.extend(Ins(int(i.kind),i.dst,i.a,i.b,i.c,i.bytes,i.lanes,i.flags) for i in ph.instructions)
            self.cache[key]=((Ins*len(instructions))(*instructions),(Phase*len(phases))(*phases),p)
        return self.cache[key]
    def predict(self,case,parameters=DEFAULT,hardware=None,atomic=None):
        if case['constructor']=='atomic':
            # Register-limit sensitivity cannot change this atomic submodel.
            # Memoize identical executions, returning a copy to keep purity.
            key=json.dumps([case,[float(v) for i,v in enumerate(parameters) if i!=NAMES.index('usable_regs')],atomic],sort_keys=True)
            if key not in self.atomic_cache:self.atomic_cache[key]=self.atomic(case,parameters,atomic)
            return dict(self.atomic_cache[key])
        if case['constructor']=='strict' and '_cohort' not in case:
            count=5 if case['mode'] in [5,6] else 4 if case['mode'] in [7,8] else 1
            results=[self.predict(dict(case,_cohort=i),parameters,hardware) for i in range(count)]
            return {**results[0],'ns':sum(r['ns'] for r in results)/count,'cohort_ns':[r['ns'] for r in results]}
        code,phases,p=self.compile(case)
        hw=hardware or dict(cores=20,partitions=4,max_waves=32,register_file=65536)
        launch=Launch(p.threads,p.group,p.shared,p.registers,**hw)
        result=self.lib.simulate(code,phases,len(phases),C.byref(launch),(C.c_double*len(parameters))(*parameters))
        return {**{n:getattr(result,n) for n,_ in Result._fields_},'ns':result.ns*p.commands}
    def atomic(self,c,p,atomic):
        a=atomic or dict(address_service=.75,line_service=.125,global_service=.05,latency=250)
        operation=c['parameters'].get('operation')
        if operation in ['dependent_failure','dependent_success','dependent_load']:
            return dict(ns=a[operation]['intercept_ns']+a[operation]['recurrence_ns']*c['it'],attempts=c['threads']*c['it'],successes=0,rounds=c['it'])
        if operation and operation in a:a=a[operation]
        n=c['n'];total=c['threads']*c['it'];dispatch=p[0]
        if not c['parameters']['cas']:
            ns=dispatch+a['latency']+max(total/n*a['address_service'],total/max(1,(n+31)//32)*a['line_service'],total*a['global_service'],c['it']*p[1])
            return dict(ns=ns,attempts=total,successes=total,rounds=0)
        if self.helpers is None:raise ValueError('CAS requires explicit strict-add expansion')
        instructions,live=self.helpers.expand('add',dict(mode=0))
        # AIR helper integer work and longest dependence path; input loads are
        # supplied by CAS, and constants here are known finite positive values.
        ready={};clock=0;service=0
        issue={0:3,1:5,2:7,3:9,4:11,9:5,16:5,17:33,18:5,19:5,20:37,21:39}
        latency={0:4,1:6,2:8,3:10,4:12,9:32,16:6,17:34,18:35,19:36,20:38,21:40}
        for i in instructions:
            if i.kind in [5,6]:continue
            start=max([clock]+[ready.get(r,0) for r in [i.a,i.b,i.c] if r>=0]);end=start+p[latency.get(int(i.kind),6)]
            if i.dst>=0:ready[i.dst]=end
            clock=start+p[1]
            if i.kind in [9,18,19]:clock=max(clock,end)
            service+=max(p[2],p[issue.get(int(i.kind),5)])
        critical=max([clock]+list(ready.values()))
        success=a.get('success',a);failure=a.get('failure',a)
        # Service sharing between returning/nonreturning atomic variants and
        # the coarse request queue are explicit, unqualified hypotheses.
        # A measured return latency alone does not certify either assumption.
        returning=max(a['dependent_failure']['recurrence_ns'],a['dependent_success']['recurrence_ns'])
        r=self.lib.simulate_cas(c['threads'],n,c['it'],20,critical,service,returning,success['address_service'],success['line_service'],success['global_service'],failure['address_service'],failure['line_service'],failure['global_service'],a['dependent_load']['recurrence_ns'])
        return dict(ns=dispatch+r.ns,attempts=r.attempts,successes=r.successes,rounds=r.rounds,helper_latency=critical,helper_service=service)
