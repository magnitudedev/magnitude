"""Behavioral checks for execution semantics, independent of measured timings."""
import ctypes as C
import unittest
from dataclasses import replace
from pathlib import Path
from machine import Machine,Ins,Phase,Launch,DEFAULT,NAMES
from helpers import Helpers

class SchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=Machine()
    def run_program(self,code,iterations=100,threads=128,group=128,shared=16,registers=8,p=None):
        ins=(Ins*len(code))(*[Ins(*i,4,32,0) for i in code]);ph=(Phase*1)(Phase(0,len(code),iterations,registers,4096));la=Launch(threads,group,shared,registers,1,4,32,65536)
        p=p or DEFAULT
        return self.m.lib.simulate(ins,ph,1,C.byref(la),(C.c_double*len(p))(*p))
    def test_raw_dependency_limits_parallelism(self):
        serial=self.run_program([(0,0,0,-1,-1)]*8)
        parallel=self.run_program([(0,i,i,-1,-1) for i in range(8)])
        self.assertEqual(serial.issued,parallel.issued)
        self.assertGreater(serial.ns,parallel.ns)
    def test_barriers_complete_all_cohorts(self):
        r=self.run_program([(0,0,0,-1,-1),(10,-1,-1,-1,-1),(0,0,0,-1,-1)],iterations=17,threads=1024,shared=32768)
        self.assertEqual(r.issued,32*17*3)
        self.assertEqual(r.barriers,32*17)
        self.assertEqual(r.resident_groups,1)
    def test_spills_count_actual_register_uses(self):
        p=list(DEFAULT);p[NAMES.index('usable_regs')]=4
        r=self.run_program([(0,4,4,-1,-1)],iterations=11,registers=5,p=p)
        self.assertEqual(r.spill_loads,44);self.assertEqual(r.spill_stores,44)
    def test_cas_conserves_updates(self):
        for destinations in [1,4,32,128]:
            r=self.m.lib.simulate_cas(128,destinations,3,4,50,10,20,1,.1,.01,1,.1,.01,20)
            self.assertEqual(r.successes,384);self.assertGreaterEqual(r.attempts,r.successes)
            self.assertGreater(r.ns,0)
    def test_uncontended_cas_has_no_retries(self):
        r=self.m.lib.simulate_cas(128,128,3,4,50,10,20,1,.1,.01,1,.1,.01,20)
        self.assertEqual(r.attempts,384)
    def test_prediction_is_immutable_and_replayable(self):
        from program import describe
        c=describe('atomic','atomic',dict(cas=False),threads=128,n=128,it=4)
        first=self.m.predict(c);expected=first['ns'];first['ns']=-1
        self.assertEqual(self.m.predict(c)['ns'],expected)
    def test_inactive_shared_binding_does_not_limit_residency(self):
        from program import describe,lower
        _,plain=lower(describe('plain','chains',dict(chains=16),shared=32768))
        _,exchange=lower(describe('exchange','sync',dict(operation='barrier'),shared=32768))
        self.assertEqual(plain.shared,0);self.assertEqual(exchange.shared,32768)

class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.h=Helpers()
    def test_native_reference_and_dynamic_path(self):
        import numpy as np
        lib=C.CDLL(None);lib.fmaf.argtypes=[C.c_float]*3;lib.fmaf.restype=C.c_float
        lib.remainderf.argtypes=[C.c_float]*2;lib.remainderf.restype=C.c_float
        def f(bits):return C.cast(C.pointer(C.c_uint32(bits)),C.POINTER(C.c_float))[0]
        for mode in range(9):
            for i in [0,1,2,3,31,32,63,64,95,96,127]:
                a,b=self.h.inputs(mode,i);x,y=f(a),f(b)
                with np.errstate(all='ignore'):
                    expected=dict(add=float(np.float32(x)+np.float32(y)),mul=float(np.float32(x)*np.float32(y)),div=float(np.float32(x)/np.float32(y)),rem=lib.remainderf(x,y),fma=lib.fmaf(x,y,x),cmp=float(x<y),convert=float(np.float16(x)))
                for op,want in expected.items():
                    trace,actual=self.h.scalar(op,a,b,a)
                    self.assertTrue(trace)
                    self.assertTrue(actual==want or np.isnan(actual) and np.isnan(want),(mode,i,op,actual,want))
    def test_lane_divergence_expands_executed_work(self):
        mixed,_=self.h.expand('fma',dict(mode=7));coherent,_=self.h.expand('fma',dict(mode=8))
        self.assertGreater(len(mixed),len(coherent))

if __name__=='__main__':unittest.main()
