"""Reproducible fixed helper compilation inputs and AIR extraction."""
from pathlib import Path
import sys,struct,json,hashlib
from program import library_prelude

ROOT=Path(__file__).parent/'results'
def prepare():
    path=ROOT/'helpers';path.mkdir(exist_ok=True)
    source=library_prelude()
    for op,expression in dict(add='f32_add(a,b)',mul='f32_mul(a,b)',div='f32_div(a,b)',rem='f32_rem(a,b)',fma='f32_mulAdd(a,b,c)',cmp='float(f32_lt(a,b))',convert='f16_to_f32(f32_to_f16(a))').items():
        source+=f'\nkernel void helper_{op}(device const float* x [[buffer(0)]],device float* y [[buffer(1)]],uint id [[thread_position_in_grid]]){{float a=x[3*id],b=x[3*id+1],c=x[3*id+2];y[id]={expression};}}\n'
    (path/'probes.metal').write_text(source)
    (path/'identity.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest())))
def extract():
    import llvmlite.binding as llvm
    for p in (ROOT/'archives').glob('*.bin'):
        b=p.read_bytes();i=b.index(bytes.fromhex('dec0170b'));_,_,off,size,_=struct.unpack_from('<5I',b,i);bc=b[i+off:i+off+size]
        mod=llvm.parse_bitcode(bc);p.with_suffix('.bc').write_bytes(bc);p.with_suffix('.ll').write_text(str(mod))
if __name__=='__main__':prepare() if sys.argv[1]=='prepare' else extract()
