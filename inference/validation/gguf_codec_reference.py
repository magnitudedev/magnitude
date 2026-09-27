#!/usr/bin/env python3
"""Fixture from the exact V3 GGUFCodec methods, executed with NumPy scalars, and
from llama.cpp's reference dequantization of the formats V3 lacked.

`--source` (the V3 tree) adds the V3 GGUFCodec cases. The llama.cpp cases need
no source: they port `dequantize_row_q4_0`, `_q5_0`, `_q5_1`, `_mxfp4` and
`_nvfp4` of ggml-quants.c at llama.cpp revision 18443257a30c (with
`ggml_e8m0_to_fp32_half` and `ggml_ue4m3_to_fp32` of ggml-impl.h) to NumPy
float32 arithmetic. `--digests` prints each llama.cpp case's FNV-1a digest of
its f32 value bits, the constants `engine/kernels/tests/import_residency.rs`
checks the engine's import against. `--gguf FILE --dump DIR` dumps real
tensors of those formats with their reference values (needs gguf-py)."""
import argparse,ast,hashlib,json,struct
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
import numpy as np

# ---------------------------------------------------------------------------
# llama.cpp reference dequantization (ggml-quants.c, revision 18443257a30c).

LLAMA_REVISION = "18443257a30c"
# kvalues_mxfp4 (= kvalues_fp4): E2M1 values doubled.
KVALUES_FP4 = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)
F32 = np.float32


def f16(block, at):
    return F32(np.frombuffer(bytes(block[at:at + 2]), dtype="<f2")[0])


def e8m0_to_fp32_half(x):
    # x < 2: the denormal patterns 0x00200000 << x; otherwise (x - 1) << 23.
    bits = (0x00200000 << x) if x < 2 else ((x - 1) << 23)
    return np.frombuffer(struct.pack("<I", bits), dtype="<f4")[0]


def ue4m3_to_fp32(x):
    # Half the UE4M3 value (the kvalues are doubled); 0 and 0x7f are zero.
    if x == 0 or x == 0x7F:
        return F32(0.0)
    exponent, mantissa = (x >> 3) & 0xF, x & 0x7
    raw = F32(np.ldexp(np.float64(mantissa), -9)) if exponent == 0 else \
        F32(np.ldexp(1.0 + mantissa / 8.0, exponent - 7))
    return F32(raw * F32(0.5))


def dequantize_q4_0(block):
    d, qs = f16(block, 0), block[2:18]
    low = [F32(int(q & 0xF) - 8) * d for q in qs]
    high = [F32(int(q >> 4) - 8) * d for q in qs]
    return low + high


def dequantize_q5(block, minimum):
    d = f16(block, 0)
    m = f16(block, 2) if minimum else None
    base = 4 if minimum else 2
    qh = int.from_bytes(bytes(block[base:base + 4]), "little")
    qs = block[base + 4:base + 20]
    out = [F32(0.0)] * 32
    for j in range(16):
        xh_0 = ((qh >> j) << 4) & 0x10
        xh_1 = (qh >> (j + 12)) & 0x10
        x0 = (int(qs[j]) & 0x0F) | xh_0
        x1 = (int(qs[j]) >> 4) | xh_1
        if minimum:
            # `x0*d + m`, which C compilers contract to one fused rounding:
            # exact in float64, then rounded once.
            out[j] = F32(np.float64(x0) * np.float64(d) + np.float64(m))
            out[j + 16] = F32(np.float64(x1) * np.float64(d) + np.float64(m))
        else:
            out[j] = F32(x0 - 16) * d
            out[j + 16] = F32(x1 - 16) * d
    return out


def dequantize_mxfp4(block):
    d, qs = e8m0_to_fp32_half(int(block[0])), block[1:17]
    low = [F32(KVALUES_FP4[q & 0xF]) * d for q in qs]
    high = [F32(KVALUES_FP4[q >> 4]) * d for q in qs]
    return low + high


def dequantize_nvfp4(block):
    out = []
    for s in range(4):
        d = ue4m3_to_fp32(int(block[s]))
        qs = block[4 + 8 * s:4 + 8 * s + 8]
        out += [F32(KVALUES_FP4[q & 0xF]) * d for q in qs]
        out += [F32(KVALUES_FP4[q >> 4]) * d for q in qs]
    return out


def golden_bytes(length, seed):
    """Byte i is the top byte of (i * 2654435761 + seed) mod 2^32."""
    index = np.arange(length, dtype=np.uint64)
    return ((index * 2654435761 + seed) % (1 << 32) >> 24).astype(np.uint8)


def sanitized(name, block):
    """Finite scale fields: f16 exponents below 31 and no E8M0/UE4M3 NaN code
    (the engine decodes NaN codes as NaN, llama.cpp as numbers)."""
    block = block.copy()
    if name in ("q4_0", "q5_0", "q5_1"):
        block[1] &= 0xBF
    if name == "q5_1":
        block[3] &= 0xBF
    if name == "mxfp4" and block[0] == 0xFF:
        block[0] = 0xFE
    if name == "nvfp4":
        for s in range(4):
            if block[s] & 0x7F == 0x7F:
                block[s] -= 1
    return block


LLAMA_FORMATS = (
    # (name, GGUF type, block bytes, block values, dequantize)
    ("q4_0", 2, 18, 32, dequantize_q4_0),
    ("q5_0", 6, 22, 32, lambda block: dequantize_q5(block, False)),
    ("q5_1", 7, 24, 32, lambda block: dequantize_q5(block, True)),
    ("mxfp4", 39, 17, 32, dequantize_mxfp4),
    ("nvfp4", 40, 36, 64, dequantize_nvfp4),
)
BLOCKS = 64


def fnv1a(values):
    digest = 0xCBF29CE484222325
    for value in values:
        bits = int(np.array(value, dtype="<f4").view("<u4"))
        digest = ((digest ^ bits) * 0x100000001B3) % (1 << 64)
    return digest


def llama_cases():
    cases = []
    for name, encoding, block_bytes, block_values, dequantize in LLAMA_FORMATS:
        raw = golden_bytes(BLOCKS * block_bytes, 12345)
        blocks = [sanitized(name, raw[b * block_bytes:(b + 1) * block_bytes]) for b in range(BLOCKS)]
        values = [value for block in blocks for value in dequantize(block)]
        assert len(values) == BLOCKS * block_values
        cases.append({
            "encoding": encoding, "name": name.upper(), "shape": [1, BLOCKS * block_values],
            "source_hex": b"".join(bytes(block) for block in blocks).hex(),
            "values": [float(value) for value in values], "fnv1a_f32": f"{fnv1a(values):#018x}",
        })
    return cases


def v3_cases(source):
    path=source/'src/engine/weights/formats/gguf.py'
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
    return records,hashlib.sha256(path.read_bytes()).hexdigest()


def dump_real_tensors(gguf_path, directory, rows):
    """For the first tensor of each llama.cpp format in `gguf_path`, write its
    first `rows` rows as raw GGUF blocks (`<stem>.blocks`), their reference
    dequantization as little-endian f32 (`<stem>.values`) and an index
    (`index.tsv`: source, rows, K, stem per line), which
    `engine/kernels/tests/import_residency.rs` reads. The reference is this
    file's port, checked against gguf-py `quants.dequantize`."""
    from gguf import GGUFReader, quants
    formats = {encoding: (name, block_bytes, block_values, dequantize)
               for name, encoding, block_bytes, block_values, dequantize in LLAMA_FORMATS}
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    seen = set()
    for tensor in GGUFReader(gguf_path).tensors:
        encoding = int(tensor.tensor_type)
        if encoding not in formats or encoding in seen:
            continue
        seen.add(encoding)
        name, block_bytes, block_values, dequantize = formats[encoding]
        k = int(tensor.shape[0])
        blocks = np.asarray(tensor.data, dtype=np.uint8).reshape(-1, k // block_values * block_bytes)[:rows]
        values = np.array([v for block in blocks.reshape(-1, block_bytes) for v in dequantize(block)],
                          dtype="<f4").reshape(len(blocks), k)
        independent = quants.dequantize(blocks, tensor.tensor_type).astype("<f4").reshape(len(blocks), k)
        mismatches = int(np.count_nonzero(values.view("<u4") != independent.view("<u4")))
        stem = f"{name}.{tensor.name}"
        (directory / f"{stem}.blocks").write_bytes(blocks.tobytes())
        (directory / f"{stem}.values").write_bytes(values.tobytes())
        index.append({"tensor": tensor.name, "format": name, "source": f"gguf_{name}",
                      "shape": [len(blocks), k], "stem": stem, "gguf_py_mismatches": mismatches})
        print(f"{tensor.name}: {name} [{len(blocks)}, {k}] gguf-py mismatches {mismatches}")
    (directory / "index.tsv").write_text("".join(
        f"{entry['source']}\t{entry['shape'][0]}\t{entry['shape'][1]}\t{entry['stem']}\n" for entry in index))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,help='the V3 tree, for the V3 GGUFCodec cases')
    p.add_argument('--output',type=Path)
    p.add_argument('--digests',action='store_true',help='print the llama.cpp cases\' digests')
    p.add_argument('--gguf',type=Path,help='a GGUF file whose real tensors to dump (with --dump)')
    p.add_argument('--dump',type=Path,help='the directory of the real-tensor dump')
    p.add_argument('--rows',type=int,default=64,help='rows dumped per tensor')
    args=p.parse_args()
    if args.gguf is not None:
        dump_real_tensors(args.gguf,args.dump,args.rows)
        return
    llama=llama_cases()
    if args.digests:
        for case in llama:
            print(f"{case['name']:6} {case['fnv1a_f32']}")
    if args.output is None:
        return
    record={'reference':'Exact V3 GGUFCodec class methods with NumPy values; decoded result rounded once to f32',
            'llama_reference':f'llama.cpp {LLAMA_REVISION} ggml-quants.c dequantize_row_* in NumPy float32',
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'numpy_version':np.__version__,'llama_cases':llama}
    if args.source is not None:
        record['cases'],record['source_sha256']=v3_cases(args.source)
    args.output.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
