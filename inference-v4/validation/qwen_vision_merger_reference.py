"""Independent Python math.erf oracle and V3 merger equation transcription."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--erf', action='store_true')
args = parser.parse_args()
source = args.source.resolve(strict=True)
provenance = {
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_sha256': {'src/engine/models/qwen35/vision.py': hashlib.sha256(
        (source / 'src/engine/models/qwen35/vision.py').read_bytes()).hexdigest()},
}

def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]

def bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]

def value(b):
    return struct.unpack('<f', struct.pack('<I', b))[0]

points = {f32(i / 64) for i in range(-640, 641)}
for boundary in [2**-28, 0.84375, 1.25, value(0x4036db6d), 6.0]:
    for delta in range(-2, 3):
        for sign in [-1, 1]:
            points.add(sign * value(bits(boundary) + delta))
for exponent in range(-149, 128, 3):
    points.update([f32(2.0**exponent), -f32(2.0**exponent)])
reference = []
for x in sorted(points):
    argument = f32(x * f32(1 / math.sqrt(2)))
    probability = f32(math.erf(argument))
    gelu = f32(f32(0.5 * x) * f32(1 + probability))
    reference.append([x, math.erf(x), gelu])
if args.erf:
    args.output.write_text(json.dumps(reference, separators=(',', ':')) + '\n')
    raise SystemExit(0)

# Float64 oracle with small nontrivial dimensions and two independent groups.
m, g, h, d = 2, 4, 3, 5
n = g*h
inputs = {
    'hidden': [((i*17)%31-15)/8 for i in range(m*n)],
    'norm_weight': [0.75, -1.5, 2.0],
    'norm_bias': [0.125, 0.25, -0.5],
    'up_weight': [((i*7)%23-11)/16 for i in range(n*n)],
    'up_bias': [i/32 for i in range(n)],
    'down_weight': [((i*13)%19-9)/16 for i in range(d*n)],
    'down_bias': [i/64 for i in range(d)],
}
normalized = []
for start in range(0, m*n, h):
    row = inputs['hidden'][start:start+h]
    mean = sum(row)/h
    variance = sum((x-mean)**2 for x in row)/h
    normalized.extend((x-mean)/math.sqrt(variance+1e-6)*w+b
                      for x,w,b in zip(row, inputs['norm_weight'], inputs['norm_bias']))
output = []
for group in range(m):
    x = normalized[group*n:(group+1)*n]
    up = [sum(v*w for v,w in zip(x, inputs['up_weight'][i*n:(i+1)*n]))+inputs['up_bias'][i]
          for i in range(n)]
    activated = [0.5*v*(1+math.erf(v/math.sqrt(2))) for v in up]
    output.extend(sum(v*w for v,w in zip(activated, inputs['down_weight'][i*n:(i+1)*n]))
                  + inputs['down_bias'][i] for i in range(d))
args.output.write_text(json.dumps({
    **provenance, 'dimensions': {'M':m,'G':g,'H':h,'D':d}, 'inputs':inputs, 'output':output,
}, indent=2) + '\n')
