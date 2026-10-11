#!/usr/bin/env python3
"""Independent float64 vision-block equations; zero coordinates isolate attention."""
import argparse
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve(strict=True)
m, heads, pairs, ffn = 3, 2, 1, 12
width = 4 * pairs
hidden = heads * width
sizes = {
    "hidden": m * hidden, "coordinates": m * 2,
    "norm1_weight": hidden, "norm1_bias": hidden,
    "qkv_weight": 3 * hidden * hidden, "qkv_bias": 3 * hidden,
    "projection_weight": hidden * hidden, "projection_bias": hidden,
    "norm2_weight": hidden, "norm2_bias": hidden,
    "up_weight": ffn * hidden, "up_bias": ffn,
    "down_weight": hidden * ffn, "down_bias": hidden,
}
inputs = {name: [((i * 7 + seed * 3) % 23 - 11) / 32 for i in range(size)]
          for seed, (name, size) in enumerate(sizes.items())}
inputs["coordinates"] = [0] * (m * 2)


def norm(row, prefix):
    mean = sum(row) / len(row)
    variance = sum((value - mean) ** 2 for value in row) / len(row)
    return [(value - mean) / math.sqrt(variance + 1e-6) * scale + bias
            for value, scale, bias in zip(row, inputs[prefix + "_weight"], inputs[prefix + "_bias"])]


def linear(row, prefix):
    weight, bias = inputs[prefix + "_weight"], inputs[prefix + "_bias"]
    return [sum(value * coefficient for value, coefficient in
                zip(row, weight[i * len(row):(i + 1) * len(row)])) + offset
            for i, offset in enumerate(bias)]


rows = [inputs["hidden"][i * hidden:(i + 1) * hidden] for i in range(m)]
qkv = [linear(norm(row, "norm1"), "qkv") for row in rows]
output = []
for row, original in enumerate(rows):
    attended = []
    for head in range(heads):
        start = head * width
        query = qkv[row][start:start + width]
        scores = [sum(a * b for a, b in zip(query, key[hidden + start:hidden + start + width]))
                  / math.sqrt(width) for key in qkv]
        probabilities = [math.exp(score - max(scores)) for score in scores]
        denominator = sum(probabilities)
        attended.extend(sum(probability * value[2 * hidden + start + column]
                            for probability, value in zip(probabilities, qkv)) / denominator
                        for column in range(width))
    residual = [a + b for a, b in zip(original, linear(attended, "projection"))]
    up = linear(norm(residual, "norm2"), "up")
    activated = [0.5 * value * (1 + math.tanh(math.sqrt(2 / math.pi)
                 * (value + 0.044715 * value ** 3))) for value in up]
    output.extend(a + b for a, b in zip(residual, linear(activated, "down")))

args.output.write_text(json.dumps({
    "provenance": __doc__,
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256": {"src/engine/models/qwen35/vision.py": hashlib.sha256(
        (source / "src/engine/models/qwen35/vision.py").read_bytes()).hexdigest()},
    "dimensions": {"M": m, "H": heads, "P": pairs, "F": ffn},
    "inputs": inputs, "output": output,
}, indent=2) + "\n")
