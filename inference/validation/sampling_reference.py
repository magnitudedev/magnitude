#!/usr/bin/env python3
"""Generate deterministic categorical draws using the V3 reference implementation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from reference_source import activate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
source = args.source.resolve(strict=True)
activate(source)

import numpy as np
from ops.tensor.ops import _sample_reference

draws = np.asarray([
    [1, 42, 0, 0, 0, 0],
    [1, 42, 0, 1, 0, 0],
    [1, 0xFFFFFFF1, 0xEFFFFFFF, 0xFFFFFFFF, 0xAAAAAAAA, 3],
    [1, 42, 0, 0, 0, 1],
    [1, 42, 0, 0, 0, 0],
    [0, 42, 0, 0, 0, 0],
], dtype=np.uint32)
logits = np.asarray([[2 * math.sin(i * 0.7) for i in range(35)]] * len(draws), dtype=np.float32)
reference = _sample_reference(logits, draws)
assert np.all(reference[:, 1] == 0)
args.output.write_text(json.dumps({
    "reference": "V3 _sample_reference, including Philox counters and F32 score publication",
    "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_sha256": {"src/ops/tensor/ops.py": hashlib.sha256(
        (source / "src/ops/tensor/ops.py").read_bytes()).hexdigest()},
    "numpy_version": np.__version__,
    "logits": logits.tolist(), "draws": draws.tolist(),
    "expected": reference[:, 0].tolist(),
}, indent=2) + "\n")
