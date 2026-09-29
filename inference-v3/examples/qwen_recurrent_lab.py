"""Production recurrent formula at artifact-derived Qwen geometry.

The operands are explicitly synthetic normalized queries/keys, finite decays and
a nonzero incoming state. The ordinary formula reference checks both outputs and
the outgoing state. This adds no numerical implementation or timing harness.
"""

from functools import partial
from pathlib import Path

import numpy as np

import ops
from ops.lab import Configuration, Fixture, MeasurementProtocol, show
from ops.tensor.primitive import round_reference
from engine import DevicePlan
from engine.models.qwen35.description import Geometry


def recurrent(geometry: Geometry, *, rows: int, plan: DevicePlan, store: Path,
              seed: int = 173) -> Configuration:
    if type(rows) is not int or not 1 <= rows <= geometry.context_limit:
        raise ValueError("recurrent rows must fit the model context")
    kh, heads, width = (geometry.recurrent_key_heads, geometry.recurrent_value_heads,
                        geometry.recurrent_width)
    dtype = geometry.activation_dtype
    signature = ops.Signature((
        ops.Argument(ops.TensorSpec((rows, kh, width), dtype), "query"),
        ops.Argument(ops.TensorSpec((rows, kh, width), dtype), "key"),
        ops.Argument(ops.TensorSpec((rows, heads, width), dtype), "value"),
        ops.Argument(ops.TensorSpec((rows, heads), ops.DType.F32), "decay"),
        ops.Argument(ops.TensorSpec((rows, heads), dtype), "beta"),
        ops.Argument(ops.TensorSpec((1, heads, width, width), ops.DType.F32),
                     "state", ops.ValueKind.RESOURCE),
        ops.Argument(ops.TensorSpec((2,), ops.DType.I32), "offsets"),
    ))

    def formula(query, key, value, decay, beta, state, offsets):
        return ops.gated_delta_recurrence(query, key, value, decay, beta, state, offsets,
                                          mapping=geometry.recurrent_head_mapping.value)

    graph = ops.trace(formula, signature)
    query, key, value, decay, beta, offsets = graph.inputs
    state, = graph.resources

    def capture(operand: ops.Value):
        if operand.id == offsets:
            return np.asarray([0, rows], dtype=np.int32)
        rng = np.random.default_rng(np.random.SeedSequence((seed, operand.id)))
        shape = operand.spec.shape
        if operand.id in (query, key):
            result = rng.normal(size=shape).astype(np.float32)
            result /= np.sqrt(np.sum(result * result, axis=-1, keepdims=True) + geometry.epsilon)
            if operand.id == query:
                result *= width ** -0.5
            return round_reference(result, operand.spec.dtype)
        if operand.id == decay:
            return rng.uniform(0.85, 1, size=shape).astype(np.float32)
        if operand.id == beta:
            return round_reference(rng.uniform(0, 1, size=shape).astype(np.float32), operand.spec.dtype)
        if operand.id in (value, state):
            return round_reference(rng.normal(0, 0.1, size=shape).astype(np.float32), operand.spec.dtype)
        raise ValueError("recurrent fixture only captures its formula boundary")

    return Configuration(
        label=f"Synthetic Qwen recurrence · {rows} rows · {heads} heads · width {width}",
        fixture=Fixture(graph, {}, capture=capture),
        device=partial(ops.DeviceRuntime.open, plan),
        options=ops.CompileOptions(mode="decode" if rows == 1 else "prefill"),
        store=store,
        protocol=MeasurementProtocol(absolute_tolerance=3e-5,
                                     relative_tolerance=8e-3 if dtype == ops.DType.BF16 else 3e-4),
        prepared_limit=1, reference_bytes=1 << 30,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="existing Qwen MLX snapshot or GGUF file")
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--store", type=Path, default=Path("runs/formula-lab.sqlite"))
    args = parser.parse_args()
    path = args.model.resolve(strict=True)
    if path.is_dir():
        from engine.weights.formats.mlx_safetensors import MLXFormat
        from engine.models.qwen35.formats.mlx import describe

        source = MLXFormat(str(path))
    else:
        from engine.weights.formats.gguf import GGUFFormat
        from engine.models.qwen35.formats.gguf import describe

        source = GGUFFormat(str(path))
    try:
        geometry = describe(source).geometry
    finally:
        source.close()
    show((recurrent(geometry, rows=args.rows, plan=DevicePlan.discover(maximum_bytes=1 << 30),
                    store=args.store),))


if __name__ == "__main__":
    main()
