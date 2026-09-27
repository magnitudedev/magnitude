"""Open the ordinary formula Lab at artifact-derived attention geometries.

Inputs are explicitly synthetic, not a captured model continuation. This uses the
production causal_attention Formula, its independent primitive reference and its
operation; it defines no benchmark equation, implementation or timing loop.
Cold fixture/reference preparation can be substantial at long contexts. It is
retained by the worker for subsequent edits and measurements within a byte budget.
"""

from functools import partial
from pathlib import Path

import numpy as np

import ops
from ops.lab import Configuration, Fixture, MeasurementProtocol, show
from engine import DevicePlan
from engine.models.qwen35.description import Geometry


def attention(geometry: Geometry, *, rows: int, history: int, plan: DevicePlan,
              store: Path, seed: int = 47) -> Configuration:
    if not 1 <= rows <= history <= geometry.context_limit:
        raise ValueError("attention rows/history must fit the model context")
    signature = ops.Signature((
        ops.Argument(ops.TensorSpec((rows, geometry.attention_heads, geometry.attention_width),
                                    geometry.activation_dtype), "query"),
        ops.Argument(ops.TensorSpec((2, history, geometry.kv_heads, geometry.attention_width),
                                    geometry.activation_dtype), "history", ops.ValueKind.RESOURCE),
        ops.Argument(ops.TensorSpec((rows, 2), ops.DType.I32), "visible"),
    ))

    def formula(query, state, visible):
        return ops.causal_attention(query, state, visible, sequence_count=1)

    graph = ops.trace(formula, signature)
    query, visible = graph.inputs
    state, = graph.resources

    def capture(value: ops.Value):
        if value.id == visible:
            ranges = np.zeros((rows, 2), dtype=np.int32)
            ranges[:, 1] = np.arange(history - rows + 1, history + 1, dtype=np.int32)
            return ranges
        if value.id not in (query, state):
            raise ValueError("this fixture only captures the attention formula inputs")
        rng = np.random.default_rng(np.random.SeedSequence((seed, value.id)))
        # Exact values in both supported activation dtypes. A nonzero value-plane
        # offset makes an accidentally cleared output fail numerical checking.
        values = rng.integers(-16, 17, size=value.spec.shape, dtype=np.int16).astype(np.float32)
        values *= 1 / 16
        if value.id == state:
            values[1] += 1 / 2
        return values

    return Configuration(
        label=f"Synthetic Qwen attention · {rows} rows · {history} history · {geometry.attention_heads} heads",
        fixture=Fixture(graph, {}, capture=capture),
        device=partial(ops.DeviceRuntime.open, plan),
        options=ops.CompileOptions(mode="decode" if rows == 1 else "prefill"),
        store=store,
        protocol=MeasurementProtocol(absolute_tolerance=3e-3, relative_tolerance=3e-3),
        prepared_limit=1,
        reference_bytes=1 << 30,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, help="existing Qwen MLX snapshot or GGUF file")
    parser.add_argument("--store", type=Path, default=Path("runs/formulas.sqlite"))
    args = parser.parse_args()
    path = args.target.resolve(strict=True)
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
    plan = DevicePlan.discover(maximum_bytes=2 << 30)
    show(tuple(attention(geometry, rows=rows, history=history, plan=plan, store=args.store)
               for history in (16384, 65536) for rows in (1, 2048)
               if history <= geometry.context_limit))


if __name__ == "__main__":
    main()
