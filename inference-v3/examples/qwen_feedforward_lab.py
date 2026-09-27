"""Production Qwen dense-FFN formula with explicitly synthetic affine inputs.

This is an ordinary Lab configuration, not a second benchmark or numerical
implementation. Geometry comes from the model; values are deterministic and are
not represented as a captured artifact execution.
"""

from functools import partial
from pathlib import Path

import numpy as np

import ops
from ops.lab import Configuration, Fixture, MeasurementProtocol
from engine import DevicePlan
from engine.models.qwen35 import equations, operations  # installs production bodies
from engine.models.qwen35.description import Geometry


def dense(geometry: Geometry, *, rows: int, plan: DevicePlan, store: Path,
          seed: int = 157) -> Configuration:
    if rows < 1 or geometry.experts is not None:
        raise ValueError("dense feed-forward preparation needs positive rows and dense geometry")
    width, intermediate, dtype = geometry.hidden, geometry.intermediate, geometry.activation_dtype
    signature = ops.Signature((
        ops.Argument(ops.TensorSpec((rows, width), dtype), "hidden"),
        *(ops.Argument(ops.TensorSpec(shape, dtype).with_representation(
            ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))),
            name, ops.ValueKind.CONSTANT)
          for name, shape in (("gate", (intermediate, width)), ("up", (intermediate, width)),
                              ("down", (width, intermediate)))),
    ))

    def formula(hidden, gate, up, down):
        return equations.dense_feedforward(hidden, equations.DenseFeedForwardTensors(gate, up, down))

    graph = ops.trace(formula, signature)
    bindings = {}
    # Every coefficient is exactly representable but the scale is not a power of
    # two. Keep only compact physical planes until the worker captures a boundary.
    scale, bias = np.float32(3 / 1024), np.float32(-22.5 / 1024)
    for identity in graph.constants:
        spec = graph.value(identity).spec
        rng = np.random.default_rng(np.random.SeedSequence((seed, identity)))
        codes = rng.integers(0, 256, size=spec.elements // 2, dtype=np.uint8).tobytes()
        groups = spec.elements // 64
        scale_bits = int(scale.view(np.uint32)) >> 16
        bias_bits = int(bias.view(np.uint32)) >> 16
        contents = (codes, np.full(groups, scale_bits, np.uint16).tobytes(),
                    np.full(groups, bias_bits, np.uint16).tobytes())
        planes = tuple(ops.SourcePlane(ops.SourceSpan(ops.MemorySource(content), 0, len(content)), group, size)
                       for content, group, size in zip(contents, (2, 64, 64), (1, 2, 2), strict=True))
        bindings[identity] = ops.Binding(spec, f"synthetic-qwen-affine-v1:{seed}:{identity}:{spec.shape}",
                                         ops.Residency.RESIDENT, planes, ops.CanonicalImport())

    def capture(value: ops.Value):
        if value.id in graph.inputs:
            rng = np.random.default_rng(np.random.SeedSequence((seed, value.id)))
            return rng.integers(-16, 17, size=value.spec.shape, dtype=np.int16).astype(np.float32) / 8
        binding = bindings[value.id]
        packed = np.frombuffer(binding.planes[0].span.source.content, dtype=np.uint8)
        result = np.empty(value.spec.elements, dtype=np.float32)
        result[::2] = packed & 15
        result[1::2] = packed >> 4
        result *= scale
        result += bias
        return result.reshape(value.spec.shape)

    return Configuration(
        label=f"Synthetic Qwen dense FFN · {rows} rows · {width}/{intermediate}",
        fixture=Fixture.from_inputs(graph, {}, bindings=bindings, capture=capture),
        device=partial(ops.DeviceRuntime.open, plan),
        options=ops.CompileOptions(mode="decode" if rows == 1 else "prefill"), store=store,
        # Independent FP32 reduction trees can straddle a BF16 rounding midpoint
        # even when the candidate is closer to the FP64 dot product. Declare
        # BF16-resolution tolerance for this synthetic development fixture;
        # artifact qualification uses its separate, unchanged paired controls.
        protocol=MeasurementProtocol(absolute_tolerance=3e-3,
                                     relative_tolerance=2**-7 if dtype == ops.DType.BF16 else 3e-3),
        prepared_limit=1, reference_bytes=1 << 30,
    )
