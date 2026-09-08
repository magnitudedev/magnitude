"""RMS reduction and native scalar composition over a cooperative Metal driver."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.computation import computation
from ..core.metal import (
    Binding,
    Float,
    GroupPosition,
    Lane,
    ReadOnly,
    RowTransform,
    Scratch,
    SIMDIndex,
    TensorSpec,
    Threadgroup,
    ThreadPosition,
)
from ..core.plan import Source
from ..core.primitive import Primitive

RMS = Source("reductions/rms.metal")


@dataclass(frozen=True)
class RMSNorm(Primitive):
    eps: float

    def infer(self, inputs):
        x, weight = inputs
        if weight.shape != (x.shape[-1],) or x.dtype != weight.dtype:
            raise ValueError("RMS requires native weights matching the feature width")
        return (x,)

    def bindings(self, values):
        x, weight = values
        if x.tensor.shape[-1] % 128:
            return ()
        width = x.tensor.shape[-1]
        threads = min(width // 4, 1024)
        return (
            Binding(
                RMS,
                "magnitude_rms",
                RowTransform(
                    x.tensor,
                    x,
                    Threadgroup(threads),
                    arguments=dict(
                        weight=ReadOnly(TensorSpec(weight)[:]),
                        eps=Float(self.eps),
                        row=GroupPosition("y"),
                        tid=ThreadPosition(),
                        lane=Lane(),
                        group=SIMDIndex(),
                        partial=Scratch(mx.float32, 32),
                    ),
                    template=(
                        x.tensor.dtype,
                        width,
                        threads,
                        (width + threads * 4 - 1) // (threads * 4),
                    ),
                ),
            ),
        )

    def lower(self, inputs):
        from ..core.graph import Graph, Node, signature
        from ..core.hooks import row_transform

        values = signature(("x", "weight"), inputs)
        output = signature(("out",), self.infer(inputs))
        node = Node(self, values, output)
        bindings = self.bindings(values)
        if not bindings:
            return mx.compile(lambda x, weight: (mx.fast.rms_norm(x, weight, self.eps),))
        return row_transform(Graph(values, output, (), (node,)), node, bindings[0])


@computation
def residual_norm(x, update, weight, eps):
    residual = (x + update).astype(x.dtype)
    return residual, RMSNorm(eps)(residual, weight)[0]


@computation
def gated_norm(hidden, gate, weight, eps):
    normalized = RMSNorm(eps)(hidden, weight)[0]
    z = gate.astype(mx.float32)
    return ((z * mx.sigmoid(z)) * normalized.astype(mx.float32)).astype(hidden.dtype)
