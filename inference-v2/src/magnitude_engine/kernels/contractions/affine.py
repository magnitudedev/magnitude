"""Canonical encoded contractions and graph-derived scalar finalization."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.fragments import lower_fragments
from ..core.graph import Graph, Node, Tensor, signature
from ..core.metal import (
    Binding,
    BlockedRows,
    ColumnStart,
    FragmentFold,
    Lane,
    MetalType,
    ReadOnly,
    RowIndices,
    TensorSpec,
)
from ..core.plan import Source
from ..core.primitive import Primitive
from .tiles import TILE, packing, row_tile

AFFINE = Source("contractions/affine.metal", (TILE, Source("core/fragments.metal")))


@dataclass(frozen=True)
class Affine(Primitive):
    """Affine encoded dot with a fixed 32-lane, packed K traversal.

    Input pack sums retain native arithmetic; encoded dot accumulation and the
    lane reduction are FP32, followed by exactly one native output conversion.
    Row/expert placement must not alter this arithmetic.
    """

    bits: int
    group_size: int

    def infer(self, inputs):
        x, w, s, b = inputs
        k, n = x.shape[-1], w.shape[-2]
        if (
            len(w.shape) != 2
            or w.dtype != mx.uint32
            or w.shape[-1] * 32 != k * self.bits
            or s != b
            or s.shape != (n, k // self.group_size)
            or s.dtype != x.dtype
            or x.dtype not in (mx.float16, mx.bfloat16, mx.float32)
            or packing(k, n, self.bits, self.group_size) is None
        ):
            raise ValueError("unsupported canonical affine geometry or encoding")
        return (Tensor((*x.shape[:-1], n), x.dtype),)

    def bindings(self, values) -> tuple[Binding, ...]:
        output = self.infer(tuple(v.tensor for v in values))[0]
        count, n = output.size // output.shape[-1], output.shape[-1]
        if not count:
            return ()
        x, w, s, b = (TensorSpec(v) for v in values)

        def whole(t):
            return ReadOnly(t[(slice(None),) * t.ndim])

        rows = row_tile(count, 4)
        pack = packing(x.shape[-1], n, self.bits, self.group_size)
        assert pack is not None  # infer already checked this encoding
        return (
            Binding(
                AFFINE,
                "magnitude_affine_fold",
                FragmentFold(
                    output,
                    BlockedRows(count, n, rows),
                    dict(x=whole(x), rows=RowIndices(), first=ColumnStart(), lane=Lane()),
                    (x.dtype, self.bits, pack, x.shape[-1], rows),
                    MetalType(
                        "AffineStep",
                        (x.dtype, self.bits, pack, x.shape[-1], n, self.group_size, rows),
                        (whole(w), whole(s), whole(b)),
                    ),
                    pack,
                ),
            ),
        )

    def lower(self, inputs):
        values = signature(("x", "w", "s", "b"), inputs)
        outputs = signature(("out",), self.infer(inputs))
        if outputs[0].tensor.size == 0:
            return lambda *args: (mx.zeros(outputs[0].tensor.shape, outputs[0].tensor.dtype),)
        node = Node(self, values, outputs)
        return lower_fragments(
            Graph(values, outputs, (), (node,)), {node: self.bindings(values)[0]}
        )
