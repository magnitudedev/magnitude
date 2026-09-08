"""Selected affine contractions and ordered route reduction are separate operations."""

from dataclasses import dataclass

from ..core.fragments import lower_fragments
from ..core.graph import Graph, Node, Tensor, signature
from ..core.kernel import Kernel
from ..core.metal import (
    ArgumentType,
    Binding,
    BlockedRows,
    ColumnStart,
    FragmentCall,
    Lane,
    MetalType,
    OrderedReduction,
    ReadOnly,
    RowIndices,
    TensorSpec,
)
from ..core.plan import Launch, Source
from ..core.primitive import Primitive
from .tiles import TILE, row_tile

SELECTED = Source("contractions/selected.metal", (TILE, Source("core/fragments.metal")))


ROUTE_STEP = Source("contractions/route.metal", (Source("core/fragments.metal"),))
ROUTE_DRIVER = Source("reductions/ordered.metal", (ROUTE_STEP,))
ROUTE_SUM = Source("contractions/route_sum.metal", (ROUTE_STEP,))


@dataclass(frozen=True)
class SelectedAffine(Primitive):
    """Fixed encoded row dot, selected by logical route; optional shared bank.

    Assignments select coefficients and weight rows. Execution order may group
    equal banks but cannot alter each dot's K traversal or native output cast.
    """

    bits: int
    group_size: int
    slots: int
    shared: bool
    per_slot: bool = False

    def infer(self, inputs):
        x, ids, order, w, s, bias, sw, ss, sb = inputs
        rows, n = ids.size, w.shape[-2]
        k = x.shape[-1]
        pack = 64 // self.bits
        if (
            k % (32 * pack)
            or self.group_size % pack
            or n % 8
            or rows % self.slots
            or x.size // k != rows // (1 if self.per_slot else self.slots)
            or order.size != rows
            or w.shape[-1] * 32 != k * self.bits
            or s != bias
            or s.shape != (*w.shape[:-1], k // self.group_size)
        ):
            raise ValueError("selected affine operands disagree with the route geometry")
        return (Tensor((rows // self.slots, self.slots, n), x.dtype),)

    def bindings(self, values):
        output = self.infer(tuple(v.tensor for v in values))[0]
        count, n = output.size // output.shape[-1], output.shape[-1]
        x, ids, order, w, s, bias, sw, ss, sb = (TensorSpec(v) for v in values)
        banks = w.shape[0]
        reuse = count // self.slots > 1 and count >= 2 * (banks + self.shared)
        rows = row_tile(count, 4) if reuse else 1

        def whole(t):
            return ReadOnly(t[(slice(None),) * t.ndim])

        return (
            Binding(
                SELECTED,
                "magnitude_selected",
                FragmentCall(
                    output,
                    BlockedRows(count, n, rows, values[2]),
                    dict(
                        x=whole(x),
                        ids=whole(ids),
                        w=whole(w),
                        s=whole(s),
                        bias=whole(bias),
                        sw=whole(sw),
                        ss=whole(ss),
                        sb=whole(sb),
                        rows=RowIndices(),
                        first=ColumnStart(),
                        lane=Lane(),
                    ),
                    (
                        x.dtype,
                        self.bits,
                        64 // self.bits,
                        x.shape[-1],
                        n,
                        self.group_size,
                        rows,
                        self.slots,
                        banks,
                        self.shared,
                        self.per_slot,
                    ),
                ),
            ),
        )

    def lower(self, inputs):
        values = signature(("x", "ids", "order", "w", "s", "bias", "sw", "ss", "sb"), inputs)
        outputs = signature(("out",), self.infer(inputs))
        node = Node(self, values, outputs)
        return lower_fragments(
            Graph(values, outputs, (), (node,)), {node: self.bindings(values)[0]}
        )


@dataclass(frozen=True)
class RouteSum(Primitive):
    """Multiply and add routes in logical slot order, rounding each step to native dtype."""

    shared: bool

    def infer(self, inputs):
        projected, scores, shared_score = inputs
        rows, slots, width = projected.shape
        if scores.shape != (rows, slots - self.shared):
            raise ValueError("route scores disagree with projected slots")
        return (Tensor((rows, width), projected.dtype),)

    def bindings(self, values):
        projected, scores, shared_score = values
        tensors = tuple(v.tensor for v in values)
        output = self.infer(tensors)[0]
        args = tuple(
            ReadOnly(TensorSpec(v)[(slice(None),) * len(v.tensor.shape)])
            for v in (scores, shared_score)
        )
        return (
            Binding(
                ROUTE_DRIVER,
                "magnitude_ordered_fragments",
                OrderedReduction(
                    projected,
                    output,
                    MetalType(
                        "RouteStep",
                        (
                            output.dtype,
                            tensors[0].shape[-2],
                            self.shared,
                            *(ArgumentType(a) for a in args),
                        ),
                        args,
                    ),
                ),
            ),
        )

    def lower(self, inputs):
        rows, slots, width = inputs[0].shape
        return Kernel(
            signature(("x", "scores", "shared_score"), inputs),
            signature(("out",), self.infer(inputs)),
            ROUTE_SUM,
            Launch((rows * width, 1, 1), (256, 1, 1)),
            (
                ("T", inputs[0].dtype),
                ("ROWS", rows),
                ("SLOTS", slots),
                ("WIDTH", width),
                ("TOPK", slots - self.shared),
                ("SHARED", self.shared),
            ),
        ).bind()
