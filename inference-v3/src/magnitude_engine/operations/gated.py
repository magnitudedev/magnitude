"""A gated pair of linear maps, with its fusion as a candidate of the composite.

The fused schedule is not a special case hidden inside a component: it is a row
of this operation's table, applicable when the representation, the rounding mode
and the shape all allow one pass over the input to produce the gated result.
"""

from __future__ import annotations

from magnitude_engine.kernels.precision import Precision
from magnitude_engine.kernels.semantics import Pointwise
from magnitude_engine.operations.activation import Elementwise
from magnitude_engine.operations.candidates import Plan, ScratchArena, Selection, realize
from magnitude_engine.operations.linear import Linear, LinearParameters
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.projections import Projections, ResidentProjections
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec

PACKED = "gated.packed"


class GatedLinear(Linear):
    def __init__(
        self,
        projections: Projections,
        input_width: int,
        precision: Precision,
        arena: ScratchArena,
    ):
        if len(projections.widths) != 2 or projections.widths[0] != projections.widths[1]:
            raise ValueError("gated projection requires two equal output widths")
        self.projections, self.precision, self.arena = projections, precision, arena
        self.context = arena.context
        self._parameters = LinearParameters(input_width, projections.widths[0])
        self.activation = Elementwise(self.context, Pointwise.SILU_PRODUCT, precision)
        self._plans: dict[tuple[int, DType, DType], Plan] = {}

    @property
    def parameters(self) -> LinearParameters:
        return self._parameters

    def plan(self, rows: int, dtype: DType, output_dtype: DType) -> Plan:
        key = rows, dtype, output_dtype
        if key not in self._plans:
            from magnitude_engine.kernels.projection.select import GATED_TABLE, GatedShape

            shape = GatedShape(
                rows,
                self._parameters.output_width,
                self._parameters.input_width,
                dtype,
                output_dtype,
            )
            representation = (
                self.projections.representation
                if isinstance(self.projections, ResidentProjections)
                else None
            )
            self._plans[key] = realize(
                "gated",
                GATED_TABLE,
                self.context,
                Selection(shape, self.precision, self.context.capability, representation),
            )
        return self._plans[key]

    def reserve(self, rows: int, dtype: DType, output_dtype: DType) -> None:
        plan = self.plan(rows, dtype, output_dtype)
        self.arena.reserve(plan.scratch)
        if not plan.executables and isinstance(self.projections, ResidentProjections):
            self.projections.reserve(rows, dtype, output_dtype)

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        rows = inputs.spec.shape[0]
        if outputs.spec.shape != (rows, self._parameters.output_width):
            raise ValueError("gated projection output geometry differs")
        plan = self.plan(rows, inputs.spec.dtype, outputs.spec.dtype)
        if plan.executables:
            fused = plan.executables[0]
            assert isinstance(self.projections, ResidentProjections)
            weights = self.projections.weight.acquire(fused.signature[1:-1])
            try:
                return (Prepared(self.context, fused, (inputs, *weights, outputs)),)
            finally:
                for tensor in weights:
                    tensor.close()
        region = plan.scratch[0]
        with Preparation(self.context) as p:
            packed = self.arena.region(p, region.name, region.spec)
            packed = p.view(
                packed, TensorSpec((rows * sum(self.projections.widths),), outputs.spec.dtype)
            )
            gate, up = self.projections.outputs(p, packed, rows)
            p.add(*self.projections.prepare(inputs, packed))
            p.add(*self.activation.prepare(gate, up, outputs))
            return p.finish()

    def close(self) -> None:
        self.activation.close()
        self.projections.close()
        self._plans.clear()
