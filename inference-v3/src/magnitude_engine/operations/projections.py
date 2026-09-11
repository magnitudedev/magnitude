"""Several linear maps of one input, with contiguous output segments.

The group exposes the algebraic sharing opportunity. Residency decides whether
it is one packed contraction or separate contractions; the model never does, and
neither does this module: it asks the projection table which schedule applies to
the representation residency produced.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import accumulate

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.operations.candidates import Plan, ScratchArena, Selection, realize
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec
from magnitude_engine.weights.residency import ResidentGroup, ResidentWeight


class Projections(ABC):
    @property
    @abstractmethod
    def widths(self) -> tuple[int, ...]: ...

    @abstractmethod
    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]: ...

    @abstractmethod
    def close(self) -> None: ...

    def outputs(self, p: Preparation, output: Tensor, rows: int) -> tuple[Tensor, ...]:
        expected = TensorSpec((rows * sum(self.widths),), output.spec.dtype)
        if output.spec != expected:
            raise ValueError("projection group storage differs from its output geometry")
        return tuple(
            p.view(
                output,
                TensorSpec((rows, width), output.spec.dtype),
                rows * start * output.spec.dtype.itemsize,
            )
            for width, start in zip(self.widths, accumulate((0, *self.widths[:-1])), strict=True)
        )


class SeparateProjections(Projections):
    """One contraction per weight, into its own segment of the packed output."""

    def __init__(self, operations: tuple[ResidentProjections, ...]):
        if not operations or len({op.inputs for op in operations}) != 1:
            raise ValueError("projection group must share one input width")
        self.operations = operations

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(width for op in self.operations for width in op.widths)

    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        with Preparation(output.context) as p:
            outputs = self.outputs(p, output, inputs.spec.shape[0])
            for op, target in zip(self.operations, outputs, strict=True):
                flat = TensorSpec((target.spec.shape[0] * target.spec.shape[1],), target.spec.dtype)
                p.add(*op.prepare(inputs, p.view(target, flat)))
            return p.finish()

    def close(self) -> None:
        for operation in self.operations:
            operation.close()


class ResidentProjections(Projections):
    """One contraction over a resident weight or a concatenated resident group."""

    def __init__(
        self,
        weight: ResidentWeight | ResidentGroup,
        precision: Precision,
        arena: ScratchArena,
    ):
        self.weight, self.precision, self.arena = weight, precision, arena
        self.context = weight.context
        if isinstance(weight, ResidentGroup):
            self._widths, self.inputs = weight.widths, weight.inputs
        else:
            if len(weight.descriptor.shape) != 2:
                raise ValueError("a projection weight is a matrix")
            self._widths = (weight.descriptor.shape[0],)
            self.inputs = weight.descriptor.shape[1]
        self._plans: dict[tuple[int, DType, DType], Plan] = {}

    @property
    def widths(self) -> tuple[int, ...]:
        return self._widths

    def plan(self, rows: int, dtype: DType, output_dtype: DType) -> Plan:
        floating(dtype)
        floating(output_dtype)
        key = rows, dtype, output_dtype
        if key not in self._plans:
            from magnitude_engine.kernels.projection.select import TABLE, ProjectionShape

            shape = ProjectionShape(rows, self._widths, self.inputs, dtype, output_dtype)
            self._plans[key] = realize(
                "projection",
                TABLE,
                self.context,
                Selection(shape, self.precision, self.context.capability, self.representation),
            )
        return self._plans[key]

    @property
    def representation(self):
        return self.weight.representation

    def reserve(self, rows: int, dtype: DType, output_dtype: DType) -> None:
        self.arena.reserve(self.plan(rows, dtype, output_dtype).scratch)

    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        if len(inputs.spec.shape) != 2 or inputs.spec.shape[1] != self.inputs:
            raise ValueError("input tensor differs from linear geometry")
        rows = inputs.spec.shape[0]
        plan = self.plan(rows, inputs.spec.dtype, output.spec.dtype)
        if output.spec != TensorSpec((rows * sum(self._widths),), output.spec.dtype):
            raise ValueError("projection output differs from its logical segments")
        contraction = plan.executables[0]
        weights = self.weight.acquire(contraction.signature[1:-1])
        try:
            with Preparation(self.context) as p:
                if len(plan.executables) == 1:
                    target = p.view(output, contraction.signature[-1])
                else:
                    region = plan.scratch[0]
                    target = self.arena.region(p, region.name, region.spec)
                    target = p.view(target, contraction.signature[-1])
                p.add(Prepared(self.context, contraction, (inputs, *weights, target)))
                for reduction in plan.executables[1:]:
                    p.add(
                        Prepared(
                            self.context,
                            reduction,
                            (
                                p.view(target, reduction.signature[0]),
                                p.view(output, reduction.signature[-1]),
                            ),
                        )
                    )
                return p.finish()
        finally:
            for tensor in weights:
                tensor.close()

    def close(self) -> None:
        self._plans.clear()
