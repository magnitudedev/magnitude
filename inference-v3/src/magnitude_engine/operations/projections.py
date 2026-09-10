"""Several linear maps of one input, with contiguous output segments.

The group exposes the algebraic sharing opportunity. Residency decides whether
it is one packed contraction or separate contractions; the model never does.
"""

from abc import ABC, abstractmethod
from itertools import accumulate

from magnitude_engine.operations.linear import Linear
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import Prepared, Tensor, TensorSpec


class Projections(ABC):
    @property
    @abstractmethod
    def widths(self) -> tuple[int, ...]: ...

    @abstractmethod
    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]: ...

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
    def __init__(self, operations: tuple[Linear, ...]):
        if not operations or len({op.parameters.input_width for op in operations}) != 1:
            raise ValueError("projection group must share one input width")
        self.operations = operations

    @property
    def widths(self) -> tuple[int, ...]:
        return tuple(op.parameters.output_width for op in self.operations)

    def prepare(self, inputs: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        with Preparation(output.context) as p:
            outputs = self.outputs(p, output, inputs.spec.shape[0])
            for op, target in zip(self.operations, outputs, strict=True):
                p.add(*op.prepare(inputs, target))
            return p.finish()
