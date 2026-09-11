"""A single linear map, as the one-segment case of a projection group.

A ``Linear`` exists because most consumers want a ``(rows, outputs)`` result
rather than a packed segment buffer. Selection, scratch and residency are the
projection group's; nothing about kernels appears here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.operations.projections import ResidentProjections
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec


@dataclass(frozen=True)
class LinearParameters:
    input_width: int
    output_width: int


class Linear(ABC):
    @property
    @abstractmethod
    def parameters(self) -> LinearParameters: ...

    @abstractmethod
    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]: ...

    @abstractmethod
    def close(self) -> None: ...


class ResidentLinear(Linear):
    def __init__(self, projection: ResidentProjections):
        if len(projection.widths) != 1:
            raise ValueError("a linear map has one output segment")
        self.projection = projection
        self.context = projection.context
        self._parameters = LinearParameters(projection.inputs, projection.widths[0])

    @property
    def parameters(self) -> LinearParameters:
        return self._parameters

    @property
    def representation(self):
        return self.projection.representation

    def plan(self, rows: int, dtype: DType, output_dtype: DType):
        return self.projection.plan(rows, dtype, output_dtype)

    def reserve(self, rows: int, dtype: DType, output_dtype: DType) -> None:
        self.projection.reserve(rows, dtype, output_dtype)

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        rows = inputs.spec.shape[0]
        if outputs.spec.shape != (rows, self.parameters.output_width):
            raise ValueError("linear output geometry differs")
        with Preparation(self.context) as p:
            flat = TensorSpec((rows * self.parameters.output_width,), outputs.spec.dtype)
            p.add(*self.projection.prepare(inputs, p.view(outputs, flat)))
            return p.finish()

    def close(self) -> None:
        self.projection.close()
