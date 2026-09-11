"""Portable activation contracts and the executables their shapes resolve to."""

from __future__ import annotations

import math

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.kernels.semantics import Pointwise
from magnitude_engine.operations.candidates import Plan, Selection, realize
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Executable,
    Prepared,
    Tensor,
    TensorSpec,
)


class Elementwise:
    def __init__(self, context: DeviceContext, kind: Pointwise, precision: Precision):
        self.context, self.kind, self.precision = context, kind, precision
        self._plans: dict[tuple[int, DType, DType, DType], Executable] = {}

    def prepare(self, first: Tensor, second: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        if first.spec.shape != second.spec.shape or first.spec.shape != output.spec.shape:
            raise ValueError("elementwise operands must have matching floating geometry")
        for tensor in (first, second, output):
            floating(tensor.spec.dtype)
        size = math.prod(first.spec.shape)
        key = size, first.spec.dtype, second.spec.dtype, output.spec.dtype
        if key not in self._plans:
            from magnitude_engine.kernels.pointwise.portable import pointwise

            self._plans[key] = self.context.specialize(
                pointwise,
                size,
                self.kind,
                capability=self.context.capability,
                precision=self.precision,
                dtype=first.spec.dtype,
                second_dtype=second.spec.dtype,
                output_dtype=output.spec.dtype,
            )
        with Preparation(self.context) as p:
            views = [
                p.view(tensor, TensorSpec((size,), tensor.spec.dtype))
                for tensor in (first, second, output)
            ]
            p.add(Prepared(self.context, self._plans[key], views))
            return p.finish()

    def close(self) -> None:
        self._plans.clear()


class RMSNorm:
    def __init__(
        self, context: DeviceContext, weight: Parameter, epsilon: float, precision: Precision
    ):
        if len(weight.spec.shape) != 1:
            raise ValueError("normalization weight must be a vector")
        self.context, self.weight, self.epsilon = context, weight, epsilon
        self.precision = precision
        self._plans: dict[tuple[int, DType, DType], Plan] = {}

    def plan(self, rows: int, dtype: DType, output_dtype: DType) -> Plan:
        key = rows, dtype, output_dtype
        if key not in self._plans:
            from magnitude_engine.kernels.norm.select import TABLE, NormShape

            shape = NormShape(rows, self.weight.spec.shape[0], self.epsilon, dtype, output_dtype)
            self._plans[key] = realize(
                "norm",
                TABLE,
                self.context,
                Selection(shape, self.precision, self.context.capability),
            )
        return self._plans[key]

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        width = self.weight.spec.shape[0]
        if inputs.spec.shape != outputs.spec.shape or inputs.spec.shape[-1] != width:
            raise ValueError("normalization operands differ from its geometry")
        floating(inputs.spec.dtype)
        floating(outputs.spec.dtype)
        rows = math.prod(inputs.spec.shape) // width
        executable = self.plan(rows, inputs.spec.dtype, outputs.spec.dtype).executables[0]
        with Preparation(self.context) as p:
            p.add(
                Prepared(
                    self.context,
                    executable,
                    [
                        p.view(inputs, executable.signature[0]),
                        p.parameter(self.weight),
                        p.view(outputs, executable.signature[-1]),
                    ],
                )
            )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
