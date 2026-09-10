"""Portable activation contracts and their workload-specific executable plans."""

import math

from magnitude_engine.numerics.policy import floating
from magnitude_engine.numerics.semantics import Pointwise
from magnitude_engine.operations.parameters import Parameter
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Executable,
    Prepared,
    Tensor,
    TensorSpec,
)


class Elementwise:
    def __init__(self, context: DeviceContext, kind: Pointwise, *, native_rounding: bool = False):
        self.native_rounding = native_rounding
        self.context, self.kind = context, kind
        self._plans: dict[tuple[int, DType, DType, DType], Executable] = {}

    def prepare(self, first: Tensor, second: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        if first.spec.shape != second.spec.shape or first.spec.shape != output.spec.shape:
            raise ValueError("elementwise operands must have matching floating geometry")
        for tensor in (first, second, output):
            floating(tensor.spec.dtype)
        size = math.prod(first.spec.shape)
        key = size, first.spec.dtype, second.spec.dtype, output.spec.dtype
        if key not in self._plans:
            from magnitude_engine.numerics.vector import pointwise

            self._plans[key] = self.context.specialize(
                pointwise,
                size,
                self.kind,
                cpu=self.context.backend == Backend.LLVM,
                dtype=first.spec.dtype,
                second_dtype=second.spec.dtype,
                output_dtype=output.spec.dtype,
                native_rounding=self.native_rounding,
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
        self,
        context: DeviceContext,
        weight: Parameter,
        epsilon: float,
        *,
        native_rounding: bool = False,
    ):
        self.native_rounding = native_rounding
        if len(weight.spec.shape) != 1:
            raise ValueError("normalization weight must be a vector")
        self.context, self.weight, self.epsilon = context, weight, epsilon
        self._plans: dict[tuple[int, DType, DType], Executable] = {}

    def prepare(self, inputs: Tensor, outputs: Tensor) -> tuple[Prepared, ...]:
        width = self.weight.spec.shape[0]
        if inputs.spec.shape != outputs.spec.shape or inputs.spec.shape[-1] != width:
            raise ValueError("normalization operands differ from its geometry")
        floating(inputs.spec.dtype)
        floating(outputs.spec.dtype)
        rows = math.prod(inputs.spec.shape) // width
        key = rows, inputs.spec.dtype, outputs.spec.dtype
        if key not in self._plans:
            from magnitude_engine.numerics.vector import rms_norm

            if self.native_rounding and self.context.backend == Backend.METAL:
                from magnitude_engine.numerics.native_bf16 import norm

                self._plans[key] = self.context.specialize(
                    norm, 1, width, ROWS=rows, epsilon=self.epsilon
                )
            else:
                self._plans[key] = self.context.specialize(
                    rms_norm,
                    rows,
                    width,
                    self.epsilon,
                    cpu=self.context.backend == Backend.LLVM,
                    dtype=inputs.spec.dtype,
                    output_dtype=outputs.spec.dtype,
                    native_rounding=self.native_rounding,
                )
        spec = TensorSpec((rows, width), inputs.spec.dtype)
        with Preparation(self.context) as p:
            p.add(
                Prepared(
                    self.context,
                    self._plans[key],
                    [
                        p.view(inputs, spec),
                        p.parameter(self.weight),
                        p.view(outputs, self._plans[key].signature[-1]),
                    ],
                )
            )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
