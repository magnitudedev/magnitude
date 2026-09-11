"""Copy a dense activation operand without exposing its native memory mapping."""

import math

from magnitude_engine.kernels.precision import floating
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.execution import (
    DeviceContext,
    DType,
    Executable,
    Prepared,
    Tensor,
    TensorSpec,
)


class Copy:
    def __init__(self, context: DeviceContext):
        self.context = context
        self._plans: dict[tuple[int, DType, DType], Executable] = {}

    def prepare(self, source: Tensor, destination: Tensor) -> tuple[Prepared, ...]:
        if source.spec.shape != destination.spec.shape:
            raise ValueError("activation copy requires matching floating geometry")
        if source.overlaps(destination):
            raise ValueError("activation copy operands must not overlap")
        floating(source.spec.dtype)
        floating(destination.spec.dtype)
        size = math.prod(source.spec.shape)
        key = size, source.spec.dtype, destination.spec.dtype
        if key not in self._plans:
            from magnitude_engine.kernels.copy.copy import copy

            self._plans[key] = self.context.specialize(
                copy,
                size,
                capability=self.context.capability,
                dtype=source.spec.dtype,
                output_dtype=destination.spec.dtype,
            )
        with Preparation(self.context) as p:
            spec = TensorSpec((size,), source.spec.dtype)
            p.add(
                Prepared(
                    self.context,
                    self._plans[key],
                    [
                        p.view(source, spec),
                        p.view(destination, TensorSpec(spec.shape, destination.spec.dtype)),
                    ],
                )
            )
            return p.finish()

    def close(self) -> None:
        self._plans.clear()
