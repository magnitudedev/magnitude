"""Small architecture parameters, dense by the time a consumer sees them.

Conversion from a stored floating tensor belongs to residency. What remains
here is the one case a container leaves encoded: a blocked vector read through
the ordinary gather.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import ExitStack

from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec
from magnitude_engine.weights.representation import Dense
from magnitude_engine.weights.residency import ResidentWeight


class Parameter(ABC):
    @property
    @abstractmethod
    def spec(self) -> TensorSpec: ...

    @abstractmethod
    def acquire(self) -> Tensor: ...

    @abstractmethod
    def close(self) -> None: ...


class ResidentParameter(Parameter):
    """Bind a norm, convolution, or gate parameter to a dense FP32 consumer.

    Matrix projections and embeddings never use this conversion; their weights
    stay in their resident representation.
    """

    def __init__(self, weight: ResidentWeight):
        self.context = weight.context
        self._spec = TensorSpec(weight.descriptor.shape, DType.F32)
        if isinstance(weight.representation, Dense):
            self._value = weight.single(self._spec)
            return
        from magnitude_engine.kernels.embedding.gather import gather

        count = math.prod(self._spec.shape)
        plan = self.context.specialize(
            gather,
            1,
            1,
            count,
            weight.representation,
            capability=self.context.capability,
        )
        with ExitStack() as cleanup:
            source = weight.single(plan.signature[1])
            cleanup.callback(source.close)
            index = self.context.upload(TensorSpec((1,), DType.I32), bytes(4))
            cleanup.callback(index.close)
            value = self.context.allocate(plan.signature[2])
            cleanup.callback(value.close)
            self.context.submit([Prepared(self.context, plan, [index, source, value])]).wait()
            self._value = value.view(self._spec)

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    def acquire(self) -> Tensor:
        return self._value.view(self._spec)

    def close(self) -> None:
        self._value.close()
