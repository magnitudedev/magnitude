"""Small architecture parameters decoded beneath their consumers' boundary."""

import math
from abc import ABC, abstractmethod
from contextlib import ExitStack

from magnitude_engine.artifacts.gguf import Encoding
from magnitude_engine.operations.weights import ResidentWeight
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, Tensor, TensorSpec


class Parameter(ABC):
    @property
    @abstractmethod
    def spec(self) -> TensorSpec: ...

    @abstractmethod
    def acquire(self) -> Tensor: ...

    @abstractmethod
    def close(self) -> None: ...


class DenseParameter(Parameter):
    """Bind a norm, convolution, or gate parameter to a dense FP32 consumer.

    Matrix projections and embeddings never use this conversion; their weights
    remain encoded. Loading conversion uses the ordinary execution owner.
    """

    def __init__(self, weight: ResidentWeight):
        self.context = weight.context
        self._spec = TensorSpec(weight.descriptor.shape, DType.F32)
        if weight.descriptor.encoding == Encoding.F32:
            self._value = weight.acquire(self._spec)
            return
        from magnitude_engine.numerics.encoded import gather

        count = math.prod(self._spec.shape)
        plan = self.context.specialize(
            gather,
            1,
            1,
            count,
            weight.descriptor.encoding,
            cpu=self.context.backend == Backend.LLVM,
            layout=weight.layout,
        )
        with ExitStack() as cleanup:
            source = weight.acquire(plan.signature[1])
            cleanup.callback(source.close)
            index = self.context.upload(TensorSpec((1,), DType.I32), bytes(4))
            cleanup.callback(index.close)
            value = self.context.allocate(plan.signature[2])
            cleanup.callback(value.close)
            ticket = self.context.submit([Prepared(self.context, plan, [index, source, value])])
            ticket.wait()
            self._value = value.view(self._spec)

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    def acquire(self) -> Tensor:
        return self._value.view(self._spec)

    def close(self) -> None:
        self._value.close()
