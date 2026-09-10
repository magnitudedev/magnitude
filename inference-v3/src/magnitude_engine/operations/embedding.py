"""Logical token lookup with representation owned below the operation boundary."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from magnitude_engine.numerics.policy import floating
from magnitude_engine.operations.weights import ResidentWeight
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Executable, Prepared, Tensor


@dataclass(frozen=True)
class EmbeddingParameters:
    vocabulary: int
    width: int


class Embedding(ABC):
    @property
    @abstractmethod
    def parameters(self) -> EmbeddingParameters: ...

    @abstractmethod
    def prepare(self, tokens: Tensor, output: Tensor) -> tuple[Prepared, ...]: ...

    @abstractmethod
    def close(self) -> None: ...


class EncodedEmbedding(Embedding):
    def __init__(self, weight: ResidentWeight):
        if len(weight.descriptor.shape) != 2:
            raise ValueError("embedding requires matrix weights")
        self.weight, self.context = weight, weight.context
        self._parameters = EmbeddingParameters(*weight.descriptor.shape)
        self._plans: dict[tuple[int, DType], Executable] = {}

    @property
    def parameters(self) -> EmbeddingParameters:
        return self._parameters

    def prepare(self, tokens: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        from magnitude_engine.numerics.encoded import gather

        if len(tokens.spec.shape) != 1:
            raise ValueError("embedding input must be a vector of token IDs")
        floating(output.spec.dtype)
        rows = tokens.spec.shape[0]
        key = rows, output.spec.dtype
        if key not in self._plans:
            self._plans[key] = self.context.specialize(
                gather,
                rows,
                self.parameters.vocabulary,
                self.parameters.width,
                self.weight.descriptor.encoding,
                cpu=self.context.backend == Backend.LLVM,
                dtype=output.spec.dtype,
                layout=self.weight.layout,
            )
        plan = self._plans[key]
        weight = self.weight.acquire(plan.signature[1])
        try:
            return (Prepared(self.context, plan, (tokens, weight, output)),)
        finally:
            weight.close()

    def close(self) -> None:
        self._plans.clear()
