"""Logical token lookup with representation owned below the operation boundary."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from magnitude_engine.kernels.precision import Precision, floating
from magnitude_engine.operations.candidates import Plan, Selection, realize
from magnitude_engine.platform.execution import DType, Prepared, Tensor
from magnitude_engine.weights.residency import ResidentWeight


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


class ResidentEmbedding(Embedding):
    def __init__(self, weight: ResidentWeight, precision: Precision):
        if len(weight.descriptor.shape) != 2:
            raise ValueError("embedding requires matrix weights")
        self.weight, self.context, self.precision = weight, weight.context, precision
        self._parameters = EmbeddingParameters(*weight.descriptor.shape)
        self._plans: dict[tuple[int, DType], Plan] = {}

    @property
    def parameters(self) -> EmbeddingParameters:
        return self._parameters

    def plan(self, rows: int, dtype: DType) -> Plan:
        key = rows, dtype
        if key not in self._plans:
            from magnitude_engine.kernels.embedding.select import TABLE, EmbeddingShape

            shape = EmbeddingShape(
                rows, self._parameters.vocabulary, self._parameters.width, dtype
            )
            self._plans[key] = realize(
                "embedding",
                TABLE,
                self.context,
                Selection(
                    shape, self.precision, self.context.capability, self.weight.representation
                ),
            )
        return self._plans[key]

    def prepare(self, tokens: Tensor, output: Tensor) -> tuple[Prepared, ...]:
        if len(tokens.spec.shape) != 1:
            raise ValueError("embedding input must be a vector of token IDs")
        floating(output.spec.dtype)
        executable = self.plan(tokens.spec.shape[0], output.spec.dtype).executables[0]
        # A row-addressed gather binds its planes first; a flat one takes the
        # token vector first. Both end with the selected rows.
        weights = self.weight.acquire(
            tuple(spec for spec in executable.signature[:-1] if spec.dtype != DType.I32)
        )
        try:
            operands = []
            index = 0
            for spec in executable.signature[:-1]:
                if spec.dtype == DType.I32:
                    operands.append(tokens)
                else:
                    operands.append(weights[index])
                    index += 1
            return (Prepared(self.context, executable, (*operands, output)),)
        finally:
            for tensor in weights:
                tensor.close()

    def close(self) -> None:
        self._plans.clear()
