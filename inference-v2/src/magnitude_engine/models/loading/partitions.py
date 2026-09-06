"""Architecture-assigned operation inputs, before choosing resident or streamed storage."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from magnitude_engine.artifacts.layouts import LogicalTensor
from magnitude_engine.artifacts.quantization import AffineEncoding
from magnitude_engine.resources.budget import MemoryBudget
from magnitude_engine.resources.lifetime import Closable


class OperationResources:
    """One model's operation allocations, all charged to the engine's ledger."""

    def __init__(self, budget: MemoryBudget):
        self.budget = budget
        self.lifetime = ExitStack()
        self._shared: dict[tuple, object] = {}

    def own[T: Closable](self, value: T) -> T:
        self.lifetime.callback(value.close)
        return value

    def shared[T](self, owner: object, geometry: tuple, create: Callable[[], T]) -> T:
        from typing import cast

        key = (owner, geometry)
        if key not in self._shared:
            self._shared[key] = create()
        return cast(T, self._shared[key])

    def close(self) -> None:
        self._shared.clear()
        self.lifetime.close()


@dataclass(frozen=True)
class EmbeddingPartition:
    name: str
    module: Any
    tensors: dict[str, LogicalTensor]
    encoding: AffineEncoding | None
    retained_for_readout: bool = False


@dataclass(frozen=True)
class ExpertPartition:
    name: str
    up: Any
    gate: Any
    down: Any
    tensors: dict[str, LogicalTensor]
    encodings: dict[str, AffineEncoding]
    activation: Callable[[mx.array, mx.array], mx.array]
