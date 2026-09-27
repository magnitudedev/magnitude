"""Backend-free operation signatures; arrays and execution state are worker types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import mlx.core as mx

    from ..execution import ExecutionScope


if TYPE_CHECKING:
    from ..loading.partitions import (
        EmbeddingPartition,
        OperationResources,
    )


class EmbeddingLookup(Protocol):
    def lookup(self, rows: mx.array, scope: ExecutionScope) -> mx.array: ...


class EmbeddingFactory(ABC):
    """Binds one architecture-assigned embedding tensor partition."""

    @abstractmethod
    def excluded(self, partition: EmbeddingPartition) -> frozenset[str]: ...

    @abstractmethod
    def bind(
        self, partition: EmbeddingPartition, resources: OperationResources
    ) -> EmbeddingLookup: ...
