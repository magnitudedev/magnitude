"""Backend-free operation signatures; arrays and execution state are worker types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import mlx.core as mx

    from ..execution import ExecutionScope


if TYPE_CHECKING:
    from ..loading.partitions import (
        ExpertPartition,
        OperationResources,
    )


class ExpertOperator(Protocol):
    def compute(
        self, hidden: mx.array, assignments: mx.array, scope: ExecutionScope
    ) -> mx.array: ...


class ExpertFactory(ABC):
    """Binds selected-expert execution; routing remains architecture-owned."""

    @abstractmethod
    def excluded(self, partition: ExpertPartition) -> frozenset[str]: ...

    @abstractmethod
    def bind(self, partition: ExpertPartition, resources: OperationResources) -> ExpertOperator: ...
