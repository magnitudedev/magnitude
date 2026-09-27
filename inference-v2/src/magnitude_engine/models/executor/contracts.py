"""Backend-free construction contracts, with explicit artifact binding stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..residency import BoundExecutor, ModelResources


class ExecutorFactory(ABC):
    @abstractmethod
    def load(self, resources: ModelResources) -> BoundExecutor: ...
