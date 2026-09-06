from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnitude_engine.models.residency import BoundExecutor, ModelResources

    from .binding import BoundGeneration
    from .methods.contracts import GenerationMethod


class MethodFactory(ABC):
    @abstractmethod
    def bind(
        self, target: BoundExecutor, resources: ModelResources
    ) -> tuple[GenerationMethod, int, str | None]: ...


class GenerationFactory(ABC):
    @abstractmethod
    def load(self, resources: ModelResources) -> BoundGeneration: ...
