"""Backend-free construction contracts, with explicit artifact binding stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..residency import BoundProgram, ModelResources
    from ..runtime import ModelStateStore

from ..contracts import ProgramSource


class StateFactory(ABC):
    @abstractmethod
    def validate(self, program: ProgramSource) -> None: ...

    @abstractmethod
    def create(
        self, program: BoundProgram, resources: ModelResources
    ) -> ModelStateStore[Any, Any]: ...
