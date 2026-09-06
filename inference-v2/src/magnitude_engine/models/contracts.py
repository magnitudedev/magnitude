"""Backend-free construction contracts, with explicit artifact binding stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magnitude_engine.artifacts.source import LocalArtifact
    from magnitude_engine.resources.io.reader import PositionalReader

    from .architectures.qwen35.mtp.loading import LoadedMTP
    from .residency import BoundProgram, ModelResources


class ProgramSource(ABC):
    @abstractmethod
    def load(self, resources: ModelResources) -> BoundProgram: ...

    def native_state_source(self) -> ProgramSource | None:
        return None

    def head_binding(self, resources: ModelResources) -> HeadBinding:
        raise ValueError("program does not expose the target binding required by a learned head")


class HeadBinding(ABC):
    @abstractmethod
    def load(
        self, artifact: LocalArtifact, reader: PositionalReader, resources: ModelResources
    ) -> LoadedMTP: ...
