from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory.contracts import MemoryPolicy
    from .runtime import Engine


class EngineInstance(ABC):
    engine: "Engine"
    properties: dict
    output_capacity: int
    budget: "MemoryPolicy"

    @abstractmethod
    def close(self) -> None: ...
