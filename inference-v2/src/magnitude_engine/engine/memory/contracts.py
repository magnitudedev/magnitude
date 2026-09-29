"""Backend-free engine policy contracts, independent of their concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from magnitude_engine.resources.budget import MemoryBudget

if TYPE_CHECKING:
    pass

from ..prefixes.contracts import PrefixIndex


class PressurePolicy(ABC):
    @abstractmethod
    def relieve(self, prefixes: PrefixIndex) -> bool: ...


class MemoryPolicy(MemoryBudget, ABC):
    pressure: PressurePolicy

    @abstractmethod
    def bind_reclaimer(self, reclaim: Callable[[], bool]) -> None: ...
