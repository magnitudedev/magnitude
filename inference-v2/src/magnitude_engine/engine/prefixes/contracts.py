"""Backend-free engine policy contracts, independent of their concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .index import Checkpoint, PrefixStore

if TYPE_CHECKING:
    pass


class RetentionPolicy(ABC):
    max_entries: int
    max_bytes: int | None

    @abstractmethod
    def select(self, eligible: tuple[Checkpoint, ...]) -> tuple[Checkpoint, ...]:
        """Select victims from physically reclaimable checkpoints in recency order."""


class PrefixIndex(PrefixStore, ABC):
    retention: RetentionPolicy

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    def maintain(self) -> None: ...
