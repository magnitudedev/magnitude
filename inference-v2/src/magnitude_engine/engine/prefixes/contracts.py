"""Backend-free engine policy contracts, independent of their concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .index import Checkpoint, PrefixStore


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

    def prefill_allowance(self, start: int, allowance: int, boundaries: tuple[int, ...]) -> int:
        """Request a checkpoint boundary only when this index retains checkpoints."""
        if self.enabled:
            return min(allowance, next((p - start for p in boundaries if p > start), allowance))
        return allowance

    @abstractmethod
    def maintain(self) -> None: ...
