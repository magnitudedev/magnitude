"""Account for allocations before creating them, including replacement peaks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import RLock

from magnitude_engine import components as c
from magnitude_engine.components import component


@dataclass(frozen=True)
class Usage:
    limit: int
    reserved: int
    peak: int
    owners: dict[str, int]


class Reservation:
    def __init__(self, budget: MemoryBudget, owner: str, size: int):
        self._budget = budget
        self.owner = owner
        self.size = size
        self._closed = False

    def close(self) -> None:
        with self._budget._lock:
            if not self._closed:
                self._budget._owners[self.owner] -= self.size
                self._budget._reserved -= self.size
                self._closed = True

    def __enter__(self) -> Reservation:
        if self._closed:
            raise RuntimeError("reservation is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@component(c.MEMORY, source=c.Source.MAG, variant="RESERVATIONS")
class MemoryBudget:
    """A worker-wide reservation ledger; observed OS usage is a separate measurement.

    The policy chooses the ceiling. Allocation owners acquire and release reservations;
    borrowing an operation does not reserve the same allocation a second time.
    """

    def __init__(self, limit: int):
        if limit < 0:
            raise ValueError("memory limit must be nonnegative")
        self.limit = limit
        self._reserved = 0
        self._peak = 0
        self._owners: Counter[str] = Counter()
        self._lock = RLock()

    def reserve(self, owner: str, size: int) -> Reservation:
        if not owner or size < 0:
            raise ValueError("reservation needs an owner and nonnegative size")
        with self._lock:
            if self._reserved + size > self.limit:
                raise MemoryError(
                    f"{owner} requires {size} bytes; {self.limit - self._reserved} free"
                )
            self._reserved += size
            self._peak = max(self._peak, self._reserved)
            self._owners[owner] += size
            return Reservation(self, owner, size)

    def snapshot(self) -> Usage:
        with self._lock:
            return Usage(self.limit, self._reserved, self._peak, +self._owners)
