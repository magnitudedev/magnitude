"""Physical reservations shared by allocations, views, and execution leases."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


class CapacityError(MemoryError):
    def __init__(self, required: int, available: int, *, constraint: str = "runtime"):
        super().__init__(f"{constraint}: allocation needs {required} bytes; {available} remain")
        self.required, self.available, self.constraint = required, available, constraint


@dataclass(frozen=True, slots=True)
class Limit:
    identity: str
    domains: frozenset[str]
    maximum: int


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    reserved_bytes: int
    peak_bytes: int
    by_constraint: tuple[tuple[str, int, int], ...]
    revision: int


@dataclass(frozen=True, slots=True)
class ConstraintMeasurement:
    identity: str
    baseline_bytes: int
    peak_bytes: int
    end_bytes: int
    maximum_bytes: int

    def __post_init__(self):
        if (not self.identity or min(self.baseline_bytes, self.end_bytes) < 0
                or self.peak_bytes < max(self.baseline_bytes, self.end_bytes)
                or self.peak_bytes > self.maximum_bytes):
            raise ValueError("invalid physical-constraint observation")


@dataclass(frozen=True, slots=True)
class MemoryMeasurement:
    """Unique reserved backing during one interval, not the lifetime high-water mark."""

    baseline_bytes: int
    peak_bytes: int
    end_bytes: int
    reserved_bytes: int
    released_bytes: int
    constraints: tuple[ConstraintMeasurement, ...]

    def __post_init__(self):
        if (min(self.baseline_bytes, self.end_bytes, self.reserved_bytes, self.released_bytes) < 0
                or self.peak_bytes < max(self.baseline_bytes, self.end_bytes)
                or self.baseline_bytes + self.reserved_bytes - self.released_bytes != self.end_bytes):
            raise ValueError("invalid memory observation accounting")
        if len({item.identity for item in self.constraints}) != len(self.constraints):
            raise ValueError("physical constraints must be unique")


class MemoryWindow:
    def __init__(self, ledger: ReservationLedger):
        self._ledger = ledger
        self._baseline = ledger.reserved
        self._peak = ledger.reserved
        self._constraint_baseline = dict(ledger._used)
        self._constraint_peak = dict(ledger._used)
        self._reserved = self._released = 0
        self._result: MemoryMeasurement | None = None

    def _change(self, size: int) -> None:
        if size > 0:
            self._reserved += size
        else:
            self._released -= size
        self._peak = max(self._peak, self._ledger.reserved)
        for identity, used in self._ledger._used.items():
            self._constraint_peak[identity] = max(self._constraint_peak[identity], used)

    def close(self) -> MemoryMeasurement:
        if self._result is None:
            ledger = self._ledger
            ledger._windows.remove(self)
            self._result = MemoryMeasurement(
                self._baseline, self._peak, ledger.reserved,
                self._reserved, self._released,
                tuple(ConstraintMeasurement(
                    limit.identity, self._constraint_baseline[limit.identity],
                    self._constraint_peak[limit.identity], ledger._used[limit.identity],
                    limit.maximum,
                ) for limit in ledger.limits),
            )
        return self._result


class Reservation:
    def __init__(self, ledger: ReservationLedger, size: int, domains: frozenset[str]):
        self.ledger, self.size, self.domains = ledger, size, domains
        self.closed = False

    def close(self) -> None:
        if not self.closed:
            self.ledger._release(self.size, self.domains)
            self.closed = True


class ReservationLedger:
    def __init__(self, limits: Iterable[Limit]):
        self.limits = tuple(limits)
        if not self.limits or any(item.maximum <= 0 or not item.domains for item in self.limits):
            raise ValueError("physical memory limits need capacity and backing domains")
        if len({item.identity for item in self.limits}) != len(self.limits):
            raise ValueError("duplicate physical constraint identity")
        self._used = {item.identity: 0 for item in self.limits}
        self.reserved = self.peak = self.revision = 0
        self._windows: set[MemoryWindow] = set()

    def observe(self) -> MemoryWindow:
        window = MemoryWindow(self)
        self._windows.add(window)
        return window

    def available(self, domains: frozenset[str]) -> int:
        matching = [item.maximum - self._used[item.identity] for item in self.limits if item.domains & domains]
        if not matching:
            raise ValueError("allocation has no physical capacity constraint")
        return min(matching)

    def reserve(self, size: int, domains: frozenset[str]) -> Reservation:
        if size <= 0 or not domains:
            raise ValueError("a reservation needs positive bytes and backing domains")
        matching = tuple(item for item in self.limits if item.domains & domains)
        if not matching:
            raise ValueError("allocation has no physical capacity constraint")
        for limit in matching:
            available = limit.maximum - self._used[limit.identity]
            if size > available:
                raise CapacityError(size, available, constraint=limit.identity)
        for limit in matching:
            self._used[limit.identity] += size
        self.reserved += size
        self.peak = max(self.peak, self.reserved)
        self.revision += 1
        for window in self._windows:
            window._change(size)
        return Reservation(self, size, domains)

    def _release(self, size: int, domains: frozenset[str]) -> None:
        for limit in self.limits:
            if limit.domains & domains:
                if self._used[limit.identity] < size:
                    raise RuntimeError("physical reservation underflow")
                self._used[limit.identity] -= size
        self.reserved -= size
        self.revision += 1
        for window in self._windows:
            window._change(-size)

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            self.reserved, self.peak,
            tuple((item.identity, self._used[item.identity], item.maximum) for item in self.limits),
            self.revision,
        )
