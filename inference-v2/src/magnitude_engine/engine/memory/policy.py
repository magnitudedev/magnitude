"""Engine-level resource policy; allocation implementations report actual reclamation."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import get_ident

from magnitude_engine.resources.budget import Reservation

from ..prefixes.contracts import PrefixIndex
from .contracts import MemoryPolicy, PressurePolicy


@dataclass(frozen=True)
class EvictPrefixesBeforeRejecting(PressurePolicy):
    def relieve(self, prefixes: PrefixIndex) -> bool:
        eligible = tuple(c for c in prefixes.eligible() if c.reclaimable)
        victims = prefixes.retention.select(eligible)
        if not victims:
            return False
        prefixes.discard(victims)
        return True


class Budgeted(MemoryPolicy):
    def __init__(self, *, limit_bytes: int, pressure: PressurePolicy):
        super().__init__(limit_bytes)
        self.pressure = pressure
        self._reclaim: Callable[[], bool] = lambda: False
        self._reclaim_thread = get_ident()

    def bind_reclaimer(self, reclaim: Callable[[], bool]) -> None:
        self._reclaim_thread = get_ident()
        self._reclaim = reclaim

    def reserve(self, owner: str, size: int) -> Reservation:
        while True:
            try:
                return super().reserve(owner, size)
            except MemoryError:
                if get_ident() != self._reclaim_thread or not self._reclaim():
                    raise
