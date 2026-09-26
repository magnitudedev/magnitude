"""Development phase times, distinct from measured operation durations."""

from contextlib import contextmanager
from time import perf_counter_ns

from .records import Phase, PhaseTime


class PhaseClock:
    def __init__(self):
        self._elapsed: dict[Phase, int] = {}

    @contextmanager
    def track(self, phase: Phase):
        started = perf_counter_ns()
        try:
            yield
        finally:
            self._elapsed[phase] = self._elapsed.get(phase, 0) + perf_counter_ns() - started

    def snapshot(self) -> tuple[PhaseTime, ...]:
        return tuple(PhaseTime(phase=phase, elapsed_ns=self._elapsed[phase])
                     for phase in Phase if phase in self._elapsed)
