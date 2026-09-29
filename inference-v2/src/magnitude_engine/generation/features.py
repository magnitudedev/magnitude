"""Owned copies of target features retained beyond the producing model advance."""

import mlx.core as mx

from magnitude_engine.resources.budget import MemoryBudget


class RetainedFeature:
    """Detach a token window so it cannot retain an entire verification activation.

    Copy remains lazy. Its producer/consumer execution must complete before close;
    it can be retained as an execution lease when consumed by a head.
    """

    def __init__(self, value: mx.array, budget: MemoryBudget):
        self.reservation = budget.reserve("generation-features", value.nbytes)
        self._value: mx.array | None = None
        try:
            self._value = mx.array(value)
        except BaseException:
            self.reservation.close()
            raise

    @property
    def value(self) -> mx.array:
        if self._value is None:
            raise RuntimeError("retained feature is closed")
        return self._value

    def close(self) -> None:
        self._value = None
        self.reservation.close()
