"""Stateless model computation with the same ownership as decoder execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from .execution import ExecutionOwner, ExecutionScope, PendingExecution, ResourceLease


class ComputationCapacityError(MemoryError):
    """Reservation failed before any graph or input state was changed."""


class Computation(Protocol):
    @property
    def owner(self) -> ExecutionOwner: ...

    @property
    def batch_key(self) -> object:
        """Same owner and key permit a physical batch; operands remain row-local."""
        ...

    def run_batch(
        self, rows: tuple[Computation, ...], scope: ExecutionScope
    ) -> tuple[tuple[mx.array, ...], ...]: ...

    def reserve(self, rows: tuple[Computation, ...]) -> ResourceLease:
        """Acquire hard working capacity before entering numerical execution."""
        ...


@dataclass(frozen=True)
class Computed:
    arrays: tuple[mx.array, ...]
    execution: PendingExecution


def evaluate(rows: tuple[Computation, ...]) -> tuple[Computed, ...]:
    if not rows or any(
        row.owner is not rows[0].owner or row.batch_key != rows[0].batch_key for row in rows
    ):
        raise ValueError("computation requires compatible work under one execution owner")
    try:
        capacity = rows[0].reserve(rows)
    except MemoryError as error:
        raise ComputationCapacityError(str(error)) from error
    transferred = False
    try:
        with rows[0].owner.scope() as scope:
            scope.acquire(lambda: capacity)
            transferred = True
            outputs = rows[0].run_batch(rows, scope)
            if len(outputs) != len(rows) or any(not values for values in outputs):
                raise RuntimeError("computation omitted a row's outputs")
            execution = scope.seal(*(value for values in outputs for value in values))
    except BaseException:
        if not transferred and not rows[0].owner.requires_disposal:
            capacity.close()
        raise
    return tuple(Computed(values, execution) for values in outputs)
