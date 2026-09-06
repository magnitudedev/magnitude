"""Backend-free operation signatures; arrays and execution state are worker types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import mlx.core as mx

    from .inputs import DeltaInputs


class DeltaRecurrence(Protocol):
    def advance(self, inputs: DeltaInputs, state: mx.array) -> tuple[mx.array, mx.array]: ...
    def reconcile(self, inputs: DeltaInputs, state: mx.array, count: int) -> mx.array: ...
