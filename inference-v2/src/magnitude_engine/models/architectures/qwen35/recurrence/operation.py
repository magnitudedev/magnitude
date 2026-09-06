"""Map Qwen recurrent layers to their per-sequence hybrid state slots."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.recurrence.gated_delta import GatedDelta
from magnitude_engine.models.state.hybrid import HybridState

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
class RecurrentMixer:
    index: int
    operation: GatedDelta

    def compute_batch(
        self, hidden: mx.array, states: tuple[HybridState, ...], scope: ExecutionScope
    ) -> mx.array:
        return self.operation.compute_batch(
            hidden,
            tuple(s.slots[self.index] for s in states),
            scope,
            committed_inputs=min(
                state.active.committed_inputs if state.active is not None else 0 for state in states
            ),
        )
