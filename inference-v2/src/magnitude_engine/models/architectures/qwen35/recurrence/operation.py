"""Map Qwen recurrent layers to their per-sequence hybrid state slots."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx

from magnitude_engine import components as c
from magnitude_engine.components import component
from magnitude_engine.models.execution import ExecutionScope
from magnitude_engine.models.recurrence.gated_delta import GatedDelta
from magnitude_engine.models.state.hybrid import HybridState

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
@component(c.QWEN_RECURRENCE, source=c.Source.MAG, variant="COMPILED_REGION")
class RecurrentMixer:
    index: int
    operation: GatedDelta

    def compute_batch(
        self, hidden: mx.array, states: tuple[HybridState, ...], scope: ExecutionScope
    ) -> mx.array:
        commitments = tuple(
            state.active.committed_inputs if state.active is not None else 0 for state in states
        )
        if not commitments or any(value != commitments[0] for value in commitments):
            raise ValueError("recurrent batch requires a uniform committed-input prefix")
        return self.operation.compute_batch(
            hidden,
            tuple(s.slots[self.index] for s in states),
            scope,
            committed_inputs=commitments[0],
        )
