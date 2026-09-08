"""Gated-delta recurrence with explicit prepared inputs and prefix reconciliation."""

from __future__ import annotations

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.kernels.recurrence import plans

from .inputs import DeltaInputs


@component("MODEL:GATED_DELTA:MAG:FUSED_UPDATE")
class MetalDelta:
    """Native recurrence; long lengths normally share one runtime-count kernel.

    Explicit prefill specialization exists for controlled kernel comparisons. Short
    verification blocks keep their fixed-width variants in either configuration.
    """

    def __init__(self, *, specialize_prefill: bool = False):
        self.specialize_prefill = specialize_prefill

    def _run(self, inputs: DeltaInputs, state: mx.array, state_only: bool):
        batch, tokens, hk, dk = inputs.keys.shape
        hv, dv = inputs.values.shape[2:]
        if (
            tokens < 1
            or dk % 32
            or hv % hk
            or state.dtype != mx.float32
            or state.shape != (batch, hv, dv, dk)
            or inputs.queries.shape != inputs.keys.shape
            or inputs.values.shape[:2] != (batch, tokens)
            or inputs.decay.shape != (batch, tokens, hv)
            or inputs.beta.shape != (batch, tokens, hv)
        ):
            raise ValueError("unsupported gated-delta geometry")
        return plans.advance(
            inputs.queries,
            inputs.keys,
            inputs.values,
            inputs.decay,
            inputs.beta,
            state,
            state_only=state_only,
            specialize_prefill=self.specialize_prefill,
        )

    def advance(self, inputs: DeltaInputs, state: mx.array) -> tuple[mx.array, mx.array]:
        final, output = self._run(inputs, state, False)
        return output, final

    def reconcile(self, inputs: DeltaInputs, state: mx.array, count: int) -> mx.array:
        if not 0 <= count <= inputs.length:
            raise ValueError("invalid recurrence prefix")
        return state if count == 0 else self._run(inputs.prefix(count), state, True)[0]
