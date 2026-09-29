"""Direct gated-delta equations for numerical qualification and fallback geometry."""

from __future__ import annotations

import mlx.core as mx

from magnitude_engine.components import component

from .inputs import DeltaInputs


@component("MODEL:GATED_DELTA:MAG:REFERENCE")
class DeltaReference:
    """Direct equations, used to qualify fused implementations and unusual geometry."""

    def advance(self, inputs: DeltaInputs, state: mx.array) -> tuple[mx.array, mx.array]:
        q, k, v = inputs.queries, inputs.keys, inputs.values
        repeats = v.shape[2] // k.shape[2]
        q = mx.repeat(q, repeats, axis=2).astype(mx.float32)
        k = mx.repeat(k, repeats, axis=2).astype(mx.float32)
        outputs = []
        for index in range(inputs.length):
            state = state * inputs.decay[:, index, :, None, None]
            error = v[:, index].astype(mx.float32) - mx.sum(
                state * k[:, index, :, None, :], axis=-1
            )
            correction = error * inputs.beta[:, index, :, None]
            state = state + correction[..., None] * k[:, index, :, None, :]
            outputs.append(mx.sum(state * q[:, index, :, None, :], axis=-1))
        return mx.stack(outputs, axis=1).astype(inputs.queries.dtype), state

    def reconcile(self, inputs: DeltaInputs, state: mx.array, count: int) -> mx.array:
        if not 0 <= count <= inputs.length:
            raise ValueError("invalid recurrence prefix")
        return state if count == 0 else self.advance(inputs.prefix(count), state)[1]
