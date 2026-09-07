"""MLX-LM recurrence under the same prepared-input contract as native kernels."""

import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel

from magnitude_engine.components import component

from .inputs import DeltaInputs


@component("MODEL:GATED_DELTA:LM:STANDARD")
class MLXDelta:
    def advance(self, inputs: DeltaInputs, state: mx.array) -> tuple[mx.array, mx.array]:
        return gated_delta_kernel(
            inputs.queries,
            inputs.keys,
            inputs.values,
            inputs.decay,
            inputs.beta,
            state,
        )

    def reconcile(self, inputs: DeltaInputs, state: mx.array, count: int) -> mx.array:
        if not 0 <= count <= inputs.length:
            raise ValueError("invalid recurrence prefix")
        return state if count == 0 else self.advance(inputs.prefix(count), state)[1]
