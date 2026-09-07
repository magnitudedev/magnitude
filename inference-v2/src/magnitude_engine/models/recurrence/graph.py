"""Pure recurrent tensor computation, independent of transactions and execution leases."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from magnitude_engine.models.projections import ParallelProjections

from .contracts import DeltaRecurrence
from .inputs import DeltaInputs
from .preparation import prepare

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
class DeltaGraph:
    projections: ParallelProjections
    convolution: Transform
    log_rates: mx.array
    time_bias: mx.array
    normalize_output: Callable[[mx.array, mx.array], mx.array]
    output: Transform
    key_heads: int
    value_heads: int
    key_width: int
    value_width: int
    window: int
    recurrence: DeltaRecurrence

    def advance(self, hidden: mx.array, conv: mx.array, memory: mx.array) -> tuple[mx.array, ...]:
        channels = 2 * self.key_heads * self.key_width + self.value_heads * self.value_width
        if (
            self.projections.packed
            and hidden.shape[1] <= 8
            and self.key_width in (32, 64, 128, 256)
            and channels % self.key_width == 0
            and isinstance(self.convolution, nn.Conv1d)
        ):
            projection = self.projections.operations[0](hidden)
            y, next_conv, gate, beta, decay = prepare(
                projection,
                conv,
                self.convolution.weight,
                self.log_rates,
                self.time_bias,
                key_heads=self.key_heads,
                key_width=self.key_width,
                value_heads=self.value_heads,
                value_width=self.value_width,
            )
            batch, count, _ = hidden.shape
            key_dim = self.key_heads * self.key_width
            q, k, v = mx.split(y, (key_dim, 2 * key_dim), axis=-1)
            q = q.reshape(batch, count, self.key_heads, self.key_width)
            k = k.reshape(q.shape)
            v = v.reshape(batch, count, self.value_heads, self.value_width)
            activation, updated = self.recurrence.advance(DeltaInputs(q, k, v, decay, beta), memory)
            result = self.output(
                self.normalize_output(activation, gate.reshape(activation.shape)).reshape(
                    batch, count, -1
                )
            )
            return result, next_conv, updated
        result = self(hidden, conv, memory)
        return result[0], mx.array(result[-2][:, -self.window :]), result[-1]

    def __call__(self, hidden: mx.array, conv: mx.array, memory: mx.array) -> tuple[mx.array, ...]:
        batch, count, _ = hidden.shape
        qkv, output_gate, beta_projection, decay_projection = self.projections(hidden)
        joined = mx.concatenate([conv, qkv], axis=1)
        convolved = nn.silu(self.convolution(joined))
        key_dim = self.key_heads * self.key_width
        q, k, v = mx.split(convolved, (key_dim, key_dim * 2), axis=-1)
        q = q.reshape(batch, count, self.key_heads, self.key_width)
        k = k.reshape(q.shape)
        inverse = self.key_width**-0.5
        q = inverse**2 * mx.fast.rms_norm(q, None, 1e-6)
        k = inverse * mx.fast.rms_norm(k, None, 1e-6)
        v = v.reshape(batch, count, self.value_heads, self.value_width)
        decay = mx.exp(
            -mx.exp(self.log_rates.astype(mx.float32))
            * nn.softplus(decay_projection + self.time_bias).astype(mx.float32)
        )
        beta = mx.sigmoid(beta_projection)
        activation, updated = self.recurrence.advance(DeltaInputs(q, k, v, decay, beta), memory)
        gate = output_gate.reshape(activation.shape)
        result = self.output(self.normalize_output(activation, gate).reshape(batch, count, -1))
        return result, q, k, v, decay, beta, joined, updated
