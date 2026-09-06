"""Pure recurrent tensor computation, independent of transactions and execution leases."""

from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .contracts import DeltaRecurrence
from .inputs import DeltaInputs

Transform = Callable[[mx.array], mx.array]


@dataclass(frozen=True)
class DeltaGraph:
    qkv: Transform
    output_gate: Transform
    beta: Transform
    decay: Transform
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
        result = self(hidden, conv, memory)
        return result[0], mx.array(result[-2][:, -self.window :]), result[-1]

    def __call__(self, hidden: mx.array, conv: mx.array, memory: mx.array) -> tuple[mx.array, ...]:
        batch, count, _ = hidden.shape
        joined = mx.concatenate([conv, self.qkv(hidden)], axis=1)
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
            * nn.softplus(self.decay(hidden) + self.time_bias).astype(mx.float32)
        )
        beta = mx.sigmoid(self.beta(hidden))
        activation, updated = self.recurrence.advance(DeltaInputs(q, k, v, decay, beta), memory)
        gate = self.output_gate(hidden).reshape(activation.shape)
        result = self.output(self.normalize_output(activation, gate).reshape(batch, count, -1))
        return result, q, k, v, decay, beta, joined, updated
