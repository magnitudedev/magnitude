"""Packed causal convolution, ordered head reductions and typed gate primitives."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import Tensor, signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Source
from ..core.primitive import Primitive

NONLINEAR = Source("reductions/nonlinear.metal")


PREPARATION = Source("recurrence/preparation.metal", (NONLINEAR,))


@dataclass(frozen=True)
class Preparation(Primitive):
    """Causal FP32 convolution; native SiLU/norm boundaries; functional state advance."""

    key_heads: int
    key_width: int
    value_heads: int
    value_width: int

    def infer(self, inputs):
        projection, state, convolution, rates, bias = inputs
        batch, count, width = projection.shape
        channels = 2 * self.key_heads * self.key_width + self.value_heads * self.value_width
        value_dim = self.value_heads * self.value_width
        if (
            self.key_width not in (32, 64, 128, 256)
            or channels % self.key_width
            or self.value_heads > channels
            or state.shape[0] != batch
            or state.shape[2] != channels
            or state.shape[1] < 1
            or width != channels + value_dim + 2 * self.value_heads
            or convolution.shape != (channels, state.shape[1] + 1, 1)
            or projection.dtype != state.dtype
            or convolution.dtype != projection.dtype
        ):
            raise ValueError("incompatible packed gated-delta preparation geometry")
        return (
            Tensor((batch, count, channels), projection.dtype),
            state,
            Tensor((batch, count, value_dim), projection.dtype),
            Tensor((batch, count, self.value_heads), projection.dtype),
            Tensor((batch, count, self.value_heads), mx.float32),
        )

    def lower(self, inputs):
        outputs = self.infer(inputs)
        batch, count, _ = inputs[0].shape
        channels, value_dim = outputs[0].shape[-1], outputs[2].shape[-1]
        return Kernel(
            signature(("proj", "state", "w", "A_log", "dt_bias"), inputs),
            signature(("y", "new_state", "z", "beta", "g"), outputs),
            PREPARATION,
            Launch((channels, batch, 1), (self.key_width, 1, 1)),
            (
                ("T", inputs[0].dtype),
                ("TT", count),
                ("CK", channels),
                ("CV", value_dim),
                ("DK", self.key_width),
                ("HV", self.value_heads),
                ("KEYDIM", self.key_heads * self.key_width),
                ("K", inputs[1].shape[1] + 1),
            ),
        ).bind()


def prepare(
    projection: mx.array,
    state: mx.array,
    convolution: mx.array,
    log_rates: mx.array,
    time_bias: mx.array,
    *,
    key_heads: int,
    key_width: int,
    value_heads: int,
    value_width: int,
) -> tuple[mx.array, ...]:
    return Preparation(key_heads, key_width, value_heads, value_width)(
        projection, state, convolution, log_rates.astype(mx.float32), time_bias
    )
