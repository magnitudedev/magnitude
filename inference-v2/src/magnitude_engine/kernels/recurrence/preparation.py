"""Packed causal convolution, ordered head reductions and typed gate primitives."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Source

NONLINEAR = Source("reductions/nonlinear.metal")
PREPARATION = Source("recurrence/preparation.metal", (NONLINEAR,))


@kernel(source=PREPARATION)
def prepare_recurrence(
    proj, state, w, A_log, dt_bias, *, key_heads, key_width, value_heads, value_width
):
    arguments = {"proj": proj, "state": state, "w": w, "A_log": A_log, "dt_bias": dt_bias}
    inputs = (
        proj.value.tensor,
        state.value.tensor,
        w.value.tensor,
        A_log.value.tensor,
        dt_bias.value.tensor,
    )
    projection, state, convolution, rates, bias = inputs
    batch, count, width = projection.shape
    channels = 2 * key_heads * key_width + value_heads * value_width
    value_dim = value_heads * value_width
    if (
        key_width not in (32, 64, 128, 256)
        or channels % key_width
        or value_heads > channels
        or (state.shape[0] != batch)
        or (state.shape[2] != channels)
        or (state.shape[1] < 1)
        or (width != channels + value_dim + 2 * value_heads)
        or (convolution.shape != (channels, state.shape[1] + 1, 1))
        or (projection.dtype != state.dtype)
        or (convolution.dtype != projection.dtype)
    ):
        raise ValueError("incompatible packed gated-delta preparation geometry")
    inferred = (
        Tensor((batch, count, channels), projection.dtype),
        state,
        Tensor((batch, count, value_dim), projection.dtype),
        Tensor((batch, count, value_heads), projection.dtype),
        Tensor((batch, count, value_heads), mx.float32),
    )
    outputs = inferred
    batch, count, _ = inputs[0].shape
    channels, value_dim = (outputs[0].shape[-1], outputs[2].shape[-1])
    return Dispatch(
        arguments,
        dict(zip(("y", "new_state", "z", "beta", "g"), outputs, strict=False)),
        Launch((channels, batch, 1), (key_width, 1, 1)),
        (
            ("T", inputs[0].dtype),
            ("TT", count),
            ("CK", channels),
            ("CV", value_dim),
            ("DK", key_width),
            ("HV", value_heads),
            ("KEYDIM", key_heads * key_width),
            ("K", inputs[1].shape[1] + 1),
        ),
    )


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
    return prepare_recurrence(
        projection,
        state,
        convolution,
        log_rates.astype(mx.float32),
        time_bias,
        key_heads=key_heads,
        key_width=key_width,
        value_heads=value_heads,
        value_width=value_width,
    )
