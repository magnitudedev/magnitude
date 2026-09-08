"""Packed gated-delta preparation: causal convolution, head norms and gates.

The short-sequence kernel preserves the MLX equation's intermediate dtype rounding.
One group owns a key-width channel block; retained convolution state is bounded.
Based on the independently qualified preparation equation in mlx-poc/kernels/gdn.py.
"""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

PREPARATION = Program("magnitude_delta_prepare", Source("recurrence/preparation.metal"))


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
    batch, count, width = projection.shape
    channels = 2 * key_heads * key_width + value_heads * value_width
    value_dim = value_heads * value_width
    if (
        key_width not in (32, 64, 128, 256)
        or channels % key_width
        or value_heads > channels
        or state.shape[0] != batch
        or state.shape[2] != channels
        or state.shape[1] < 1
        or width != channels + value_dim + 2 * value_heads
        or convolution.shape != (channels, state.shape[1] + 1, 1)
        or projection.dtype != state.dtype
        or convolution.dtype != projection.dtype
    ):
        raise ValueError("incompatible packed gated-delta preparation geometry")
    result = KernelPlan(
        program=PREPARATION,
        inputs=(
            Input("proj", projection),
            Input("state", state),
            Input("w", convolution),
            Input("A_log", log_rates.astype(mx.float32)),
            Input("dt_bias", time_bias),
        ),
        outputs=(
            Output("y", (batch, count, channels), projection.dtype),
            Output("new_state", state.shape, state.dtype),
            Output("z", (batch, count, value_dim), projection.dtype),
            Output("beta", (batch, count, value_heads), projection.dtype),
            Output("g", (batch, count, value_heads), mx.float32),
        ),
        launch=Launch((channels, batch, 1), (key_width, 1, 1)),
        template=(
            ("T", projection.dtype),
            ("CK", channels),
            ("CV", value_dim),
            ("HV", value_heads),
            ("K", state.shape[1] + 1),
            ("KEYDIM", key_heads * key_width),
            ("DK", key_width),
            ("TT", count),
        ),
    ).run()
    return tuple(result)
