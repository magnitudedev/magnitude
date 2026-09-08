"""Functional bounded KV append; allocation and visibility belong to the caller."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

APPEND = Program("magnitude_append_tail", Source("state/append.metal"))


def update_tail(
    buffer: mx.array, new_keys: mx.array, new_values: mx.array, offsets: mx.array, capacity: int
) -> mx.array:
    """One bounded write over contiguous K and V regions of the same allocation."""
    batch, heads, count, key_width = new_keys.shape
    value_width = new_values.shape[-1]
    return KernelPlan(
        program=APPEND,
        inputs=(
            Input("previous", buffer),
            Input("keys", new_keys),
            Input("values", new_values),
            Input("offsets", offsets),
        ),
        outputs=(Output("output", buffer.shape, buffer.dtype),),
        launch=Launch((buffer.size, 1, 1), (256, 1, 1)),
        template=(
            ("B", batch),
            ("H", heads),
            ("C", capacity),
            ("K", key_width),
            ("V", value_width),
            ("N", count),
        ),
    ).run()[0]
