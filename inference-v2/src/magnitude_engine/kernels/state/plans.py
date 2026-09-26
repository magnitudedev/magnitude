"""Functional bounded KV append; allocation and visibility belong to the caller."""

import mlx.core as mx

from .. import kernel
from ..core.metal import Dispatch
from ..core.plan import Launch, Scalar, Source

APPEND = Source("state/append.metal")


@kernel(source=APPEND)
def append_tail(previous, keys, values, offsets, *, capacity):
    arguments = {"previous": previous, "keys": keys, "values": values, "offsets": offsets}
    inputs = (previous.value.tensor, keys.value.tensor, values.value.tensor, offsets.value.tensor)
    previous, keys, values, offsets = inputs
    batch, heads, count, width = keys.shape
    if (
        values.shape[:3] != (batch, heads, count)
        or offsets.size != batch
        or previous.size != batch * heads * capacity * (width + values.shape[-1])
        or (previous.dtype != keys.dtype)
        or (keys.dtype != values.dtype)
    ):
        raise ValueError("KV append operands disagree with the allocation geometry")
    inferred = (previous,)
    batch, heads, count, dk = inputs[1].shape
    constants = dict(
        BATCH=batch,
        HEADS=heads,
        COUNT=count,
        DK=dk,
        DV=inputs[2].shape[-1],
        CAPACITY=capacity,
        SIZE=inputs[0].size,
    )
    return Dispatch(
        arguments,
        dict(zip(("output",), inferred, strict=True)),
        Launch((inputs[0].size, 1, 1), (256, 1, 1)),
        constants=tuple((Scalar(k, v) for k, v in constants.items())),
    )


def update_tail(
    buffer: mx.array, new_keys: mx.array, new_values: mx.array, offsets: mx.array, capacity: int
) -> mx.array:
    return append_tail(buffer, new_keys, new_values, offsets, capacity=capacity)
