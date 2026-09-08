"""Functional bounded KV append; allocation and visibility belong to the caller."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Scalar, Source
from ..core.primitive import Primitive

APPEND = Source("state/append.metal")


@dataclass(frozen=True)
class AppendTail(Primitive):
    """Copy the old allocation, replacing only each row's bounded append interval."""

    capacity: int

    def infer(self, inputs):
        previous, keys, values, offsets = inputs
        batch, heads, count, width = keys.shape
        if (
            values.shape[:3] != (batch, heads, count)
            or offsets.size != batch
            or previous.size != batch * heads * self.capacity * (width + values.shape[-1])
            or previous.dtype != keys.dtype
            or keys.dtype != values.dtype
        ):
            raise ValueError("KV append operands disagree with the allocation geometry")
        return (previous,)

    def lower(self, inputs):
        batch, heads, count, dk = inputs[1].shape
        constants = dict(
            BATCH=batch,
            HEADS=heads,
            COUNT=count,
            DK=dk,
            DV=inputs[2].shape[-1],
            CAPACITY=self.capacity,
            SIZE=inputs[0].size,
        )
        return Kernel(
            signature(("previous", "keys", "values", "offsets"), inputs),
            signature(("output",), self.infer(inputs)),
            APPEND,
            Launch((inputs[0].size, 1, 1), (256, 1, 1)),
            constants=tuple(Scalar(k, v) for k, v in constants.items()),
        ).bind()


def update_tail(
    buffer: mx.array, new_keys: mx.array, new_values: mx.array, offsets: mx.array, capacity: int
) -> mx.array:
    return AppendTail(capacity)(buffer, new_keys, new_values, offsets)[0]
