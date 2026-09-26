"""Canonical short-query affine projection with separately selected scheduling."""

import mlx.core as mx

from .. import compile
from .affine import affine, prepare_input
from .tiles import packing


@compile
def _projection(inputs, weight, scales, biases, *, bits, group_size):
    rows, width = inputs.size // inputs.shape[-1], inputs.shape[-1]
    prepared = rows > 1 and weight.shape[-2] >= 4 * width
    values, sums = (
        prepare_input(inputs, bits=bits, pack=packing(width, weight.shape[-2], bits, group_size))
        if prepared
        else (inputs, inputs)
    )
    return affine(
        values, weight, scales, biases, sums, bits=bits, group_size=group_size, prepared=prepared
    )


def apply(
    inputs: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    bits: int,
    group_size: int,
) -> mx.array | None:
    width, outputs = inputs.shape[-1], weight.shape[-2]
    if (
        packing(width, outputs, bits, group_size) is None
        or inputs.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or scales.dtype != inputs.dtype
        or biases.dtype != inputs.dtype
    ):
        return None
    if not inputs.size:
        return mx.zeros((*inputs.shape[:-1], outputs), inputs.dtype)
    return _projection(inputs, weight, scales, biases, bits=bits, group_size=group_size)
