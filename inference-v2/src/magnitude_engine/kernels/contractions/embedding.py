"""Gather and affine-dequantize resident vocabulary rows in one dispatch."""

import mlx.core as mx

from .. import kernel
from ..core.graph import Tensor
from ..core.metal import Dispatch
from ..core.plan import Launch, Source

EMBEDDING = Source("contractions/embedding.metal")


@kernel(source=EMBEDDING)
def embedding(tok, weight, scales, biases, *, bits, group_size):
    arguments = {"tok": tok, "weight": weight, "scales": scales, "biases": biases}
    inputs = (tok.value.tensor, weight.value.tensor, scales.value.tensor, biases.value.tensor)
    rows, weight, scales, biases = inputs
    width = weight.shape[-1] * 32 // bits
    if (
        weight.dtype != mx.uint32
        or scales != biases
        or width % group_size
        or (scales.shape != (weight.shape[0], width // group_size))
    ):
        raise ValueError("embedding operands disagree with the affine encoding")
    inferred = (Tensor((*rows.shape, width), scales.dtype),)
    output = inferred
    width = output[0].shape[-1]
    return Dispatch(
        arguments,
        dict(zip(("out",), output, strict=False)),
        Launch((width, inputs[0].size, 1), (min(width, 256), 1, 1)),
        (
            ("T", inputs[2].dtype),
            ("WIDTH", width),
            ("VOCAB", inputs[1].shape[0]),
            ("BITS", bits),
            ("GROUP", group_size),
        ),
    )


def lookup(rows, weight, scales, biases, *, bits: int, group_size: int) -> mx.array:
    return embedding(rows, weight, scales, biases, bits=bits, group_size=group_size)
