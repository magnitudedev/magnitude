"""Gather and affine-dequantize resident vocabulary rows in one dispatch."""

from dataclasses import dataclass

import mlx.core as mx

from ..core.graph import Tensor, signature
from ..core.kernel import Kernel
from ..core.plan import Launch, Source
from ..core.primitive import Primitive

EMBEDDING = Source("contractions/embedding.metal")


@dataclass(frozen=True)
class Embedding(Primitive):
    """Select an encoded row; decode each group with one FP32 FMA and native cast."""

    bits: int
    group_size: int

    def infer(self, inputs):
        rows, weight, scales, biases = inputs
        width = weight.shape[-1] * 32 // self.bits
        if (
            weight.dtype != mx.uint32
            or scales != biases
            or width % self.group_size
            or scales.shape != (weight.shape[0], width // self.group_size)
        ):
            raise ValueError("embedding operands disagree with the affine encoding")
        return (Tensor((*rows.shape, width), scales.dtype),)

    def lower(self, inputs):
        output = self.infer(inputs)
        width = output[0].shape[-1]
        return Kernel(
            signature(("tok", "weight", "scales", "biases"), inputs),
            signature(("out",), output),
            EMBEDDING,
            Launch((width, inputs[0].size, 1), (min(width, 256), 1, 1)),
            (
                ("T", inputs[2].dtype),
                ("WIDTH", width),
                ("VOCAB", inputs[1].shape[0]),
                ("BITS", self.bits),
                ("GROUP", self.group_size),
            ),
        ).bind()


def lookup(rows, weight, scales, biases, *, bits: int, group_size: int) -> mx.array:
    return Embedding(bits, group_size)(rows, weight, scales, biases)[0]
