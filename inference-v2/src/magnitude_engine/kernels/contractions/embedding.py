"""Gather and affine-dequantize resident vocabulary rows in one dispatch."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

EMBEDDING = Program("magnitude_embedding", Source("contractions/embedding.metal"))


def lookup(
    rows: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
    *,
    bits: int,
    group_size: int,
) -> mx.array:
    width = weight.shape[-1] * 32 // bits
    return KernelPlan(
        program=EMBEDDING,
        inputs=(
            Input("tok", rows),
            Input("weight", weight),
            Input("scales", scales),
            Input("biases", biases),
        ),
        outputs=(Output("out", (*rows.shape, width), scales.dtype),),
        launch=Launch((width, rows.size, 1), (min(width, 256), 1, 1)),
        template=(
            ("T", scales.dtype),
            ("D", width),
            ("GROUP", group_size),
            ("BITS", bits),
            ("VOCAB", weight.shape[0]),
        ),
    ).run()[0]
