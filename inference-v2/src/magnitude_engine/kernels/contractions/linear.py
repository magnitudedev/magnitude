"""Affine short-query contractions: fixed row arithmetic, shared encoded loads."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

from .tiles import TILE, packing, row_tile

PREPARE = Program(
    "magnitude_affine_input",
    Source("contractions/affine_input.metal", (Source("contractions/encoded.metal"),)),
)
LINEAR = Program("magnitude_affine_linear", Source("contractions/linear.metal", (TILE,)))


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
    pack = packing(width, outputs, bits, group_size)
    if (
        pack is None
        or inputs.dtype not in (mx.float16, mx.bfloat16, mx.float32)
        or scales.dtype != inputs.dtype
        or biases.dtype != inputs.dtype
    ):
        return None
    rows = inputs.size // width
    if not rows:
        return mx.zeros((*inputs.shape[:-1], outputs), inputs.dtype)
    tile = row_tile(rows, maximum=4)
    # Very wide projections reuse each input pack across many output tiles.
    # Materialize its affine preparation once rather than repeat dtype arithmetic.
    prepared = rows > 1 and outputs >= 4 * width
    values = inputs
    sums = mx.zeros((1,), mx.float32)
    if prepared:
        values, sums = KernelPlan(
            program=PREPARE,
            inputs=(Input("x", inputs),),
            outputs=(
                Output("prepared", inputs.shape, mx.float32),
                Output("sums", (rows, width // pack), mx.float32),
            ),
            launch=Launch((rows * width // pack, 1, 1), (256, 1, 1)),
            template=(
                ("T", inputs.dtype),
                ("M", rows),
                ("K", width),
                ("BITS", bits),
                ("PACK", pack),
            ),
        ).run()
    return KernelPlan(
        program=LINEAR,
        inputs=(
            Input("x", values),
            Input("sums", sums),
            Input("w", weight),
            Input("scales", scales),
            Input("biases", biases),
        ),
        outputs=(Output("out", (*inputs.shape[:-1], outputs), inputs.dtype),),
        launch=Launch(
            (64, (outputs + 7) // 8, (rows + tile - 1) // tile),
            (64, 1, 1),
        ),
        template=(
            ("T", inputs.dtype),
            ("K", width),
            ("N", outputs),
            ("M", rows),
            ("R", tile),
            ("X", values.dtype),
            ("PREPARED", prepared),
            ("BITS", bits),
            ("GROUP", group_size),
            ("PACK", pack),
        ),
    ).run()[0]
