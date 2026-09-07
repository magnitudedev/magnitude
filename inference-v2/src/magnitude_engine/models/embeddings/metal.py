"""Gather and affine-dequantize resident vocabulary rows in one dispatch."""

from functools import cache
from typing import Any

import mlx.core as mx

_SOURCE = """
    uint d = thread_position_in_grid.x;      // element in [0, D)
    if (d >= D) return;
    uint b = thread_position_in_grid.y;      // token row
    constexpr int KW = (D * BITS) / 32;
    constexpr int G = D / GROUP;
    int token = int(tok[b]);
    uint row = uint(token < 0 ? token + VOCAB : token);
    const device uint* wrow = weight + row * KW;
    uint bp = d * BITS;
    uint wi = bp >> 5, sh = bp & 31u;
    uint v = wrow[wi] >> sh;
    if (sh + BITS > 32u) v |= wrow[wi + 1] << (32u - sh);
    uint qv = v & ((1u << BITS) - 1u);
    T sc = scales[row * G + d / GROUP];
    T bi = biases[row * G + d / GROUP];
    out[b * D + d] = T(metal::fma(float(sc), float(qv), float(bi)));

"""


@cache
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="magnitude_embedding",
        input_names=["tok", "weight", "scales", "biases"],
        output_names=["out"],
        source=_SOURCE,
    )


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
    return _kernel()(
        inputs=[rows, weight, scales, biases],
        template=[
            ("T", scales.dtype),
            ("D", width),
            ("GROUP", group_size),
            ("BITS", bits),
            ("VOCAB", weight.shape[0]),
        ],
        grid=(width, rows.size, 1),
        threadgroup=(min(width, 256), 1, 1),
        output_shapes=[(*rows.shape, width)],
        output_dtypes=[scales.dtype],
    )[0]
