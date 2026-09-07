"""Fused activations with explicit intermediate rounding under MLX compilation."""

from functools import cache
from typing import Any

import mlx.core as mx


@cache
def _sigmoid_gate() -> Any:
    return mx.fast.metal_kernel(
        name="sigmoid_gate",
        input_names=["values", "gates"],
        output_names=["output"],
        source="""
            uint i = thread_position_in_grid.x;
            if (i >= N) return;
            T x = gates[SHARED ? i / WIDTH : i];
            // Preserve MLX's dtype arithmetic and its precise unary exponential.
            // Default exp can round differently after fusion, including for BF16.
            T exponential = T(metal::precise::exp(metal::abs(x)));
            auto y = 1 / (1 + exponential);
            T gate = T((x < 0) ? y : 1 - y);
            output[i] = values[i] * gate;
        """,
    )


def sigmoid_gate(values: mx.array, gates: mx.array) -> mx.array:
    """Multiply by sigmoid, with an elementwise or final-axis shared gate."""
    if (
        values.ndim < 1
        or gates.ndim != values.ndim
        or gates.shape[:-1] != values.shape[:-1]
        or gates.shape[-1] not in (1, values.shape[-1])
        or values.dtype != gates.dtype
        or values.dtype not in (mx.float16, mx.bfloat16, mx.float32)
    ):
        raise ValueError("sigmoid gate requires aligned floating tensors")
    if values.size == 0:
        return values
    return _sigmoid_gate()(
        inputs=[values, gates],
        template=[
            ("T", values.dtype),
            ("N", values.size),
            ("WIDTH", values.shape[-1]),
            ("SHARED", gates.shape[-1] == 1),
        ],
        grid=(values.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[values.dtype],
    )[0]
