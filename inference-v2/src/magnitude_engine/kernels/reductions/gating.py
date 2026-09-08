"""Fused activations with explicit intermediate rounding under MLX compilation."""

import mlx.core as mx

from magnitude_engine.kernels.core import Input, KernelPlan, Launch, Output, Program, Source

SIGMOID_GATE = Program("sigmoid_gate", Source("reductions/sigmoid_gate.metal"))


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
    return KernelPlan(
        program=SIGMOID_GATE,
        inputs=(
            Input("values", values),
            Input("gates", gates),
        ),
        outputs=(Output("output", values.shape, values.dtype),),
        launch=Launch((values.size, 1, 1), (256, 1, 1)),
        template=(
            ("T", values.dtype),
            ("N", values.size),
            ("WIDTH", values.shape[-1]),
            ("SHARED", gates.shape[-1] == 1),
        ),
    ).run()[0]
