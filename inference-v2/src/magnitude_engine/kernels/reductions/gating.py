"""Fused activations with explicit intermediate rounding under MLX compilation."""

import mlx.core as mx

from .. import compile, kernel, metal


@kernel(source="nonlinear.metal", function="magnitude_sigmoid")
def native_sigmoid(x):
    domain = metal.Domain(item=x.size)
    (i,) = domain.indices
    return metal.TileCall(
        domain, metal.Thread(), {"x": metal.Load(x[i])}, metal.Replicated((i,), x.dtype)
    )


@compile
def _sigmoid_gate(values, gates):
    shape = values.shape
    x = mx.broadcast_to(gates, shape).reshape(-1)
    return (values.reshape(-1) * native_sigmoid(x)).reshape(shape)


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
    return _sigmoid_gate(values, gates)
