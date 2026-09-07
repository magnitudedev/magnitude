"""Fused gates preserve eager MLX rounding, including under outer compilation."""

import mlx.core as mx
import pytest

from magnitude_engine.models.activations import sigmoid_gate


def equal(a, b):
    mx.eval(a, b)
    assert bool(mx.all((a == b) | (mx.isnan(a) & mx.isnan(b))).item())


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_complete_reduced_precision_sigmoid_domain(dtype):
    gates = mx.arange(65536, dtype=mx.uint16).view(dtype)
    mx.random.seed(713)
    values = mx.random.normal(gates.shape).astype(dtype)
    expected = values * mx.sigmoid(gates)
    equal(sigmoid_gate(values, gates), expected)
    equal(mx.compile(sigmoid_gate)(values, gates), expected)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("shared", [False, True])
def test_batched_gate_and_strided_inputs(dtype, shared):
    mx.random.seed(714)
    values = mx.random.normal((3, 7, 66)).astype(dtype)[..., ::2]
    gates = mx.random.normal((3, 7, 1 if shared else 33)).astype(dtype)
    expected = values * mx.sigmoid(gates)
    equal(sigmoid_gate(values, gates), expected)
    equal(mx.compile(sigmoid_gate)(values, gates), expected)
