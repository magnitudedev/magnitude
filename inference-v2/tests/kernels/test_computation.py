"""Composition and numerical contracts, tested through actual MLX execution."""

from unittest.mock import patch

import mlx.core as mx
import pytest

from magnitude_engine.kernels.core.computation import computation
from magnitude_engine.kernels.core.elementwise import Elementwise
from magnitude_engine.kernels.core.scheduling import MLX


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("shape,shared", [((3, 8), False), ((3, 8), True), ((2, 3, 8), True)])
def test_nested_composition_and_schedules(dtype, shape, shared):
    @computation
    def product(x, y):
        return (mx.sigmoid(y) * x).astype(x.dtype)

    @computation
    def region(x, y, *, scale):
        return product(x, y) * scale + x

    x = mx.random.normal(shape, key=mx.random.key(7)).astype(dtype)
    y = mx.random.normal((*shape[:-1], 1) if shared else shape, key=mx.random.key(9)).astype(dtype)
    mx.eval(x, y)
    expected = (mx.sigmoid(y) * x).astype(dtype) * 0.25 + x
    mx.eval(expected)
    # A child with a cached executable still expands into its parent's graph.
    mx.eval(product(x, y))
    for tile in (1, 4):
        bound = region.with_schedule(Elementwise(tile)).specialize(x, y, scale=0.25)
        actual = mx.compile(bound)(x, y, scale=0.25)
        assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6).item()
        assert all(n.operation != "CustomKernel" for n in bound.graph.nodes)


def test_warm_compilation_does_not_capture_or_lower_again():
    @computation
    def region(x, y):
        return x * mx.sigmoid(y)

    x = mx.ones((2, 8))
    compiled = mx.compile(lambda x, y: mx.tanh(region(x, y)))
    mx.eval(compiled(x, x))
    with patch("magnitude_engine.kernels.core.computation.capture", side_effect=AssertionError):
        for i in range(4):
            value = x + i
            assert mx.allclose(compiled(value, x), mx.tanh(value * mx.sigmoid(x))).item()


def test_output_tree_constants_and_specialization():
    constant = mx.array([0.25])

    @computation
    def region(x, *, scale=1.0):
        return {"scaled": x * scale, "biased": x + constant}

    x = mx.ones((4, 8))
    bound = region.specialize(x, scale=2.0)
    assert region.specialize(x + 2, scale=2.0) is bound
    assert region.specialize(x, scale=3.0) is not bound
    result = bound(x, scale=2.0)
    assert mx.array_equal(result["scaled"], x * 2).item()
    assert mx.array_equal(result["biased"], x + constant).item()
    with pytest.raises(ValueError, match="specialization"):
        bound(x, scale=3.0)


def test_mlx_retained_region_and_generated_composition():
    @computation
    def region(x, weight):
        return mx.tanh(x @ weight)

    x = mx.ones((2, 8))
    weight = mx.ones((8, 16))
    actual = region.with_schedule(MLX())(x, weight)
    assert mx.array_equal(actual, mx.tanh(x @ weight)).item()


def test_values_views_and_reduced_precision_boundary():
    @computation
    def region(x):
        return x.astype(mx.bfloat16).astype(mx.float32) * 0.25

    x = mx.random.normal((8, 8))
    for view in (x, x.T, x[:, ::2], mx.broadcast_to(x[:1], (8, 8))):
        assert mx.array_equal(
            region(view), view.astype(mx.bfloat16).astype(mx.float32) * 0.25
        ).item()


def test_capture_rejects_data_dependent_host_extraction():
    @computation
    def region(x):
        return x if x.sum().item() else -x

    with pytest.raises(ValueError, match="eval"):
        region(mx.ones((2, 8)))


def test_affine_composition_schedules_and_mixed_mlx():
    from magnitude_engine.kernels.contractions.affine import Affine
    from magnitude_engine.kernels.core.execution import ExecutionPlan

    dot = Affine(4, 64)

    @computation
    def region(x, wg, sg, bg, wu, su, bu):
        g = dot(x, wg, sg, bg)[0]
        u = dot(x, wu, su, bu)[0]
        return (g * mx.sigmoid(g)).astype(x.dtype) * u

    x = mx.random.normal((3, 512), key=mx.random.key(14)).astype(mx.bfloat16)
    wg, sg, bg = mx.quantize(
        mx.random.normal((512, 512), key=mx.random.key(1)).astype(x.dtype), group_size=64, bits=4
    )
    wu, su, bu = mx.quantize(
        mx.random.normal((512, 512), key=mx.random.key(2)).astype(x.dtype), group_size=64, bits=4
    )
    args = x, wg, sg, bg, wu, su, bu
    mx.eval(*args)
    g = mx.concatenate(
        [mx.quantized_matmul(row[None], wg, sg, bg, group_size=64, bits=4) for row in x]
    )
    u = mx.concatenate(
        [mx.quantized_matmul(row[None], wu, su, bu, group_size=64, bits=4) for row in x]
    )
    expected = (g * mx.sigmoid(g)).astype(x.dtype) * u
    # Different schedules and independently executed leaves share the same math.
    bound = region.specialize(*args)
    assert sum(isinstance(n.operation, Affine) for n in bound.graph.nodes) == 2
    assert mx.array_equal(bound(*args), expected).item()
    bound = region.specialize(*args)
    assert isinstance(bound.call, ExecutionPlan)
    assert mx.array_equal(bound(*args), expected).item()

    @computation
    def mixed(x, wg, sg, bg):
        h = dot(x, wg, sg, bg)[0]
        return mx.fast.rms_norm(h, None, eps=1e-6) * mx.sigmoid(h)

    bound = mixed.specialize(x, wg, sg, bg)
    assert {r.backend for r in bound.call.regions} == {"MLX", "METAL"}
    assert mx.allclose(
        mx.compile(bound)(x, wg, sg, bg),
        mx.fast.rms_norm(g, None, eps=1e-6) * mx.sigmoid(g),
        atol=1e-6,
    ).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "op", [mx.negative, mx.exp, mx.log, mx.sin, mx.cos, mx.sqrt, mx.rsqrt, mx.tanh]
)
def test_unary_special_values_match_mlx(dtype, op):
    x = mx.array([0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0], dtype)
    bound = computation(lambda x: op(x)).specialize(x)
    actual, expected = bound(x).astype(mx.float32), op(x).astype(mx.float32)
    same = (actual.view(mx.uint32) == expected.view(mx.uint32)) | (
        mx.isnan(actual) & mx.isnan(expected)
    )
    assert mx.all(same).item()


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16, mx.int32])
@pytest.mark.parametrize("op", [mx.maximum, mx.minimum])
def test_minmax_special_values_match_mlx(dtype, op):
    x = mx.array([0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0]).astype(dtype)
    y = mx.array([-0.0, 0.0, 1.0, float("nan"), float("nan"), float("nan")]).astype(dtype)
    actual = computation(lambda x, y: op(x, y))(x, y).astype(mx.float32)
    expected = op(x, y).astype(mx.float32)
    same = (actual.view(mx.uint32) == expected.view(mx.uint32)) | (
        mx.isnan(actual) & mx.isnan(expected)
    )
    assert mx.all(same).item()
