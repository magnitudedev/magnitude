"""Exercise the public declaration API through actual Metal and native MLX compilation."""

from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import pytest

from magnitude_engine import kernels
from magnitude_engine.kernels import metal
from magnitude_engine.kernels.contractions.affine import affine
from magnitude_engine.kernels.reductions.normalization import rms_norm

SOURCE = str(Path(__file__).parent / "metal/composition.metal")


@kernels.kernel(source=SOURCE, function="bump")
def bump(x):
    domain = metal.Domain(item=x.size)
    (i,) = domain.indices
    return metal.TileCall(
        domain, metal.SIMDGroup(), {"x": metal.Load(x[i])}, metal.Distributed((i,), x.dtype)
    )


@kernels.kernel(source=SOURCE, function="pair_subtract")
def paired(x):
    domain = metal.Domain(item=x.size)
    (i,) = domain.indices
    return metal.TileCall(
        domain,
        metal.SIMDGroup(),
        {"x": metal.Load(x[i]), "partner": metal.Load(x[i ^ 1])},
        metal.Distributed((i,), x.dtype),
    )


def test_new_function_composes_without_planner_registration():
    region = kernels.compile(lambda x: mx.tanh(bump(x)))
    x = mx.arange(32).astype(mx.float32)
    assert mx.allclose(region(x), mx.tanh(x + 2)).item()
    assert len(kernels.artifact(region)["regions"]) == 1


def test_local_exchange_and_mlx_between_custom_calls():
    region = kernels.compile(lambda x: paired(mx.tanh(bump(x))))
    x = mx.arange(64).astype(mx.float32) / 31
    expected = mx.tanh(x + 2)
    expected = expected - expected.reshape(-1, 2)[:, ::-1].reshape(x.shape)
    assert mx.allclose(region(x), expected, atol=1e-6).item()
    assert len(kernels.artifact(region)["regions"]) == 1
    assert "1 local exchanges" in kernels.explain(region)


def test_invalid_index_rejected_before_launch():
    @kernels.kernel(source=SOURCE, function="bump")
    def invalid(x):
        domain = metal.Domain(item=x.size)
        (i,) = domain.indices
        return metal.TileCall(
            domain, metal.Thread(), {"x": metal.Load(x[i + 1])}, metal.Replicated((i,), x.dtype)
        )

    with pytest.raises(ValueError, match="tensor domain"):
        invalid(mx.ones(32))
    with pytest.raises(ValueError, match="32-element"):
        domain = metal.Domain(item=31)
        metal.TileCall(domain, metal.Thread(), {}, metal.Distributed(domain.indices, mx.float32))


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_shared_affine_driver_fanout_and_warm_execution(rows, dtype):
    x = mx.random.normal((rows, 512), key=mx.random.key(1)).astype(dtype)
    w = mx.quantize(
        mx.random.normal((512, 512), key=mx.random.key(2)).astype(dtype), bits=4, group_size=64
    )
    u = mx.quantize(
        mx.random.normal((512, 512), key=mx.random.key(3)).astype(dtype), bits=4, group_size=64
    )

    def dot(x, weight):
        return affine(x, *weight, x, bits=4, group_size=64)

    @kernels.compile
    def region(x, w, u):
        g, h = dot(x, w), dot(x, u)
        return g, g * mx.sigmoid(g) * h

    expected_g, expected_h = dot(x, w), dot(x, u)
    g, actual = region(x, w, u)
    assert mx.array_equal(g, expected_g).item()
    assert mx.allclose(actual, expected_g * mx.sigmoid(expected_g) * expected_h, atol=1e-5).item()
    assert len(kernels.artifact(region)["regions"]) == 1
    assert "1 shared pack drivers" in kernels.explain(region)
    with patch("magnitude_engine.kernels.core.compiler.snapshot", side_effect=AssertionError):
        mx.eval(region(x + 1, w, u))


def test_dependent_contractions_keep_global_boundary():
    x = mx.ones((1, 512))
    w = mx.quantize(mx.eye(512), bits=4, group_size=64)

    @kernels.compile
    def region(x, w):
        y = affine(x, *w, x, bits=4, group_size=64)
        return affine(y, *w, y, bits=4, group_size=64)

    assert mx.allclose(region(x, w), x, atol=1e-5).item()
    assert len(kernels.artifact(region)["regions"]) == 2


def test_public_row_example_preserves_residual_fanout():
    @kernels.compile
    def region(x, update, w):
        residual = x + update
        return residual, rms_norm(residual, w)

    x = mx.random.normal((3, 256), key=mx.random.key(93)).astype(mx.bfloat16)
    w = mx.ones(256, mx.bfloat16)
    residual, actual = region(x, x, w)
    assert mx.array_equal(residual, x + x).item()
    assert mx.array_equal(actual, rms_norm(x + x, w)).item()
    assert len(kernels.artifact(region)["regions"]) == 1


def test_new_cooperative_driver_uses_same_declaration():
    @kernels.kernel(source=SOURCE, function="row_transform")
    def transform(x):
        return metal.RowTransform(
            metal.Tensor(x.shape, x.dtype),
            x.value,
            metal.Threadgroup(32),
            dict(row=metal.GroupPosition("y"), tid=metal.ThreadPosition()),
            (x.shape[-1], 32),
        )

    region = kernels.compile(lambda x: mx.tanh(transform(x + 3)))
    x = mx.random.normal((2, 64), key=mx.random.key(93))
    assert mx.allclose(region(x), mx.tanh((x + 3) * 2 + 1)).item()
    assert len(kernels.artifact(region)["regions"]) == 1


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("outputs", [2048, 2053])
def test_prepared_wide_projection_retains_inline_arithmetic(dtype, outputs):
    from magnitude_engine.kernels.contractions.linear import _projection, apply

    x = mx.random.normal((3, 512), key=mx.random.key(14)).astype(dtype)
    weights = mx.quantize(
        mx.random.normal((outputs, 512), key=mx.random.key(17)).astype(dtype), bits=4, group_size=64
    )
    inline = affine(x, *weights, x, bits=4, group_size=64)
    actual = apply(x, *weights, bits=4, group_size=64)
    assert mx.array_equal(actual, inline).item()
    assert len(kernels.artifact(_projection)["regions"]) == 2
