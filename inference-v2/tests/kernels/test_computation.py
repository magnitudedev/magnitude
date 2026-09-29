"""MLX scalar semantics at native compilation and owned-kernel boundaries."""

from pathlib import Path

import mlx.core as mx
import pytest

from magnitude_engine import kernels
from magnitude_engine.kernels import metal

SOURCE = str(Path(__file__).parent / "metal/composition.metal")


@kernels.kernel(source=SOURCE, function="identity")
def identity(x):
    domain = metal.Domain(item=x.size)
    (i,) = domain.indices
    return metal.TileCall(
        domain, metal.Thread(), {"x": metal.Load(x[i])}, metal.Replicated((i,), x.dtype)
    )


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "op", [mx.negative, mx.exp, mx.log, mx.sin, mx.cos, mx.sqrt, mx.rsqrt, mx.tanh]
)
def test_unary_special_values_match_mlx(dtype, op):
    x = mx.array([0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0, -1.0], dtype)
    compiled = kernels.compile(lambda x: op(identity(x)))
    actual, expected = compiled(x).astype(mx.float32), op(x).astype(mx.float32)
    same = (actual.view(mx.uint32) == expected.view(mx.uint32)) | (
        mx.isnan(actual) & mx.isnan(expected)
    )
    assert mx.all(same).item()
    assert len(kernels.artifact(compiled)["regions"]) == 1


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16, mx.int32])
@pytest.mark.parametrize("op", [mx.maximum, mx.minimum])
def test_minmax_special_values_match_mlx(dtype, op):
    x = mx.array([0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.0]).astype(dtype)
    y = mx.array([-0.0, 0.0, 1.0, float("nan"), float("nan"), float("nan")]).astype(dtype)
    compiled = kernels.compile(lambda x, y: op(identity(x), y))
    actual, expected = compiled(x, y).astype(mx.float32), op(x, y).astype(mx.float32)
    same = (actual.view(mx.uint32) == expected.view(mx.uint32)) | (
        mx.isnan(actual) & mx.isnan(expected)
    )
    assert mx.all(same).item()


def test_data_dependent_host_extraction_preserves_mlx_error():
    f = kernels.compile(lambda x: x if x.item() > 0 else -x)
    with pytest.raises(ValueError, match="item|eval"):
        f(mx.array(1.0))
