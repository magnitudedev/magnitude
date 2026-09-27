"""Fused transitions preserve the upstream reduction and observable residuals."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated

from magnitude_engine.models.normalization import GatedRMSNorm, residual_norm


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("width", [64, 128, 256, 2048, 5120])
def test_residual_stream_preserves_norm_and_feature_values(dtype, width):
    x = mx.random.normal((2, 3, width), key=mx.random.key(815)).astype(dtype)
    update = mx.random.normal(x.shape, key=mx.random.key(816)).astype(dtype)
    norm = nn.RMSNorm(width, eps=1e-6)
    norm.weight = mx.random.uniform(shape=(width,), key=mx.random.key(817)).astype(dtype)
    residual, normalized = mx.compile(lambda x, y: residual_norm(x, y, norm))(x, update)
    assert mx.array_equal(residual, x + update).item()
    expected = norm(x + update)
    if dtype == mx.float32:
        assert mx.allclose(normalized, expected, atol=1e-6, rtol=1e-6).item()
    else:
        assert mx.array_equal(normalized, expected).item()


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_gated_norm_matches_upstream_without_changing_reduction_order(dtype, width):
    x = mx.random.normal((2, 3, 4, width), key=mx.random.key(819)).astype(dtype)
    gate = mx.random.normal(x.shape, key=mx.random.key(820)).astype(dtype)
    upstream = Qwen3NextRMSNormGated(width)
    upstream.weight = mx.random.uniform(shape=(width,), key=mx.random.key(821)).astype(dtype)
    owned = GatedRMSNorm(upstream.weight, upstream.eps)
    actual, expected = owned(x, gate), upstream(x, gate)
    if dtype == mx.float32:
        assert mx.allclose(actual, expected, atol=1e-6, rtol=1e-6).item()
    else:
        assert mx.array_equal(actual, expected).item()
