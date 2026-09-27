import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

from magnitude_engine.models.architectures.qwen35.attention.rotary import QwenRotary


@pytest.mark.parametrize('dtype,bits', [(mx.float32, 23), (mx.float16, 10), (mx.bfloat16, 7)])
@pytest.mark.parametrize('batch,count,width,rotated,offset', [
    (1, 1, 32, 8, 0),
    (1, 32, 256, 64, 0),
    (3, 1, 256, 64, 16384),
    (2, 4, 256, 64, 32768),
    (1, 512, 256, 64, 16384),
    (2, 3, 64, 64, 131071),
])
def test_rotary_matches_stock_and_rotation_equation(
    dtype, bits, batch, count, width, rotated, offset,
):
    mx.random.seed(912)
    # Transposed Q/K are the model's natural layout. Reverse the feature stride
    # too, so correctness does not depend on contiguous storage or copying it.
    q = mx.random.normal((batch, count, 8, width)).astype(dtype).transpose(0, 2, 1, 3)
    k = mx.random.normal((batch, count, 2, width)).astype(dtype).transpose(0, 2, 1, 3)
    if count == 3:
        q, k = q[..., ::-1], k[..., ::-1]
    offsets = offset if batch == 1 else mx.array([offset + i * 109 for i in range(batch)], mx.int32)
    if count == 32:
        offsets = mx.array(offsets, mx.int32)  # Scalar device offset, as well as host/row offsets.
    operation = QwenRotary(nn.RoPE(rotated, traditional=False, base=10_000_000))
    actual = operation(q, k, offset=offsets)
    reference = Qwen3_5RotaryEmbedding(rotated, base=10_000_000, mrope_section=[rotated // 2, 0, 0])
    positions = mx.array(offsets).reshape(-1, 1) + mx.arange(count)[None, :]
    expected = reference.apply_rotary(q, k, positions)
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected, strict=True))
    assert operation.rotation is not None
    frequencies = np.array(operation.rotation.inv_freq)
    np.testing.assert_allclose(
        frequencies, 10_000_000 ** (-np.arange(0, rotated, 2, dtype=np.float64) / rotated),
        rtol=2e-7, atol=0,
    )
    # Independent FP64 rotation of the specified FP32 angles. The error budget
    # allows the final activation rounding and FP32 transcendental arithmetic.
    angles = (np.array(positions).astype(np.float32)[..., None] * frequencies).astype(np.float64)
    cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
    half = rotated // 2
    for source, result in zip((q, k), actual, strict=True):
        source = np.array(source.astype(mx.float32)).astype(np.float64)
        oracle = source.copy()
        x, y = source[..., :half], source[..., half:rotated]
        oracle[..., :half] = x * cos - y * sin
        oracle[..., half:rotated] = y * cos + x * sin
        error = np.abs(np.array(result.astype(mx.float32)) - oracle)
        budget = np.abs(oracle) * 2 ** (-bits - 1) + 3e-6 * (1 + np.abs(source))
        assert np.all(error <= budget)
        assert np.array_equal(
            np.array(result.astype(mx.float32))[..., rotated:], source[..., rotated:],
        )


@pytest.mark.parametrize('traditional,scale', [(True, 1.0), (False, 0.5)])
def test_noncanonical_position_operators_keep_their_declared_semantics(traditional, scale):
    source = nn.RoPE(16, traditional=traditional, base=10_000, scale=scale)
    operation = QwenRotary(source)
    q = mx.random.normal((2, 4, 3, 32))
    k = mx.random.normal((2, 2, 3, 32))
    offsets = mx.array([3, 1000], mx.int32)
    actual = operation(q, k, offset=offsets)
    expected = source(q, offset=offsets), source(k, offset=offsets)
    assert all(mx.array_equal(a, b).item() for a, b in zip(actual, expected, strict=True))
