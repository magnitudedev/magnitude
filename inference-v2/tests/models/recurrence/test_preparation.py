import mlx.core as mx
import mlx.nn as nn
import pytest

from magnitude_engine.models.recurrence.preparation import prepare


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "batch,count,key_width,value_width", [(1, 1, 32, 32), (3, 4, 128, 128), (2, 8, 64, 128)]
)
def test_packed_preparation_preserves_mlx_equation(dtype, batch, count, key_width, value_width):
    mx.random.seed(491)
    hk, hv, window = 2, 4, 3
    kd, vd = hk * key_width, hv * value_width
    channels = 2 * kd + vd
    proj = mx.random.normal((batch, count, channels + vd + 2 * hv)).astype(dtype)
    state = mx.random.normal((batch, window, channels)).astype(dtype)
    weights = (mx.random.normal((channels, window + 1, 1)) * 0.1).astype(dtype)
    rates = mx.random.normal((hv,)).astype(dtype)
    bias = mx.random.normal((hv,)).astype(dtype)
    y, next_state, gate, beta, decay = prepare(
        proj,
        state,
        weights,
        rates,
        bias,
        key_heads=hk,
        key_width=key_width,
        value_heads=hv,
        value_width=value_width,
    )
    raw, expected_gate, b, a = mx.split(
        proj, (channels, channels + vd, channels + vd + hv), axis=-1
    )
    joined = mx.concatenate((state, raw), axis=1)
    convolved = nn.silu(mx.conv1d(joined, weights, groups=channels))
    q, k, v = mx.split(convolved, (kd, 2 * kd), axis=-1)
    q = mx.fast.rms_norm(q.reshape(batch, count, hk, key_width), None, 1e-6) * (1 / key_width)
    k = mx.fast.rms_norm(k.reshape(batch, count, hk, key_width), None, 1e-6) * (key_width**-0.5)
    expected = mx.concatenate(
        (q.reshape(batch, count, kd), k.reshape(batch, count, kd), v), axis=-1
    )
    expected_decay = mx.exp(
        -mx.exp(rates.astype(mx.float32)) * nn.softplus(a + bias).astype(mx.float32)
    )
    mx.eval(y, next_state, gate, beta, decay, expected, expected_decay)
    # Same gates/rounding as the production reference; no relaxed BF16 allowance.
    assert mx.allclose(y, expected, atol=2e-5, rtol=2e-5).item()
    assert mx.array_equal(next_state, joined[:, -window:]).item()
    assert mx.array_equal(gate, expected_gate).item()
    assert mx.allclose(beta, mx.sigmoid(b), atol=2e-5, rtol=2e-5).item()
    assert mx.allclose(decay, expected_decay, atol=2e-5, rtol=2e-5).item()
