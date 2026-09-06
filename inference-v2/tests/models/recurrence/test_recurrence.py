import mlx.core as mx
import pytest
from mlx_lm.models.gated_delta import gated_delta_kernel

from magnitude_engine.models.recurrence.inputs import DeltaInputs
from magnitude_engine.models.recurrence.metal import MetalDelta
from magnitude_engine.models.recurrence.reference import DeltaReference


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 32, 2, 8),
        (2, 5, 2, 64, 4, 7),
        (1, 7, 2, 128, 4, 32),
        (1, 33, 2, 32, 4, 32),
        (2, 129, 2, 64, 4, 16),
    ],
)
def test_owned_delta_kernel_matches_library_and_prefix_replay(dtype, shape):
    b, t, hk, dk, hv, dv = shape
    mx.random.seed(11)
    q = mx.random.normal((b, t, hk, dk)).astype(dtype) / dk
    k = mx.random.normal(q.shape).astype(dtype) / dk**0.5
    v = mx.random.normal((b, t, hv, dv)).astype(dtype)
    g = mx.random.uniform(shape=(b, t, hv))
    beta = mx.random.uniform(shape=g.shape).astype(dtype)
    state = mx.random.normal((b, hv, dv, dk)) * 0.1
    inputs = DeltaInputs(q, k, v, g, beta)
    actual, final = MetalDelta().advance(inputs, state)
    expected, expected_final = gated_delta_kernel(q, k, v, g, beta, state)
    mx.eval(actual, final, expected, expected_final)
    assert mx.allclose(actual, expected, atol=1e-5).item()
    assert mx.allclose(final, expected_final, atol=1e-5).item()
    for count in (0, 1, t - 1, t):
        prefix = MetalDelta().reconcile(inputs, state, count)
        expected = (
            state
            if count == 0
            else gated_delta_kernel(
                q[:, :count], k[:, :count], v[:, :count], g[:, :count], beta[:, :count], state
            )[1]
        )
        assert mx.allclose(prefix, expected, atol=1e-5).item()
    if dtype == mx.float32:
        reference, reference_state = DeltaReference().advance(inputs, state)
        assert mx.allclose(reference, actual, atol=1e-5).item()
        assert mx.allclose(reference_state, final, atol=1e-5).item()
