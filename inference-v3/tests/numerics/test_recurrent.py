import os

import numpy as np
import pytest

from magnitude_engine.numerics.recurrent import delta_sequence, prepare_sequence
from magnitude_engine.numerics.semantics import HeadMapping
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context


@pytest.mark.device
@pytest.mark.parametrize("width", [37, 128])
@pytest.mark.parametrize("steps", [1, 2, 17, 128])
def test_recurrent_preparation_history_normalization_and_extreme_gates(width, steps):
    batch, kh, vh, window = 2, 2, 6, 4
    channels = (kh * 2 + vh) * width
    rng = np.random.default_rng(347)
    x = rng.normal(size=(batch * steps, channels)).astype(np.float32)
    weights = rng.normal(size=(channels, window)).astype(np.float32)
    previous = rng.normal(size=(batch, channels, window - 1)).astype(np.float32)
    x[0, :width] = 0
    previous[0, :width] = 0
    alpha = np.linspace(-100, 100, batch * steps * vh, dtype=np.float32).reshape(batch * steps, vh)
    beta_input = alpha[::-1].copy()
    decay_weights = -rng.uniform(0.1, 2, vh).astype(np.float32)
    bias = rng.normal(size=vh).astype(np.float32)
    combined = np.concatenate(
        (previous.astype(np.float64), x.reshape(batch, steps, channels).transpose(0, 2, 1)), axis=-1
    )
    convolved = np.stack(
        [np.sum(combined[:, :, step : step + window] * weights, axis=-1) for step in range(steps)],
        axis=1,
    )
    activated = (convolved / (1 + np.exp(-convolved))).reshape(batch * steps, 2 * kh + vh, width)
    q, k, v = np.split(activated, [kh, 2 * kh], axis=1)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6) / np.sqrt(width)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
    expected = [
        combined[:, :, -(window - 1) :],
        q,
        k,
        v,
        1 / (1 + np.exp(-beta_input.astype(np.float64))),
        np.exp(decay_weights * np.logaddexp(0, alpha.astype(np.float64) + bias)),
    ]
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 8 * 1024**2, 0)
    arrays = (x, weights, previous, alpha, beta_input, decay_weights, bias)
    inputs = [context.upload(TensorSpec(a.shape, DType.F32), a.tobytes()) for a in arrays]
    outputs = [context.allocate(TensorSpec(value.shape, DType.F32)) for value in expected]
    arguments = [*inputs[:3], outputs[0], *inputs[3:], *outputs[1:]]
    program = prepare_sequence(
        batch, steps, kh, vh, width, window, 1e-6, cpu=backend == Backend.LLVM
    )
    ticket = context.submit([Prepared(context, context.compile(program), arguments)])
    for output, reference in zip(outputs, expected, strict=True):
        actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(
            reference.shape
        )
        np.testing.assert_allclose(actual, reference, rtol=6e-6, atol=2e-6)
    unchanged = np.frombuffer(context.read(inputs[2], after=ticket), np.float32).reshape(
        previous.shape
    )
    np.testing.assert_array_equal(unchanged, previous)
    for tensor in [*inputs, *outputs]:
        tensor.close()
    context.close()


@pytest.mark.device
@pytest.mark.parametrize("mapping", tuple(HeadMapping))
@pytest.mark.parametrize("steps", [1, 17, 128])
@pytest.mark.parametrize("key_width,value_width", [(37, 9), (128, 128)])
def test_delta_nonzero_state_and_independent_batch_rows(mapping, key_width, value_width, steps):
    batch, kh, vh = 2, 2, 6
    rng = np.random.default_rng(438)
    q = rng.normal(size=(batch * steps, kh, key_width))
    k = rng.normal(size=q.shape)
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(key_width)
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    v = rng.normal(size=(batch * steps, vh, value_width))
    decay = rng.uniform(0, 1, (batch * steps, vh))
    beta = rng.uniform(0, 1, (batch * steps, vh))
    previous = rng.normal(size=(batch, vh, value_width, key_width))
    arrays = [a.astype(np.float32) for a in (q, k, v, decay, beta, previous)]
    q, k, v, decay, beta, previous = (a.astype(np.float64) for a in arrays)
    indices = np.arange(vh) % kh if mapping == HeadMapping.TILED else np.arange(vh) // (vh // kh)
    expected_state = previous.copy()
    expected_output = np.empty_like(v)
    for step in range(steps):
        rows = np.arange(batch) * steps + step
        decayed = expected_state * decay[rows, :, None, None]
        residual = (v[rows] - np.einsum("bhvk,bhk->bhv", decayed, k[rows][:, indices])) * beta[
            rows, :, None
        ]
        expected_state = decayed + residual[:, :, :, None] * k[rows][:, indices, None, :]
        expected_output[rows] = np.einsum("bhvk,bhk->bhv", expected_state, q[rows][:, indices])
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 16 * 1024**2, 0)
    tensors = [context.upload(TensorSpec(a.shape, DType.F32), a.tobytes()) for a in arrays]
    next_state = context.allocate(TensorSpec(previous.shape, DType.F32))
    output = context.allocate(TensorSpec(v.shape, DType.F32))
    if backend == Backend.METAL:
        from magnitude_engine.numerics.metal_recurrent import delta_sequence as metal_delta

        assert context.subgroup_width is not None
        program = metal_delta(
            batch,
            steps,
            kh,
            vh,
            key_width,
            value_width,
            mapping,
            subgroup_width=context.subgroup_width,
        )
    else:
        program = delta_sequence(
            batch, steps, kh, vh, key_width, value_width, mapping, cpu=backend == Backend.LLVM
        )
    ticket = context.submit(
        [Prepared(context, context.compile(program), [*tensors, next_state, output])]
    )
    actual_state = np.frombuffer(context.read(next_state, after=ticket), np.float32).reshape(
        previous.shape
    )
    actual_output = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(v.shape)
    untouched = np.frombuffer(context.read(tensors[-1], after=ticket), np.float32).reshape(
        previous.shape
    )
    np.testing.assert_array_equal(untouched, arrays[-1])
    np.testing.assert_allclose(actual_state, expected_state, rtol=3e-6, atol=2e-6)
    np.testing.assert_allclose(actual_output, expected_output, rtol=3e-6, atol=2e-6)
    for tensor in [*tensors, next_state, output]:
        tensor.close()
    assert context.allocated_bytes == 0
    context.close()
