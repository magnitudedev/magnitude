import os

import numpy as np
import pytest

from magnitude_engine.numerics.rotary import prepare_attention
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context


@pytest.mark.device
def test_query_key_norm_and_interleaved_multiaxis_rotary_at_long_positions():
    rows, qh, kh, width, rotary = 5, 4, 2, 256, 64
    rng = np.random.default_rng(974)
    query_gate = rng.normal(size=(rows, qh * 2 * width)).astype(np.float32)
    keys = rng.normal(size=(rows, kh * width)).astype(np.float32)
    qw, kw = (rng.uniform(0.5, 1.5, width).astype(np.float32) for _ in range(2))
    coordinates = np.array(
        [
            [0, 0, 0],
            [65535, 65535, 65535],
            [131072, 65536, 40],
            [1, 57, 2],
            [262143, 262140, 262139],
        ],
        np.int32,
    )
    query = query_gate.reshape(rows, qh, 2, width)[:, :, 0, :]
    gate = query_gate.reshape(rows, qh, 2, width)[:, :, 1, :]
    frequencies = (
        1 / np.power(1e7, np.arange(rotary // 2, dtype=np.float64) / (rotary // 2))
    ).astype(np.float32)
    axes = np.zeros(rotary // 2, np.int32)
    axes[1 : 11 * 3 : 3] = 1
    axes[2 : 10 * 3 : 3] = 2
    angles = (coordinates[:, axes].astype(np.float32) * frequencies).astype(np.float64)

    def reference(x, w):
        x = x.astype(np.float64)
        normalized = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * w
        first, second = normalized[:, :, : rotary // 2], normalized[:, :, rotary // 2 : rotary]
        result = normalized.copy()
        result[:, :, : rotary // 2] = first * np.cos(angles[:, None]) - second * np.sin(
            angles[:, None]
        )
        result[:, :, rotary // 2 : rotary] = second * np.cos(angles[:, None]) + first * np.sin(
            angles[:, None]
        )
        return result

    expected = (reference(query, qw), reference(keys.reshape(rows, kh, width), kw), gate)
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    context = open_context(backend, 4 * 1024**2, 0)
    inputs = [
        context.upload(TensorSpec(a.shape, dtype), a.tobytes())
        for a, dtype in (
            (query_gate, DType.F32),
            (keys, DType.F32),
            (qw, DType.F32),
            (kw, DType.F32),
            (coordinates, DType.I32),
        )
    ]
    outputs = [context.allocate(TensorSpec(a.shape, DType.F32)) for a in expected]
    program = prepare_attention(
        rows, qh, kh, width, rotary, 1e7, (11, 11, 10, 0), 1e-6, cpu=backend == Backend.LLVM
    )
    ticket = context.submit([Prepared(context, context.compile(program), [*inputs, *outputs])])
    for output, target in zip(outputs, expected, strict=True):
        actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(target.shape)
        np.testing.assert_allclose(actual, target, rtol=3e-6, atol=2e-6)
    for tensor in [*inputs, *outputs]:
        tensor.close()
    context.close()
