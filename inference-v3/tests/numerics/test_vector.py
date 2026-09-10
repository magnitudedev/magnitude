import os

import numpy as np
import pytest

from magnitude_engine.numerics.semantics import Pointwise
from magnitude_engine.numerics.vector import pointwise, rms_norm
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context


def execute(factory, inputs, shape):
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    arrays = [np.asarray(value, dtype=np.float32) for value in inputs]
    context = open_context(
        backend, sum(a.nbytes for a in arrays) + np.prod(shape).item() * 4 + 1024**2, 0
    )
    tensors = [context.upload(TensorSpec(a.shape, DType.F32), a.tobytes()) for a in arrays]
    output = context.allocate(TensorSpec(shape, DType.F32))
    kernel = context.compile(factory(backend == Backend.LLVM))
    ticket = context.submit([Prepared(context, kernel, [*tensors, output])])
    for tensor in tensors:
        tensor.close()
    actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(shape)
    output.close()
    assert context.allocated_bytes == 0
    context.close()
    return actual


@pytest.mark.device
@pytest.mark.parametrize("kind", tuple(Pointwise))
def test_pointwise(kind):
    a = np.linspace(-30, 30, 259, dtype=np.float32)
    b = np.random.default_rng(3).normal(size=a.shape).astype(np.float32)
    expected = {
        Pointwise.ADD: a + b,
        Pointwise.MULTIPLY: a * b,
        Pointwise.SILU_PRODUCT: a.astype(np.float64) / (1 + np.exp(-a.astype(np.float64))) * b,
        Pointwise.SIGMOID_PRODUCT: a / (1 + np.exp(-b.astype(np.float64))),
    }[kind]
    actual = execute(lambda cpu: pointwise(a.size, kind, cpu=cpu), (a, b), a.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.device
@pytest.mark.parametrize("width", [127, 2560])
def test_norm(width):
    a = np.random.default_rng(23).normal(size=(3, width)).astype(np.float32)
    a[0] = 0
    a[1] *= 1e-5
    w = np.random.default_rng(65).uniform(0.5, 1.5, width).astype(np.float32)
    expected = (
        a.astype(np.float64)
        / np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=-1, keepdims=True) + 1e-6)
        * w
    )
    actual = execute(lambda cpu: rms_norm(3, width, 1e-6, cpu=cpu), (a, w), a.shape)
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=1e-7)


@pytest.mark.device
@pytest.mark.parametrize("output_dtype", (DType.F32, DType.BF16))
def test_residual_add_keeps_fp32_accumulation_with_bf16_update(output_dtype):
    from contextlib import ExitStack

    from magnitude_engine.operations.activation import Elementwise

    first = np.linspace(0.9999, 1.0001, 259, dtype=np.float32)
    update = np.full(first.shape, 1 / 1024, np.float32)
    packed = (update.view(np.uint32) >> 16).astype(np.uint16)
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")), 1024**2, 0
        )
        cleanup.callback(context.close)
        a = context.upload(TensorSpec(first.shape, DType.F32), first.tobytes())
        b = context.upload(TensorSpec(first.shape, DType.BF16), packed.tobytes())
        output = context.allocate(TensorSpec(first.shape, output_dtype))
        for tensor in (a, b, output):
            cleanup.callback(tensor.close)
        operation = Elementwise(context, Pointwise.ADD)
        cleanup.callback(operation.close)
        ticket = context.submit(operation.prepare(a, b, output))
        content = context.read(output, after=ticket)
        expected = first + update
        if output_dtype == DType.BF16:
            # Independent round-to-nearest-even at the declared storage edge.
            bits = expected.view(np.uint32)
            expected_bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
            np.testing.assert_array_equal(np.frombuffer(content, np.uint16), expected_bits)
        else:
            np.testing.assert_array_equal(np.frombuffer(content, np.float32), expected)
