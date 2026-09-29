"""Vision's added tensor mathematics, independently qualified on each backend."""

import math
import os

import numpy as np
import pytest

import ops
from engine import DevicePlan


def fixture(width):
    rng = np.random.default_rng(48)
    values = rng.normal(100, 0.4, (3, width)).astype(np.float32)
    values[1] = 3.0  # Constant rows must not acquire a spurious variance.
    gain = rng.normal(1, 0.1, width).astype(np.float32)
    bias = rng.normal(0, 0.2, width).astype(np.float32)
    signature = ops.Signature(
        tuple(
            ops.Argument(ops.TensorSpec(x.shape, ops.DType.F32), name)
            for x, name in zip((values, gain, bias), ("value", "gain", "bias"), strict=True)
        )
    )
    # Independent float64 reference, deliberately sensitive to cancellation in E[x²]-E[x]².
    x = values.astype(np.float64)
    expected = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)
    return signature, (values, gain, bias), (expected * gain + bias).astype(np.float32)


@pytest.mark.parametrize("width", [1, 7, 33, 1152])
def test_centered_normalization_reference(width):
    signature, values, expected = fixture(width)
    graph = ops.trace(ops.layer_norm, signature)
    (actual,) = ops.evaluate_reference(
        graph, dict(zip(("value", "gain", "bias"), values, strict=True))
    ).outputs
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("approximate", [False, True])
def test_gelu_reference_keeps_its_declared_approximation(approximate):
    values = np.linspace(-5, 5, 65).astype(np.float32)
    function = ops.gelu_tanh if approximate else ops.gelu
    signature = ops.Signature((ops.Argument(ops.TensorSpec(values.shape, ops.DType.F32), "x"),))
    graph = ops.trace(function, signature)
    (actual,) = ops.evaluate_reference(graph, {"x": values}).outputs
    expected = np.array(
        [
            0.5
            * x
            * (
                1
                + (
                    math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
                    if approximate
                    else math.erf(x / math.sqrt(2))
                )
            )
            for x in values.astype(np.float64)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-7)


@pytest.mark.device
@pytest.mark.parametrize("width", [7, 33, 1152])
def test_centered_normalization_and_both_activations_on_device(width):
    signature, values, expected = fixture(width)

    def function(value, gain, bias):
        normalized = ops.layer_norm(value, gain, bias)
        return normalized, ops.gelu(normalized), ops.gelu_tanh(normalized)

    with ops.DeviceRuntime.open(
        DevicePlan.discover(
            backend=os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"), maximum_bytes=128 << 20
        )
    ) as device:
        resources = [
            device.upload(argument.spec, value.tobytes())
            for argument, value in zip(signature.args, values, strict=True)
        ]
        program = None
        try:
            program = ops.compile(
                function,
                signature=signature,
                device=device,
                constants={},
                options=ops.CompileOptions(mode="prefill"),
            )
            execution = program.submit(*resources)
            execution.completion.wait()
            try:
                actual = [
                    np.frombuffer(device.read(output), np.float32).reshape(expected.shape)
                    for output in execution.outputs
                ]
                np.testing.assert_allclose(actual[0], expected, rtol=2e-3, atol=2e-4)
                for index, approximate in enumerate((False, True), 1):
                    x = actual[0].astype(np.float64)
                    expected_activation = (
                        0.5
                        * x
                        * (
                            1
                            + (
                                np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
                                if approximate
                                else np.vectorize(math.erf)(x / math.sqrt(2))
                            )
                        )
                    )
                    np.testing.assert_allclose(
                        actual[index], expected_activation, rtol=2e-5, atol=1e-6
                    )
            finally:
                for output in execution.outputs:
                    output.close()
        finally:
            if program is not None:
                program.close()
            for resource in resources:
                resource.close()
