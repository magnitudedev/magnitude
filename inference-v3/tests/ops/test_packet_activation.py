"""Shared nibble preparation preserves the original FP32 product range."""

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan


@pytest.mark.device
@pytest.mark.parametrize("dtype,exponent,scale_exponent", [
    (ops.DType.F16, -24, 8),
    (ops.DType.BF16, -120, 100),
    (ops.DType.BF16, -114, 100),
    (ops.DType.BF16, 100, -100),
    (ops.DType.F32, -120, 100),
])
def test_affine_packet_preparation_preserves_small_and_large_products(dtype, exponent, scale_exponent):
    if not torch.backends.mps.is_available():
        pytest.skip("requires a Metal device")
    width, outputs = 512, 5
    source = (np.arange(width, dtype=np.float32) % 7 * np.float32(2.0**exponent)).reshape(1, width)
    source_tensor = torch.from_numpy(source).to(getattr(torch, dtype.value))
    codes = (np.arange(outputs * width).reshape(outputs, width) % 16).astype(np.uint8)
    encoded = codes.ravel()[::2] | codes.ravel()[1::2] << 4
    scale = np.float32(2.0**scale_exponent)
    coefficients = np.full(codes.size // 64, int(scale.view(np.uint32)) >> 16, np.uint16)
    payload = encoded.tobytes() + coefficients.tobytes() + bytes(coefficients.nbytes)
    weight_spec = ops.TensorSpec(codes.shape, dtype).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16)))
    expected = source_tensor.float().numpy() @ (codes.astype(np.float32) * scale).T
    expected = torch.from_numpy(expected).to(getattr(torch, dtype.value)).float().numpy()
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=1 << 20)) as device:
        hidden = device.upload(ops.TensorSpec(source.shape, dtype), source_tensor.view(torch.uint8).numpy().tobytes())
        weight = device.upload(weight_spec, payload)
        program = execution = None
        try:
            program = ops.compile(
                ops.linear,
                signature=ops.Signature((ops.Argument(hidden.spec, "x"),
                                         ops.Argument(weight.spec, "weight", ops.ValueKind.CONSTANT))),
                device=device, constants={"weight": weight}, options=ops.CompileOptions(mode="decode"),
            )
            execution = program.submit(hidden)
            execution.completion.wait()
            actual = torch.frombuffer(bytearray(device.read(execution.outputs[0])), dtype=getattr(torch, dtype.value))
            np.testing.assert_array_equal(actual.float().numpy().reshape(expected.shape), expected)
        finally:
            if execution is not None:
                for output in execution.outputs:
                    output.close()
            if program is not None:
                program.close()
            weight.close()
            hidden.close()
