"""Non-exact affine coefficients, tail tiles and an FP32 publication boundary."""

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.kernels.schedules import OperandPreparation


class Factored:
    def __init__(self, rows):
        self.rows = rows

    def select(self, request, default):
        return next(
            s
            for s in request.candidates
            if s.rows == self.rows and s.operands.preparation == OperandPreparation.FACTORED
        )


@pytest.mark.device
@pytest.mark.parametrize("backend", ["metal", "cuda"])
@pytest.mark.parametrize("dtype", [ops.DType.F16, ops.DType.BF16])
@pytest.mark.parametrize("rows", [9, 33])
@pytest.mark.parametrize("physical_rows", [16, 64])
def test_factored_coefficients_keep_fp32_precision_across_tails(
    backend, dtype, rows, physical_rows
):
    if backend == "metal" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    rng = np.random.default_rng(641)
    shape = (257, 512)
    codes = rng.integers(0, 16, shape, dtype=np.uint8)
    scales = torch.from_numpy(rng.uniform(0.002, 0.009, (257, 8)).astype(np.float32)).bfloat16()
    biases = torch.from_numpy(rng.uniform(-0.08, -0.01, (257, 8)).astype(np.float32)).bfloat16()
    packed = (codes.ravel()[::2] | codes.ravel()[1::2] << 4).tobytes()
    contents = (
        packed
        + scales.view(torch.uint16).numpy().tobytes()
        + biases.view(torch.uint16).numpy().tobytes()
    )
    source = torch.from_numpy(rng.normal(0, 0.2, (rows, 512)).astype(np.float32)).to(
        torch.float16 if dtype == ops.DType.F16 else torch.bfloat16
    )
    decoded = codes.astype(np.float64) * scales.float().numpy().repeat(
        64, axis=1
    ) + biases.float().numpy().repeat(64, axis=1)
    reference = source.float().numpy().astype(np.float64) @ decoded.T
    weight_spec = ops.TensorSpec(shape, dtype).with_representation(
        ops.Affine(ops.Code(4), 64, ops.DirectCoefficients(ops.DType.BF16, ops.DType.BF16))
    )
    input_spec = ops.TensorSpec((rows, 512), dtype)
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend=backend, maximum_bytes=32 << 20)
    ) as device:
        inputs = (
            device.upload(input_spec, source.view(torch.uint8).numpy().tobytes()),
            device.upload(weight_spec, contents),
        )
        program = None
        try:
            program = ops.compile(
                lambda x, w: ops.linear(x, w, output_dtype=ops.DType.F32),
                signature=ops.Signature(
                    (ops.Argument(input_spec, "x"), ops.Argument(weight_spec, "w"))
                ),
                device=device,
                constants={},
                options=ops.CompileOptions(mode="prefill", schedules=Factored(physical_rows)),
            )
            execution = program.submit(*inputs)
            execution.completion.wait()
            try:
                actual = np.frombuffer(device.read(execution.outputs[0]), np.float32).reshape(
                    reference.shape
                )
                np.testing.assert_allclose(actual, reference, rtol=3e-5, atol=3e-5)
            finally:
                for output in execution.outputs:
                    output.close()
        finally:
            if program is not None:
                program.close()
            for resource in inputs:
                resource.close()
