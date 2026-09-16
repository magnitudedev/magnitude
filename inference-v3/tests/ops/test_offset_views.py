"""Model-owned slices retain their physical byte offsets across the packed ABI."""

import numpy as np
import pytest

import ops
from engine import DevicePlan
from ops.compiler.program import annotation


def test_contiguous_port_keeps_a_symbolic_offset():
    import tilelang.language as T

    value = annotation(T, ops.TensorSpec((3, 8), ops.DType.F32))
    assert type(value.elem_offset).__name__ == "Var"
    assert tuple(map(int, value.strides)) == (8, 1)
    assert value.offset_factor == 1


@pytest.mark.device
@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_sliced_input_and_retained_output_can_feed_another_program(backend):
    import torch

    available = (
        torch.backends.mps.is_available() if backend == "metal" else torch.cuda.is_available()
    )
    if not available:
        pytest.skip(f"{backend} unavailable")
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend=backend, maximum_bytes=8 << 20)
    ) as device:
        host = np.arange(128, dtype=np.float32).reshape(4, 32)
        source = device.upload(ops.TensorSpec(host.shape, ops.DType.F32), host.tobytes())
        spec = ops.TensorSpec((2, 32), ops.DType.F32)
        sliced = source.view(spec, 32 * 4)
        program = ops.compile(
            lambda x: x + x,
            signature=ops.Signature((ops.Argument(spec, "x"),)),
            constants={},
            device=device,
            options=ops.CompileOptions(mode="decode"),
        )
        execution = program.submit(sliced)
        execution.completion.wait()
        np.testing.assert_array_equal(
            np.frombuffer(device.read(execution.outputs[0]), np.float32).reshape(2, 32),
            host[1:3] * 2,
        )
        second_spec = ops.TensorSpec((1, 32), ops.DType.F32)
        retained = execution.outputs[0].view(second_spec, 32 * 4)
        second = ops.compile(
            lambda x: x + x,
            signature=ops.Signature((ops.Argument(second_spec, "x"),)),
            constants={},
            device=device,
            options=ops.CompileOptions(mode="decode"),
        )
        following = second.submit(retained)
        following.completion.wait()
        np.testing.assert_array_equal(
            np.frombuffer(device.read(following.outputs[0]), np.float32), host[2] * 4
        )
        for run in (following, execution):
            for value in run.outputs:
                value.close()
        retained.close()
        sliced.close()
        source.close()
        second.close()
        program.close()


@pytest.mark.device
@pytest.mark.parametrize("backend", ["metal", "cuda"])
def test_prebound_constant_view_preserves_its_storage_origin(backend):
    import torch

    if backend == "metal" and not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    values = np.arange(32, dtype=np.float32).reshape(4, 8) / 32
    weights = np.arange(64, dtype=np.float32).reshape(8, 8) / 64
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend=backend, maximum_bytes=4 << 20)
    ) as device:
        source = device.upload(ops.TensorSpec(values.shape, ops.DType.F32), values.tobytes())
        backing = device.upload(
            ops.TensorSpec((16, 8), ops.DType.F32),
            np.full((8, 8), -7, np.float32).tobytes() + weights.tobytes(),
        )
        weight = backing.view(ops.TensorSpec(weights.shape, ops.DType.F32), weights.nbytes)
        try:
            program = ops.compile(
                lambda x, w: ops.linear(x, w),
                signature=ops.Signature(
                    (
                        ops.Argument(source.spec, "x"),
                        ops.Argument(weight.spec, "w", ops.ValueKind.CONSTANT),
                    )
                ),
                device=device,
                constants={"w": weight},
                options=ops.CompileOptions(mode="decode"),
            )
            try:
                execution = program.submit(source)
                execution.completion.wait()
                try:
                    answer = np.frombuffer(device.read(execution.outputs[0]), np.float32)
                    np.testing.assert_allclose(
                        answer.reshape(4, 8), values @ weights.T, rtol=2e-6, atol=2e-6
                    )
                finally:
                    for output in execution.outputs:
                        output.close()
            finally:
                program.close()
        finally:
            weight.close()
            backing.close()
            source.close()
