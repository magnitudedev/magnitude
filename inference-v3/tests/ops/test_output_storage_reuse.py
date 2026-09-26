"""Escaped output leases must outlive reusable frames and compiled owners."""
from contextlib import ExitStack
import struct

import pytest
import torch

import ops
from engine import DevicePlan


@pytest.mark.device
def test_reusable_outputs_preserve_escaped_results_and_survive_reclamation():
    if not torch.backends.mps.is_available():
        pytest.skip('requires a Metal worker')
    with ExitStack() as owned:
        device = ops.DeviceRuntime.open(DevicePlan.discover(backend='metal', maximum_bytes=1 << 26))
        owned.callback(device.close)
        spec = ops.TensorSpec((64,), ops.DType.F32)
        source = device.upload(spec, struct.pack('=64f', *range(64)))
        owned.callback(source.close)
        compiled = ops.compile(lambda value: value + value,
                               signature=ops.Signature((ops.Argument(spec, 'value'),)),
                               device=device, constants={}, options=ops.CompileOptions(mode='decode'))
        owned.callback(compiled.close)
        compiled.reuse_output_storage(max_frames=2)

        def execute(value):
            result = compiled.submit(value)
            result.completion.wait()
            output = result.outputs[0]
            owned.callback(output.close)
            return output

        first = execute(source)
        checkpoint = first.fork()
        owned.callback(checkpoint.close)
        first.close()
        second = execute(checkpoint)
        third = execute(second)
        expected = tuple(float(i * 2) for i in range(64))
        assert struct.unpack('=64f', device.read(checkpoint)) == expected
        assert struct.unpack('=64f', device.read(second)) == tuple(float(i * 4) for i in range(64))
        assert struct.unpack('=64f', device.read(third)) == tuple(float(i * 8) for i in range(64))
        # Both cached frames escape. Dropping the pool must preserve their leases.
        compiled.release_output_storage()
        assert struct.unpack('=64f', device.read(checkpoint)) == expected
        second.close()
        third.close()
        fourth = execute(checkpoint)
        fourth.close()
        charged = device.allocated_bytes
        for _ in range(5):
            result = execute(checkpoint)
            result.close()
            assert device.allocated_bytes == charged
        # The retained caller lease also survives destruction of the executable.
        compiled.close()
        assert struct.unpack('=64f', device.read(checkpoint)) == expected
