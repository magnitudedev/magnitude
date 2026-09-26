"""Final-gate coverage of recurring source I/O through production operations."""

from contextlib import ExitStack

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.tensor.primitive import evaluate_reference
from ops.runtime.observation import Activity


def source(value):
    content = value.tobytes()
    return ops.Binding(ops.TensorSpec(value.shape, ops.DType.F32), "fixture:" + str(value.shape),
                       ops.Residency.STREAMED,
                       (ops.SourcePlane(ops.SourceSpan(ops.MemorySource(content), 0, len(content)), 1, 4),),
                       ops.DenseImport(ops.DType.F32))


@pytest.mark.device
@pytest.mark.parametrize("kind", ["projection", "projection-vector", "projection-batched", "embedding"])
def test_bounded_source_consumers_include_tail_io_without_invocation_compilation(kind):
    if not torch.backends.mps.is_available():
        pytest.skip("requires the designated Metal gate device")
    weights = (np.arange(257 * 32, dtype=np.float32).reshape(257, 32) % 13) / 16
    binding = source(weights)
    if kind.startswith("projection"):
        values = np.eye(32, dtype=np.float32)[:2]
        if kind == "projection-vector":
            values = values[0]
        elif kind == "projection-batched":
            values = values.reshape(1, 2, 32)
        function, dtype = ops.linear, ops.DType.F32
        expected = values @ weights.T
    else:
        values = np.array([256, 0, 128, 256, 7], np.int32)
        function, dtype = ops.embedding, ops.DType.I32
        expected = weights[values]
    signature = ops.Signature((ops.Argument(ops.TensorSpec(values.shape, dtype), "value"),
                               ops.Argument(binding.spec, "weight", ops.ValueKind.CONSTANT)))
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=128 << 10)) as device, ExitStack() as owned:
        operand = device.upload(signature.args[0].spec, values.tobytes())
        owned.callback(operand.close)
        compiled = ops.compile(function, signature=signature, device=device,
                               constants={"weight": binding}, options=ops.CompileOptions(mode="prefill"))
        owned.callback(compiled.close)
        for _ in range(2):
            with device.observe() as capture:
                execution = compiled.submit(operand)
                execution.completion.wait()
            try:
                actual = np.frombuffer(device.read(execution.outputs[0]), np.float32).reshape(expected.shape)
                np.testing.assert_array_equal(actual, expected)
                source_bytes = capture.result.completed_bytes(Activity.SOURCE_READ)
                if kind.startswith("projection"):
                    assert source_bytes == weights.nbytes
                else:
                    assert 0 < source_bytes <= values.size * weights.shape[1] * weights.dtype.itemsize
                assert not any(item.kind == Activity.COMPILE for item in capture.result.activities)
            finally:
                for output in execution.outputs:
                    output.close()


@pytest.mark.device
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_streamed_experts_reuse_prepared_regions_and_preserve_rank_sum(mode):
    if not torch.backends.mps.is_available():
        pytest.skip("requires the designated Metal gate device")
    rows, width, experts, selected = 3, 32, 3, 2
    hidden = (np.arange(rows * width, dtype=np.float32).reshape(rows, width) % 7 - 3) / 16
    routes = np.array([[2, 0], [0, 2], [2, 1]], np.int32)
    scores = np.array([[0.2, 0.8], [0.7, 0.3], [0.5, 0.5]], np.float32)
    weights = np.stack([np.eye(width, dtype=np.float32) * ((index + 1) / 4) for index in range(experts)])
    banks = {name: source(weights) for name in ("gate", "up", "down")}
    signature = ops.Signature(tuple(ops.Argument(ops.TensorSpec(value.shape, dtype), name)
                                   for name, value, dtype in (("hidden", hidden, ops.DType.F32),
                                                             ("routes", routes, ops.DType.I32),
                                                             ("scores", scores, ops.DType.F32))),
                              {name: ops.Argument(binding.spec, name, ops.ValueKind.CONSTANT)
                               for name, binding in banks.items()})
    graph = ops.trace(ops.routed_experts, signature)
    expected = evaluate_reference(graph, dict(zip((*graph.inputs, *graph.constants),
                                                 (hidden, routes, scores, weights, weights, weights), strict=True))).outputs[0]
    with ops.DeviceRuntime.open(DevicePlan.discover(backend="metal", maximum_bytes=64 << 20)) as device, ExitStack() as owned:
        operands = []
        for value, identity in zip((hidden, routes, scores), graph.inputs, strict=True):
            resource = device.upload(graph.value(identity).spec, value.tobytes())
            owned.callback(resource.close)
            operands.append(resource)
        program = ops.compile(ops.routed_experts, signature=signature, device=device,
                              constants=banks, options=ops.CompileOptions(mode=mode))
        owned.callback(program.close)
        for _ in range(2):
            with device.observe() as capture:
                execution = program.submit(*operands)
                execution.completion.wait()
            try:
                actual = np.frombuffer(device.read(execution.outputs[0]), np.float32).reshape(hidden.shape)
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-6)
                assert capture.result.completed_bytes(Activity.SOURCE_READ) > 0
                assert not any(activity.kind == Activity.COMPILE for activity in capture.result.activities)
            finally:
                for output in execution.outputs:
                    output.close()
