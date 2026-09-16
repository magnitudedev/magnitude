"""Preparation ownership checks, without a native compiler or numerical backend."""

from types import SimpleNamespace

import numpy as np
import pytest

import ops
from ops.lab import Fixture, MeasurementProtocol, preparation
from tests.ops.test_runtime_observation import Runtime as MemoryRuntime


@ops.formula
def read_state(state):
    return state + state


@ops.formula
def write_state(state, ranges):
    return ops.kv_copy(state, ranges, max_count=1)


class Runtime(MemoryRuntime):
    compiler_target = ops.CompilerTarget(32, 256, 32768)
    compiler_identity = "preparation-test"
    runtime_identity = "preparation-test"


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_only_written_state_needs_new_conditioning_storage(monkeypatch, writable, selected):
    spec = ops.TensorSpec((2, 2, 1, 1), ops.DType.F32)
    arguments = [ops.Argument(spec, "state", ops.ValueKind.RESOURCE)]
    if writable:
        arguments.append(ops.Argument(ops.TensorSpec((1, 3), ops.DType.I32), "ranges"))
    graph = ops.trace(write_state if writable else read_state, ops.Signature(tuple(arguments)))
    values = {graph.resources[0]: np.arange(4, dtype=np.float32).reshape(spec.shape)}
    if writable:
        values[graph.inputs[0]] = np.array([[0, 1, 1]], dtype=np.int32)
    fixture = Fixture(graph, values)
    boundary = fixture.boundary(ops.FormulaTree(graph).roots[0])
    resolver = object() if selected else None
    def analyze(graph, **kwargs):
        assert kwargs["options"].schedules is resolver
        assert kwargs["device_identity"] == (device.evidence_identity if selected else None)
        return SimpleNamespace(graph=graph, operations=())
    monkeypatch.setattr(preparation, "analyze_graph", analyze)
    monkeypatch.setattr(preparation, "materialize", lambda plan, **kwargs:
                        SimpleNamespace(graph=plan.graph, code_dependencies=(), close=lambda: None))
    with ops.DeviceRuntime(Runtime(), budget_bytes=4096, schedules=resolver) as device:
        prepared = preparation.PreparedFormula(boundary, device, ops.CompileOptions(mode="decode"))
        try:
            identity, = boundary.isolated.graph.resources
            if writable:
                assert prepared._state_content[identity] is boundary.inputs[identity].physical
            with prepared.inputs() as first:
                before = first.state[identity]._lease.allocation
                assert device.read(first.state[identity]) == boundary.inputs[identity].physical
                if writable:
                    before.native.content = bytes(spec.storage_nbytes)
            with prepared.inputs() as second:
                after = second.state[identity]._lease.allocation
                assert (before is after) is not writable
                assert device.read(second.state[identity]) == boundary.inputs[identity].physical
                if not writable:
                    expected = boundary.reference.outputs[0].astype(np.float32).tobytes()

                    def execute(_):
                        return SimpleNamespace(
                            outputs=(device.upload(spec, expected),),
                            completion=SimpleNamespace(wait=lambda: None),
                        )

                    monkeypatch.setattr(prepared, "execute", execute)
                    prepared.check(second, MeasurementProtocol())
                    after.native.content = bytes(spec.storage_nbytes)
                    with pytest.raises(AssertionError, match="modified a read-only"):
                        prepared.check(second, MeasurementProtocol())
        finally:
            prepared.close()
        assert device.allocated_bytes == 0
