"""Consolidated gate: populated production hierarchy and actual streamed source."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.models.qwen35.tensor_program import InvocationSpecs, define, weight_roles
from ops.lab import Lab, MeasurementProtocol, ObservationStore
from ops.lab.records import Outcome
from ops.performance.resources import Resource
from ops.runtime.observation import Activity
from tests.models.test_qwen35_tensor_program import _description


@pytest.fixture
def file_source(tmp_path):
    from engine.platform.storage import FileSource

    path = tmp_path / "streamed-source.bin"
    path.write_bytes(np.arange(65536, dtype=np.uint8).tobytes())
    with FileSource(path) as source:
        yield source


@pytest.mark.device
@pytest.mark.performance
def test_production_qwen_tree_has_independent_measurements_and_resource_models(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("designated Metal qualification device required")
    description = _description()
    rows = 2
    specs = InvocationSpecs(
        batch=1, tokens=ops.TensorSpec((rows,), ops.DType.I32),
        coordinates=ops.TensorSpec((rows, 3), ops.DType.I32),
        recurrent_offsets=ops.TensorSpec((2,), ops.DType.I32),
        output_rows=None, draws=None,
        destinations=(ops.TensorSpec((rows,), ops.DType.I32),),
        visible=(ops.TensorSpec((rows, 2), ops.DType.I32),),
        attention_state=(ops.TensorSpec((2, 16, 1, 4), ops.DType.F16),),
        convolution_state=(ops.TensorSpec((1, 16, 2), ops.DType.F16),),
        delta_state=(ops.TensorSpec((1, 2, 4, 4), ops.DType.F32),),
    )
    rng = np.random.default_rng(44)
    weights, values = {}, {}
    for descriptor, dtype in weight_roles(description):
        weights[descriptor.name] = ops.TensorSpec(descriptor.shape, dtype)
        values[descriptor.name] = (
            np.ones(descriptor.shape, dtype=dtype.value) if descriptor.name.endswith("norm") else
            np.full(descriptor.shape, -.1, dtype=dtype.value) if descriptor.name.endswith("decay") else
            rng.normal(0, .05, descriptor.shape).astype(dtype.value))
    definition = define(description, weights, "prefill", specs, precision="reference")
    values.update({
        "tokens": np.array([1, 2], np.int32),
        "coordinates": np.repeat(np.arange(rows, dtype=np.int32)[:, None], 3, axis=1),
        "recurrent_offsets": np.array([0, rows], np.int32),
        "attention.0.destinations": np.arange(rows, dtype=np.int32),
        "attention.0.visible": np.array([[0, 1], [0, 2]], np.int32),
        "attention.0.state": np.zeros((2, 16, 1, 4), np.float16),
        "recurrent.0.0.convolution": np.zeros((1, 16, 2), np.float16),
        "recurrent.0.0.delta": np.zeros((1, 2, 4, 4), np.float32),
    })
    fixture = definition.fixture(values)
    plan = DevicePlan.discover(backend="metal", maximum_bytes=1 << 30)
    path = Path(os.environ.get("MAGNITUDE_ROOFLINE_STORE", tmp_path / "production-tree.sqlite"))
    # Match the existing hybrid production check, not a new tolerance relaxation.
    protocol = MeasurementProtocol(absolute_tolerance=.08, relative_tolerance=.08)
    with Lab(fixture=fixture, device=lambda: ops.DeviceRuntime.open(plan), store=path,
             options=definition.options, protocol=protocol, label="Qwen hybrid production composition · small fixture") as lab:
        root, = lab.formulas.roots
        targets = lab.subtree(root)
        assert len(targets) > 20
        for ticket in lab.measure_subtree(root):
            result = ticket.result.result(timeout=300)
            assert result.outcome == Outcome.COMPLETE, (ticket.target.definition, result.error)
        states = lab.snapshot().states
        assert all(state.history.latest_success.roofline is not None for state in states)
        assert all(state.history.latest_success.checked for state in states)
        profile = lab.snapshot().characterization
        assert all(state.history.latest_success.roofline.characterization == profile.identity for state in states)
        print("Populated production tree:", path)
        for state in states:
            from ops.lab.tui import label
            print(label(state).plain)
    with ObservationStore(path) as store:
        record = store.configurations()[0]
        assert len(store.recorded_rooflines(record)) == len(targets)


@pytest.mark.device
@pytest.mark.performance
def test_streamed_source_has_matching_path_evidence(tmp_path, file_source):
    if not torch.backends.mps.is_available():
        pytest.skip("designated Metal qualification device required")
    from ops.lab.characterization import copy
    from ops.lab.fixtures import Fixture

    size = 65536
    data = np.arange(size, dtype=np.uint8)
    source = file_source
    spec = ops.TensorSpec((size,), ops.DType.U8)
    graph = ops.trace(copy, ops.Signature((ops.Argument(spec, "source", ops.ValueKind.CONSTANT),
                                          ops.Argument(spec, "destination", ops.ValueKind.RESOURCE),
                                          ops.Argument(ops.TensorSpec((2,), ops.DType.I64), "extent"))))
    ids = {value.name: value.id for value in graph.values if value.producer is None}
    binding = ops.Binding(spec, "streamed-source-gate", ops.Residency.STREAMED,
                          (ops.SourcePlane(ops.SourceSpan(source, 0, size), 1, 1),), ops.DenseImport(ops.DType.U8))
    fixture = Fixture(graph, {ids["source"]: data, ids["destination"]: np.zeros(size, np.uint8),
                              ids["extent"]: np.array([0, size], np.int64)}, bindings={ids["source"]: binding})
    plan = DevicePlan.discover(backend="metal", maximum_bytes=1 << 30)
    store_path = Path(os.environ.get("MAGNITUDE_ROOFLINE_STORE", tmp_path / "streamed.sqlite"))
    with Lab(fixture=fixture, device=lambda: ops.DeviceRuntime.open(plan), store=store_path,
             options=ops.CompileOptions(mode="prefill")) as lab:
        root, = lab.formulas.roots
        result = lab.measure(root).result.result(timeout=300)
        assert result.outcome == Outcome.COMPLETE, result.error
        observed = lab.inspect(root).result().latest_success
        term, = (item for item in observed.roofline.limits if item.demand.resource == Resource.SOURCE_IMPORT)
        assert term.demand.source == source.info
        assert term.demand.lower == size
        assert all(sample.completed_bytes(Activity.SOURCE_READ) == size for sample in observed.samples)
