"""Actual artifact region, extracted from production composition, not a new equation."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from engine.models.qwen35.equations import routed_feedforward
from engine.models.qwen35.formats.gguf import describe
from engine.models.qwen35.tensor_program import define, weight_roles
from engine.qualification import QualificationCase, invocation_specs
from engine.weights.formats.gguf import GGUFFormat
from engine.weights.tensor_residency import describe_binding
from ops.lab import Fixture, Lab, MeasurementProtocol, ObservationStore
from ops.lab.records import Outcome
from ops.lab.tui import label
from ops.tensor.primitive import round_reference


@pytest.mark.model
@pytest.mark.device
@pytest.mark.performance
def test_artifact_ffn_tree_is_populated_from_production_formulas(tmp_path):
    path = os.environ.get("MAGNITUDE_ROOFLINE_GGUF")
    if path is None or not torch.backends.mps.is_available():
        pytest.skip("requires designated Metal device and MAGNITUDE_ROOFLINE_GGUF")
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    format = GGUFFormat(path)
    try:
        description = describe(format)
        bindings = {role.name: describe_binding(format, role, dtype)
                    for role, dtype in weight_roles(description)}
        case = QualificationCase(name="roofline-artifact", mode="prefill", rows=2048,
                                 batch=1, context=2048, logits=False)
        definition = define(description, {name: binding.spec for name, binding in bindings.items()},
                            "prefill", invocation_specs(description, case, slots=1))
        graph = ops.trace(definition.function, definition.signature)
        selected = ops.FormulaTree(graph).occurrences(routed_feedforward)[0]
        isolated = selected.isolate()
        physical = {port.local: bindings[graph.value(port.original).name] for port in isolated.inputs
                    if graph.value(port.original).name in bindings}
        names = {port.local: graph.value(port.original).name for port in isolated.inputs if port.local in physical}
        hidden_port, = (port for port in isolated.inputs if port.local not in physical)
        hidden = round_reference(np.random.default_rng(42).normal(0, .25, hidden_port.spec.shape), hidden_port.spec.dtype)

        def capture(value):
            entry = format.directory.tensor(names[value.id])
            raw = format.source.read(format.directory.data_offset + entry.offset, entry.nbytes)
            # Independent artifact decoding supplies reference values only. All
            # measured numerical execution uses the production TileLang path.
            return dequantize(np.frombuffer(raw, dtype=np.uint8), GGMLQuantizationType(int(entry.encoding))).reshape(entry.shape)

        fixture = Fixture.from_inputs(isolated.graph, {hidden_port.local: hidden}, bindings=physical, capture=capture)
        plan = DevicePlan.discover(backend="metal", maximum_bytes=16 << 30)
        store_path = Path(os.environ.get("MAGNITUDE_ROOFLINE_STORE", tmp_path / "artifact-tree.sqlite"))
        # Retain the already-established artifact check, not a relaxed gate.
        protocol = MeasurementProtocol(absolute_tolerance=.003, relative_tolerance=.0078125)
        with Lab(fixture=fixture, device=lambda: ops.DeviceRuntime.open(plan), store=store_path,
                 options=definition.options, protocol=protocol, reference_bytes=8 << 30, prepared_limit=16,
                 label="Qwen35 MoE layer-zero FFN · actual GGUF weights · synthetic hidden · 2048 rows") as lab:
            root, = lab.formulas.roots
            targets = lab.subtree(root)
            for ticket in lab.measure_subtree(root):
                result = ticket.result.result(timeout=600)
                assert result.outcome == Outcome.COMPLETE, (ticket.target.definition, result.error)
            for state in lab.snapshot().states:
                observed = state.history.latest_success
                assert observed.roofline is not None
                print(label(state).plain)
                # An empirical reference is not qualified merely by existing.
                # A large above-reference result invalidates its applicability,
                # not numerical correctness and not the fast implementation.
                assert observed.roofline.seconds <= observed.median_seconds * 1.1, label(state).plain
            print("Artifact-backed populated tree:", store_path)
        with ObservationStore(store_path) as store:
            configuration = store.configurations()[0]
            assert len(store.recorded_rooflines(configuration)) == len(targets)
    finally:
        format.close()
