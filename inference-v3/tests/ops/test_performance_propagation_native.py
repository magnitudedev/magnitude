"""A real schedule change, an enclosing prediction, then an independent GPU check."""

import numpy as np
import pytest
import torch
from formula_performance.evidence import evaluate
from formula_performance.records import Publication, Transfer

import ops
from engine import DevicePlan
from ops.lab.fixtures import Fixture
from ops.lab.records import MeasurementProtocol
from ops.lab.runner import MeasurementRunner
from ops.lab.store import ObservationStore


@pytest.fixture(autouse=True)
def machine_ownership():
    from ops.lab.ownership import exclusive_measurement

    with exclusive_measurement():
        yield


@ops.formula(id="qualification.performance.model", metric="tokens", rows="x")
def composed(x, weight):
    return ops.linear(x, weight), ops.silu(x)


class MatrixSchedule:
    def __init__(self, alternative):
        self.alternative = alternative
        self.choices = []

    def select(self, request, default):
        selected = default
        if self.alternative and request.semantic_kernel == "matrix.dense":
            selected = next(c for c in reversed(request.candidates) if c != default)
        self.choices.append((request.semantic_kernel, selected))
        return selected


@pytest.mark.device
def test_native_component_change_predicts_and_checks_enclosing_boundary(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("requires Metal native timestamps")
    graph = ops.trace(
        composed,
        ops.Signature(
            (
                ops.Argument(ops.TensorSpec((16, 512), ops.DType.F32)),
                ops.Argument(
                    ops.TensorSpec((512, 512), ops.DType.F32), kind=ops.ValueKind.CONSTANT
                ),
            )
        ),
    )
    rng = np.random.default_rng(42)
    fixture = Fixture.from_inputs(
        graph,
        {
            v: rng.normal(0, 0.1, graph.value(v).spec.shape).astype(np.float32)
            for v in (*graph.inputs, *graph.constants)
        },
    )
    root = ops.FormulaTree(graph).roots[0]
    child = root.children[0]
    protocols = MeasurementProtocol(
        samples=3, warmups=1, kernel_limit=32, absolute_tolerance=1e-4, relative_tolerance=1e-4
    )
    plan = DevicePlan.discover(backend="metal", maximum_bytes=256 << 20)
    publications = []
    schedules = [MatrixSchedule(False), MatrixSchedule(True)]
    with (
        ops.DeviceRuntime.open(plan) as device,
        ObservationStore(tmp_path / "native.sqlite") as store,
    ):
        for index, targets in enumerate(((root,), (child, root))):
            runner = MeasurementRunner(
                fixture,
                device,
                store,
                ops.CompileOptions(
                    mode="prefill", precision="reference", schedules=schedules[index]
                ),
                protocol=protocols,
            )
            try:
                for target in targets:
                    m = runner.measure(target).measurement
                    assert m.checked, m.error
                    p = Publication.model_validate_json(
                        store.artifact(m.artifacts["formula-performance"])
                    )
                    # Both programs use this exact schedule policy over one fixture.
                    # Preserve their independently compiled identities in raw evidence.
                    p = p.model_copy(
                        update={
                            "observations": tuple(
                                o.model_copy(
                                    update={
                                        "implementation": "baseline" if index == 0 else "candidate",
                                        "coordinates": {"fixture": graph.fingerprint},
                                    }
                                )
                                for o in p.observations
                            )
                        }
                    )
                    publications.append(p)
            finally:
                runner.close()
    assert schedules[0].choices != schedules[1].choices
    first = evaluate(publications[:1])
    child_key = next(k for k, r in first["components"].items() if r["definition"] == "linear")
    captured = publications[0].captures[0]
    contribution = first["components"][child_key]["contributions"][0]
    assert contribution["complete_boundary"]
    transfer = Transfer(
        identity="controlled-matrix-schedule",
        child=child_key,
        parent="",
        baseline_child=captured.identity + ":" + child_key,
        baseline_parent=captured.identity + ":",
        contribution=contribution["inclusive_seconds"],
        relationship="conditional-serial",
        assumptions=("only the matrix tile changes; sibling and enclosing costs remain fixed",),
        evidence=(captured.identity,),
    )
    contract = Publication(transfers=(transfer,))
    before = evaluate((*publications[:2], contract))
    predictions = before["components"][""]["predictions"]
    prediction = next(p for p in predictions if p["baseline"] == transfer.baseline_parent)
    assert not prediction["validation"]
    after = evaluate((*publications, contract))
    checked = next(
        p
        for p in after["components"][""]["predictions"]
        if p["baseline"] == transfer.baseline_parent
    )
    assert checked["seconds"] == prediction["seconds"]
    assert checked["validation"]
    (tmp_path / "propagation.json").write_text(__import__("json").dumps(after, indent=2))


@pytest.mark.device
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_bounded_workload_check_uses_actual_production_outputs(phase):
    from types import SimpleNamespace

    from roofline.integrations.magnitude import Magnitude

    from engine.models.qwen35.runtime import DenseRuntime
    from tests.models.test_qwen35_tensor_program import ArrayResidency, _description

    if not torch.backends.mps.is_available():
        pytest.skip("requires Metal")
    with ops.DeviceRuntime.open(
        DevicePlan.discover(backend="metal", maximum_bytes=256 << 20)
    ) as device:
        description = _description()
        data = description.model_dump()
        # Use complete subgroup-width heads for this integration fixture.
        data["geometry"].update(
            hidden=64,
            intermediate=128,
            attention_width=32,
            rotary_width=32,
            rotary_sections=(8, 8, 0, 0),
            recurrent_width=32,
        )

        def dimensions(value):
            if isinstance(value, dict):
                if "shape" in value and "name" in value:
                    value["shape"] = tuple(
                        {4: 32, 8: 64, 16: 128}.get(d, d) for d in value["shape"]
                    )
                for item in value.values():
                    dimensions(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    dimensions(item)

        dimensions(data)
        description = type(description).model_validate(data)
        weights = ArrayResidency(device)
        weights.identity = description.artifact_identity
        owner = object.__new__(Magnitude)
        owner.device = device
        owner.experiment = SimpleNamespace(
            scope=phase, context=2, steps=2, protocol=SimpleNamespace(samples=1, warmups=0)
        )
        owner.prefill_rows = 2
        owner.tokens = lambda: SimpleNamespace(prompt=(1, 2), continuation=(3, 4))
        owner.decode_weight = lambda value: weights.arrays[value.name]
        owner.model = DenseRuntime(
            description, device, weights, max_sequences=1, prefill_rows=2, context_capacity=16
        )
        try:
            checks = owner.check_workload()
            assert len(checks) == (1 if phase == "prefill" else 2)
            assert all(check["passed"] for check in checks)
        finally:
            owner.model.close()
            for resource in weights.resources:
                resource.close()
