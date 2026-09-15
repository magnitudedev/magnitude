from pathlib import Path

import numpy as np
import pytest
import torch

import ops
from engine import DevicePlan
from ops.lab.comparison import compare_prepared
from ops.lab.fixtures import Fixture
from ops.lab.preparation import PreparedFormula
from ops.lab.records import MeasurementProtocol
from ops.lab.retention import load_boundary, save_boundary
from ops.lab.store import ObservationStore
from tests.ops.test_model_evidence import run


@ops.formula(id="test.performance.upstream")
def upstream(value):
    return ops.silu(ops.tanh(value))


def fixture():
    graph = ops.trace(
        upstream, ops.Signature((ops.Argument(ops.TensorSpec((64,), ops.DType.F32), "value"),))
    )
    return Fixture.from_inputs(graph, {graph.inputs[0]: np.linspace(-2, 2, 64, dtype=np.float32)})


def test_retained_boundary_roundtrip_and_corruption(tmp_path):
    original = fixture()
    (target,) = ops.FormulaTree(original.root).occurrences(ops.silu)
    boundary = original.boundary(target)
    path = tmp_path / "boundary.zip"
    save_boundary(boundary, path, provenance={"workload": "fixed-test"})
    restored, provenance = load_boundary(original, target, path)
    assert restored.identity == boundary.identity
    assert provenance == {"workload": "fixed-test"}
    np.testing.assert_array_equal(restored.reference.outputs[0], boundary.reference.outputs[0])
    assert not next(iter(restored.inputs.values())).reference.flags.writeable
    other = ops.FormulaTree(original.root).roots[0]
    with pytest.raises(ValueError, match="trace"):
        load_boundary(original, other, path)


@pytest.mark.device
def test_production_capture_and_paired_replay_on_gpu(tmp_path, monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal device unavailable")
    source = fixture()
    (target,) = ops.FormulaTree(source.root).occurrences(ops.silu)
    plan = DevicePlan.discover(backend="metal", maximum_bytes=128 << 20)
    options = ops.CompileOptions(mode="decode")
    protocol = MeasurementProtocol(
        samples=2, warmups=1, absolute_tolerance=1e-6, relative_tolerance=1e-6, kernel_limit=32
    )
    with (
        ops.DeviceRuntime.open(plan) as device,
        ObservationStore(tmp_path / "evidence.sqlite") as store,
    ):
        boundary, dependencies, prefix = source.capture_boundary(target, device, options)
        np.testing.assert_allclose(
            next(iter(boundary.inputs.values())).reference,
            np.tanh(np.linspace(-2, 2, 64, dtype=np.float32)),
            atol=1e-6,
        )
        assert dependencies and prefix != source.root.fingerprint
        candidates = {name: PreparedFormula(boundary, device, options) for name in ("a", "b")}
        try:
            results = compare_prepared(
                candidates, context=run().context, store=store, protocol=protocol, blocks=2
            )
            assert len(results) == 2
            for result in results:
                m = store.measurement(result.measurements[0])
                assert m.checked
                assert m.samples[0].kernels.busy_ns is not None
                assert m.samples[0].kernels.attribution == "compiled-order-and-symbols"
                assert m.samples[0].kernels.activities[0].origins
            from ops.lab.preparation import NumericalMismatch

            def mismatch(*args):
                raise NumericalMismatch("deliberately invalid candidate")

            with monkeypatch.context() as changed:
                changed.setattr(candidates["b"], "check", mismatch)
                exploratory = compare_prepared(
                    candidates,
                    context=run().context,
                    store=store,
                    protocol=protocol.model_copy(update={"measure_invalid": True}),
                    blocks=2,
                )
                assert exploratory[1].status == "complete"
                assert exploratory[1].correctness == "failed"
                assert store.measurement(exploratory[1].measurements[0]).median_seconds is None

            def broken(*args, **kwargs):
                raise RuntimeError("deliberately broken sample")

            with monkeypatch.context() as changed:
                changed.setattr(candidates["b"], "sample", broken)
                with pytest.raises(RuntimeError, match="broken sample"):
                    compare_prepared(
                        candidates, context=run().context, store=store, protocol=protocol, blocks=2
                    )
                incomplete = [r for r in store.runs() if r.status == "incomplete"]
                assert len(incomplete) == 2
                assert all(
                    store.measurement(r.measurements[0]).observed_seconds is None
                    for r in incomplete
                )
        finally:
            for candidate in candidates.values():
                candidate.close()


def configuration():
    from ops.lab import Configuration

    return Configuration(
        label="Production prefix replay integration",
        fixture=fixture(),
        device=lambda: ops.DeviceRuntime.open(
            DevicePlan.discover(backend="metal", maximum_bytes=128 << 20)
        ),
        options=ops.CompileOptions(mode="decode"),
        store=Path("unused.sqlite"),
        protocol=MeasurementProtocol(samples=1, warmups=0, kernel_limit=32),
    )


@pytest.mark.device
def test_headless_factory_retains_and_restores_without_requiring_analysis(tmp_path, monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("Metal device unavailable")
    from ops.lab.execution import MeasurementRequest, describe_configuration, execute_request
    from ops.lab.runner import MeasurementRunner

    factory = __name__ + ":configuration"
    # Discovery must not construct a device or compile the trace.
    with monkeypatch.context() as guard:
        guard.setattr(
            ops.DeviceRuntime, "open", lambda *a, **k: pytest.fail("discovery opened a device")
        )
        description = describe_configuration(factory, {})
    target = next(t for t in description["scopes"] if t["formula"] == ops.silu.ref.id)
    request = MeasurementRequest(
        context=run().context,
        occurrences=(target["occurrence"],),
        semantics={target["occurrence"]: target["semantics"]},
        repeats=2,
        protocol=MeasurementProtocol(
            samples=1,
            warmups=0,
            kernel_limit=32,
            inputs="production",
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
        ),
        save_boundaries=str(tmp_path / "boundaries"),
    )

    def unavailable(*args):
        raise ValueError("deliberately unsupported resource analysis")

    monkeypatch.setattr(MeasurementRunner, "_quantities", unavailable)
    path = tmp_path / "history.sqlite"
    execute_request(factory, request, path)
    with ObservationStore(path, read_only=True) as store:
        results = store.runs()
        assert len(results) == 2
        measured = [store.measurement(r.measurements[0]) for r in results]
        assert all(m.checked and m.observed_seconds is not None for m in measured)
        assert all(m.preparation["boundary_reused"] for m in measured)
        assert all(any(u.name == "useful-quantities" for u in m.unavailable) for m in measured)
        assert all(m.artifacts for m in measured)
    restored = request.model_copy(
        update={
            "repeats": 1,
            "save_boundaries": None,
            "retained_boundaries": {
                target["occurrence"]: str(tmp_path / "boundaries" / f"{target['occurrence']}.zip")
            },
        }
    )
    execute_request(factory, restored, path)
    with ObservationStore(path, read_only=True) as store:
        latest = store.measurement(store.runs()[0].measurements[0])
        assert latest.preparation["restored_fixed_fixture"] is not None

    from ops.lab.bundles import export_bundle, import_bundle

    bundle = tmp_path / "portable.json"
    with ObservationStore(path, read_only=True) as store:
        export_bundle(store, bundle)
    with ObservationStore(tmp_path / "imported.sqlite") as imported:
        assert import_bundle(imported, bundle) == 3
        for digest in latest.artifacts.values():
            assert imported.artifact(digest)

    from ops.lab.preparation import NumericalMismatch

    def mismatch(*args):
        raise NumericalMismatch("exploratory request")

    with monkeypatch.context() as exploratory:
        exploratory.setattr(PreparedFormula, "check", mismatch)
        invalid = restored.model_copy(
            update={
                "repeats": 2,
                "protocol": restored.protocol.model_copy(update={"measure_invalid": True}),
            }
        )
        execute_request(factory, invalid, path)
    with ObservationStore(path, read_only=True) as store:
        assert len(store.runs()) == 5
        assert all(r.status == "complete" and r.correctness == "failed" for r in store.runs()[:2])
