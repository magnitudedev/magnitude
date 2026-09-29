"""Worker lifecycle/queue tests using no native compiler or device execution."""

from threading import Event, get_ident

import numpy as np
import pytest

import ops
from ops.lab import worker
from ops.lab.fixtures import Fixture
from ops.lab.records import Outcome
from ops.lab.store import ObservationStore


@ops.formula(id="test.lab.identity")
def identity(value):
    return value + value


def fixture():
    graph = ops.trace(identity, ops.Signature((ops.Argument(ops.TensorSpec((2,), ops.DType.F32)),)))
    return Fixture(graph, {graph.inputs[0]: np.array([1, 2], dtype=np.float32)})


class Device:
    evidence_identity = "test-device"
    def __init__(self):
        self.owner = get_ident()
        self.closed = False

    def check(self):
        assert self.owner == get_ident()

    def drain(self):
        self.check()

    def close(self):
        self.check()
        self.closed = True


class FailingRunner:
    def __init__(self, fixture, device, *args, **kwargs):
        self.device = device

    def refresh(self):
        self.device.check()
        return ()

    def pending_sources(self):
        return ()

    def measure(self, *args, **kwargs):
        self.device.check()
        raise ValueError("fixture could not be prepared")

    def close(self):
        self.device.check()


def test_configuration_forwards_bounded_preparation_capacity(monkeypatch, tmp_path):
    from ops.lab import Configuration

    received = []

    class CapturingRunner(FailingRunner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            received.append((kwargs["prepared_limit"], kwargs["reference_bytes"]))

    monkeypatch.setattr(worker, "MeasurementRunner", CapturingRunner)
    configuration = Configuration(
        label="large prepared boundary", fixture=fixture(), device=Device,
        options=ops.CompileOptions(mode="decode"), store=tmp_path / "capacity.sqlite",
        prepared_limit=2, reference_bytes=1 << 30,
    )
    with configuration.open() as lab:
        lab.ready.result(timeout=5)
        assert received == [(2, 1 << 30)]


@pytest.mark.parametrize("limits", [{"reference_bytes": 0}, {"reference_bytes": -1},
                                    {"prepared_limit": 0}, {"prepared_limit": -1}])
def test_invalid_preparation_capacity_does_not_start_a_device(tmp_path, limits):
    from ops.lab import Configuration

    def forbidden():
        raise AssertionError("invalid configuration must not open a device")

    arguments = dict(fixture=fixture(), device=forbidden,
                     options=ops.CompileOptions(mode="decode"), store=tmp_path / "invalid.sqlite")
    with pytest.raises(ValueError, match="positive"):
        Configuration(label="invalid", **arguments, **limits)
    with pytest.raises(ValueError, match="positive"):
        worker.Lab(**arguments, **limits)


def test_history_inspection_never_derives_reference_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "MeasurementRunner", FailingRunner)
    prepared = fixture()

    def forbidden(*args):
        raise AssertionError("history must not prepare a boundary")

    monkeypatch.setattr(prepared, "boundary", forbidden)
    path = tmp_path / "history.sqlite"
    with worker.Lab(fixture=prepared, device=Device, store=path,
                    options=ops.CompileOptions(mode="decode")) as lab:
        assert lab.inspect(lab.formulas.roots[0]).result(timeout=5) is None
    with ObservationStore(path) as store:
        record, = store.configurations()
        assert record.device == Device.evidence_identity
        assert record.formulas[0].definition == identity.ref


def test_device_owned_by_worker_and_preparation_failure_persisted(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "MeasurementRunner", FailingRunner)
    created = []
    caller = get_ident()

    def device():
        result = Device()
        created.append(result)
        return result

    path = tmp_path / "observations.sqlite"
    with worker.Lab(fixture=fixture(), device=device, store=path,
                    options=ops.CompileOptions(mode="decode")) as lab:
        target = lab.formulas.roots[0]
        result = lab.measure(target).result.result(timeout=5)
        assert result.outcome == Outcome.FAILED
        assert result.measurement is None
        assert "fixture could not be prepared" in result.error
        assert created[0].owner != caller
        receipt = lab.acknowledge(result, client="test-api")
        assert receipt.request_to_visible_ns >= result.active_ns
    assert created[0].closed
    with ObservationStore(path) as store:
        assert store.jobs()[0] == result
        assert not store.unfinished_jobs()
        assert store.visibility(result.identity) == (receipt,)


def test_cancel_queued_request_does_not_prepare(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "MeasurementRunner", FailingRunner)
    release = Event()

    def device():
        assert release.wait(timeout=5)
        return Device()

    lab = worker.Lab(fixture=fixture(), device=device, store=tmp_path / "observations.sqlite",
                     options=ops.CompileOptions(mode="decode"))
    try:
        target = lab.formulas.roots[0]
        ticket = lab.measure(target)
        assert lab.measure(target) is ticket
        ticket.cancel()
        release.set()
        result = ticket.result.result(timeout=5)
        assert result.outcome == Outcome.CANCELLED
        assert result.measurement is None
    finally:
        release.set()
        lab.close()


def test_requests_reject_string_selection_and_foreign_occurrences(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "MeasurementRunner", FailingRunner)
    with worker.Lab(fixture=fixture(), device=Device, store=tmp_path / "observations.sqlite",
                    options=ops.CompileOptions(mode="decode")) as lab:
        with pytest.raises(ValueError, match="typed occurrence"):
            lab.measure("test.lab.identity")
        other = ops.FormulaTree(fixture().root).roots[0]
        with pytest.raises(ValueError, match="typed occurrence"):
            lab.measure(other)


def test_subtree_scope_is_typed_parent_first_and_does_not_measure_on_preview(monkeypatch, tmp_path):
    from tests.ops.test_connected_roofline import boundary

    monkeypatch.setattr(worker, "MeasurementRunner", FailingRunner)
    fixture, root = boundary()
    with worker.Lab(fixture=fixture, device=Device, store=tmp_path / "subtree.sqlite",
                    options=ops.CompileOptions(mode="prefill")) as lab:
        scope = lab.subtree(root)
        assert scope[0] == root
        assert len(scope) == 3
        assert all(state.job is None for state in lab.snapshot().states)
        with pytest.raises(ValueError, match="typed occurrence"):
            lab.subtree("qualification.roofline.composed")
