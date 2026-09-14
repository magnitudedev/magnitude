from datetime import UTC, datetime
from dataclasses import replace
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from ops.formula import FormulaRef, units
from ops.lab.records import (
    Implementation, Measurement, MeasurementProtocol, Outcome, Series, UsefulQuantity,
)
from ops.lab.store import ObservationStore
from ops.runtime.memory import MemoryMeasurement
from ops.runtime.observation import KernelActivity, KernelObservation, ObservationStatus, RuntimeObservation


def test_workers_can_initialize_one_new_observation_store_concurrently(tmp_path):
    barrier = Barrier(8)
    path = tmp_path / "simultaneous.sqlite"

    def open_store(index):
        barrier.wait(timeout=5)
        with ObservationStore(path) as store:
            assert store._connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
            store.publish(measurement(f"worker-{index}", "kernel", 100 + index))

    with ThreadPoolExecutor(max_workers=8) as workers:
        tuple(workers.map(open_store, range(8)))
    with ObservationStore(path) as store:
        assert len(store.history(series()).observations) == 8


def test_journal_transition_retries_only_bounded_sqlite_lock_errors(monkeypatch):
    from types import SimpleNamespace
    from ops.lab import store

    clock = [0.0]
    monkeypatch.setattr(store, "monotonic", lambda: clock[0])
    monkeypatch.setattr(store, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Connection:
        mode = "delete"
        attempts = 0
        code = sqlite3.SQLITE_BUSY
        succeed = True

        def execute(self, query):
            if query == "PRAGMA journal_mode=WAL":
                self.attempts += 1
                if self.succeed and self.attempts == 2:
                    self.mode = "wal"
                else:
                    error = sqlite3.OperationalError("journal transition")
                    error.sqlite_errorcode = self.code
                    raise error
            return SimpleNamespace(fetchone=lambda: (self.mode,))

    connection = Connection()
    store._enable_wal(connection)
    assert connection.attempts == 2
    store._enable_wal(connection)
    assert connection.attempts == 2  # Existing WAL requires no transition.
    blocked = Connection()
    blocked.succeed = False
    with pytest.raises(sqlite3.OperationalError):
        store._enable_wal(blocked)
    assert clock[0] >= 5
    broken = Connection()
    broken.code = sqlite3.SQLITE_READONLY
    with pytest.raises(sqlite3.OperationalError):
        store._enable_wal(broken)
    assert broken.attempts == 1


def series():
    return Series(
        formula=FormulaRef("projection", 1), semantics="same-equation", fixture="same-values",
        device="same-device", geometry="same-shape", precision="fp32",
        protocol=MeasurementProtocol(samples=1, warmups=0),
    )


def measurement(identity, source, elapsed, *, outcome=Outcome.COMPLETE):
    return Measurement(
        identity=identity, created=datetime.now(UTC), series=series(),
        implementation=Implementation(fingerprint=source, compiler="test", dependencies=(source,)),
        outcome=outcome, checked=outcome == Outcome.COMPLETE,
        samples=(RuntimeObservation(
            ObservationStatus.COMPLETE, elapsed, (), MemoryMeasurement(0, 0, 0, 0, 0, ()), None,
        ),) if outcome == Outcome.COMPLETE else (),
        error="cancelled" if outcome != Outcome.COMPLETE else None,
    )


def test_implementation_rewrite_shares_history_and_failure_preserves_last_success(tmp_path):
    first = measurement("first", "old-kernel", 200)
    second = measurement("second", "new-kernels", 100)
    cancelled = measurement("third", "new-kernels", 0, outcome=Outcome.CANCELLED)
    with ObservationStore(tmp_path / "measurements.sqlite") as store:
        store.publish(first)
        store.publish(second)
        store.publish(cancelled)
        history = store.history(series(), limit=1)
        assert history.latest == cancelled
        assert history.latest_success == second
        assert history.best == second
        assert len(history.observations) == 1
        assert not history.stale(second.implementation)
        assert history.stale(first.implementation)
        assert store.series() == (series(),)
    with ObservationStore(tmp_path / "measurements.sqlite") as store:
        assert store.history(series()).latest_success == second


def test_records_are_immutable_and_identical_publication_is_idempotent(tmp_path):
    record = measurement("same", "kernel", 100)
    with ObservationStore(tmp_path / "measurements.sqlite") as store:
        store.publish(record)
        store.publish(record)
        assert len(store.history(series()).observations) == 1
        with pytest.raises(ValueError, match="immutable"):
            store.publish(measurement("same", "other", 200))
        assert store.history(series()).latest == record


def test_incomplete_runtime_sample_is_not_a_successful_measurement():
    record = measurement("one", "kernel", 100).model_dump()
    record["samples"] = (RuntimeObservation(
        ObservationStatus.INCOMPLETE, 100, (), MemoryMeasurement(0, 0, 0, 0, 0, ()), None,
    ),)
    with pytest.raises(ValueError, match="unfinished runtime"):
        Measurement.model_validate(record)


def test_series_identity_changes_with_conditions_not_implementation():
    base = series()
    assert base.identity == measurement("one", "kernel-a", 100).series.identity
    assert base.identity == measurement("two", "kernel-b", 100).series.identity
    changed = Series.model_validate({**base.model_dump(), "fixture": "different-values"})
    assert base.identity != changed.identity


def native_measurement():
    record = measurement("native", "two-kernels", 1000)
    sample = replace(record.samples[0], kernels=KernelObservation("test-native-clock", (
        KernelActivity("first", 100), KernelActivity("second", 150),
    )))
    return Measurement.model_validate({**record.model_dump(), "samples": (sample,), "quantities": (
        UsefulQuantity(name="arithmetic", amount=200, unit=units.flop, basis="formula useful work"),
    )})


def test_historical_protocol_retains_exact_series_key():
    payload = series().model_dump(mode="json")
    payload["protocol"].pop("kernel_limit")
    payload["protocol"]["version"] = 1
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    historical = Series.model_validate(payload)
    assert historical.protocol.kernel_limit is None
    assert historical.model_dump(mode="json") == payload
    assert historical.identity == expected
    assert historical.identity != series().identity
    with pytest.raises(ValueError, match="version 1"):
        MeasurementProtocol(version=1, kernel_limit=2)
    disabled = MeasurementProtocol(kernel_limit=None)
    assert MeasurementProtocol.model_validate_json(disabled.model_dump_json()) == disabled
    assert disabled.kernel_limit is None


def test_historical_observation_is_not_rewritten_when_republished(tmp_path):
    payload = measurement("old", "old-kernel", 1000).model_dump(mode="json")
    payload["series"]["protocol"].pop("kernel_limit")
    payload["series"]["protocol"]["version"] = 1
    payload["samples"][0].pop("kernels")
    original = json.dumps(payload)
    record = Measurement.model_validate_json(original)
    with ObservationStore(tmp_path / "old.sqlite") as store:
        store.publish(record)
        # Fixture representing the pre-extension serialized observation.
        store._connection.execute("UPDATE formula_measurements SET record=? WHERE identity=?", (original, record.identity))
        store._connection.commit()
        store.publish(store.history(record.series).latest_success)
        saved, = store._connection.execute("SELECT record FROM formula_measurements").fetchone()
        assert saved == original


def test_native_and_complete_operation_rates_roundtrip_in_one_history(tmp_path):
    evidence = native_measurement()
    metrics = {metric.name: metric for metric in evidence.metrics}
    assert metrics["elapsed"].value == 1e-6
    assert metrics["kernel-device-time"].value == 250e-9
    assert metrics["kernel-count"].value == 2
    assert metrics["rate:arithmetic"].value == 200e6
    assert metrics["kernel-rate:arithmetic"].value == 800e6
    with ObservationStore(tmp_path / "native.sqlite") as store:
        store.publish(evidence)
    with ObservationStore(tmp_path / "native.sqlite") as store:
        assert store.history(evidence.series).latest_success == evidence


def test_native_timing_cannot_be_partially_present_or_exceed_protocol():
    evidence = native_measurement()
    payload = evidence.model_dump()
    payload["series"]["protocol"]["samples"] = 2
    payload["samples"] = (evidence.samples[0], replace(evidence.samples[0], kernels=None))
    with pytest.raises(ValueError, match="partially available"):
        Measurement.model_validate(payload)
    payload = evidence.model_dump()
    payload["series"]["protocol"]["kernel_limit"] = 1
    with pytest.raises(ValueError, match="exceed"):
        Measurement.model_validate(payload)
