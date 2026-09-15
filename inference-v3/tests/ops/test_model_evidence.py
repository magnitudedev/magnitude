import json
import sqlite3
from datetime import UTC, datetime

import pytest

from ops.formula import units
from ops.lab.bundles import export_bundle, import_bundle
from ops.lab.evidence import (
    ExecutionContext,
    Model,
    ObservedMetric,
    PairedComparison,
    RunEvidence,
    Scope,
    Workload,
)
from ops.lab.model_view import ModelApp, model_details
from ops.lab.store import ObservationStore
from ops.runtime.observation import KernelActivity, KernelObservation
from tests.ops.test_observation_store import measurement


def run(identity="run", hardware="same-device", implementation="old"):
    return RunEvidence(
        identity=identity,
        created=datetime.now(UTC),
        context=ExecutionContext(
            model=Model(identity="qwen-4b", label="Qwen 4B"),
            workload=Workload(kind="synthetic", recipe={"shape": [4]}, realization="fixture"),
            engine="v3",
            artifact="weights",
            numerical_contract="fp32",
            hardware=hardware,
            host="test-host",
            implementation=implementation,
        ),
        scope=Scope(kind="prefill"),
        protocol={"pair_ids": ["a", "b"]},
        status="complete",
        correctness="passed",
        metrics=(
            ObservedMetric(
                name="latency",
                unit=units.second,
                samples=(2.0, 3.0),
                boundary="complete",
                basis="wall time",
            ),
        ),
    )


def test_models_share_semantics_not_hardware_timings(tmp_path):
    a = run()
    b = run("other", "other-device", "new")
    with ObservationStore(tmp_path / "db") as store:
        store.publish_run(a)
        store.publish_run(b)
        assert len(store.models()) == 1
        assert len(store.runs("qwen-4b")) == 2
        assert a.comparison_key != b.comparison_key
        with pytest.raises(ValueError, match="hardware"):
            PairedComparison.from_runs(a, b, "latency", "complete")
        assert "2 condition series" in model_details(store, a.context.model).plain


def test_remote_bundle_is_idempotent_atomic_and_hardware_qualified(tmp_path):
    source = tmp_path / "source"
    bundle = tmp_path / "bundle.json"
    with ObservationStore(source) as store:
        store.publish(measurement("m", "body", 100))
        evidence = run().model_copy(update={"measurements": ("m",)})
        store.publish_run(evidence)
        export_bundle(store, bundle)
    with ObservationStore(tmp_path / "destination") as destination:
        assert import_bundle(destination, bundle) == 1
        assert import_bundle(destination, bundle) == 0
        assert destination.runs() == (evidence,)
        destination.publish_run(run("conflict"))
        with ObservationStore(source) as store:
            store.publish_run(run("new"))
            store.publish_run(run("conflict", implementation="different"))
            export_bundle(store, bundle)
        with pytest.raises(ValueError, match="immutable"):
            import_bundle(destination, bundle)
        assert {r.identity for r in destination.runs()} == {"run", "conflict"}
        envelope = json.loads(bundle.read_text())
        envelope["sha256"] = "bad"
        bundle.write_text(json.dumps(envelope))
        with pytest.raises(ValueError, match="checksum"):
            import_bundle(destination, bundle)


@pytest.mark.asyncio
async def test_model_browser_is_read_only_and_requires_no_workload(tmp_path):
    path = tmp_path / "db"
    with ObservationStore(path) as store:
        store.publish_run(run())
    with ObservationStore(path, read_only=True) as store:
        app = ModelApp(store)
        async with app.run_test() as pilot:
            await pilot.press("down")
            assert "Qwen 4B" in str(app.query_one("#details").render())
            await pilot.press("r")
            await pilot.press("q")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            store.publish_run(run("forbidden"))


def test_interval_union_preserves_overlap_and_unknown_history():
    native = KernelObservation(
        "one-clock",
        (
            KernelActivity("same-name", 10, 5, 15, 0),
            KernelActivity("same-name", 15, 10, 25, 1),
            KernelActivity("third", 5, 30, 35, 2),
        ),
    )
    assert native.elapsed_ns == 30
    assert native.busy_ns == 25
    assert native.overlap_ns == 5
    assert KernelObservation("old", (KernelActivity("a", 20),)).busy_ns is None
    with pytest.raises(ValueError, match="once"):
        KernelObservation(
            "clock", (KernelActivity("a", 1, 0, 1, 0), KernelActivity("b", 1, 1, 2, 0))
        )


def test_pairs_require_shared_identity_but_history_does_not():
    a = run()
    b = run("b", implementation="new").model_copy(
        update={
            "metrics": (
                ObservedMetric(
                    name="latency",
                    unit=units.second,
                    samples=(1.0, 2.0),
                    boundary="complete",
                    basis="wall time",
                ),
            )
        }
    )
    assert PairedComparison.from_runs(a, b, "latency", "complete").deltas == (1.0, 1.0)
    later = b.model_copy(update={"protocol": {"pair_ids": ["c", "d"]}})
    assert later.comparison_key == a.comparison_key
    with pytest.raises(ValueError, match="pair identities"):
        PairedComparison.from_runs(a, later, "latency", "complete")
    other_host = b.model_copy(
        update={"context": b.context.model_copy(update={"host": "elsewhere"})}
    )
    with pytest.raises(ValueError, match="hardware"):
        PairedComparison.from_runs(a, other_host, "latency", "complete")


def test_import_order_does_not_rewrite_formula_chronology(tmp_path):
    from datetime import timedelta

    now = datetime.now(UTC)
    new = measurement("new", "new", 100).model_copy(update={"created": now})
    old = measurement("old", "old", 200).model_copy(update={"created": now - timedelta(days=1)})
    with ObservationStore(tmp_path / "db") as store:
        store.publish(new)
        store.publish(old)
        assert store.history(new.series).latest == new
        assert store.history(new.series).latest_success == new


def test_exploratory_failed_numerics_retain_time_without_becoming_best(tmp_path):
    from ops.lab.records import Outcome

    m = measurement("wrong", "fast", 10).model_copy(
        update={
            "outcome": Outcome.FAILED,
            "checked": False,
            "error": "NumericalMismatch: wrong result",
        }
    )
    evidence = run().model_copy(update={"correctness": "failed", "measurements": (m.identity,)})
    with ObservationStore(tmp_path / "db") as store:
        store.publish(m)
        store.publish_run(evidence)
        assert m.observed_seconds == 1e-8
        assert store.history(m.series).best is None
        assert store.runs()[0].status == "complete"
        assert store.runs()[0].correctness == "failed"
