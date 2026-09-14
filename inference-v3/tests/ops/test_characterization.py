"""Device rates must remain linked to checked, compatible measured evidence."""

from datetime import UTC, datetime

import pytest

from ops.formula import units
from ops.lab.characterization import Characterization, ProbeProtocol, Rate, Resource
from ops.lab.records import MeasurementProtocol, UsefulQuantity
from ops.lab.store import ObservationStore
from ops.tensor.types import DType
from tests.ops.test_observation_store import measurement


def test_resource_conditioning_is_explicit_without_changing_ordinary_series_keys():
    ordinary = MeasurementProtocol()
    assert "minimum_warmup_seconds" not in ordinary.model_dump(mode="json")
    conditioned = ProbeProtocol().measurement
    assert conditioned.minimum_warmup_seconds == 1
    assert MeasurementProtocol.model_validate_json(conditioned.model_dump_json()) == conditioned
    assert conditioned != ordinary


def test_same_device_with_changed_runtime_is_a_different_evidence_scope():
    from ops.runtime.resources import DeviceRuntime
    from tests.ops.test_compiler import Runtime

    first, second = Runtime(), Runtime()
    second.runtime_identity = "updated-driver"
    with DeviceRuntime(first, budget_bytes=1024) as before, DeviceRuntime(second, budget_bytes=1024) as after:
        assert before.capabilities == after.capabilities
        assert before.compiler_identity == after.compiler_identity
        assert before.evidence_identity != after.evidence_identity


def test_characterized_rate_must_match_its_published_measurement(tmp_path):
    evidence = measurement("copy-probe", "copy-body", 200).model_copy(update={
        "quantities": (UsefulQuantity(name="copy-bytes", amount=32, unit=units.byte,
                                      basis="complete copy source and destination extent"),),
    })
    metric = next(item for item in evidence.metrics if item.name == "rate:copy-bytes")
    rate = Rate(resource=Resource.EXECUTION_COPY, value=metric.value, unit=metric.unit,
                measurement=evidence.identity, metric=metric.name, working_set_bytes=32,
                dtype=DType.U8, conditions=("resident", "complete-operation wall timing"))
    profile = Characterization(identity="profile", key="device-probes-v1", created=datetime.now(UTC),
                               device=evidence.series.device, compiler=evidence.implementation.compiler,
                               capabilities="test", protocol=ProbeProtocol(measurement=evidence.series.protocol), rates=(rate,))
    with ObservationStore(tmp_path / "evidence.sqlite") as store:
        with pytest.raises(ValueError, match="missing measured evidence"):
            store.publish_characterization(profile)
        store.publish(evidence)
        store.publish_characterization(profile)
        assert store.characterization(profile.key) == profile
        assert store.characterization("other-device") is None
        invented = profile.model_copy(update={"identity": "invented", "rates": (
            rate.model_copy(update={"value": rate.value * 2}),
        )})
        with pytest.raises(ValueError, match="does not match"):
            store.publish_characterization(invented)
        assert store.characterization(profile.key) == profile
