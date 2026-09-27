"""Selection persistence is qualified by identity and reference validation."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ops.compiler.selection import SelectionIdentity, calibrate, load_selection


@pytest.fixture
def identity():
    return SelectionIdentity(
        *(
            "semantic",
            "family",
            "construction",
            "workload",
            "numerical",
            "validation",
            "compiler",
            "target",
            "physical",
        )
    )


def test_requires_complete_identity(identity):
    with pytest.raises(ValueError):
        replace(identity, physical_device="")


def test_qualified_roundtrip_and_every_identity_component_invalidates(
    tmp_path, monkeypatch, identity
):
    monkeypatch.setattr("tilelang.cache.compiler_identity", lambda: "compiler")
    tuner = SimpleNamespace(
        profile_args=SimpleNamespace(skip_check=False, ref_prog=lambda: None, backend="wall"),
        compile_args=SimpleNamespace(target="target"),
        configs=[{"candidate": 2}],
        run=lambda **kw: SimpleNamespace(config={"candidate": 2}, latency=0.1),
    )
    expected = calibrate(tuner, tmp_path, identity)
    assert load_selection(tmp_path, identity) == expected
    for field in identity.__dataclass_fields__:
        assert load_selection(tmp_path, replace(identity, **{field: "different"})) is None
    tuner.profile_args.skip_check = True
    with pytest.raises(ValueError, match="reference"):
        calibrate(tuner, tmp_path, identity)
    tuner.profile_args.skip_check = False
    tuner.profile_args.ref_prog = None
    with pytest.raises(ValueError, match="reference"):
        calibrate(tuner, tmp_path, identity)


@pytest.mark.parametrize("contents", ["{", "null", "[]", '{"version":2}', '{"version":1}'])
def test_corrupt_records_are_misses(tmp_path, identity, contents):
    (tmp_path / (identity.key + ".json")).write_text(contents)
    assert load_selection(tmp_path, identity) is None


def test_identity_contents_checked_even_at_matching_filename(tmp_path, identity):
    (tmp_path / (identity.key + ".json")).write_text(
        json.dumps(
            dict(version=1, identity={}, config={"candidate": 0}, latency_ms=1, timing_basis="wall")
        )
    )
    assert load_selection(tmp_path, identity) is None


def test_runtime_reloads_magnitude_record_without_tuning(tmp_path, monkeypatch):
    from ops.compiler.schedules import QualifiedSchedules, ScheduleRequest

    monkeypatch.setattr("tilelang.cache.compiler_identity", lambda: "compiler")
    request = ScheduleRequest("projection", "family", "construction", "shape", "model", (8, 16))
    profile = QualifiedSchedules(str(tmp_path), "compiler", "target", "device", "reference")
    tuner = SimpleNamespace(
        profile_args=SimpleNamespace(skip_check=False, ref_prog=lambda: None, backend="wall"),
        compile_args=SimpleNamespace(target="target"),
        configs=[{"candidate": 1}],
        run=lambda **kw: SimpleNamespace(config={"candidate": 1}, latency=0.1),
    )
    calibrate(tuner, tmp_path, profile.identity(request))

    def unexpected(**kwargs):
        pytest.fail("loading a selected schedule attempted calibration")

    tuner.run = unexpected
    assert profile.select(request, 8) == 16
    with pytest.raises(ValueError, match="calibration required"):
        profile.select(replace(request, workload="other-shape"), 8)
