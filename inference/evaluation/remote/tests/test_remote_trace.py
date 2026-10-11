"""Thermal trace recording and hardware facts (ported from session-bench's thermal checks)."""

import json

from magnitude_remote.host import facts
from magnitude_remote.host.probes import LinuxProbe, Reading, linux_group
from magnitude_remote.host.trace import SUMMARY, TRACE, ThermalRecorder


class Probe:
    source = "test-SMC"
    grouping = "test grouping"

    def read(self):
        return Reading(
            {"Tp01": 65.0, "Tg01": 45.0}, {"Tp01": "cpu", "Tg01": "gpu"}, {"cpu": 1.0}, "nominal"
        )

    def close(self):
        pass


def test_recorder_writes_trace_and_time_weighted_summary(tmp_path):
    with ThermalRecorder(tmp_path, interval_seconds=0.02, probe_factory=Probe) as recorder:
        pass
    summary = json.loads((tmp_path / SUMMARY).read_text())
    assert summary == recorder.summary
    assert summary["status"] == "available"
    assert summary["source"] == "test-SMC"
    assert summary["channels"]["cpu_mean"]["start_c"] == 65.0
    assert summary["channels"]["gpu_mean"]["end_c"] == 45.0
    assert summary["channels"]["other_mean"]["valid_samples"] == 0
    lines = [json.loads(line) for line in (tmp_path / TRACE).read_text().splitlines()]
    assert lines[0]["phase"] == "start" and lines[-1]["phase"] == "end"
    assert lines[0]["sensors_c"] == {"Tp01": 65.0, "Tg01": 45.0}
    assert lines[0]["utilization_pct"] == {"cpu": 1.0}
    assert lines[0]["at"].endswith("+00:00")


def test_unavailable_probe_is_an_explicit_error_not_zero(tmp_path):
    def broken():
        raise OSError("no SMC here")

    with ThermalRecorder(tmp_path, probe_factory=broken) as recorder:
        pass
    assert recorder.summary["status"] == "unavailable"
    assert recorder.summary["errors"] == {"probe: OSError: no SMC here": 2}
    line = json.loads((tmp_path / TRACE).read_text().splitlines()[0])
    assert line["sensors_c"] == {} and line["groups_c"]["cpu_mean"] is None


def test_linux_probe_reads_zones_hwmon_and_skips_duplicates(tmp_path, monkeypatch):
    zone = tmp_path / "thermal" / "thermal_zone0"
    zone.mkdir(parents=True)
    (zone / "type").write_text("x86_pkg_temp\n")
    (zone / "temp").write_text("51000\n")
    for name, chip, label, value in (
        ("x86_pkg_temp", "hwmon0", None, "51000"),
        ("nvme", "hwmon1", "Composite", "38500"),
        ("amdgpu", "hwmon2", "edge", "44000"),
    ):
        path = tmp_path / "hwmon" / chip
        path.mkdir(parents=True)
        (path / "name").write_text(name)
        (path / "temp1_input").write_text(value)
        if label:
            (path / "temp1_label").write_text(label)
    monkeypatch.setattr("magnitude_remote.host.probes.nvidia_smi", lambda: [("0", 40.0, 3.0)])
    ticks = iter([(100, 1000), (130, 1100)])
    monkeypatch.setattr("magnitude_remote.host.probes._proc_stat_ticks", lambda: next(ticks))
    reading = LinuxProbe(tmp_path).read()
    assert reading.sensors_c == {
        "zone:thermal_zone0:x86_pkg_temp": 51.0,
        "hwmon:hwmon1:nvme:Composite": 38.5,
        "hwmon:hwmon2:amdgpu:edge": 44.0,
        "nvidia:0": 40.0,
    }
    assert reading.sensor_groups == {
        "zone:thermal_zone0:x86_pkg_temp": "cpu",
        "hwmon:hwmon1:nvme:Composite": "other",
        "hwmon:hwmon2:amdgpu:edge": "gpu",
        "nvidia:0": "gpu",
    }
    assert reading.utilization_pct["gpu0"] == 3.0
    assert reading.utilization_pct["cpu"] == 30.0
    assert reading.errors == {}


def test_linux_grouping():
    assert linux_group("coretemp Package id 0") == "cpu"
    assert linux_group("acpitz") == "other"


def test_local_hardware_facts_are_typed_with_explicit_errors():
    record = facts.capture("alias-x").to_json()
    assert record["alias"] == "alias-x"
    assert record["hostname"]
    assert record["memory_bytes"] is None or record["memory_bytes"] > 0
    assert record["cpu_cores_logical"] > 0
    assert isinstance(record["errors"], list)
    for key in ("chip", "memory_bytes", "cpu_cores_physical", "gpus", "model"):
        assert record[key] is not None or any(e.startswith(key) for e in record["errors"])
