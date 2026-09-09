import hashlib
import json
import threading

import pytest

from performance.temperature import AppleSMC, _Data, sensor_group
from performance.thermals import TemperatureSeries, ThermalRecorder


class Probe:
    source = "test-sensors"

    def __init__(self):
        self.closed = False
        self.sampled = threading.Event()

    def read(self):
        self.sampled.set()
        return {"sensors_c": {"Tp01": 60.0, "Tp02": 80.0, "Tg01": 50.0}, "errors": {}}

    def close(self):
        self.closed = True


def test_weighted_mean_excludes_missing_intervals_and_keeps_endpoints():
    series = TemperatureSeries()
    for seconds, value in [(0, None), (1, 40), (3, 60), (4, None), (6, 90), (9, 90), (10, None)]:
        series.add(seconds, value)
    assert series.summary() == {
        "start_c": None,
        "end_c": None,
        "mean_c": 74.0,
        "min_c": 40,
        "max_c": 90,
        "valid_samples": 4,
        "covered_seconds": 5.0,
    }


@pytest.mark.parametrize("exception", [None, ValueError, KeyboardInterrupt])
def test_trace_and_summary_survive_exit(tmp_path, exception):
    probe = Probe()
    recorder = ThermalRecorder(tmp_path, interval_seconds=0.01, probe_factory=lambda: probe)
    try:
        with recorder:
            probe.sampled.clear()
            assert probe.sampled.wait(2), "background sampler did not run"
            if exception:
                raise exception("interrupted workload")
    except (ValueError, KeyboardInterrupt):
        assert exception is not None
    assert probe.closed
    assert recorder._thread is not None and not recorder._thread.is_alive()
    trace = (tmp_path / "thermals.jsonl").read_bytes()
    rows = [json.loads(line) for line in trace.splitlines()]
    assert rows[0]["phase"] == "start" and rows[-1]["phase"] == "end"
    assert any(row["phase"] == "periodic" for row in rows)
    assert all(row["probe_duration_ms"] >= 0 for row in rows)
    assert rows[0]["groups_c"] == {
        "cpu_mean": 70.0,
        "cpu_max": 80.0,
        "gpu_mean": 50.0,
        "gpu_max": 50.0,
    }
    summary = json.loads((tmp_path / "thermals.json").read_text())
    assert summary == recorder.summary
    assert summary["status"] == "available"
    assert summary["channels"]["cpu_mean"]["mean_c"] == pytest.approx(70.0)
    assert summary["channels"]["cpu_max"]["max_c"] == 80.0
    assert summary["trace_sha256"] == hashlib.sha256(trace).hexdigest()


def test_probe_unavailable_is_evidence_not_benchmark_failure(tmp_path):
    def missing():
        raise OSError("sensor denied")

    with ThermalRecorder(tmp_path, probe_factory=missing) as recorder:
        pass
    assert recorder.summary["status"] == "unavailable"
    assert recorder.summary["sample_count"] == 2
    assert recorder.summary["errors"] == {"probe: OSError: sensor denied": 2}
    assert all(c["mean_c"] is None for c in recorder.summary["channels"].values())


def test_partial_sensor_failure_does_not_invent_zero_or_bridge_gaps(tmp_path):
    class Intermittent(Probe):
        def read(self):
            return {"sensors_c": {"Tp01": 60.0}, "errors": {"Tg01": "read failed"}}

    with ThermalRecorder(tmp_path, probe_factory=Intermittent) as recorder:
        pass
    assert recorder.summary["status"] == "partial"
    assert recorder.summary["channels"]["gpu_mean"]["start_c"] is None
    assert recorder.summary["channels"]["gpu_mean"]["valid_samples"] == 0
    assert recorder.summary["channels"]["cpu_mean"]["mean_c"] == pytest.approx(60)


def test_smc_rejects_invalid_temperatures_and_keeps_other_sensors():
    import struct

    probe = AppleSMC.__new__(AppleSMC)
    probe._keys = dict.fromkeys(["Tp01", "Tp02", "Tg01", "Tg02"])
    values = dict(zip(probe._keys, [60.0, float("nan"), 0.0, 151.0], strict=True))

    def read(key, **kwargs):
        result = _Data()
        result.data[:4] = struct.pack("<f", values[key])
        return result

    probe._read = read
    result = probe.read()
    assert result["sensors_c"] == {"Tp01": 60.0}
    assert set(result["errors"]) == {"Tp02", "Tg01", "Tg02"}
    assert sensor_group("Tp01") == "cpu"
    assert sensor_group("Tg01") == "gpu"
    assert sensor_group("F0Ac") is None
