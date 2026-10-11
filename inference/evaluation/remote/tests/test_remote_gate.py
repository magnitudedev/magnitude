"""Thermal gate stability logic over synthetic readings."""

import json

import pytest

from magnitude_remote.host.gate import GateSettings, GateTimeout, Sample, judge, wait
from magnitude_remote.host.probes import Reading

SETTINGS = GateSettings(window_seconds=60, interval_seconds=2, timeout_seconds=1)


def reading(cpu=50.0, gpu=45.0, cpu_busy=2.0, gpu_busy=0.0, pressure="nominal", errors=None):
    sensors, groups = {}, {}
    if cpu is not None:
        sensors.update({"Tp01": cpu, "Tp02": cpu + 1})
        groups.update({"Tp01": "cpu", "Tp02": "cpu"})
    if gpu is not None:
        sensors["Tg01"], groups["Tg01"] = gpu, "gpu"
    utilization = {"cpu": cpu_busy}
    if gpu_busy is not None:
        utilization["gpu0"] = gpu_busy
    return Reading(sensors, groups, utilization, pressure, errors or {})


def series(make, seconds=70, step=2):
    return [Sample(float(t), make(t)) for t in range(0, seconds + 1, step)]


def test_flat_idle_passes():
    verdict = judge(series(lambda t: reading(cpu=50 + 0.2 * (t % 4 == 0))), SETTINGS)
    assert verdict.passed, verdict.reasons
    assert set(verdict.channels) == {"cpu_mean_c", "gpu_mean_c", "cpu_pct", "gpu0_pct"}


def test_cooling_down_is_not_rising_but_must_settle():
    cooling = judge(series(lambda t: reading(cpu=60 - 0.01 * t)), SETTINGS)
    assert cooling.passed, cooling.reasons
    fast = judge(series(lambda t: reading(cpu=70 - 0.1 * t)), SETTINGS)
    assert not fast.passed and any("spread" in r for r in fast.reasons)


def test_upward_trend_fails_even_with_small_spread():
    verdict = judge(series(lambda t: reading(gpu=45 + 0.012 * t)), SETTINGS)
    assert len(verdict.reasons) == 1 and "gpu rising" in verdict.reasons[0]


def test_busy_cpu_or_gpu_fails():
    assert "cpu busy" in judge(series(lambda t: reading(cpu_busy=40)), SETTINGS).reasons[0]
    assert "gpu0 busy" in judge(series(lambda t: reading(gpu_busy=30)), SETTINGS).reasons[0]


def test_window_must_be_covered():
    verdict = judge(series(lambda t: reading(), seconds=30), SETTINGS)
    assert not verdict.passed and "window not yet covered" in verdict.reasons[0]


def test_missing_readings_never_pass():
    gap = judge(series(lambda t: reading(gpu=None if t == 40 else 45.0)), SETTINGS)
    assert not gap.passed
    none = judge(series(lambda t: reading(cpu=None, gpu=None, errors={"smc": "unavailable"})), SETTINGS)
    assert any("no temperature sensors" in r for r in none.reasons)
    failed = judge(
        series(lambda t: reading(gpu_busy=None, errors={"utilization:gpu": "ioreg failed"})), SETTINGS
    )
    assert "utilization:gpu: ioreg failed" in failed.reasons


def test_a_sensor_dropping_out_does_not_shift_the_group_mean():
    def make(t):
        sample = reading(cpu=50.0)
        if t == 40:  # one glitched sweep loses the hotter sensor
            del sample.sensors_c["Tp02"]
        return sample

    verdict = judge(series(make), SETTINGS)
    assert verdict.passed, verdict.reasons
    assert verdict.channels["cpu_mean_c"]["latest"] == 50.0


def test_thermal_pressure_must_stay_nominal():
    verdict = judge(series(lambda t: reading(pressure="moderate" if t == 50 else "nominal")), SETTINGS)
    assert verdict.reasons == ["thermal pressure ['moderate', 'nominal']"]


class FixedProbe:
    source = grouping = "test"

    def __init__(self, make):
        self.make = make

    def read(self):
        return self.make()

    def close(self):
        pass


def test_wait_passes_and_logs_every_sample(tmp_path):
    settings = GateSettings(window_seconds=0.2, interval_seconds=0.05, timeout_seconds=5)
    verdict = wait(FixedProbe(reading), settings, tmp_path / "gate.jsonl")
    assert verdict.passed
    lines = [json.loads(line) for line in (tmp_path / "gate.jsonl").read_text().splitlines()]
    assert len(lines) >= 4 and lines[-1]["verdict"]["passed"] is True
    assert all(line["phase"] == "gate" and line["sensors_c"]["Tp01"] == 50.0 for line in lines)


def test_wait_times_out_with_readings(tmp_path):
    settings = GateSettings(window_seconds=0.2, interval_seconds=0.05, timeout_seconds=0.5)
    with pytest.raises(GateTimeout) as raised:
        wait(FixedProbe(lambda: reading(cpu_busy=90)), settings, tmp_path / "gate.jsonl")
    assert raised.value.latest.utilization_pct["cpu"] == 90
    assert "cpu busy" in str(raised.value)
