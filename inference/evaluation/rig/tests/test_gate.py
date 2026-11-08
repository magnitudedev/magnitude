"""Thermal gate stability logic over synthetic readings."""

import json
from dataclasses import replace

import pytest

from magnitude_rig.host.gate import GateSettings, GateTimeout, Sample, judge, wait
from magnitude_rig.host.probes import Reading

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
    return Reading(sensors, groups, utilization, {}, pressure, errors or {})


def series(make, seconds=70, step=2):
    return [Sample(float(t), make(t)) for t in range(0, seconds + 1, step)]


def test_flat_idle_passes():
    verdict = judge(series(lambda t: reading(cpu=50 + 0.2 * (t % 4 == 0))), SETTINGS, cpu_own=True)
    assert verdict.passed, verdict.reasons
    assert set(verdict.channels) == {"cpu_own", "cpu_mean_c", "gpu_mean_c", "cpu_pct", "gpu0_pct"}


def test_on_a_shared_machine_only_the_runs_own_gpu_is_judged():
    """Other tenants move the machine's CPU temperatures and load: on a host whose CPUs are not
    all the run's, they are not judged; its GPU still is, and a host with nothing of the run's
    own to judge never passes."""
    tenants = lambda t: reading(cpu=50 + (3.0 if t % 8 == 0 else 0.0), cpu_busy=60)
    assert not judge(series(tenants), SETTINGS, cpu_own=True).passed
    shared = judge(series(tenants), SETTINGS, cpu_own=False)
    assert shared.passed, shared.reasons
    assert set(shared.channels) == {"cpu_own", "gpu_mean_c", "gpu0_pct"}
    hot = judge(series(lambda t: reading(cpu=60, gpu=45 + 0.012 * t)), SETTINGS, cpu_own=False)
    assert hot.reasons and "gpu rising" in hot.reasons[0]
    gpuless = judge(series(lambda t: reading(gpu=None, gpu_busy=None)), SETTINGS, cpu_own=False)
    assert not gpuless.passed and "no temperature sensors of the run's own" in gpuless.reasons[0]


def test_a_containers_cpus_are_its_cpuset_bounded_by_every_quota_above_it(tmp_path):
    from magnitude_rig.host import facts

    mount = tmp_path / "cgroup"
    (mount / "tenant" / "job").mkdir(parents=True)
    (mount / "tenant" / "cpu.max").write_text("400000 100000\n")  # 4 CPUs for the tenant
    (mount / "tenant" / "job" / "cpu.max").write_text("max 100000\n")
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/tenant/job\n")
    assert facts._cgroup_quotas(membership, mount) == [None, 4.0, None]
    # A container without its own namespace names the host's path; its mount's root is its own.
    (mount / "cpu.max").write_text("800000 100000\n")
    membership.write_text("0::/docker/abc\n")
    assert facts._cgroup_quotas(membership, mount) == [None, None, 8.0]
    v1 = tmp_path / "v1"
    (v1 / "cpu,cpuacct").mkdir(parents=True)
    (v1 / "cpu,cpuacct" / "cpu.cfs_quota_us").write_text("-1\n")
    membership.write_text("4:cpu,cpuacct:/\n2:memory:/\n")
    assert facts._cgroup_quotas(membership, v1) == [None]


def test_cooling_down_is_not_rising_but_must_settle():
    cooling = judge(series(lambda t: reading(cpu=60 - 0.01 * t)), SETTINGS, cpu_own=True)
    assert cooling.passed, cooling.reasons
    fast = judge(series(lambda t: reading(cpu=70 - 0.1 * t)), SETTINGS, cpu_own=True)
    assert not fast.passed and any("spread" in r for r in fast.reasons)


def test_upward_trend_fails_even_with_small_spread():
    verdict = judge(series(lambda t: reading(gpu=45 + 0.012 * t)), SETTINGS, cpu_own=True)
    assert len(verdict.reasons) == 1 and "gpu rising" in verdict.reasons[0]


def test_busy_cpu_or_gpu_fails():
    assert "cpu busy" in judge(series(lambda t: reading(cpu_busy=40)), SETTINGS, cpu_own=True).reasons[0]
    assert "gpu0 busy" in judge(series(lambda t: reading(gpu_busy=30)), SETTINGS, cpu_own=True).reasons[0]


def test_window_must_be_covered():
    verdict = judge(series(lambda t: reading(), seconds=30), SETTINGS, cpu_own=True)
    assert not verdict.passed and "window not yet covered" in verdict.reasons[0]


def test_missing_readings_never_pass():
    gap = judge(series(lambda t: reading(gpu=None if t == 40 else 45.0)), SETTINGS, cpu_own=True)
    assert not gap.passed
    none = judge(series(lambda t: reading(cpu=None, gpu=None, errors={"smc": "unavailable"})), SETTINGS, cpu_own=True)
    assert any("no temperature sensors" in r for r in none.reasons)
    failed = judge(
        series(lambda t: reading(gpu_busy=None, errors={"utilization:gpu": "ioreg failed"})), SETTINGS, cpu_own=True
    )
    assert "utilization:gpu: ioreg failed" in failed.reasons


def test_a_sensor_dropping_out_does_not_shift_the_group_mean():
    def make(t):
        sample = reading(cpu=50.0)
        if t == 40:  # one glitched sweep loses the hotter sensor
            del sample.sensors_c["Tp02"]
        return sample

    verdict = judge(series(make), SETTINGS, cpu_own=True)
    assert verdict.passed, verdict.reasons
    assert verdict.channels["cpu_mean_c"]["latest"] == 50.0


def test_thermal_pressure_must_stay_nominal():
    verdict = judge(series(lambda t: reading(pressure="moderate" if t == 50 else "nominal")), SETTINGS, cpu_own=True)
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
    verdict = wait(FixedProbe(reading), settings, tmp_path / "gate.jsonl", cpu_own=True)
    assert verdict.passed
    lines = [json.loads(line) for line in (tmp_path / "gate.jsonl").read_text().splitlines()]
    assert len(lines) >= 4 and lines[-1]["verdict"]["passed"] is True
    assert all(line["phase"] == "gate" and line["sensors_c"]["Tp01"] == 50.0 for line in lines)


def test_wait_times_out_with_readings(tmp_path):
    settings = GateSettings(window_seconds=0.2, interval_seconds=0.05, timeout_seconds=0.5)
    with pytest.raises(GateTimeout) as raised:
        wait(FixedProbe(lambda: reading(cpu_busy=90)), settings, tmp_path / "gate.jsonl", cpu_own=True)
    assert raised.value.latest.utilization_pct["cpu"] == 90
    assert "cpu busy" in str(raised.value)


def test_a_host_without_sensors_is_judged_on_idleness():
    settings = replace(SETTINGS, temperatures=False)
    idle = judge(series(lambda t: reading(cpu=None, gpu=None, errors={"thermal": "none"})), settings, cpu_own=True)
    assert idle.passed, idle.reasons
    busy = judge(series(lambda t: reading(cpu=None, gpu=None, cpu_busy=40)), settings, cpu_own=True)
    assert not busy.passed and "cpu busy" in busy.reasons[0]


def test_a_host_with_nothing_of_the_runs_own_to_judge_fails():
    settings = replace(SETTINGS, temperatures=False)
    verdict = judge(series(lambda t: reading(cpu=None, gpu=None, gpu_busy=None)), settings, cpu_own=False)
    assert not verdict.passed
    assert verdict.reasons == ["nothing of the run's own to judge: no temperature or utilization reading"]
