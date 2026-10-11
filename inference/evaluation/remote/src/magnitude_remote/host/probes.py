"""Per-OS thermal and load probes: one reading per call, failures kept as explicit errors.

- macOS: AppleSMC temperatures, thermal pressure level, CPU ticks, IOAccelerator utilization.
- Linux: /sys/class/thermal zones, hwmon temperatures, nvidia-smi, /proc/stat.
- Windows: nvidia-smi and system CPU times; other temperatures are reported unsupported.

Utilization is measured since the previous reading (the first since the probe opened).
Sensor groups describe CPU/GPU associations, not documented core identities.
"""

from __future__ import annotations

import ctypes as c
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

GROUPS = ("cpu", "gpu", "other")


@dataclass(frozen=True)
class Reading:
    sensors_c: Dict[str, float]
    sensor_groups: Dict[str, str]
    utilization_pct: Dict[str, float]
    thermal_pressure: Optional[str]
    errors: Dict[str, str] = field(default_factory=dict)

    def group_values(self, group: str) -> List[float]:
        return [v for k, v in self.sensors_c.items() if self.sensor_groups[k] == group]

    def group_sensors(self, group: str) -> List[str]:
        return [k for k in self.sensors_c if self.sensor_groups[k] == group]

    def to_json(self) -> dict:
        return {
            "sensors_c": self.sensors_c,
            "sensor_groups": self.sensor_groups,
            "utilization_pct": self.utilization_pct,
            "thermal_pressure": self.thermal_pressure,
            "errors": self.errors,
        }


class Probe(Protocol):
    source: str
    grouping: str

    def read(self) -> Reading: ...

    def close(self) -> None: ...


def _failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def nvidia_smi() -> Optional[List[Tuple[str, Optional[float], Optional[float]]]]:
    """(index, temperature, utilization) per GPU, or None when nvidia-smi is not installed."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,temperature.gpu,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        raise OSError(f"nvidia-smi exited {result.returncode}: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.strip().splitlines():
        index, temperature, utilization = (part.strip() for part in line.split(","))
        rows.append((index, number(temperature), number(utilization)))
    return rows


def number(text: str) -> Optional[float]:
    """nvidia-smi's numeric field, or None for [N/A] and similar."""
    try:
        return float(text)
    except ValueError:
        return None


def _add_nvidia(sensors, groups, utilization, errors) -> None:
    try:
        rows = nvidia_smi()
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        errors["nvidia-smi"] = _failure(error)
        return
    for index, temperature, busy in rows or ():
        key = f"nvidia:{index}"
        if temperature is None:
            errors[key] = "temperature not reported"
        else:
            sensors[key], groups[key] = temperature, "gpu"
        if busy is None:
            errors[f"utilization:gpu{index}"] = "utilization not reported"
        else:
            utilization[f"gpu{index}"] = busy


class _Ticks:
    """Busy fraction between successive (busy, total) tick samples."""

    def __init__(self, sample: Callable[[], Tuple[int, int]]):
        self._sample = sample
        self._previous = sample()

    def percent(self) -> float:
        busy, total = self._sample()
        previous_busy, previous_total = self._previous
        self._previous = (busy, total)
        if total <= previous_total:
            raise OSError("CPU tick counters did not advance")
        return 100.0 * (busy - previous_busy) / (total - previous_total)


class MacProbe:
    source = "macOS: AppleSMC, thermal pressure, host CPU ticks, IOAccelerator"
    grouping = "SMC prefixes: CPU-associated Tp/Te/Ts; GPU-associated Tg"
    _PRESSURE = {0: "nominal", 1: "moderate", 2: "heavy", 3: "trapping", 4: "sleeping"}

    def __init__(self):
        from .smc import AppleSMC, sensor_group

        self._group = sensor_group
        self._system = c.CDLL("/usr/lib/libSystem.B.dylib")
        self._system.mach_host_self.restype = c.c_uint32
        self._system.host_statistics.argtypes = [
            c.c_uint32, c.c_int, c.POINTER(c.c_uint32 * 4), c.POINTER(c.c_uint32)
        ]
        self._host = self._system.mach_host_self()
        self._pressure_token = c.c_int()
        code = self._system.notify_register_check(
            b"com.apple.system.thermalpressurelevel", c.byref(self._pressure_token)
        )
        if code:
            raise OSError(f"notify_register_check(thermalpressurelevel) failed: {code}")
        self._cpu = _Ticks(self._cpu_ticks)
        self._smc = None
        self._smc_error = None
        try:
            self._smc = AppleSMC()
        except OSError as error:
            self._smc_error = _failure(error)

    def _cpu_ticks(self) -> Tuple[int, int]:
        ticks, count = (c.c_uint32 * 4)(), c.c_uint32(4)
        code = self._system.host_statistics(self._host, 3, c.byref(ticks), c.byref(count))
        if code:
            raise OSError(f"host_statistics(HOST_CPU_LOAD_INFO) failed: {code}")
        user, system, idle, nice = ticks
        return user + system + nice, user + system + idle + nice

    def read(self) -> Reading:
        sensors, groups, utilization, errors = {}, {}, {}, {}
        if self._smc is None:
            errors["smc"] = self._smc_error
        else:
            reading = self._smc.read()
            sensors.update(reading["sensors_c"])
            groups.update({key: self._group(key) for key in sensors})
            errors.update(reading["errors"])
        pressure = None
        state = c.c_uint64()
        code = self._system.notify_get_state(self._pressure_token, c.byref(state))
        if code:
            errors["thermal_pressure"] = f"notify_get_state failed: {code}"
        else:
            pressure = self._PRESSURE.get(state.value, f"level-{state.value}")
        try:
            utilization["cpu"] = self._cpu.percent()
        except OSError as error:
            errors["utilization:cpu"] = _failure(error)
        try:
            utilization.update(self._gpu_utilization())
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            errors["utilization:gpu"] = _failure(error)
        return Reading(sensors, groups, utilization, pressure, errors)

    @staticmethod
    def _gpu_utilization() -> Dict[str, float]:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise OSError(f"ioreg exited {result.returncode}: {result.stderr.strip()}")
        values = re.findall(r'"Device Utilization %"=(\d+)', result.stdout)
        if not values:
            raise ValueError("IOAccelerator reports no Device Utilization %")
        return {f"gpu{index}": float(value) for index, value in enumerate(values)}

    def close(self) -> None:
        if self._smc is not None:
            self._smc.close()
        self._system.notify_cancel(self._pressure_token)


def linux_group(name: str) -> str:
    lowered = name.lower()
    if any(
        word in lowered
        for word in ("cpu", "coretemp", "k10temp", "zenpower", "x86_pkg", "soc", "package")
    ):
        return "cpu"
    if any(word in lowered for word in ("gpu", "amdgpu", "nouveau", "radeon")):
        return "gpu"
    return "other"


def _proc_stat_ticks() -> Tuple[int, int]:
    fields = Path("/proc/stat").read_text().splitlines()[0].split()
    if fields[0] != "cpu":
        raise OSError("/proc/stat does not start with the aggregate cpu line")
    values = [int(v) for v in fields[1:9]]
    idle = values[3] + values[4]
    return sum(values) - idle, sum(values)


def _millidegrees(path: Path) -> float:
    return int(path.read_text().strip()) / 1000.0


class LinuxProbe:
    source = "Linux: /sys/class/thermal, hwmon, nvidia-smi, /proc/stat"
    grouping = (
        "Names containing cpu/coretemp/k10temp/zenpower/x86_pkg/soc/package are CPU; "
        "gpu/amdgpu/nouveau/radeon and nvidia-smi are GPU; everything else is other. "
        "hwmon chips duplicating a thermal zone type are skipped."
    )

    def __init__(self, sysfs: Path = Path("/sys/class")):
        self._sysfs = sysfs
        self._cpu = _Ticks(_proc_stat_ticks)

    def read(self) -> Reading:
        sensors, groups, utilization, errors = {}, {}, {}, {}
        zone_types = set()
        for zone in sorted((self._sysfs / "thermal").glob("thermal_zone*")):
            try:
                kind = (zone / "type").read_text().strip()
                zone_types.add(kind)
                key = f"zone:{zone.name}:{kind}"
                sensors[key], groups[key] = _millidegrees(zone / "temp"), linux_group(kind)
            except (OSError, ValueError) as error:
                errors[f"zone:{zone.name}"] = _failure(error)
        for chip in sorted((self._sysfs / "hwmon").glob("hwmon*")):
            try:
                name = (chip / "name").read_text().strip()
            except OSError as error:
                errors[f"hwmon:{chip.name}"] = _failure(error)
                continue
            if name in zone_types:
                continue
            for sensor in sorted(chip.glob("temp*_input")):
                channel = sensor.name[: -len("_input")]
                label_path = chip / f"{channel}_label"
                try:
                    label = label_path.read_text().strip() if label_path.exists() else channel
                    key = f"hwmon:{chip.name}:{name}:{label}"
                    sensors[key] = _millidegrees(sensor)
                    groups[key] = linux_group(f"{name} {label}")
                except (OSError, ValueError) as error:
                    errors[f"hwmon:{chip.name}:{channel}"] = _failure(error)
        _add_nvidia(sensors, groups, utilization, errors)
        try:
            utilization["cpu"] = self._cpu.percent()
        except (OSError, ValueError) as error:
            errors["utilization:cpu"] = _failure(error)
        return Reading(sensors, groups, utilization, None, errors)

    def close(self) -> None:
        pass


class _FileTime(c.Structure):
    _fields_ = [("low", c.c_uint32), ("high", c.c_uint32)]

    @property
    def value(self) -> int:
        return (self.high << 32) | self.low


def _windows_ticks() -> Tuple[int, int]:
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    if not c.windll.kernel32.GetSystemTimes(c.byref(idle), c.byref(kernel), c.byref(user)):
        raise OSError("GetSystemTimes failed")
    total = kernel.value + user.value  # kernel time includes idle time
    return total - idle.value, total


class WindowsProbe:
    source = "Windows: nvidia-smi, GetSystemTimes"
    grouping = "nvidia-smi GPUs are GPU; other temperatures are unsupported on Windows"

    def __init__(self):
        self._cpu = _Ticks(_windows_ticks)

    def read(self) -> Reading:
        sensors, groups, utilization = {}, {}, {}
        errors = {"temperature:cpu": "CPU temperatures are unsupported on Windows"}
        _add_nvidia(sensors, groups, utilization, errors)
        try:
            utilization["cpu"] = self._cpu.percent()
        except OSError as error:
            errors["utilization:cpu"] = _failure(error)
        return Reading(sensors, groups, utilization, None, errors)

    def close(self) -> None:
        pass


def open_probe() -> Probe:
    if sys.platform == "darwin":
        return MacProbe()
    if sys.platform.startswith("linux"):
        return LinuxProbe()
    if sys.platform == "win32":
        return WindowsProbe()
    raise OSError(f"no thermal probe for platform {sys.platform}")
