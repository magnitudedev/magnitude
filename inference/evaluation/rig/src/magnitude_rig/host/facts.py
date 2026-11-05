"""The typed hardware record of the executing host, captured before timed work.

Unavailable values are null and every failed probe keeps its error. Hostname identifies
the executing machine; the ssh alias, when there is one, is recorded beside it.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from .probes import number, windows_memory_status

T = TypeVar("T")


@dataclass(frozen=True)
class Gpu:
    name: str
    cores: Optional[int]
    memory_bytes: Optional[int]


@dataclass(frozen=True)
class HostFacts:
    hostname: str
    alias: Optional[str]
    os: str
    os_version: str
    architecture: str
    model: Optional[str]
    chip: Optional[str]
    cpu_cores_physical: Optional[int]
    cpu_cores_logical: Optional[int]
    # The logical CPUs this host's runs may use (`cpus_allowed`), of `cpu_cores_logical`.
    cpus_allowed: Optional[float]
    memory_bytes: Optional[int]
    gpus: Optional[List[Gpu]]
    errors: List[str]

    def to_json(self) -> dict:
        return asdict(self)


def cpus_allowed() -> float:
    """The logical CPUs this process may run on: on Linux its affinity (the container's cpuset),
    bounded by its cgroup's CPU quota on the way to the root; elsewhere the machine's. Fewer than
    the machine's (`os.cpu_count()`, every logical CPU, a container's too) means the machine's
    CPUs, and so its CPU temperatures and load, are shared with work that is not this host's."""
    if not sys.platform.startswith("linux"):
        return float(os.cpu_count())
    allowed = float(len(os.sched_getaffinity(0)))
    quotas = [quota for quota in _cgroup_quotas() if quota is not None]
    return min([allowed, *quotas])


def _cgroup_quotas(membership: Path = Path("/proc/self/cgroup"),
                   mount: Path = Path("/sys/fs/cgroup")) -> List[Optional[float]]:
    """Each CPU quota (CPUs) of this process's cgroup and its ancestors, None where unlimited.
    A container without its own cgroup namespace names its host's path, which its own mount
    lacks: the walk up still reaches the mount's root, the container's own cgroup."""
    quotas: List[Optional[float]] = []
    for line in membership.read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if controllers == "":  # v2: `cpu.max` is "<quota> <period>" or "max <period>"
            root, quota_of = mount, _v2_quota
        elif "cpu" in controllers.split(","):  # v1: cfs quota and period, -1 unlimited
            root = next((mount / name for name in ("cpu,cpuacct", "cpu") if (mount / name).is_dir()), None)
            if root is None:
                continue
            quota_of = _v1_quota
        else:
            continue
        directory = root / path.lstrip("/")
        while True:
            quotas.append(quota_of(directory))
            if directory == root:
                break
            directory = directory.parent
    return quotas


def _v2_quota(directory: Path) -> Optional[float]:
    limit = directory / "cpu.max"
    if not limit.exists():  # the root, or the cpu controller not enabled there: no limit
        return None
    quota, period = limit.read_text().split()
    return None if quota == "max" else int(quota) / int(period)


def _v1_quota(directory: Path) -> Optional[float]:
    limit = directory / "cpu.cfs_quota_us"
    if not limit.exists():
        return None
    quota = int(limit.read_text())
    return None if quota < 0 else quota / int((directory / "cpu.cfs_period_us").read_text())


def _run(argv: List[str]) -> str:
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise OSError(f"{' '.join(argv)} exited {result.returncode}: {result.stderr.strip()}")
    return result.stdout


def _sysctl(name: str) -> str:
    value = _run(["sysctl", "-n", name]).strip()
    if not value:
        raise OSError(f"sysctl {name} is empty")
    return value


def _cpuinfo_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    names = re.findall(r"^Model name:\s*(.+)$", _run(["lscpu"]), re.MULTILINE)
    if not names:
        raise OSError("neither /proc/cpuinfo nor lscpu reports a model name")
    return " + ".join(dict.fromkeys(name.strip() for name in names))


def _linux_physical_cores() -> int:
    cores = set()
    for cpu in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        topology = cpu / "topology"
        if topology.is_dir():
            cores.add(
                (
                    (topology / "physical_package_id").read_text().strip(),
                    (topology / "core_id").read_text().strip(),
                )
            )
    if not cores:
        raise OSError("no CPU topology in /sys/devices/system/cpu")
    return len(cores)


def _linux_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise OSError("/proc/meminfo has no MemTotal")


def _windows_memory() -> int:
    return windows_memory_status().total_physical


def _mac_gpus() -> List[Gpu]:
    text = _run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
    names = re.findall(r'"model" = "([^"]+)"', text)
    cores = re.findall(r'"gpu-core-count" = (\d+)', text)
    if not names:
        raise OSError("IOAccelerator reports no GPU model")
    return [
        Gpu(name, int(cores[i]) if i < len(cores) else None, None) for i, name in enumerate(names)
    ]


def _nvidia_gpus() -> List[Gpu]:
    """NVIDIA GPUs; none when nvidia-smi is not installed. Unified memory reports no total."""
    try:
        text = _run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        )
    except FileNotFoundError:
        return []
    gpus = []
    for line in text.strip().splitlines():
        name, memory = (part.strip() for part in line.split(","))
        mebibytes = number(memory)
        gpus.append(Gpu(name, None, None if mebibytes is None else int(mebibytes) * 1024 * 1024))
    return gpus


def capture(alias: Optional[str]) -> HostFacts:
    errors: List[str] = []

    def attempt(what: str, read: Callable[[], T]) -> Optional[T]:
        try:
            return read()
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
            errors.append(f"{what}: {type(error).__name__}: {error}")
            return None

    system = platform.system()
    if sys.platform == "darwin":
        version = platform.mac_ver()[0]
        model = attempt("model", lambda: _sysctl("hw.model"))
        chip = attempt("chip", lambda: _sysctl("machdep.cpu.brand_string"))
        physical = attempt("cpu_cores_physical", lambda: int(_sysctl("hw.physicalcpu")))
        memory = attempt("memory_bytes", lambda: int(_sysctl("hw.memsize")))
        gpus = attempt("gpus", _mac_gpus)
    elif sys.platform.startswith("linux"):
        version = platform.release()
        model = attempt(
            "model", lambda: Path("/sys/devices/virtual/dmi/id/product_name").read_text().strip()
        )
        chip = attempt("chip", _cpuinfo_model)
        physical = attempt("cpu_cores_physical", _linux_physical_cores)
        memory = attempt("memory_bytes", _linux_memory)
        gpus = attempt("gpus", _nvidia_gpus)
    elif sys.platform == "win32":
        version = platform.version()
        model = None
        errors.append("model: unsupported on Windows")
        chip = platform.processor() or None
        if chip is None:
            errors.append("chip: platform.processor() is empty")
        physical = None
        errors.append("cpu_cores_physical: unsupported on Windows")
        memory = attempt("memory_bytes", _windows_memory)
        gpus = attempt("gpus", _nvidia_gpus)
    else:
        raise OSError(f"no hardware facts for platform {sys.platform}")
    return HostFacts(
        hostname=platform.node(),
        alias=alias,
        os=system,
        os_version=version,
        architecture=platform.machine(),
        model=model,
        chip=chip,
        cpu_cores_physical=physical,
        cpu_cores_logical=os.cpu_count(),
        cpus_allowed=attempt("cpus_allowed", cpus_allowed),
        memory_bytes=memory,
        gpus=gpus,
        errors=errors,
    )
