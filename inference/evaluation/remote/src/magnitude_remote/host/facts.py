"""The typed hardware record of the executing host, captured before timed work.

Unavailable values are null and every failed probe keeps its error. Hostname identifies
the executing machine; the ssh alias, when there is one, is recorded beside it.
"""

from __future__ import annotations

import ctypes as c
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

from .probes import number

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
    memory_bytes: Optional[int]
    gpus: Optional[List[Gpu]]
    errors: List[str]

    def to_json(self) -> dict:
        return asdict(self)


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


class _MemoryStatus(c.Structure):
    _fields_ = [
        ("length", c.c_uint32),
        ("load", c.c_uint32),
        ("total_physical", c.c_uint64),
        ("available_physical", c.c_uint64),
        ("total_page_file", c.c_uint64),
        ("available_page_file", c.c_uint64),
        ("total_virtual", c.c_uint64),
        ("available_virtual", c.c_uint64),
        ("available_extended_virtual", c.c_uint64),
    ]


def _windows_memory() -> int:
    status = _MemoryStatus(length=c.sizeof(_MemoryStatus))
    if not c.windll.kernel32.GlobalMemoryStatusEx(c.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.total_physical


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
        memory_bytes=memory,
        gpus=gpus,
        errors=errors,
    )
