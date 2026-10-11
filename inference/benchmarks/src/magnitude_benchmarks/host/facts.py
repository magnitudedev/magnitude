"""The benchmark hardware record, captured once per run before preparation."""

import platform
import subprocess
import sys
from pathlib import Path

import psutil

from ..fixtures.records import Record


class HardwareReport(Record):
    hostname: str
    os: str
    os_version: str
    chip: str | None
    memory_bytes: int
    errors: tuple[str, ...]


def chip() -> tuple[str | None, str | None]:
    """The processor model name and, when it cannot be read, why."""
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), None
        return None, f"sysctl machdep.cpu.brand_string failed: {result.stderr.strip()}"
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip(), None
        return None, "/proc/cpuinfo has no model name"
    name = platform.processor()
    return (name, None) if name else (None, f"no processor name on {sys.platform}")


def capture_hardware() -> HardwareReport:
    name, error = chip()
    return HardwareReport(
        hostname=platform.node(),
        os=platform.system(),
        os_version=platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release(),
        chip=name,
        memory_bytes=psutil.virtual_memory().total,
        errors=() if error is None else (error,),
    )
