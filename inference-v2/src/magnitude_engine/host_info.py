"""Host hardware evidence, collected without initializing the inference backend."""

import json
import platform
import subprocess

import psutil
from pydantic import BaseModel, ConfigDict, Field


class GPUInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    cores: int | None = Field(default=None, gt=0)


class HardwareInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    hostname: str
    os: str
    os_version: str
    model: str | None = None
    chip: str | None = None
    cpu_cores: int | None = Field(default=None, gt=0)
    memory_bytes: int = Field(gt=0)
    gpus: tuple[GPUInfo, ...] | None = None
    errors: tuple[str, ...] = ()


def _read(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def capture_hardware() -> HardwareInfo:
    """Call once before measured work. Unavailable Mac probes remain explicit evidence."""
    model, chip = None, platform.processor() or None
    cpu_cores = psutil.cpu_count(logical=False)
    memory_bytes = psutil.virtual_memory().total
    gpus = None
    errors = []
    if platform.system() == "Darwin":
        try:
            values = _read(
                "/usr/sbin/sysctl",
                "-n",
                "hw.model",
                "machdep.cpu.brand_string",
                "hw.physicalcpu",
                "hw.memsize",
            ).splitlines()
            model, chip, cores, memory = values
            cpu_cores, memory_bytes = int(cores), int(memory)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            errors.append(f"sysctl: {error}")
        try:
            displays = json.loads(
                _read(
                    "/usr/sbin/system_profiler",
                    "-json",
                    "SPDisplaysDataType",
                )
            )["SPDisplaysDataType"]
            gpus = tuple(
                GPUInfo(name=display["sppci_model"], cores=display.get("sppci_cores"))
                for display in displays
            )
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
            errors.append(f"system_profiler: {error}")
    return HardwareInfo(
        hostname=platform.node(),
        os=platform.system(),
        os_version=platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release(),
        model=model,
        chip=chip,
        cpu_cores=cpu_cores,
        memory_bytes=memory_bytes,
        gpus=gpus,
        errors=tuple(errors),
    )
