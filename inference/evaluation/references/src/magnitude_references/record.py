"""Result-record identity fields (`evaluation-result/1`): build, host, timestamps."""
from __future__ import annotations

from datetime import UTC, datetime
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utc_stamp() -> str:
    """Run-directory timestamp."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build() -> dict:
    """The local working tree's contract `build` object, from the remote package, which owns build
    identity. Runs on hosts without a git checkout receive it from the controller (`--build-identity`)."""
    from magnitude_remote.build import identify
    return identify().to_json()


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", name], check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _memory_bytes() -> int | None:
    if platform.system() == "Darwin":
        value = _sysctl("hw.memsize")
        return int(value) if value else None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def _chip() -> str | None:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def host(device: torch.device) -> dict:
    import torch
    gpus = None
    if device.type == "cuda":
        gpus = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    elif device.type == "mps":
        gpus = [_chip()]
    return {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}", "chip": _chip(),
            "cpu_cores": os.cpu_count(), "memory_bytes": _memory_bytes(), "gpus": gpus, "device": str(device)}
