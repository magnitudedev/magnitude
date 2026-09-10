"""Hardware report projection required by the preserved session-bench format."""

import platform

from magnitude_engine.platform.machine import discover
from magnitude_engine.platform.topology import Description, DeviceKind, MemoryKind


class HardwareReport(Description):
    hostname: str
    os: str
    os_version: str
    chip: str
    memory_bytes: int
    errors: tuple[str, ...]


def capture_hardware() -> HardwareReport:
    machine = discover()
    return HardwareReport(
        hostname=platform.node(),
        os=platform.system(),
        os_version=platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release(),
        chip=next(device.name for device in machine.devices if device.kind == DeviceKind.CPU),
        memory_bytes=sum(
            domain.capacity_bytes for domain in machine.memory if domain.kind == MemoryKind.HOST
        ),
        errors=machine.diagnostics,
    )
