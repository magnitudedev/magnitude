"""Native hwloc discovery, retaining its NUMA and cache relationships."""

import ctypes as C
from xml.etree import ElementTree as XML

from magnitude_engine.platform.host.native import library_path
from magnitude_engine.platform.host.topology import (
    ComputeDevice,
    CpuCache,
    DeviceId,
    DeviceKind,
    MemoryDomain,
    MemoryId,
    MemoryKind,
)


def cpu_set(raw: str) -> tuple[int, ...]:
    bitmap = int(raw.replace(",", "").replace("0x", ""), 16)
    return tuple(i for i in range(bitmap.bit_length()) if bitmap & (1 << i))


def snapshot_xml() -> str:
    library = C.CDLL(library_path("hwloc"))
    library.hwloc_topology_init.argtypes = [C.POINTER(C.c_void_p)]
    library.hwloc_topology_load.argtypes = [C.c_void_p]
    library.hwloc_topology_destroy.argtypes = [C.c_void_p]
    library.hwloc_topology_export_xmlbuffer.argtypes = [
        C.c_void_p,
        C.POINTER(C.c_void_p),
        C.POINTER(C.c_int),
        C.c_ulong,
    ]
    library.hwloc_free_xmlbuffer.argtypes = [C.c_void_p, C.c_void_p]
    topology = C.c_void_p()
    if library.hwloc_topology_init(C.byref(topology)) != 0:
        raise RuntimeError("hwloc topology initialization failed")
    try:
        if library.hwloc_topology_load(topology) != 0:
            raise RuntimeError("hwloc topology discovery failed")
        buffer, length = C.c_void_p(), C.c_int()
        if library.hwloc_topology_export_xmlbuffer(topology, C.byref(buffer), C.byref(length), 0):
            raise RuntimeError("hwloc topology export failed")
        try:
            return C.string_at(buffer, length.value - 1).decode("utf-8")
        finally:
            library.hwloc_free_xmlbuffer(topology, buffer)
    finally:
        library.hwloc_topology_destroy(topology)


def interpret(xml: str) -> tuple[ComputeDevice, tuple[MemoryDomain, ...], tuple[CpuCache, ...]]:
    root = XML.fromstring(xml)
    machine = root.find("object")
    if machine is None or machine.get("type") != "Machine":
        raise ValueError("hwloc returned no machine topology")
    allowed = cpu_set(machine.attrib["allowed_cpuset"])
    model = root.find(".//info[@name='CPUModel']")
    cpu = ComputeDevice(
        id=DeviceId("cpu:process"),
        name=model.attrib["value"] if model is not None else "Process CPU domain",
        kind=DeviceKind.CPU,
        cpu_indices=allowed,
    )
    memory = tuple(
        MemoryDomain(
            id=MemoryId(f"host:numa:{node.attrib['os_index']}"),
            kind=MemoryKind.HOST,
            capacity_bytes=int(node.attrib["local_memory"]),
            local_cpus=cpu_set(node.attrib["cpuset"]),
        )
        for node in root.findall(".//object[@type='NUMANode']")
        if int(node.attrib["local_memory"]) > 0
    )
    if not memory:
        raise RuntimeError("hwloc did not report host memory backing")
    caches = tuple(
        CpuCache(
            level=int(node.attrib["depth"]),
            bytes=int(node.attrib["cache_size"]),
            line_bytes=int(node.attrib["cache_linesize"]),
            cpu_indices=cpu_set(node.attrib["cpuset"]),
            instruction_only=node.attrib["cache_type"] == "2",
        )
        for node in root.iter("object")
        if "cache_size" in node.attrib
    )
    return cpu, memory, caches
