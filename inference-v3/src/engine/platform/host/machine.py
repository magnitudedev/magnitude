"""Discover the process-visible machine; selection does not erase inventory."""

import sys

from engine.platform.backend import Backend
from engine.platform.host import hwloc
from engine.devices import (
    AccessKind,
    AllocationKind,
    AllocationMode,
    ConstraintId,
    ConstraintKind,
    Endpoint,
    EndpointId,
    DeviceTopology,
    MemoryAccess,
    MemoryConstraint,
)


def discover() -> DeviceTopology:
    cpu, memory, caches = hwloc.interpret(hwloc.snapshot_xml())
    capacity = sum(domain.capacity_bytes for domain in memory)
    domains = tuple(domain.id for domain in memory)
    host_budget = MemoryConstraint(
        id=ConstraintId("host:physical"),
        domains=domains,
        maximum_bytes=capacity,
        kind=ConstraintKind.HARD,
        source="hwloc NUMA local_memory",
    )
    host_mode = AllocationMode(
        kind=AllocationKind.HOST,
        domains=domains,
        constraints=(host_budget.id,),
        max_allocation_bytes=capacity,
        access=(
            MemoryAccess(
                device=cpu.id, kind=AccessKind.DIRECT, host_synchronization_required=False
            ),
        ),
    )
    cpu_endpoint = Endpoint(
        id=EndpointId("llvm:process"),
        device=cpu.id,
        backend=Backend.LLVM,
        ordinal=0,
        modes=(host_mode,),
    )
    devices, endpoints, constraints = (cpu,), (cpu_endpoint,), (host_budget,)
    diagnostics = ()
    if sys.platform == "darwin":
        from engine.platform.host.darwin import metal_inventory

        gpu_devices, gpu_endpoints, gpu_memory, gpu_budgets = metal_inventory(
            cpu, memory, host_budget.id
        )
        devices += gpu_devices
        endpoints += gpu_endpoints
        memory += gpu_memory
        constraints += gpu_budgets
    else:
        diagnostics = ("GPU and process-limit discovery is not implemented for this host OS",)
    return DeviceTopology(
        devices=devices,
        endpoints=endpoints,
        memory=memory,
        constraints=constraints,
        cpu_caches=caches,
        diagnostics=diagnostics,
    )


def choose_endpoint(backend: Backend | None = None, ordinal: int = 0) -> Endpoint:
    """Select from the actual process-visible inventory, preserving diagnostics."""
    return discover().select(backend, ordinal)
