"""Discover the process-visible machine; selection does not erase inventory."""

import sys

from magnitude_engine.platform import hwloc
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.topology import (
    AccessKind,
    AllocationKind,
    AllocationMode,
    BudgetId,
    BudgetKind,
    Endpoint,
    EndpointId,
    Machine,
    MemoryAccess,
    MemoryBudget,
)


def discover() -> Machine:
    cpu, memory, caches = hwloc.interpret(hwloc.snapshot_xml())
    capacity = sum(domain.capacity_bytes for domain in memory)
    domains = tuple(domain.id for domain in memory)
    host_budget = MemoryBudget(
        id=BudgetId("host:physical"),
        domains=domains,
        limit_bytes=capacity,
        kind=BudgetKind.HARD,
        source="hwloc NUMA local_memory",
    )
    host_mode = AllocationMode(
        kind=AllocationKind.HOST,
        domains=domains,
        budgets=(host_budget.id,),
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
    devices, endpoints, budgets = (cpu,), (cpu_endpoint,), (host_budget,)
    diagnostics = ()
    if sys.platform == "darwin":
        from magnitude_engine.platform.darwin import metal_inventory

        gpu_devices, gpu_endpoints, gpu_memory, gpu_budgets = metal_inventory(
            cpu, memory, host_budget.id
        )
        devices += gpu_devices
        endpoints += gpu_endpoints
        memory += gpu_memory
        budgets += gpu_budgets
    else:
        diagnostics = ("GPU and process-limit discovery is not implemented for this host OS",)
    return Machine(
        devices=devices,
        endpoints=endpoints,
        memory=memory,
        budgets=budgets,
        cpu_caches=caches,
        diagnostics=diagnostics,
    )


def choose_endpoint(backend: Backend | None = None, ordinal: int = 0) -> Endpoint:
    """Select from the actual process-visible inventory, preserving diagnostics."""
    machine = discover()
    candidates = tuple(
        endpoint
        for endpoint in machine.endpoints
        if not endpoint.unavailable
        and endpoint.ordinal == ordinal
        and (backend is None or endpoint.backend == backend)
    )
    if not candidates:
        raise RuntimeError(
            "no available endpoint matches execution selection: " + "; ".join(machine.diagnostics)
        )
    # Prefer a discovered accelerator to the CPU fallback. Performance/profile
    # qualification is separate; this does not invent undiscovered GPU backends.
    return min(candidates, key=lambda endpoint: endpoint.backend == Backend.LLVM)


def open_context(backend: Backend, budget_bytes: int, ordinal: int):
    from magnitude_engine.platform.execution import DeviceContext

    inventory = discover()
    endpoint = next(
        (
            endpoint
            for endpoint in inventory.endpoints
            if endpoint.backend == backend and endpoint.ordinal == ordinal
        ),
        None,
    )
    if endpoint is None or endpoint.unavailable:
        detail = (
            "; ".join(endpoint.unavailable) if endpoint is not None else "endpoint not discovered"
        )
        raise RuntimeError(f"cannot open {backend.value}:{ordinal}: {detail}")
    if backend == Backend.METAL:
        from magnitude_engine.platform.metal import MetalDriver

        device = next(device for device in inventory.devices if device.id == endpoint.device)
        if device.registry_id is None:
            raise RuntimeError("Metal endpoint has no native registry identity")
        driver = MetalDriver(device.registry_id)
    else:
        from magnitude_engine.platform.torch_runtime import TorchDriver

        driver = TorchDriver(backend, ordinal)
    return DeviceContext(driver, budget_bytes)
