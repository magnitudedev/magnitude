"""Backend-neutral construction of accelerator topology from native driver facts."""

from dataclasses import dataclass
from importlib import import_module

from engine.devices import (
    AccessKind,
    AllocationKind,
    AllocationMode,
    ComputeDevice,
    ConstraintId,
    ConstraintKind,
    DeviceId,
    DeviceKind,
    Endpoint,
    EndpointId,
    GpuLimits,
    MemoryAccess,
    MemoryConstraint,
    MemoryDomain,
    MemoryId,
    MemoryKind,
)
from engine.platform.backend import Backend


@dataclass(frozen=True, slots=True)
class Accelerator:
    identity: str
    name: str
    ordinal: int
    capacity_bytes: int
    integrated: bool
    max_threads_per_group: int
    max_shared_bytes: int
    pci_address: str | None = None


def _pci_address(properties) -> str | None:
    values = tuple(
        getattr(properties, name, None)
        for name in ("pci_domain_id", "pci_bus_id", "pci_device_id")
    )
    if any(value is None for value in values):
        return None
    domain, bus, device = values
    return f"{domain:04x}:{bus:02x}:{device:02x}.0"


def torch_accelerators(
    *, torch=None
) -> tuple[Backend | None, tuple[Accelerator, ...], str | None]:
    """Read the accelerator runtime already owned by this engine process."""
    try:
        torch = torch or import_module("torch")
        runtime = torch.cuda
        if not runtime.is_available():
            return None, (), "Torch reports no process-visible CUDA or HIP devices"
        backend = Backend.HIP if getattr(torch.version, "hip", None) is not None else Backend.CUDA
        devices = []
        for ordinal in range(runtime.device_count()):
            properties = runtime.get_device_properties(ordinal)
            uuid = getattr(properties, "uuid", None)
            pci = _pci_address(properties)
            identity = (
                f"uuid:{uuid}"
                if uuid is not None
                else f"pci:{pci}"
                if pci
                else f"ordinal:{ordinal}"
            )
            shared = getattr(properties, "shared_memory_per_block_optin", None)
            if shared is None:
                shared = properties.shared_memory_per_block
            devices.append(
                Accelerator(
                    identity=identity,
                    name=properties.name,
                    ordinal=ordinal,
                    capacity_bytes=int(properties.total_memory),
                    integrated=bool(properties.is_integrated),
                    max_threads_per_group=int(properties.max_threads_per_block),
                    max_shared_bytes=int(shared),
                    pci_address=pci,
                )
            )
        return backend, tuple(devices), None
    except (AttributeError, OSError, RuntimeError) as error:
        return None, (), f"Torch accelerator inventory unavailable: {error}"


def inventory(
    backend: Backend,
    discovered: tuple[Accelerator, ...],
    cpu: ComputeDevice,
    host: tuple[MemoryDomain, ...],
    host_budget: ConstraintId,
):
    """Translate driver facts into physical devices, endpoints and backing domains."""
    devices, endpoints, memory, constraints = [], [], [], []
    for item in discovered:
        identity = DeviceId(f"{backend.value}:{item.identity}")
        devices.append(
            ComputeDevice(
                id=identity,
                name=item.name,
                kind=DeviceKind.GPU,
                pci_address=item.pci_address,
            )
        )
        if item.integrated:
            domains = tuple(domain.id for domain in host)
            budgets = (host_budget,)
        else:
            domain = MemoryDomain(
                id=MemoryId(f"{backend.value}:memory:{item.identity}"),
                kind=MemoryKind.DEVICE,
                capacity_bytes=item.capacity_bytes,
            )
            budget = MemoryConstraint(
                id=ConstraintId(f"{backend.value}:physical:{item.identity}"),
                domains=(domain.id,),
                maximum_bytes=item.capacity_bytes,
                kind=ConstraintKind.HARD,
                source=f"{backend.value.upper()} device total memory",
            )
            memory.append(domain)
            constraints.append(budget)
            domains, budgets = (domain.id,), (budget.id,)
        endpoints.append(
            Endpoint(
                id=EndpointId(f"{backend.value}:{item.identity}"),
                device=identity,
                backend=backend,
                ordinal=item.ordinal,
                modes=(
                    AllocationMode(
                        kind=AllocationKind.PRIVATE,
                        domains=domains,
                        constraints=budgets,
                        access=(
                            MemoryAccess(
                                device=identity,
                                kind=AccessKind.DIRECT,
                                host_synchronization_required=True,
                            ),
                            MemoryAccess(
                                device=cpu.id,
                                kind=AccessKind.COPY,
                                host_synchronization_required=True,
                            ),
                        ),
                        max_allocation_bytes=item.capacity_bytes,
                    ),
                ),
                gpu=GpuLimits(
                    max_threads_per_group=(item.max_threads_per_group, 1, 1),
                    max_shared_bytes=item.max_shared_bytes,
                    max_buffer_bytes=item.capacity_bytes,
                ),
            )
        )
    return tuple(devices), tuple(endpoints), tuple(memory), tuple(constraints)
