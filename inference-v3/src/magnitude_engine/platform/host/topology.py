"""Portable descriptions of physical resources and API exposure.

Inventory is independent of endpoint selection and kernel qualification. Memory
domains describe backing; budgets constrain it without adding more capacity.
"""

from enum import StrEnum
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from magnitude_engine.platform.backend import Backend

DeviceId = NewType("DeviceId", str)
EndpointId = NewType("EndpointId", str)
MemoryId = NewType("MemoryId", str)
BudgetId = NewType("BudgetId", str)


class Description(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class DeviceKind(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    ACCELERATOR = "accelerator"


class ComputeDevice(Description):
    id: DeviceId
    name: str
    kind: DeviceKind
    cpu_indices: tuple[int, ...] = ()
    pci_address: str | None = None
    registry_id: int | None = None


class CpuCache(Description):
    level: int = Field(gt=0)
    bytes: int = Field(gt=0)
    line_bytes: int = Field(gt=0)
    cpu_indices: tuple[int, ...]
    instruction_only: bool


class MemoryKind(StrEnum):
    HOST = "host"
    DEVICE = "device"


class MemoryDomain(Description):
    id: MemoryId
    kind: MemoryKind
    capacity_bytes: int = Field(gt=0)
    local_cpus: tuple[int, ...] = ()


class BudgetKind(StrEnum):
    HARD = "hard"
    RECOMMENDED = "recommended"


class MemoryBudget(Description):
    id: BudgetId
    domains: tuple[MemoryId, ...]
    limit_bytes: int = Field(gt=0)
    kind: BudgetKind
    source: str


class AllocationKind(StrEnum):
    HOST = "host"
    SHARED = "shared"
    PRIVATE = "private"
    PINNED = "pinned"
    MANAGED = "managed"


class AccessKind(StrEnum):
    DIRECT = "direct"
    COPY = "copy"


class MemoryAccess(Description):
    device: DeviceId
    kind: AccessKind
    host_synchronization_required: bool


class AllocationMode(Description):
    kind: AllocationKind
    domains: tuple[MemoryId, ...]
    budgets: tuple[BudgetId, ...]
    access: tuple[MemoryAccess, ...]
    max_allocation_bytes: int = Field(gt=0)


class GpuLimits(Description):
    max_threads_per_group: tuple[int, int, int]
    max_shared_bytes: int = Field(gt=0)
    max_buffer_bytes: int = Field(gt=0)


class Endpoint(Description):
    id: EndpointId
    device: DeviceId
    backend: Backend
    ordinal: int | None = Field(ge=0)
    modes: tuple[AllocationMode, ...]
    gpu: GpuLimits | None = None
    unavailable: tuple[str, ...] = ()


class Machine(Description):
    devices: tuple[ComputeDevice, ...]
    endpoints: tuple[Endpoint, ...]
    memory: tuple[MemoryDomain, ...]
    budgets: tuple[MemoryBudget, ...]
    cpu_caches: tuple[CpuCache, ...]
    diagnostics: tuple[str, ...] = ()

    @model_validator(mode="after")
    def relationships(self):
        def identities(items):
            ids = {item.id for item in items}
            if len(ids) != len(items):
                raise ValueError("duplicate resource identity")
            return ids

        devices, domains = identities(self.devices), identities(self.memory)
        budgets = identities(self.budgets)
        identities(self.endpoints)
        for budget in self.budgets:
            if not budget.domains or not set(budget.domains) <= domains:
                raise ValueError("budget must refer to existing memory domains")
        for endpoint in self.endpoints:
            if endpoint.device not in devices:
                raise ValueError("endpoint must refer to an existing compute device")
            if endpoint.ordinal is None and not endpoint.unavailable:
                raise ValueError("unexposed runtime endpoint requires an unavailable diagnostic")
            for mode in endpoint.modes:
                if not mode.domains or not set(mode.domains) <= domains:
                    raise ValueError("allocation must refer to existing backing domains")
                if not set(mode.budgets) <= budgets:
                    raise ValueError("allocation must refer to existing budgets")
                if not mode.access or any(access.device not in devices for access in mode.access):
                    raise ValueError("allocation access must refer to existing devices")
        return self

    def endpoint(self, identity: EndpointId) -> Endpoint:
        for endpoint in self.endpoints:
            if endpoint.id == identity:
                return endpoint
        raise KeyError(identity)
