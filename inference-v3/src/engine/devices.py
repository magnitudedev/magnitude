"""Portable descriptions of physical resources and API exposure.

Inventory is independent of endpoint selection and kernel qualification. Memory
domains describe backing; constraints constrain it without adding more capacity.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import NewType

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine.platform.backend import Backend

DeviceId = NewType("DeviceId", str)
EndpointId = NewType("EndpointId", str)
MemoryId = NewType("MemoryId", str)
ConstraintId = NewType("ConstraintId", str)


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


class ConstraintKind(StrEnum):
    HARD = "hard"
    RECOMMENDED = "recommended"


class MemoryConstraint(Description):
    id: ConstraintId = ConstraintId("")
    domains: tuple[MemoryId, ...]
    maximum_bytes: int = Field(gt=0)
    kind: ConstraintKind = ConstraintKind.HARD
    source: str = "configuration"

    @model_validator(mode="after")
    def identity(self):
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("a memory constraint needs distinct backing domains")
        if not self.id:
            identity = hashlib.sha256(repr((tuple(sorted(self.domains)), self.source)).encode())
            object.__setattr__(self, "id", ConstraintId(f"constraint:{identity.hexdigest()[:20]}"))
        return self


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
    constraints: tuple[ConstraintId, ...]
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


class DeviceTopology(Description):
    devices: tuple[ComputeDevice, ...]
    endpoints: tuple[Endpoint, ...]
    memory: tuple[MemoryDomain, ...]
    constraints: tuple[MemoryConstraint, ...]
    cpu_caches: tuple[CpuCache, ...]
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def discover(cls) -> DeviceTopology:
        from .platform.host.machine import discover

        return discover()

    @model_validator(mode="after")
    def relationships(self):
        def identities(items):
            ids = {item.id for item in items}
            if len(ids) != len(items):
                raise ValueError("duplicate resource identity")
            return ids

        devices, domains = identities(self.devices), identities(self.memory)
        constraints = identities(self.constraints)
        identities(self.endpoints)
        for budget in self.constraints:
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
                if not set(mode.constraints) <= constraints:
                    raise ValueError("allocation must refer to existing constraints")
                if not mode.access or any(access.device not in devices for access in mode.access):
                    raise ValueError("allocation access must refer to existing devices")
        return self

    def endpoint(self, identity: EndpointId) -> Endpoint:
        for endpoint in self.endpoints:
            if endpoint.id == identity:
                return endpoint
        raise KeyError(identity)

    def select(self, backend: Backend | None = None, ordinal: int = 0) -> Endpoint:
        candidates = tuple(
            endpoint for endpoint in self.endpoints
            if not endpoint.unavailable and endpoint.ordinal == ordinal
            and (backend is None or endpoint.backend == backend)
        )
        if not candidates:
            raise ValueError(f"no available endpoint for {backend!r}, ordinal {ordinal}: {self.diagnostics}")
        return min(candidates, key=lambda endpoint: (endpoint.backend == Backend.LLVM, endpoint.id))


class DevicePlan(Description):
    """Immutable deployment selection; the topology remains the canonical inventory."""

    topology: DeviceTopology
    endpoints: tuple[EndpointId, ...]
    constraints: tuple[MemoryConstraint, ...] = ()

    @model_validator(mode="after")
    def selection(self):
        if not self.endpoints or len(set(self.endpoints)) != len(self.endpoints):
            raise ValueError("a device plan needs distinct execution endpoints")
        for identity in self.endpoints:
            endpoint = self.topology.endpoint(identity)
            if endpoint.ordinal is None or endpoint.unavailable or not endpoint.modes:
                raise ValueError(f"endpoint {identity!r} is unavailable: {endpoint.unavailable}")
        domains = {domain.id for domain in self.topology.memory}
        ids = set()
        for constraint in self.constraints:
            if not set(constraint.domains) <= domains:
                raise ValueError("plan constraint references an unknown backing domain")
            if constraint.id in ids:
                raise ValueError("duplicate plan constraint identity")
            if constraint.kind != ConstraintKind.HARD:
                raise ValueError("deployment constraints must be hard; guidance belongs in topology")
            ids.add(constraint.id)
        if ids & {item.id for item in self.topology.constraints}:
            raise ValueError("deployment constraints must not replace discovered constraints")
        return self

    @property
    def selected_endpoints(self) -> tuple[Endpoint, ...]:
        return tuple(self.topology.endpoint(identity) for identity in self.endpoints)

    @property
    def memory_domains(self) -> tuple[MemoryDomain, ...]:
        return self.topology.memory

    @property
    def memory_constraints(self) -> tuple[MemoryConstraint, ...]:
        return tuple(
            item for item in self.topology.constraints if item.kind == ConstraintKind.HARD
        ) + self.constraints

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    @classmethod
    def discover(
        cls, *, maximum_bytes: int, backend: Backend | None = None, ordinal: int = 0,
    ) -> DevicePlan:
        if backend == "auto":
            backend = None
        elif backend is not None:
            backend = Backend(backend)
        topology = DeviceTopology.discover()
        endpoint = topology.select(backend, ordinal)
        domains = tuple(dict.fromkeys(domain for mode in endpoint.modes for domain in mode.domains))
        return cls(
            topology=topology,
            endpoints=(endpoint.id,),
            constraints=(MemoryConstraint(domains=domains, maximum_bytes=maximum_bytes),),
        )
