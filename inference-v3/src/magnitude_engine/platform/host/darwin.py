"""Metal framework discovery via native Objective-C APIs."""

import ctypes as C

from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.host.topology import (
    AccessKind,
    AllocationKind,
    AllocationMode,
    BudgetId,
    BudgetKind,
    ComputeDevice,
    DeviceId,
    DeviceKind,
    Endpoint,
    EndpointId,
    GpuLimits,
    MemoryAccess,
    MemoryBudget,
    MemoryDomain,
)


class _Size(C.Structure):
    _fields_ = [("x", C.c_ulong), ("y", C.c_ulong), ("z", C.c_ulong)]


class _ObjC:
    def __init__(self):
        self.library = C.CDLL("/usr/lib/libobjc.A.dylib")
        self.library.sel_registerName.argtypes = [C.c_char_p]
        self.library.sel_registerName.restype = C.c_void_p
        address = C.cast(self.library.objc_msgSend, C.c_void_p).value
        if address is None:
            raise RuntimeError("Objective-C message dispatcher is unavailable")
        self.address = address

    def send(
        self,
        object: int,
        selector: str,
        result: type | None = C.c_void_p,
        types: tuple[type, ...] = (),
        arguments: tuple[object, ...] = (),
    ):
        function = C.CFUNCTYPE(result, C.c_void_p, C.c_void_p, *types)(self.address)
        return function(object, self.library.sel_registerName(selector.encode()), *arguments)


def metal_inventory(cpu: ComputeDevice, host: tuple[MemoryDomain, ...], host_budget: BudgetId):
    objc = _ObjC()
    library = C.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    library.MTLCopyAllDevices.restype = C.c_void_p
    array = library.MTLCopyAllDevices()
    if not array:
        raise RuntimeError("MTLCopyAllDevices returned no inventory")
    devices, endpoints, memory, budgets = [], [], [], []
    try:
        for ordinal in range(objc.send(array, "count", C.c_ulong)):
            device = objc.send(array, "objectAtIndex:", types=(C.c_ulong,), arguments=(ordinal,))
            registry = objc.send(device, "registryID", C.c_ulong)
            identity = DeviceId(f"metal:registry:{registry}")
            name = objc.send(objc.send(device, "name"), "UTF8String", C.c_char_p).decode("utf-8")
            devices.append(
                ComputeDevice(id=identity, name=name, kind=DeviceKind.GPU, registry_id=registry)
            )
            unified = objc.send(device, "hasUnifiedMemory", C.c_bool)
            maximum = objc.send(device, "maxBufferLength", C.c_ulong)
            working_set = objc.send(device, "recommendedMaxWorkingSetSize", C.c_ulong)
            limits = objc.send(device, "maxThreadsPerThreadgroup", _Size)
            gpu = GpuLimits(
                max_threads_per_group=(limits.x, limits.y, limits.z),
                max_shared_bytes=objc.send(device, "maxThreadgroupMemoryLength", C.c_ulong),
                max_buffer_bytes=maximum,
            )
            modes = []
            unavailable = []
            if unified:
                domains = tuple(domain.id for domain in host)
                budget = BudgetId(f"metal:working-set:{registry}")
                budgets.append(
                    MemoryBudget(
                        id=budget,
                        domains=domains,
                        limit_bytes=working_set,
                        kind=BudgetKind.RECOMMENDED,
                        source="MTLDevice.recommendedMaxWorkingSetSize",
                    )
                )
                for kind in (AllocationKind.PRIVATE, AllocationKind.SHARED):
                    access = (
                        MemoryAccess(
                            device=identity,
                            kind=AccessKind.DIRECT,
                            host_synchronization_required=True,
                        ),
                    )
                    if kind == AllocationKind.SHARED:
                        access += (
                            MemoryAccess(
                                device=cpu.id,
                                kind=AccessKind.DIRECT,
                                host_synchronization_required=True,
                            ),
                        )
                    modes.append(
                        AllocationMode(
                            kind=kind,
                            domains=domains,
                            budgets=(host_budget, budget),
                            access=access,
                            max_allocation_bytes=maximum,
                        )
                    )
            else:
                # Metal's working-set guidance is not a physical VRAM capacity.
                # IOKit reconciliation must establish that backing before use.
                unavailable.append("discrete Metal backing requires IOKit memory reconciliation")
            endpoints.append(
                Endpoint(
                    id=EndpointId(f"metal:{registry}"),
                    device=identity,
                    backend=Backend.METAL,
                    ordinal=ordinal,
                    modes=tuple(modes),
                    gpu=gpu,
                    unavailable=tuple(unavailable),
                )
            )
    finally:
        objc.send(array, "release", None)
    return tuple(devices), tuple(endpoints), tuple(memory), tuple(budgets)
