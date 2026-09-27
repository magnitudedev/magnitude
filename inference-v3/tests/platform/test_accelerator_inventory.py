from types import SimpleNamespace

from engine.devices import (
    AllocationKind,
    ComputeDevice,
    ConstraintId,
    DeviceId,
    DeviceKind,
    MemoryDomain,
    MemoryId,
    MemoryKind,
)
from engine.platform.backend import Backend
from engine.platform.host.accelerator import Accelerator, inventory, torch_accelerators


def _host():
    cpu = ComputeDevice(id=DeviceId("cpu"), name="CPU", kind=DeviceKind.CPU)
    memory = (
        MemoryDomain(
            id=MemoryId("host"), kind=MemoryKind.HOST, capacity_bytes=32 * 1024**3
        ),
    )
    return cpu, memory, ConstraintId("host-budget")


def _device(*, integrated: bool):
    return Accelerator(
        "uuid:00000000-0000-0000-0000-000000000001",
        "test GPU",
        0,
        16 * 1024**3,
        integrated,
        1024,
        101_376,
        "0000:01:02.0",
    )


def test_integrated_accelerator_uses_host_physical_backing():
    cpu, host, budget = _host()
    devices, endpoints, memory, constraints = inventory(
        Backend.CUDA, (_device(integrated=True),), cpu, host, budget
    )

    assert len(devices) == len(endpoints) == 1
    assert memory == constraints == ()
    endpoint = endpoints[0]
    assert endpoint.backend is Backend.CUDA
    assert endpoint.modes[0].kind is AllocationKind.PRIVATE
    assert endpoint.modes[0].domains == ("host",)
    assert endpoint.modes[0].constraints == ("host-budget",)
    assert endpoint.gpu is not None and endpoint.gpu.max_shared_bytes == 101_376


def test_discrete_accelerator_owns_device_memory_for_every_backend():
    cpu, host, budget = _host()
    for backend in (Backend.CUDA, Backend.HIP):
        devices, endpoints, memory, constraints = inventory(
            backend, (_device(integrated=False),), cpu, host, budget
        )
        assert len(devices) == len(endpoints) == len(memory) == len(constraints) == 1
        assert memory[0].kind is MemoryKind.DEVICE
        assert endpoints[0].modes[0].domains == (memory[0].id,)
        assert endpoints[0].modes[0].constraints == (constraints[0].id,)


class _Runtime:
    def is_available(self):
        return True

    def device_count(self):
        return 1

    def get_device_properties(self, ordinal):
        assert ordinal == 0
        return SimpleNamespace(
            uuid="00000000-0000-0000-0000-000000000001",
            name="test GPU",
            total_memory=16 * 1024**3,
            is_integrated=True,
            max_threads_per_block=1024,
            shared_memory_per_block=64 * 1024,
            pci_domain_id=0,
            pci_bus_id=1,
            pci_device_id=2,
        )


def test_torch_build_determines_cuda_or_hip_backend():
    cuda = SimpleNamespace(cuda=_Runtime(), version=SimpleNamespace(hip=None))
    hip = SimpleNamespace(cuda=_Runtime(), version=SimpleNamespace(hip="7.0"))

    assert torch_accelerators(torch=cuda)[0] is Backend.CUDA
    assert torch_accelerators(torch=hip)[0] is Backend.HIP
