"""Generic physical resources, compiled entrypoints, and completion lifetime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import get_ident
from typing import Any, Protocol

from ..compiler.lowering import Capabilities
from ..tensor.types import DType, TensorSpec


class CapacityError(MemoryError):
    def __init__(self, required: int, available: int):
        super().__init__(f"allocation needs {required} bytes; {available} bytes remain")
        self.required = required
        self.available = available


class NativeAllocation(Protocol):
    @property
    def allocated_bytes(self) -> int: ...

    def view(self, spec: TensorSpec, offset: int = 0) -> Any: ...

    def close(self) -> None: ...


class NativeCompletion(Protocol):
    def ready(self) -> bool: ...

    def wait(self) -> None: ...


class NativeBoundEntrypoint(Protocol):
    def submit(self, dynamic: tuple[Any, ...]) -> NativeCompletion: ...

    def close(self) -> None: ...


class NativeExecutable(Protocol):
    def bind(
        self,
        static: Mapping[int, Any],
        dynamic_indices: tuple[int, ...],
    ) -> NativeBoundEntrypoint: ...

    def close(self) -> None: ...


class NativeRuntime(Protocol):
    @property
    def capabilities(self) -> Capabilities: ...

    @property
    def compiler_identity(self) -> str: ...

    def allocate(self, size: int, alignment: int) -> NativeAllocation: ...

    def upload(self, spec: TensorSpec, content: bytes) -> NativeAllocation: ...

    def download(self, value: Any) -> bytes: ...

    def compile(
        self,
        program: object,
        signature: tuple[TensorSpec, ...],
    ) -> NativeExecutable: ...

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion: ...

    def close(self) -> None: ...


class _Allocation:
    def __init__(self, device: Device, native: NativeAllocation, usable_bytes: int):
        self.device = device
        self.native = native
        self.usable_bytes = usable_bytes
        self.charged_bytes = native.allocated_bytes
        self.claims = 0
        self.closed = False

    def acquire(self) -> _Lease:
        self.device._check()
        if self.closed:
            raise RuntimeError("allocation has been reclaimed")
        self.claims += 1
        return _Lease(self)

    def release(self) -> None:
        self.device._check_thread()
        if self.claims <= 0:
            raise RuntimeError("allocation claim underflow")
        self.claims -= 1
        if self.claims == 0:
            self.native.close()
            self.closed = True
            self.device._allocated -= self.charged_bytes


class _Lease:
    def __init__(self, allocation: _Allocation):
        self.allocation = allocation
        self.closed = False

    def fork(self) -> _Lease:
        self.check()
        return self.allocation.acquire()

    def check(self) -> None:
        if self.closed:
            raise RuntimeError("resource lease is closed")

    def close(self) -> None:
        if not self.closed:
            self.allocation.release()
            self.closed = True


class Resource:
    """A typed view whose allocation is retained independently of its owner."""

    def __init__(self, lease: _Lease, spec: TensorSpec, offset: int = 0):
        if not spec.static:
            raise ValueError("physical resources require a concrete specification")
        if (
            offset < 0
            or offset % spec.dtype.itemsize
            or offset + spec.storage_nbytes > lease.allocation.usable_bytes
        ):
            raise ValueError("resource view exceeds or misaligns its allocation")
        self._lease = lease
        self.spec = spec
        self.offset = offset

    @property
    def device(self) -> Device:
        return self._lease.allocation.device

    @property
    def native(self) -> Any:
        self._lease.check()
        return self._lease.allocation.native.view(self.spec, self.offset)

    @property
    def allocated_bytes(self) -> int:
        self._lease.check()
        return self._lease.allocation.charged_bytes

    def view(self, spec: TensorSpec, offset: int = 0) -> Resource:
        if offset < 0 or offset + spec.storage_nbytes > self.spec.storage_nbytes:
            raise ValueError("resource subview exceeds parent")
        return Resource(self._lease.fork(), spec, self.offset + offset)

    def fork(self) -> Resource:
        return Resource(self._lease.fork(), self.spec, self.offset)

    def close(self) -> None:
        self._lease.close()


class Completion:
    def __init__(
        self,
        device: Device,
        native: NativeCompletion,
        retained: tuple[object, ...],
        on_release=None,
    ):
        self.device = device
        self._native = native
        self._retained = retained
        self._on_release = on_release
        self._released = False

    def ready(self) -> bool:
        ready = self._native.ready()
        if ready:
            self._release()
        return ready

    @property
    def done(self) -> bool:
        return self.ready()

    def wait(self) -> None:
        self._native.wait()
        self._release()

    def completion_waiter(self):
        """Return a thread-safe native wait for an external owner loop.

        Resource release remains on the device owner thread when ``wait`` is
        subsequently called there.
        """
        return self._native.wait

    def _release(self) -> None:
        if self._released:
            return
        for value in reversed(self._retained):
            close = getattr(value, "close", None)
            if close is not None:
                close()
        self._retained = ()
        if self._on_release is not None:
            self._on_release()
            self._on_release = None
        self._released = True


@dataclass(frozen=True, slots=True)
class Execution:
    outputs: tuple[Resource, ...]
    completion: Completion


class Device:
    def __init__(self, runtime: NativeRuntime, *, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("device budget must be positive")
        self.runtime = runtime
        self.budget_bytes = budget_bytes
        self._allocated = 0
        self._thread = get_ident()
        self._closed = False

    @property
    def capabilities(self) -> Capabilities:
        return self.runtime.capabilities

    @property
    def compiler_identity(self) -> str:
        return self.runtime.compiler_identity

    @property
    def allocated_bytes(self) -> int:
        return self._allocated

    @property
    def available_bytes(self) -> int:
        return self.budget_bytes - self._allocated

    def _check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("device resources are thread-confined")

    def _check(self) -> None:
        self._check_thread()
        if self._closed:
            raise RuntimeError("device is closed")

    def check(self) -> None:
        self._check()

    def check_thread(self) -> None:
        self._check_thread()

    def allocate(self, spec: TensorSpec, *, alignment: int | None = None) -> Resource:
        self._check()
        native = self._allocate(spec.storage_nbytes, max(spec.dtype.itemsize, alignment or 1))
        return Resource(native.acquire(), spec)

    def allocate_temporary(self, size: int, alignment: int) -> Resource:
        allocation = self._allocate(size, alignment)
        return Resource(allocation.acquire(), TensorSpec((size,), DType.U8))

    def _allocate(self, size: int, alignment: int) -> _Allocation:
        self._check()
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        if size > self.available_bytes:
            raise CapacityError(size, self.available_bytes)
        native = self.runtime.allocate(size, alignment)
        allocation = _Allocation(self, native, size)
        if allocation.charged_bytes > self.available_bytes:
            native.close()
            raise CapacityError(allocation.charged_bytes, self.available_bytes)
        self._allocated += allocation.charged_bytes
        return allocation

    def upload(self, spec: TensorSpec, content: bytes) -> Resource:
        self._check()
        if len(content) != spec.storage_nbytes:
            raise ValueError("upload byte count differs from tensor specification")
        native = self.runtime.upload(spec, content)
        allocation = _Allocation(self, native, spec.storage_nbytes)
        if allocation.charged_bytes > self.available_bytes:
            native.close()
            raise CapacityError(allocation.charged_bytes, self.available_bytes)
        self._allocated += allocation.charged_bytes
        return Resource(allocation.acquire(), spec)

    def read(self, resource: Resource, *, after: Completion | None = None) -> bytes:
        self._check()
        if resource.device is not self or (after is not None and after.device is not self):
            raise ValueError("read operands belong to another device")
        if after is not None:
            after.wait()
        return self.runtime.download(resource.native)

    def close(self) -> None:
        self._check()
        if self._allocated:
            raise RuntimeError(f"device still owns {self._allocated} charged bytes")
        self.runtime.close()
        self._closed = True


class BoundEntrypoint:
    def __init__(self, native: NativeBoundEntrypoint, dynamic_indices: tuple[int, ...]):
        self.native = native
        self.dynamic_indices = dynamic_indices

    def submit(self, arguments: Mapping[int, Resource]) -> NativeCompletion:
        return self.native.submit(tuple(arguments[index].native for index in self.dynamic_indices))

    def close(self) -> None:
        self.native.close()
