"""Allocation and submission lifetimes, independent of the native backend.

Prepared commands hold their operands. Submitted tickets keep those claims even
if callers discard every output; only proven completion permits reclamation.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from threading import get_ident
from typing import TYPE_CHECKING, Protocol

from magnitude_engine.data import Record
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.measurement import clock_ns
from magnitude_engine.platform.specialization import Specialization
from magnitude_engine.platform.storage import ByteSource, ZeroSource

if TYPE_CHECKING:
    from tilelang.jit import PrimFunc


class DType(StrEnum):
    U8 = "uint8"
    U16 = "uint16"
    U32 = "uint32"
    I32 = "int32"
    F16 = "float16"
    BF16 = "bfloat16"
    F32 = "float32"

    @property
    def itemsize(self) -> int:
        return _ITEM_BYTES[self]


_ITEM_BYTES = {
    DType.U8: 1,
    DType.U16: 2,
    DType.U32: 4,
    DType.I32: 4,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self):
        if not isinstance(self.dtype, DType):
            raise TypeError("tensor dtype must be a DType")
        if not isinstance(self.shape, tuple):
            raise TypeError("tensor shape must be an immutable tuple")
        if not self.shape or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError("tensor shape must contain positive integer extents")

    @cached_property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


class NativeBuffer(Protocol):
    @property
    def allocated_bytes(self) -> int: ...

    def close(self) -> None: ...


class NativeCompletion(Protocol):
    def ready(self) -> bool: ...
    def wait(self) -> None: ...


class NativeInterval(Protocol):
    def finish(self) -> None: ...
    def seconds(self) -> float | None: ...


class NativeExecutable(Protocol):
    def bind(
        self, arguments: tuple[tuple[NativeBuffer, TensorSpec, int], ...]
    ) -> NativeCommand: ...


class NativeCommand(Protocol):
    def close(self) -> None: ...


class ResourceLease(Protocol):
    """Execution pin for a logical resource such as a reusable pool extent."""

    @property
    def context(self) -> DeviceContext: ...

    def fork(self) -> ResourceLease: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Executable:
    signature: tuple[TensorSpec, ...]
    native: NativeExecutable
    driver: Driver
    dispatches: int = 1


class ExecutionOrder(StrEnum):
    ORDERED = "ordered"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class InputReference:
    index: int
    offset: int


class Driver(Protocol):
    backend: Backend

    @property
    def subgroup_width(self) -> int | None: ...

    def compile(self, program: PrimFunc) -> Executable: ...
    def bind_sequence(
        self,
        kernels: tuple[NativeExecutable, ...],
        commands: tuple[NativeCommand, ...],
        references: tuple[tuple[InputReference | None, ...], ...],
        order: ExecutionOrder,
    ) -> NativeExecutable: ...
    def dispatch(self, commands: tuple[NativeCommand, ...]) -> None: ...
    def interval(self) -> NativeInterval: ...
    def allocate(self, size: int) -> NativeBuffer: ...
    def upload(self, content: bytes) -> NativeBuffer: ...
    def write(self, buffer: NativeBuffer, offset: int, content: bytes) -> None: ...
    def read(self, buffer: NativeBuffer, offset: int, size: int) -> bytes: ...
    def record(self, *, timing: bool = False) -> NativeCompletion: ...
    def drain(self) -> None: ...


class CapacityError(MemoryError):
    def __init__(self, required: int, available: int):
        super().__init__(f"allocation needs {required} bytes; {available} bytes remain in budget")
        self.required = required
        self.available = available


class _Allocation:
    def __init__(self, context: DeviceContext, native: NativeBuffer, size: int):
        self.context, self.native, self.size = context, native, size
        self.charged_bytes = native.allocated_bytes
        self.claims = 0
        self.reclaimed = False

    def acquire(self) -> Lease:
        self.context.check()
        if self.reclaimed:
            raise RuntimeError("allocation has been reclaimed")
        self.claims += 1
        return Lease(self)

    def release(self) -> None:
        self.context.check_thread()
        self.claims -= 1
        if self.claims == 0:
            self.native.close()
            self.reclaimed = True
            self.context._allocated -= self.charged_bytes


class Lease:
    def __init__(self, allocation: _Allocation):
        self._allocation = allocation
        self._closed = False

    @property
    def context(self) -> DeviceContext:
        return self._allocation.context

    def check(self) -> None:
        if self._closed:
            raise RuntimeError("resource lease is closed")

    def fork(self) -> Lease:
        self.check()
        return self._allocation.acquire()

    def close(self) -> None:
        if not self._closed:
            self._allocation.release()
            self._closed = True


def reclaimable_bytes(tensors: tuple[Tensor, ...]) -> int:
    """Backing freed if exactly these direct tensor claims were released now.

    Region pins can carry additional ownership, so they are conservatively
    excluded. Owners of reusable regions account those regions themselves.
    This is a read-only ownership query, not a reservation or a release.
    """
    allocations: dict[_Allocation, set[int]] = {}
    for tensor in tensors:
        tensor._lease.check()
        tensor.context.check_thread()
        if not tensor._pins:
            allocations.setdefault(tensor._lease._allocation, set()).add(id(tensor._lease))
    return sum(
        allocation.charged_bytes
        for allocation, claims in allocations.items()
        if allocation.claims == len(claims)
    )


class Tensor:
    """Contiguous logical view with its own claim on the underlying allocation."""

    def __init__(
        self,
        lease: Lease,
        spec: TensorSpec,
        offset: int = 0,
        *,
        pins: tuple[ResourceLease, ...] = (),
    ):
        if offset < 0 or offset % spec.dtype.itemsize:
            raise ValueError("invalid tensor byte offset")
        if offset + spec.nbytes > lease._allocation.size:
            raise ValueError("tensor view exceeds allocation")
        self._lease, self._spec, self._offset = lease, spec, offset
        self._pins = pins

    @property
    def spec(self) -> TensorSpec:
        return self._spec

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def context(self) -> DeviceContext:
        return self._lease.context

    @property
    def exclusive_allocation(self) -> bool:
        """No other view or execution claim can access this backing allocation."""
        self._lease.check()
        return not self._pins and self._lease._allocation.claims == 1

    def overlaps(self, other: Tensor) -> bool:
        return (
            self._lease._allocation is other._lease._allocation
            and self.offset < other.offset + other.spec.nbytes
            and other.offset < self.offset + self.spec.nbytes
        )

    def view(self, spec: TensorSpec, offset: int = 0) -> Tensor:
        if offset < 0 or offset + spec.nbytes > self.spec.nbytes:
            raise ValueError("tensor subview exceeds parent view")
        lease = self._lease.fork()
        pins: list[ResourceLease] = []
        try:
            for pin in self._pins:
                pins.append(pin.fork())
            return Tensor(lease, spec, self.offset + offset, pins=tuple(pins))
        except BaseException:
            for pin in reversed(pins):
                pin.close()
            lease.close()
            raise

    def pin(self, *resources: ResourceLease) -> Tensor:
        """A new view retaining reusable regions through every derived consumer."""
        if any(resource.context is not self._lease.context for resource in resources):
            raise ValueError("tensor resource belongs to another execution owner")
        view = self.view(self.spec)
        pins: list[ResourceLease] = []
        try:
            for resource in resources:
                pins.append(resource.fork())
            view._pins = (*view._pins, *pins)
            return view
        except BaseException:
            for pin in pins:
                pin.close()
            view.close()
            raise

    def close(self) -> None:
        self._lease.close()
        for pin in self._pins:
            pin.close()


class Prepared:
    def __init__(
        self,
        context: DeviceContext,
        kernel: Executable,
        tensors: Sequence[Tensor],
        *,
        resources: Sequence[ResourceLease] = (),
    ):
        context.check()
        if kernel.driver is not context.driver:
            raise ValueError("executable belongs to another runtime owner")
        if tuple(tensor.spec for tensor in tensors) != kernel.signature:
            raise ValueError("operand specifications differ from executable signature")
        leases: list[Lease] = []
        pins: list[ResourceLease] = []
        operand_claims: list[tuple[ResourceLease, ...]] = []
        try:
            for tensor in tensors:
                if tensor._lease._allocation.context is not context:
                    raise ValueError("operand belongs to another device context")
                leases.append(tensor._lease.fork())
                start = len(pins)
                for pin in tensor._pins:
                    pins.append(pin.fork())
                operand_claims.append((leases[-1], *pins[start:]))
            extra_start = len(pins)
            for resource in resources:
                if resource.context is not context:
                    raise ValueError("resource belongs to another execution owner")
                pins.append(resource.fork())
            arguments = tuple(
                (lease._allocation.native, tensor.spec, tensor.offset)
                for lease, tensor in zip(leases, tensors, strict=True)
            )
            command = kernel.native.bind(arguments)
        except BaseException:
            for pin in reversed(pins):
                pin.close()
            for lease in reversed(leases):
                lease.close()
            raise
        self.context, self.kernel = context, kernel.native
        self.dispatches = kernel.dispatches
        self._leases: tuple[ResourceLease, ...] = (*leases, *pins)
        self._arguments = tuple(
            (tensor._lease._allocation, tensor.spec, tensor.offset) for tensor in tensors
        )
        self._operand_claims = tuple(operand_claims)
        self._extra_claims = tuple(pins[extra_start:])
        self._command = command
        self._consumed = False
        self._submission: Ticket | None = None

    @property
    def submission(self) -> Ticket | None:
        """The owner-issued ticket that consumed this command, retained after completion."""
        return self._submission

    def close(self) -> None:
        if not self._consumed:
            self._command.close()
            for lease in self._leases:
                lease.close()
            self._consumed = True


class SubmissionError(RuntimeError):
    def __init__(self, ticket: Ticket, cause: BaseException):
        super().__init__("submission failed; the ticket owns potentially submitted work")
        self.ticket, self.cause = ticket, cause


class Ticket:
    def __init__(
        self,
        context: DeviceContext,
        leases: tuple[ResourceLease, ...],
        executables: tuple[NativeExecutable, ...],
        commands: tuple[NativeCommand, ...],
    ):
        self.context, self._leases = context, leases
        self._executables = executables
        self._commands = commands
        self._completion: NativeCompletion | None = None
        self._interval: NativeInterval | None = None
        self._error: BaseException | None = None
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    @property
    def device_seconds(self) -> float | None:
        if not self._done:
            raise RuntimeError("timing requires completed work")
        if self._interval is None:
            return None
        return self._interval.seconds()

    def completion_waiter(self) -> Callable[[], None]:
        """Thread-safe native wait; reconciliation still belongs to the owner.

        The returned bound method retains the native completion object. It does
        not release leases, update ticket state, or authorize model commitment.
        """
        self.context.check_thread()
        if self._completion is None:
            raise RuntimeError("submission has no native completion to await")
        return self._completion.wait

    def wait(self) -> None:
        self.context.check_thread()
        if not self._done:
            try:
                if self._completion is None:
                    self.context.driver.drain()
                else:
                    self._completion.wait()
            except BaseException:
                # Ownership remains intact when completion cannot be established.
                self.context._failed = True
                raise
            for command in self._commands:
                command.close()
            self._commands = ()
            for lease in self._leases:
                lease.close()
            self._leases = ()
            self._executables = ()
            self._done = True
            self.context._pending.remove(self)
        if self._error is not None:
            raise SubmissionError(self, self._error) from self._error

    def poll(self) -> bool:
        self.context.check_thread()
        if self._done:
            return True
        try:
            ready = self._completion is not None and self._completion.ready()
        except BaseException:
            self.context._failed = True
            raise
        if ready:
            self.wait()
            return True
        return False


class SpecializationStatistics(Record):
    requests: int
    reused: int
    constructed: int
    construction_ns: int


class DeviceContext:
    """One ordered execution owner and an explicit allocation budget.

    This budget is an engine admission limit, never a claim about free hardware
    memory. Native allocation failure remains possible below it.
    """

    def __init__(self, driver: Driver, budget_bytes: int):
        if budget_bytes <= 0:
            raise ValueError("device allocation budget must be positive")
        self.driver, self.budget_bytes = driver, budget_bytes
        self._thread = get_ident()
        self._allocated = 0
        self._peak_allocated = 0
        self._pending: set[Ticket] = set()
        self._code: dict[int, list[tuple[PrimFunc, Executable]]] = {}
        self._specializations: dict[Specialization, Executable] = {}
        self._specialization_requests = self._specialization_reused = 0
        self._construction_ns = 0
        self._failed = False
        self._closed = False

    def check_thread(self) -> None:
        if get_ident() != self._thread:
            raise RuntimeError("device work belongs to its execution owner thread")

    def check(self) -> None:
        self.check_thread()
        if self._closed or self._failed:
            raise RuntimeError("device context is unavailable")

    @property
    def allocated_bytes(self) -> int:
        return self._allocated

    @property
    def peak_allocated_bytes(self) -> int:
        """Lifetime high-water mark of successfully charged native backing.

        Aliased views count once. Host objects, compiler allocations and refused
        native allocations are outside this execution-owner observation.
        """
        return self._peak_allocated

    @property
    def backend(self) -> Backend:
        return self.driver.backend

    @property
    def subgroup_width(self) -> int | None:
        return self.driver.subgroup_width

    @property
    def specialization_statistics(self) -> SpecializationStatistics:
        self.check_thread()
        return SpecializationStatistics(
            requests=self._specialization_requests,
            reused=self._specialization_reused,
            constructed=len(self._specializations),
            construction_ns=self._construction_ns,
        )

    def specialize[**P](
        self, factory: Callable[P, PrimFunc], *args: P.args, **kwargs: P.kwargs
    ) -> Executable:
        """Resolve a kernel definition before constructing its TileLang IR.

        Equivalent layer shapes share code, never weights or invocation operands.
        Cache lifetime is this execution owner; direct IR compilation remains the
        lower compiler boundary used for already constructed programs.
        """
        self.check()
        key = Specialization.bind(factory, *args, **kwargs)
        self._specialization_requests += 1
        if key in self._specializations:
            self._specialization_reused += 1
            return self._specializations[key]
        started = clock_ns()
        program = factory(*args, **kwargs)
        self._construction_ns += clock_ns() - started
        executable = self.compile(program)
        self._specializations[key] = executable
        return executable

    def compile(self, program: PrimFunc) -> Executable:
        self.check()
        from tilelang import tvm

        # A context fixes the endpoint and compiler policy. Structural equality
        # permits sharing code across weight instances without borrowing their
        # operands, and resolves hash collisions before reusing an executable.
        identity = tvm.ir.structural_hash(program)
        for previous, executable in self._code.get(identity, ()):
            if tvm.ir.structural_equal(previous, program):
                return executable
        executable = self.driver.compile(program)
        self._code.setdefault(identity, []).append((program, executable))
        return executable

    def _allocate(self, spec: TensorSpec, make: Callable[[], NativeBuffer]) -> Tensor:
        self.check()
        available = self.budget_bytes - self._allocated
        if spec.nbytes > available:
            raise CapacityError(spec.nbytes, available)
        native = make()
        if native.allocated_bytes > available:
            required = native.allocated_bytes
            native.close()
            raise CapacityError(required, available)
        allocation = _Allocation(self, native, spec.nbytes)
        self._allocated += allocation.charged_bytes
        self._peak_allocated = max(self._peak_allocated, self._allocated)
        return Tensor(allocation.acquire(), spec)

    def allocate(self, spec: TensorSpec) -> Tensor:
        return self._allocate(spec, lambda: self.driver.allocate(spec.nbytes))

    def zeros(self, spec: TensorSpec) -> Tensor:
        return self.upload_source(spec, ZeroSource(spec.nbytes), 0)

    def initialize_zero(self, tensor: Tensor) -> None:
        """Initialize unborrowed storage; never overwrite a live consumer's data."""
        self.check()
        if tensor.context is not self or not tensor.exclusive_allocation:
            raise ValueError("initialization requires exclusively owned backing")
        chunk = bytes(min(tensor.spec.nbytes, 8 * 1024**2))
        for offset in range(0, tensor.spec.nbytes, len(chunk)):
            size = min(len(chunk), tensor.spec.nbytes - offset)
            self.driver.write(
                tensor._lease._allocation.native, tensor.offset + offset, chunk[:size]
            )

    def indices(self, values: Sequence[int], shape: tuple[int, ...] | None = None) -> Tensor:
        spec = TensorSpec((len(values),) if shape is None else shape, DType.I32)
        return self.upload(spec, struct.pack(f"={len(values)}i", *values))

    def upload(self, spec: TensorSpec, content: bytes) -> Tensor:
        if len(content) != spec.nbytes:
            raise ValueError("uploaded byte count differs from tensor specification")
        return self._allocate(spec, lambda: self.driver.upload(content))

    def upload_source(
        self, spec: TensorSpec, source: ByteSource, offset: int, *, chunk_bytes: int = 8 * 1024**2
    ) -> Tensor:
        """Publish a new tensor only after bounded, synchronous staging completes.

        The destination is private to loading until this returns. No mutation API
        is exposed on a tensor that might already have submitted consumers.
        """
        if offset < 0 or offset + spec.nbytes > source.size or chunk_bytes <= 0:
            raise ValueError("invalid artifact upload range or staging size")
        tensor = self.allocate(spec)
        try:
            for start in range(0, spec.nbytes, chunk_bytes):
                length = min(chunk_bytes, spec.nbytes - start)
                content = source.read(offset + start, length)
                if len(content) != length:
                    raise ValueError("artifact source returned a short read")
                self.driver.write(tensor._lease._allocation.native, start, content)
            return tensor
        except BaseException:
            tensor.close()
            raise

    def submit(
        self, commands: Sequence[Prepared], *, after: Sequence[Ticket] = (), timing: bool = False
    ) -> Ticket:
        self.check()
        for dependency in after:
            if dependency.context is not self or dependency._error is not None:
                raise ValueError("dependency must be successful work on the same endpoint")
        if len({id(command) for command in commands}) != len(commands):
            raise ValueError("a prepared command cannot be submitted twice")
        for command in commands:
            if command.context is not self or command._consumed:
                raise ValueError("invalid or consumed prepared command")
        ticket = Ticket(
            self,
            tuple(lease for command in commands for lease in command._leases),
            tuple(command.kernel for command in commands),
            tuple(command._command for command in commands),
        )
        self._pending.add(ticket)
        for command in commands:
            command._consumed = True
            command._submission = ticket
        try:
            if timing:
                ticket._interval = self.driver.interval()
            self.driver.dispatch(ticket._commands)
            if ticket._interval is not None:
                ticket._interval.finish()
            ticket._completion = self.driver.record()
        except BaseException as error:
            self._failed = True
            ticket._error = error
            raise SubmissionError(ticket, error) from error
        return ticket

    def read(self, tensor: Tensor, *, after: Ticket) -> bytes:
        if after.context is not self or tensor._lease._allocation.context is not self:
            raise ValueError("readback belongs to another context")
        after.wait()
        tensor._lease.check()
        self.check()
        return self.driver.read(tensor._lease._allocation.native, tensor.offset, tensor.spec.nbytes)

    def reap(self) -> None:
        self.check()
        for ticket in tuple(self._pending):
            ticket.poll()

    def close(self) -> None:
        self.check_thread()
        if self._closed:
            return
        failures = []
        for ticket in tuple(self._pending):
            try:
                ticket.wait()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise BaseExceptionGroup("device shutdown failed", failures)
        self._specializations.clear()
        self._code.clear()
        self._closed = True
