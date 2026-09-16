"""Generic physical resources, compiled entrypoints, and completion lifetime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import get_ident
from typing import TYPE_CHECKING, Any, Protocol

from ..compiler.lowering import CompilerTarget
from ..tensor.types import DType, TensorSpec
from .configuration import DeviceConfiguration
from .memory import CapacityError, Limit, Reservation, ReservationLedger
from .observation import Activity, KernelActivity, RuntimeCapture, RuntimeRecorder

if TYPE_CHECKING:
    from ..binding import SourceSpan
    from ..compiler.schedules import ScheduleResolver


class NativeAllocation(Protocol):
    @property
    def allocated_bytes(self) -> int: ...

    def view(self, spec: TensorSpec, offset: int = 0) -> Any: ...

    def close(self) -> None: ...


class NativeCompletion(Protocol):
    def ready(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass(frozen=True)
class NativeUpload:
    allocation: NativeAllocation
    completion: NativeCompletion
    staging: object


class NativeSubmissionError(RuntimeError):
    """A launch failed after potentially submitting work; completion still owns it."""

    def __init__(self, cause: BaseException, completion: NativeCompletion):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.completion = completion


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


class NativeKernelCapture(Protocol):
    clock: str

    def start(self) -> None: ...

    def finish(self) -> tuple[KernelActivity, ...]: ...

    def close(self) -> None: ...


class NativeRuntime(Protocol):
    @property
    def compiler_target(self) -> CompilerTarget: ...

    @property
    def compiler_identity(self) -> str: ...

    @property
    def runtime_identity(self) -> str: ...

    def allocate(self, size: int, alignment: int) -> NativeAllocation: ...

    def upload(self, spec: TensorSpec, content: bytes) -> NativeAllocation: ...

    def upload_async(self, spec: TensorSpec, content: bytes) -> NativeUpload: ...

    def download(self, value: Any) -> bytes: ...

    def compile(
        self,
        program: object,
        signature: tuple[TensorSpec, ...],
    ) -> NativeExecutable: ...

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion: ...

    def capture_kernels(self, limit: int) -> NativeKernelCapture | None: ...

    def close(self) -> None: ...


class _TrackedCompletion:
    """Keep measurement pending until native completion, including joined work."""

    def __init__(self, device: DeviceRuntime, native: NativeCompletion):
        self.device, self.native = device, native
        self._done = False
        device._submissions[id(self)] = self

    def ready(self) -> bool:
        if self._done:
            return True
        ready = self.native.ready()
        if ready and get_ident() == self.device._thread:
            self._finish()
        return ready

    def wait(self) -> None:
        # External owner loops may block on the native wait. Only the device
        # thread mutates observation/lifetime bookkeeping on its subsequent wait.
        if get_ident() != self.device._thread:
            self.native.wait()
            return
        if not self._done:
            with self.device.observations.span(Activity.WAIT, target=(self.device.endpoint_id,)):
                self.native.wait()
            self._finish()

    def _finish(self) -> None:
        self._done = True
        self.device._submissions.pop(id(self), None)


class _CompletionGroup:
    def __init__(self, device: DeviceRuntime, children: tuple[NativeCompletion, ...]):
        self.device, self.children = device, children
        self._done = False
        self.native = device.runtime.join(tuple(
            child.native if isinstance(child, (_TrackedCompletion, _CompletionGroup)) else child
            for child in children
        ))

    def _finish(self) -> None:
        for child in self.children:
            if isinstance(child, (_TrackedCompletion, _CompletionGroup)):
                child._finish()
        self._done = True

    def ready(self) -> bool:
        if self._done:
            return True
        ready = self.native.ready()
        if ready and get_ident() == self.device._thread:
            self._finish()
        return ready

    def wait(self) -> None:
        if self._done:
            return
        if get_ident() != self.device._thread:
            self.native.wait()
            return
        with self.device.observations.span(Activity.WAIT, target=(self.device.endpoint_id,)):
            self.native.wait()
        # One observed native wait, whether the adapter uses an aggregate event
        # or several child events internally. Don't count both wrapper levels.
        self._finish()


class _Allocation:
    def __init__(
        self, device: DeviceRuntime, native: NativeAllocation, usable_bytes: int,
        reservation: Reservation,
    ):
        self.device = device
        self.native = native
        self.usable_bytes = usable_bytes
        self.charged_bytes = native.allocated_bytes
        self.reservation = reservation
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
        if self.claims == 1:
            with self.device.observations.span(
                Activity.RELEASE, source=tuple(sorted(self.reservation.domains)),
                size=self.charged_bytes,
            ) as activity:
                self.native.close()
                if activity is not None:
                    activity.bytes_completed = self.charged_bytes
            self.closed = True
            self.reservation.close()
        self.claims -= 1


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


@dataclass(frozen=True, slots=True)
class _ResourceView:
    """Validated immutable geometry shared by independent allocation leases."""
    spec: TensorSpec
    offset: int


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
        self._view = _ResourceView(spec, offset)

    @property
    def spec(self) -> TensorSpec:
        return self._view.spec

    @property
    def offset(self) -> int:
        return self._view.offset

    @property
    def device(self) -> DeviceRuntime:
        return self._lease.allocation.device

    @property
    def native(self) -> Any:
        self._lease.check()
        return self._lease.allocation.native.view(self.spec, self.offset)

    @property
    def allocated_bytes(self) -> int:
        self._lease.check()
        return self._lease.allocation.charged_bytes

    @property
    def sole_owner(self) -> bool:
        """Whether this lease is the only owner, including completion pins."""
        self.device._check_thread()
        self._lease.check()
        return self._lease.allocation.claims == 1

    def view(self, spec: TensorSpec, offset: int = 0) -> Resource:
        if offset == 0 and spec == self.spec:
            return self.fork()
        if offset < 0 or offset + spec.storage_nbytes > self.spec.storage_nbytes:
            raise ValueError("resource subview exceeds parent")
        lease = self._lease.fork()
        try:
            return Resource(lease, spec, self.offset + offset)
        except BaseException:
            lease.close()
            raise

    def fork(self) -> Resource:
        # Only the lease is new. The immutable view already passed all bounds,
        # alignment and concrete-geometry checks when it was constructed.
        resource = object.__new__(Resource)
        resource._lease = self._lease.fork()
        resource._view = self._view
        return resource

    def close(self) -> None:
        self._lease.close()


class Completion:
    def __init__(
        self,
        device: DeviceRuntime,
        native: NativeCompletion,
        retained: tuple[object, ...],
        on_release=None,
    ):
        self.device = device
        self._native = native
        self._retained = retained
        self._on_release = on_release
        self._released = False
        device._completions.add(self)

    @classmethod
    def join(cls, completions: tuple[Completion, ...]) -> Completion:
        if not completions:
            raise ValueError("completion join requires submitted work")
        device = completions[0].device
        if any(completion.device is not device for completion in completions):
            raise ValueError("joined completions belong to different devices")
        native = device.join(tuple(completion._native for completion in completions))

        def release():
            for completion in completions:
                completion._release()

        return cls(device, native, completions, release)

    def ready(self) -> bool:
        self.device._check_thread()
        ready = self._native.ready()
        if ready:
            self._release()
        return ready

    @property
    def done(self) -> bool:
        return self.ready()

    def wait(self) -> None:
        self.device._check_thread()
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
        self.device._check_thread()
        for value in reversed(self._retained):
            close = getattr(value, "close", None)
            if close is not None:
                close()
        self._retained = ()
        if self._on_release is not None:
            self._on_release()
            self._on_release = None
        self._released = True
        self.device._completions.discard(self)


@dataclass(frozen=True, slots=True)
class Execution:
    outputs: tuple[Resource, ...]
    completion: Completion


class DeviceRuntime:
    """One live owner for execution, reservations and completion-retained resources."""

    def __init__(
        self, runtime: NativeRuntime, *, budget_bytes: int,
        configuration: DeviceConfiguration | None = None,
        schedules: ScheduleResolver | None = None,
    ):
        if budget_bytes <= 0:
            raise ValueError("device budget must be positive")
        self.runtime = runtime
        self.configuration = configuration
        self.schedules = schedules
        self.endpoint_id = "injected"
        self.device_id = "injected"
        self.domains = frozenset({"injected"})
        self.host_domains = self.domains
        self.maximum_allocation_bytes = budget_bytes
        limits = [Limit("injected", self.domains, budget_bytes)]
        if configuration is not None:
            endpoint = configuration.selected_endpoints[0]
            self.endpoint_id, self.device_id = endpoint.id, endpoint.device
            mode = endpoint.modes[0]
            self.domains = frozenset(mode.domains)
            self.host_domains = frozenset(domain.id for domain in configuration.memory_domains
                                          if domain.kind == "host")
            if not self.host_domains:
                raise ValueError("runtime configuration must identify source-staging host memory")
            self.maximum_allocation_bytes = mode.max_allocation_bytes
            limits = [
                Limit(item.id, frozenset(item.domains), item.maximum_bytes)
                for item in configuration.memory_constraints
            ]
        self.memory = ReservationLedger(limits)
        self.budget_bytes = self.memory.available(self.domains)
        self._thread = get_ident()
        self._closed = False
        self._completions: set[Completion] = set()
        self._submissions: dict[int, NativeCompletion] = {}
        self._executables: set[object] = set()
        self._bindings: dict[str, Resource] = {}
        self._imports: dict[str, object] = {}
        self._import_programs: dict[object, object] = {}
        self._source_kernels: dict[str, tuple[object, object]] = {}
        self.observations = RuntimeRecorder(self)
        self._characterization = None

    @property
    def characterization(self):
        """Last explicitly loaded/measured resource evidence, or no evidence yet."""
        self.check()
        return self._characterization

    @property
    def evidence_identity(self) -> str:
        hardware = (self.configuration.fingerprint if self.configuration is not None else
                    f"{self.device_id}:{self.compiler_target.identity}")
        return f"{hardware}:{self.runtime.runtime_identity}"

    def characterize(self, store, *, protocol=None, refresh=False, cancellation=None):
        from ..lab.characterization import ProbeProtocol, characterize
        from ..lab.ownership import exclusive_measurement

        self.check()
        with exclusive_measurement():
            self._characterization = characterize(
                self, store, protocol=protocol if protocol is not None else ProbeProtocol.for_capacity(self.available_bytes),
                refresh=refresh, cancellation=cancellation,
            )
        return self._characterization

    def observe(self, *, kernel_limit: int | None = None) -> RuntimeCapture:
        """Capture an isolated complete execution, with no implicit synchronization."""
        return self.observations.capture(kernel_limit)

    def characterize_sources(self, store, spans, *, cancellation=None):
        from ..lab.characterization import characterize_sources
        from ..lab.ownership import exclusive_measurement

        self.check()
        with exclusive_measurement():
            if self._characterization is None:
                self.characterize(store, cancellation=cancellation)
            self._characterization = characterize_sources(
                self, store, self._characterization, spans, cancellation=cancellation,
            )
        return self._characterization

    def compile_native(self, program: object, signature: tuple[TensorSpec, ...]) -> NativeExecutable:
        self._check()
        with self.observations.span(Activity.COMPILE, target=(self.endpoint_id,)):
            return self.runtime.compile(program, signature)

    def submit_native(self, entrypoint: NativeBoundEntrypoint, arguments: tuple[Any, ...]) -> NativeCompletion:
        self._check()
        with self.observations.span(Activity.SUBMIT, target=(self.endpoint_id,)):
            try:
                native = entrypoint.submit(arguments)
            except NativeSubmissionError as error:
                error.completion = _TrackedCompletion(self, error.completion)
                raise
        return _TrackedCompletion(self, native)

    def join(self, completions: tuple[NativeCompletion, ...]) -> NativeCompletion:
        self._check()
        return _CompletionGroup(self, completions)

    def read_source(self, span: SourceSpan, *, value_identity: str) -> bytearray:
        """Observe source API traffic without claiming a physical disk read."""
        self._check()
        with self.observations.span(
            Activity.SOURCE_READ, source=(value_identity,),
            target=tuple(sorted(self.host_domains)), size=span.length, offset=span.offset,
            source_info=span.source.info,
        ) as activity:
            content = bytearray(span.length)
            count = span.source.read_into(span.offset, memoryview(content))
            if activity is not None:
                activity.bytes_completed = count
            if count != span.length:
                raise IOError("immutable source returned a short bounded read")
            return content

    def prepare_binding(self, binding):
        from .imports import PreparedImport, plan_import

        self._check()
        key = binding.fingerprint
        prepared = self._imports.get(key)
        if prepared is None or prepared.plan.binding is not binding:
            # Equivalent provenance can come from a newly opened artifact. Reuse
            # geometry programs, not a closed prior source object's handle.
            prepared = PreparedImport(self, plan_import(binding))
            try:
                prepared.prepare()
            except BaseException:
                prepared.close()
                raise
            self._imports[key] = prepared
        return prepared

    def resolve(self, binding, *, transient: bool = False) -> Resource:
        from ..binding import Residency

        self._check()
        transient = transient or binding.residency == Residency.STREAMED
        key = binding.fingerprint
        existing = self._bindings.get(key)
        if existing is not None and not transient:
            return existing.fork()
        result = self.prepare_binding(binding).load()
        if transient:
            return result
        self._bindings[key] = result
        return result.fork()

    def evict_binding(self, binding) -> None:
        """Release the cache claim; execution/compiled leases still retain backing."""
        self.evict_bindings((binding,))

    def evict_bindings(self, bindings) -> None:
        """Retire one set of logical weight roots and its derived source regions.

        Compiled operations retain their own prepared imports and resource leases.
        Shared conversion programs belong to the runtime, not to an artifact.
        """
        self._check()
        bindings = tuple(bindings)
        roots = {binding.root_identity for binding in bindings}
        keys = {binding.fingerprint for binding in bindings}
        for key, importer in tuple(self._imports.items()):
            if importer.plan.binding.root_identity in roots:
                keys.add(key)
                del self._imports[key]
        for key in keys:
            resource = self._bindings.pop(key, None)
            if resource is not None:
                resource.close()

    def invalidate_imports(self, changed: frozenset[tuple[str, str]]) -> None:
        """Retire edited conversion code and its derived resident cache values.

        Caller first retires dependent prepared operations. This never evicts an
        engine prefix: these are runtime-owned immutable import cache claims.
        """
        self._check()
        self.drain()
        programs = {key: program for key, program in self._import_programs.items()
                    if any((item.module, item.symbol) in changed for item in program.code_dependencies)}
        if not programs:
            return
        affected = set(programs.values())
        for key, importer in tuple(self._imports.items()):
            if any(program in affected for program in importer._programs.values()):
                resource = self._bindings.pop(key, None)
                if resource is not None:
                    resource.close()
                importer.close()
                del self._imports[key]
        for key, program in programs.items():
            program.close()
            del self._import_programs[key]

    @classmethod
    def open(cls, configuration: DeviceConfiguration, *, schedules: ScheduleResolver | None = None) -> DeviceRuntime:
        endpoints = configuration.selected_endpoints
        if len(endpoints) != 1:
            raise ValueError("this runtime requires one execution endpoint; distributed execution is not implemented")
        endpoint = endpoints[0]
        if endpoint.ordinal is None or endpoint.unavailable or not endpoint.modes:
            raise ValueError(f"endpoint {endpoint.id!r} is not executable")
        domains = frozenset(endpoint.modes[0].domains)
        capacities = [
            item.maximum_bytes for item in configuration.memory_constraints
            if domains.intersection(item.domains)
        ]
        if not capacities:
            raise ValueError("execution endpoint has no backing-memory constraint")
        from .tilelang import TileLangRuntime

        native = TileLangRuntime(str(endpoint.backend), ordinal=endpoint.ordinal)
        try:
            return cls(native, budget_bytes=min(capacities), configuration=configuration, schedules=schedules)
        except BaseException:
            native.close()
            raise

    @property
    def compiler_target(self) -> CompilerTarget:
        return self.runtime.compiler_target

    @property
    def compiler_identity(self) -> str:
        return self.runtime.compiler_identity

    @property
    def allocated_bytes(self) -> int:
        return self.memory.reserved

    @property
    def available_bytes(self) -> int:
        return self.memory.available(self.domains)

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
        represented_alignment = 4 if spec.representation is not None else 1
        native = self._allocate(
            spec.storage_nbytes,
            max(spec.dtype.itemsize, represented_alignment, alignment or 1),
        )
        return Resource(native.acquire(), spec)

    def allocate_temporary(self, size: int, alignment: int) -> Resource:
        allocation = self._allocate(size, alignment)
        return Resource(allocation.acquire(), TensorSpec((size,), DType.U8))

    def _allocate(self, size: int, alignment: int) -> _Allocation:
        self._check()
        if size <= 0 or alignment <= 0:
            raise ValueError("allocation size and alignment must be positive")
        charge = (size + alignment - 1) // alignment * alignment
        if charge > self.maximum_allocation_bytes:
            raise CapacityError(charge, self.maximum_allocation_bytes, constraint="maximum allocation")
        reservation = self.memory.reserve(charge, self.domains)
        native = None
        try:
            with self.observations.span(
                Activity.ALLOCATE, target=tuple(sorted(self.domains)), size=charge,
            ) as activity:
                native = self.runtime.allocate(size, alignment)
                if activity is not None:
                    activity.bytes_completed = native.allocated_bytes
            if native.allocated_bytes != charge:
                raise RuntimeError("allocator returned a charge different from its reserved aligned size")
            return _Allocation(self, native, size, reservation)
        except BaseException:
            if native is not None:
                native.close()
            reservation.close()
            raise

    def upload(self, spec: TensorSpec, content: bytes) -> Resource:
        self._check()
        if len(content) != spec.storage_nbytes:
            raise ValueError("upload byte count differs from tensor specification")
        from ..representations import Dense

        packed = spec.representation is not None and not isinstance(spec.representation, Dense)
        charge = (spec.storage_nbytes + 3) // 4 * 4 if packed else spec.storage_nbytes
        if charge > self.maximum_allocation_bytes:
            raise CapacityError(charge, self.maximum_allocation_bytes, constraint="maximum allocation")
        reservation = self.memory.reserve(charge, self.domains)
        stage = None
        native = None
        try:
            stage = self.memory.reserve(charge, self.host_domains)
            with self.observations.span(
                Activity.UPLOAD, source=tuple(sorted(self.host_domains)),
                target=tuple(sorted(self.domains)), size=len(content),
            ) as activity:
                native = self.runtime.upload(spec, content)
                if activity is not None:
                    activity.bytes_completed = len(content)
            if native.allocated_bytes != charge:
                raise RuntimeError("upload returned storage different from its reserved size")
            allocation = _Allocation(self, native, spec.storage_nbytes, reservation)
            return Resource(allocation.acquire(), spec)
        except BaseException:
            if native is not None:
                native.close()
            reservation.close()
            raise
        finally:
            if stage is not None:
                stage.close()

    def upload_async(self, spec: TensorSpec, content: bytes) -> Execution:
        """Submit an ordered transfer with completion-owned staging and output.

        Later device work may consume the output on the same execution domain.
        The caller joins or waits for the transfer before considering its physical
        boundary complete, including when it abandons the output without using it.
        """
        self._check()
        if len(content) != spec.storage_nbytes:
            raise ValueError("upload byte count differs from tensor specification")
        from ..representations import Dense

        packed = spec.representation is not None and not isinstance(spec.representation, Dense)
        charge = (spec.storage_nbytes + 3) // 4 * 4 if packed else spec.storage_nbytes
        if charge > self.maximum_allocation_bytes:
            raise CapacityError(charge, self.maximum_allocation_bytes, constraint="maximum allocation")
        reservation = self.memory.reserve(charge, self.domains)
        stage = None
        completion = None
        try:
            stage = self.memory.reserve(charge, self.host_domains)
            with self.observations.span(
                Activity.UPLOAD, source=tuple(sorted(self.host_domains)),
                target=tuple(sorted(self.domains)), size=len(content),
            ) as activity:
                native = self.runtime.upload_async(spec, content)
            tracked = _TrackedCompletion(self, native.completion)
            allocation = _Allocation(self, native.allocation, spec.storage_nbytes, reservation)
            transferred_bytes = len(content)

            def completed():
                if activity is not None:
                    activity.complete_bytes(transferred_bytes)

            completion = Completion(
                self, tracked, (stage, native.staging, allocation.acquire()), completed
            )
            if native.allocation.allocated_bytes != charge:
                raise RuntimeError("upload returned storage different from its reserved size")
            lease = allocation.acquire()
            try:
                output = Resource(lease, spec)
            except BaseException:
                lease.close()
                raise
            return Execution((output,), completion)
        except BaseException:
            if completion is not None:
                # If waiting itself fails, the registered completion continues
                # to own the source, destination and reservations for a retry.
                completion.wait()
            else:
                if stage is not None:
                    stage.close()
                reservation.close()
            raise

    def read(self, resource: Resource, *, after: Completion | None = None) -> bytes:
        self._check()
        if resource.device is not self or (after is not None and after.device is not self):
            raise ValueError("read operands belong to another device")
        if after is not None:
            after.wait()
        with self.observations.span(
            Activity.DOWNLOAD, source=tuple(sorted(self.domains)),
            target=tuple(sorted(self.host_domains)), size=resource.spec.storage_nbytes,
        ) as activity:
            content = self.runtime.download(resource.native)
            if activity is not None:
                activity.bytes_completed = len(content)
            return content[: resource.spec.storage_nbytes]

    def drain(self) -> None:
        self._check()
        for completion in tuple(self._completions):
            completion.wait()
        for native in tuple(self._submissions.values()):
            native.wait()

    def close(self) -> None:
        self._check_thread()
        if self._closed:
            return
        self.drain()
        for prepared in tuple(self._imports.values()):
            prepared.close()
        self._imports.clear()
        for executable, entrypoint in tuple(self._source_kernels.values()):
            entrypoint.close()
            executable.close()
        self._source_kernels.clear()
        for executable in tuple(self._executables):
            executable.close()
        for binding in tuple(self._bindings.values()):
            binding.close()
        self._bindings.clear()
        self._import_programs.clear()
        if self.allocated_bytes:
            raise RuntimeError(f"runtime still owns {self.allocated_bytes} charged bytes through live resource leases")
        self.runtime.close()
        self._closed = True

    def __enter__(self) -> DeviceRuntime:
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
