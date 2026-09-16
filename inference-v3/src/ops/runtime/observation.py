"""Opt-in observations of actual execution, independent of formula arithmetic.

Host activities describe API boundaries, not hardware traffic. Optional native
kernel timestamps are separate. Lab attaches both to one typed formula occurrence.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
from time import perf_counter_ns
from typing import TYPE_CHECKING

from ..binding import SourceInfo
from .memory import MemoryMeasurement

if TYPE_CHECKING:
    from .resources import DeviceRuntime


class Activity(StrEnum):
    SOURCE_READ = "source-read"
    ALLOCATE = "allocate"
    RELEASE = "release"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    COMPILE = "compile"
    SUBMIT = "submit"
    WAIT = "wait"


class ObservationStatus(StrEnum):
    COMPLETE = "complete"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


@dataclass(frozen=True, slots=True)
class HostActivity:
    index: int
    parent: int | None
    kind: Activity
    started_ns: int
    elapsed_ns: int
    source: tuple[str, ...]
    target: tuple[str, ...]
    bytes_requested: int | None
    bytes_completed: int | None
    source_offset: int | None
    error: str | None
    source_info: SourceInfo | None = None

    def __post_init__(self):
        if min(self.index, self.started_ns, self.elapsed_ns) < 0:
            raise ValueError("activity identifiers and host timestamps must be nonnegative")
        if self.parent is not None and not 0 <= self.parent < self.index:
            raise ValueError("activity parent must precede its child")
        if any(value is not None and value < 0 for value in
               (self.bytes_requested, self.bytes_completed, self.source_offset)):
            raise ValueError("activity byte counts and source offsets must be nonnegative")


@dataclass(frozen=True, slots=True)
class KernelActivity:
    name: str
    elapsed_ns: int
    started_ns: int | None = None
    ended_ns: int | None = None
    dispatch: int | None = None
    origins: tuple[int, ...] = ()
    owner: int | None = None
    invocation: int | None = None
    graph: str | None = None

    def __post_init__(self):
        if not self.name or type(self.elapsed_ns) is not int or self.elapsed_ns < 0:
            raise ValueError("kernel timestamps require a name and nonnegative nanoseconds")
        if (self.started_ns is None) != (self.ended_ns is None):
            raise ValueError("kernel interval needs both endpoints")
        if self.started_ns is not None and self.ended_ns is not None and (
            self.started_ns < 0 or self.ended_ns - self.started_ns != self.elapsed_ns
        ):
            raise ValueError("kernel endpoints disagree with duration")
        if self.dispatch is not None and self.dispatch < 0:
            raise ValueError("dispatch identity must be nonnegative")


@dataclass(frozen=True, slots=True)
class KernelObservation:
    clock: str
    activities: tuple[KernelActivity, ...]
    attribution: str = "unavailable"

    def __post_init__(self):
        if not self.clock:
            raise ValueError("kernel timestamps require their native clock method")
        ids = [a.dispatch for a in self.activities if a.dispatch is not None]
        if len(ids) != len(set(ids)):
            raise ValueError("dynamic dispatch is recorded once per capture")

    @property
    def elapsed_ns(self) -> int:
        return sum(activity.elapsed_ns for activity in self.activities)

    @property
    def busy_ns(self) -> int | None:
        if any(a.started_ns is None for a in self.activities):
            return None
        total, end = 0, 0
        for activity in sorted(self.activities, key=lambda a: a.started_ns if a.started_ns is not None else 0):
            assert activity.started_ns is not None and activity.ended_ns is not None
            total += max(0, activity.ended_ns - max(end, activity.started_ns))
            end = max(end, activity.ended_ns)
        return total

    @property
    def overlap_ns(self) -> int | None:
        busy = self.busy_ns
        return None if busy is None else self.elapsed_ns - busy


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    status: ObservationStatus
    elapsed_ns: int
    activities: tuple[HostActivity, ...]
    memory: MemoryMeasurement
    error: str | None
    kernels: KernelObservation | None = None

    def __post_init__(self):
        if self.elapsed_ns < 0:
            raise ValueError("observation duration must be nonnegative")
        indices = {item.index for item in self.activities}
        if len(indices) != len(self.activities):
            raise ValueError("physical activities are recorded once")
        if any(item.started_ns + item.elapsed_ns > self.elapsed_ns for item in self.activities):
            raise ValueError("host activity exceeds the observed interval")
        if any(item.parent is not None and item.parent not in indices for item in self.activities):
            raise ValueError("host activity refers to a missing parent")
        if self.status == ObservationStatus.COMPLETE and self.error is not None:
            raise ValueError("a completed observation cannot have a boundary error")

    def completed_bytes(self, kind: Activity) -> int:
        """Bytes crossing this API; no inference about physical buses or storage."""
        return sum(item.bytes_completed or 0 for item in self.activities if item.kind == kind)


class _Span(AbstractContextManager):
    def __init__(
        self, capture: RuntimeCapture, kind: Activity, *, source: tuple[str, ...],
        target: tuple[str, ...], size: int | None, offset: int | None,
        source_info: SourceInfo | None,
    ):
        self._capture = capture
        self._kind, self._source, self._target = kind, source, target
        self._size, self._offset = size, offset
        self._source_info = source_info
        self.bytes_completed: int | None = None
        self._record_index: int | None = None

    def complete_bytes(self, size: int) -> None:
        """Attach transfer completion without extending its host API interval.

        An already finalized incomplete capture remains historical evidence;
        a later drain cannot turn it into a completed measurement.
        """
        self.bytes_completed = size
        if self._record_index is not None and self._capture._result is None:
            record = self._capture._activities[self._record_index]
            self._capture._activities[self._record_index] = replace(record, bytes_completed=size)

    def __enter__(self) -> _Span:
        capture = self._capture
        self._index = capture._next_index
        capture._next_index += 1
        self._parent = capture._stack[-1] if capture._stack else None
        capture._stack.append(self._index)
        self._started = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        finished = perf_counter_ns()
        capture = self._capture
        capture._stack.pop()
        self._record_index = len(capture._activities)
        capture._activities.append(HostActivity(
            self._index, self._parent, self._kind,
            self._started - capture._started, finished - self._started,
            self._source, self._target, self._size,
            self.bytes_completed, self._offset,
            exc_type.__name__ if exc_type is not None else None,
            self._source_info,
        ))


class RuntimeCapture(AbstractContextManager):
    """One isolated interval. The caller must finish submitted work before exiting.

    Capturing never synchronizes or replays execution. Optional counter storage
    is prepared before the observed host interval, separate from operation backing.
    Incomplete/failed results remain diagnostic evidence, not successful latency.
    """

    def __init__(self, recorder: RuntimeRecorder, kernel_limit: int | None = None):
        if kernel_limit is not None and (type(kernel_limit) is not int or not 1 <= kernel_limit <= 65536):
            raise ValueError("kernel capture capacity must be between 1 and 65536")
        self._recorder = recorder
        self._kernel_limit = kernel_limit
        self._native = None
        self._activities: list[HostActivity] = []
        self._stack: list[int] = []
        self._next_index = 0
        self._entered = False
        self._result: RuntimeObservation | None = None
        self._expected = []
        self._invocations = 0
        self._mapping_complete = True

    def program(self, graph, units) -> None:
        """Retain actual compiled call order; qualify it against native dispatches.

        Dynamic source loops are intentionally not guessed from static counts.
        A mismatch leaves timing valid and attribution unavailable.
        """
        invocation = self._invocations
        self._invocations += 1
        for unit in units:
            if unit.source_loop is not None:
                self._mapping_complete = False
                continue
            for call in unit.unit.calls:
                operation = call.operation
                definition = operation.definition
                if definition is None:
                    self._mapping_complete = False
                    continue
                nodes = set(operation.nodes)
                touched = tuple(c.occurrence for c in graph.formulas if nodes.intersection(c.nodes))
                containing = [c for c in graph.formulas if nodes <= set(c.nodes)]
                owner = min(containing, key=lambda c: (len(c.nodes), -c.occurrence)).occurrence if containing else None
                self._expected.extend((definition.name, touched, owner, invocation, graph.fingerprint)
                                      for _ in range(operation.kernel_count))

    def _attribute(self, activities):
        assert self._native is not None
        if not self._mapping_complete or len(activities) != len(self._expected):
            return KernelObservation(self._native.clock, activities, "unavailable: dynamic/count mismatch")
        for activity, (name, *_) in zip(activities, self._expected, strict=True):
            if activity.name != name and not activity.name.startswith(name + "_"):
                return KernelObservation(self._native.clock, activities, "unavailable: compiled/native name mismatch")
        attributed = tuple(replace(activity, origins=origins, owner=owner, invocation=invocation, graph=graph)
                           for activity, (_, origins, owner, invocation, graph)
                           in zip(activities, self._expected, strict=True))
        return KernelObservation(self._native.clock, attributed, "compiled-order-and-symbols")

    @property
    def result(self) -> RuntimeObservation:
        if self._result is None:
            raise RuntimeError("runtime observation has not finished")
        return self._result

    def __enter__(self) -> RuntimeCapture:
        recorder = self._recorder
        recorder.device.check()
        if self._entered:
            raise RuntimeError("a runtime capture is single-use")
        if recorder._active is not None:
            raise RuntimeError("measure one physical boundary at a time; nested events are attributed once")
        if recorder.device._submissions or recorder.device._completions:
            raise RuntimeError("finish earlier invocations before isolated measurement")
        self._entered = True
        try:
            if self._kernel_limit is not None:
                self._native = recorder.device.runtime.capture_kernels(self._kernel_limit)
                if self._native is not None:
                    self._native.start()
            self._window = recorder.device.memory.observe()
        except BaseException as error:
            if self._native is not None:
                try:
                    self._native.close()
                except BaseException as cleanup:
                    error.add_note(f"Kernel capture cleanup also failed: {cleanup}")
            raise
        self._started = perf_counter_ns()
        recorder._active = self
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        recorder = self._recorder
        recorder.device.check_thread()
        elapsed = perf_counter_ns() - self._started
        incomplete = bool(recorder.device._submissions or recorder.device._completions or self._stack)
        status = (ObservationStatus.FAILED if exc_type is not None else
                  ObservationStatus.INCOMPLETE if incomplete else ObservationStatus.COMPLETE)
        recorder._active = None
        kernels, failure = None, None
        try:
            if self._native is not None and status == ObservationStatus.COMPLETE:
                kernels = self._attribute(self._native.finish())
        except BaseException as error:
            status, failure = ObservationStatus.FAILED, error
        finally:
            if self._native is not None:
                try:
                    self._native.close()
                except BaseException as cleanup:
                    status = ObservationStatus.FAILED
                    original = exc if exc is not None else failure
                    if original is not None:
                        original.add_note(f"Kernel capture cleanup also failed: {cleanup}")
                    else:
                        failure = cleanup
            self._result = RuntimeObservation(
                status, elapsed, tuple(sorted(self._activities, key=lambda item: item.index)),
                self._window.close(),
                exc_type.__name__ if exc_type is not None else type(failure).__name__ if failure is not None else None,
                kernels,
            )
        if failure is not None:
            raise failure
        if incomplete and exc_type is None:
            raise RuntimeError("observation ended before execution completed; result is incomplete")


class RuntimeRecorder:
    def __init__(self, device: DeviceRuntime):
        self.device = device
        self._active: RuntimeCapture | None = None
        self._disabled = nullcontext(None)

    def capture(self, kernel_limit: int | None = None) -> RuntimeCapture:
        return RuntimeCapture(self, kernel_limit)

    def program(self, graph, units) -> None:
        if self._active is not None:
            self._active.program(graph, units)

    def span(
        self, kind: Activity, *, source: tuple[str, ...] = (),
        target: tuple[str, ...] = (), size: int | None = None,
        offset: int | None = None, source_info: SourceInfo | None = None,
    ) -> AbstractContextManager[_Span | None]:
        if self._active is None:
            return self._disabled
        return _Span(self._active, kind, source=source, target=target, size=size, offset=offset,
                     source_info=source_info)
