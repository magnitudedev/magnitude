"""Opt-in observations of actual execution, independent of formula arithmetic.

Host activities describe API boundaries, not hardware traffic. Optional native
kernel timestamps are separate. Lab attaches both to one typed formula occurrence.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
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

    def __post_init__(self):
        if not self.name or type(self.elapsed_ns) is not int or self.elapsed_ns < 0:
            raise ValueError("kernel timestamps require a name and nonnegative nanoseconds")


@dataclass(frozen=True, slots=True)
class KernelObservation:
    clock: str
    activities: tuple[KernelActivity, ...]

    def __post_init__(self):
        if not self.clock:
            raise ValueError("kernel timestamps require their native clock method")

    @property
    def elapsed_ns(self) -> int:
        return sum(activity.elapsed_ns for activity in self.activities)


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
                kernels = KernelObservation(self._native.clock, self._native.finish())
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

    def span(
        self, kind: Activity, *, source: tuple[str, ...] = (),
        target: tuple[str, ...] = (), size: int | None = None,
        offset: int | None = None, source_info: SourceInfo | None = None,
    ) -> AbstractContextManager[_Span | None]:
        if self._active is None:
            return self._disabled
        return _Span(self._active, kind, source=source, target=target, size=size, offset=offset,
                     source_info=source_info)
