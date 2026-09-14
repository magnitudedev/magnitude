"""Persistent device-owner worker and typed, cancellable formula requests."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread, current_thread
from time import perf_counter_ns
from uuid import uuid4

from ..compiler.compilation import CompileOptions
from ..formula import FormulaHandle, FormulaTree
from ..runtime.resources import DeviceRuntime
from .fixtures import Fixture
from .records import History, JobResult, JobStart, MeasurementProtocol, Outcome, Phase, PhaseTime, Visibility
from .runner import MeasurementRunner
from .store import ObservationStore
from .ownership import exclusive_measurement
from .characterization import Characterization


class Status(StrEnum):
    UNMEASURED = "unmeasured"
    RECORDED = "recorded"
    CURRENT = "current"
    STALE = "stale"
    QUEUED = "queued"
    RUNNING = "running"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FormulaState:
    target: FormulaHandle
    status: Status = Status.UNMEASURED
    history: History | None = None
    job: JobResult | None = None
    phase: Phase | None = None
    completed: int = 0
    total: int = 1
    error: str | None = None
    visibility: Visibility | None = None
    source_pending: bool = False
    dependencies: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Snapshot:
    revision: int
    states: tuple[FormulaState, ...]
    ready: bool
    closed: bool
    error: str | None
    characterization: Characterization | None = None
    characterizing: bool = False
    source_error: str | None = None


@dataclass(frozen=True, slots=True)
class Ticket:
    identity: str
    target: FormulaHandle
    result: Future[JobResult]
    _cancel: Event

    def cancel(self) -> None:
        # Future.cancel() would discard the required drained/persisted outcome.
        self._cancel.set()


@dataclass(frozen=True, slots=True)
class _Measure:
    ticket: Ticket
    requested: datetime
    started_ns: int


@dataclass(frozen=True, slots=True)
class _Invalidate:
    targets: tuple[FormulaHandle, ...]
    result: Future[tuple[FormulaHandle, ...]]


@dataclass(frozen=True, slots=True)
class _Inspect:
    target: FormulaHandle
    result: Future[History | None]


@dataclass(frozen=True, slots=True)
class _Characterize:
    refresh: bool
    result: Future[Characterization]
    cancel: Event


class Lab:
    """One worker, runtime, fixture cache and store for API and TUI clients.

    The device factory runs on the worker. Passing a runtime created on the UI
    thread is deliberately unsupported: live ownership must not move threads.
    Fixture construction is explicit and happens once, not once per sample.
    """

    def __init__(
        self, *, fixture: Fixture, device: Callable[[], DeviceRuntime], store: Path,
        options: CompileOptions, protocol: MeasurementProtocol = MeasurementProtocol(),
        prepared_limit: int = 8, reference_bytes: int = 256 << 20,
        label: str = "Prepared configuration",
    ):
        if any(type(value) is not int or value < 1 for value in (prepared_limit, reference_bytes)):
            raise ValueError("preparation count and reference byte limits must be positive integers")
        self.formulas = FormulaTree(fixture.root)
        self._fixture, self._device_factory, self._path = fixture, device, store
        self._options, self._protocol, self._prepared_limit = options, protocol, prepared_limit
        self._reference_bytes = reference_bytes
        self._label = label
        self._recorded = None
        self._lock = Lock()
        self._queue: Queue[_Measure | _Invalidate | _Inspect | _Characterize | Visibility | None] = Queue()
        self._profile_request: _Characterize | None = None
        self._characterization: Characterization | None = None
        self._source_error = None
        self._states = {target: FormulaState(target) for target in self.formulas}
        self._pending: dict[FormulaHandle, Ticket] = {}
        self._revision = 0
        self._accepting = True
        self._closed = False
        self._error: str | None = None
        self._identity = str(uuid4())
        self._deliveries: dict[str, tuple[int, int]] = {}
        self.ready: Future[FormulaTree] = Future()
        self._thread = Thread(target=self._run, name="formula-lab", daemon=False)
        self._thread.start()

    def snapshot(self) -> Snapshot:
        with self._lock:
            return Snapshot(self._revision, tuple(self._states.values()),
                            self.ready.done() and self._error is None, self._closed, self._error,
                            self._characterization, self._profile_request is not None, self._source_error)

    def _poll_sources(self, runner):
        try:
            pending = set(runner.pending_sources())
            error = None
        except OSError as failure:
            # A file can be temporarily absent during an editor's atomic save.
            # Keep evidence; the explicit request still requires successful refresh.
            pending = None
            error = f"Source scan unavailable: {failure}"
        with self._lock:
            changed = error != self._source_error
            self._source_error = error
            if pending is not None:
                for target, state in self._states.items():
                    source_pending = target in pending
                    if state.source_pending != source_pending:
                        self._states[target] = replace(state, source_pending=source_pending)
                        changed = True
            if changed:
                self._revision += 1

    def characterize(self, *, refresh: bool = False) -> Future[Characterization]:
        """Explicitly load matching device evidence, or measure it if absent."""
        with self._lock:
            if not self._accepting:
                raise RuntimeError(self._error or "Lab is closing")
            if self._profile_request is not None:
                return self._profile_request.result
            result: Future[Characterization] = Future()
            result.set_running_or_notify_cancel()
            request = _Characterize(refresh, result, Event())
            self._profile_request = request
            self._queue.put(request)
            self._revision += 1
            return result

    def _check_target(self, target: FormulaHandle) -> None:
        if not isinstance(target, FormulaHandle) or target not in self._states:
            raise ValueError("Lab requests require a typed occurrence from this fixture")

    def measure(self, target: FormulaHandle) -> Ticket:
        self._check_target(target)
        with self._lock:
            if not self._accepting:
                raise RuntimeError(self._error or "Lab is closing")
            if target in self._pending:
                return self._pending[target]
            ticket = Ticket(str(uuid4()), target, Future(), Event())
            ticket.result.set_running_or_notify_cancel()
            self._pending[target] = ticket
            self._states[target] = replace(self._states[target], status=Status.QUEUED,
                                           phase=None, error=None)
            self._revision += 1
            self._queue.put(_Measure(ticket, datetime.now(UTC), perf_counter_ns()))
            return ticket

    def measure_affected(self, target: FormulaHandle) -> tuple[Ticket, ...]:
        """Measure the explicitly displayed dependency scope, not an implicit sweep."""
        self._check_target(target)
        return tuple(self.measure(item) for item in self.formulas.affected((target,)))

    def subtree(self, target: FormulaHandle) -> tuple[FormulaHandle, ...]:
        """Typed parent-first scope; each entry is an independent measurement."""
        self._check_target(target)
        selected = {target}
        for item in self.formulas:
            if item.parent in selected:
                selected.add(item)
        return tuple(item for item in self.formulas if item in selected and item.call.complete)

    def measure_subtree(self, target: FormulaHandle) -> tuple[Ticket, ...]:
        return tuple(self.measure(item) for item in self.subtree(target))

    def invalidate(self, changed: tuple[FormulaHandle, ...]) -> Future[tuple[FormulaHandle, ...]]:
        for target in changed:
            self._check_target(target)
        result: Future[tuple[FormulaHandle, ...]] = Future()
        result.set_running_or_notify_cancel()
        with self._lock:
            if not self._accepting:
                raise RuntimeError(self._error or "Lab is closing")
            self._queue.put(_Invalidate(changed, result))
        return result

    def inspect(self, target: FormulaHandle) -> Future[History | None]:
        """Load a selected boundary's history without compilation or measurement."""
        self._check_target(target)
        result: Future[History | None] = Future()
        result.set_running_or_notify_cancel()
        with self._lock:
            if not self._accepting:
                raise RuntimeError(self._error or "Lab is closing")
            self._queue.put(_Inspect(target, result))
        return result

    def cancel(self, target: FormulaHandle | None = None) -> None:
        with self._lock:
            if target is None and self._profile_request is not None:
                self._profile_request.cancel.set()
            if target is not None:
                self._check_target(target)
            for handle, ticket in self._pending.items():
                if target is None or handle == target:
                    ticket.cancel()

    def acknowledge(self, result: JobResult, *, client: str) -> Visibility:
        """Called after the client has actually displayed/consumed this result."""
        visible = perf_counter_ns()
        with self._lock:
            if not self._accepting:
                raise RuntimeError("Lab is closing")
            interval = self._deliveries.get(result.identity)
            if interval is None:
                raise ValueError("job has no recent delivery from this worker")
            started, completed = interval
            visibility = Visibility(job=result.identity, client=client,
                                    request_to_visible_ns=visible - started,
                                    completed_to_visible_ns=visible - completed)
            self._queue.put(visibility)
        return visibility

    def close(self, *, wait: bool = True) -> None:
        with self._lock:
            if self._accepting:
                self._accepting = False
                for ticket in self._pending.values():
                    ticket.cancel()
                if self._profile_request is not None:
                    self._profile_request.cancel.set()
                self._queue.put(None)
        if wait and current_thread() is not self._thread:
            self._thread.join()

    def show(self) -> None:
        from .tui import PerformanceApp

        PerformanceApp(self).run()

    def __enter__(self) -> Lab:
        self.ready.result()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _update(self, target, **changes):
        with self._lock:
            self._states[target] = replace(self._states[target], **changes)
            self._revision += 1

    def _measure(self, request: _Measure, runner: MeasurementRunner, store: ObservationStore):
        target, ticket = request.ticket.target, request.ticket
        started = perf_counter_ns()
        store.begin_job(JobStart(identity=ticket.identity, worker=self._identity,
                                 requested=request.requested, formula=target.definition,
                                 semantics=target.semantic_identity))
        measurement = None
        publication_ns = 0
        phases = ()
        outcome, error = Outcome.CANCELLED, "measurement cancelled before preparation"
        if not ticket._cancel.is_set():
            self._update(target, status=Status.RUNNING)
            try:
                run, refresh_time = self._execute_request(ticket, runner)
                measurement = run.measurement
                publication_ns = run.publish_ns
                phases = (refresh_time, *measurement.phases)
                outcome, error = measurement.outcome, measurement.error
            except Exception as failure:
                outcome, error = Outcome.FAILED, f"{type(failure).__name__}: {failure}"
                # Failure before Series construction is a real job failure, not
                # a fabricated measurement. A failed drain is fatal to this owner.
                runner.device.drain()
        finished = perf_counter_ns()
        result = JobResult(
            identity=ticket.identity, requested=request.requested, finished=datetime.now(UTC),
            formula=target.definition, semantics=target.semantic_identity, outcome=outcome,
            measurement=measurement.identity if measurement is not None else None,
            queue_ns=started - request.started_ns, active_ns=finished - started,
            publication_ns=publication_ns, phases=phases, error=error,
        )
        store.publish_job(result)
        changes = dict(status={Outcome.COMPLETE: Status.CURRENT, Outcome.FAILED: Status.FAILED,
                               Outcome.CANCELLED: Status.CANCELLED}[outcome],
                       job=result, phase=None, error=error, visibility=None)
        if measurement is not None:
            store.link_series(self._recorded, target.call.occurrence, measurement.series)
            changes["history"] = store.history(measurement.series)
        with self._lock:
            self._pending.pop(target, None)
            self._deliveries[ticket.identity] = request.started_ns, perf_counter_ns()
            while len(self._deliveries) > 256:
                del self._deliveries[next(iter(self._deliveries))]
            self._states[target] = replace(self._states[target], **changes)
            self._revision += 1
        ticket.result.set_result(result)

    def _execute_request(self, ticket: Ticket, runner: MeasurementRunner):
        # Lock contention is a failed request, not worker death or an unreported
        # wait. The lock covers Python refresh as well as device measurement.
        with exclusive_measurement():
            target = ticket.target
            self._update(target, phase=Phase.REFRESH, completed=0, total=1)
            refresh_started = perf_counter_ns()
            affected = runner.refresh()
            self._poll_sources(runner)
            refresh_time = PhaseTime(phase=Phase.REFRESH, elapsed_ns=perf_counter_ns() - refresh_started)
            for affected_target in affected:
                if affected_target != target:
                    with self._lock:
                        state = self._states[affected_target]
                        self._states[affected_target] = replace(
                            state, status=Status.QUEUED if affected_target in self._pending else Status.STALE,
                        )
                        self._revision += 1
            run = runner.measure(
                target, cancellation=ticket._cancel,
                progress=lambda phase, current, total: self._update(
                    target, phase=phase, completed=current, total=total,
                ),
            )
            with self._lock:
                self._characterization = runner.device.characterization
                self._revision += 1
            return run, refresh_time

    def _run(self):
        device = runner = store = None
        request = None
        try:
            device = self._device_factory()
            device.check()
            store = ObservationStore(self._path)
            from .archive import RecordedConfiguration

            self._recorded = RecordedConfiguration.capture(self.formulas, label=self._label,
                                                           device=device.evidence_identity)
            store.publish_configuration(self._recorded)
            dependencies = {item.occurrence: item.dependencies for item in self._recorded.formulas}
            with self._lock:
                self._states = {target: replace(state, dependencies=dependencies[target.call.occurrence])
                                for target, state in self._states.items()}
            runner = MeasurementRunner(self._fixture, device, store, self._options,
                                       protocol=self._protocol, prepared_limit=self._prepared_limit,
                                       reference_bytes=self._reference_bytes)
            self.ready.set_result(self.formulas)
            with self._lock:
                self._revision += 1
            while True:
                try:
                    request = self._queue.get(timeout=1)
                except Empty:
                    self._poll_sources(runner)
                    continue
                if request is None:
                    break
                if isinstance(request, _Measure):
                    self._measure(request, runner, store)
                elif isinstance(request, _Characterize):
                    try:
                        with exclusive_measurement():
                            evidence = device.characterize(store, refresh=request.refresh,
                                                           cancellation=request.cancel)
                        with self._lock:
                            self._characterization = evidence
                            self._profile_request = None
                            self._revision += 1
                        request.result.set_result(evidence)
                    except Exception as failure:
                        device.drain()
                        with self._lock:
                            self._profile_request = None
                            self._revision += 1
                        request.result.set_exception(failure)
                    finally:
                        with self._lock:
                            if self._profile_request is request:
                                self._profile_request = None
                            self._revision += 1
                elif isinstance(request, Visibility):
                    store.publish_visibility(request)
                    with self._lock:
                        for target, state in self._states.items():
                            if state.job is not None and state.job.identity == request.job:
                                self._states[target] = replace(state, visibility=request)
                                self._revision += 1
                elif isinstance(request, _Inspect):
                    try:
                        # Reading history must not derive reference inputs. A new
                        # live configuration establishes comparability on its first
                        # actual measurement; archived configurations remain browsable.
                        history = store.recorded_history(self._recorded, request.target.call.occurrence)
                        with self._lock:
                            state = self._states[request.target]
                            self._states[request.target] = replace(
                                state, history=history,
                                status=(Status.RECORDED if history is not None and history.latest is not None and
                                        state.status == Status.UNMEASURED else state.status),
                            )
                            self._revision += 1
                        request.result.set_result(history)
                    except Exception as failure:
                        request.result.set_exception(failure)
                else:
                    affected = runner.invalidate(request.targets)
                    for target in affected:
                        with self._lock:
                            state = self._states[target]
                            self._states[target] = replace(
                                state, status=Status.QUEUED if target in self._pending else Status.STALE,
                            )
                            self._revision += 1
                    request.result.set_result(affected)
        except BaseException as failure:
            with self._lock:
                self._error = f"{type(failure).__name__}: {failure}"
                self._accepting = False
                self._revision += 1
                pending = tuple(self._pending.values())
                self._pending.clear()
            if not self.ready.done():
                self.ready.set_exception(failure)
            for ticket in pending:
                if not ticket.result.done():
                    ticket.result.set_exception(failure)
                self._update(ticket.target, status=Status.FAILED, phase=None, error=str(failure))
            if isinstance(request, (_Invalidate, _Inspect, _Characterize)) and not request.result.done():
                request.result.set_exception(failure)
            # Invalidation callers must not hang after owner failure either.
            while not self._queue.empty():
                request = self._queue.get_nowait()
                if isinstance(request, (_Invalidate, _Inspect, _Characterize)):
                    request.result.set_exception(failure)
        finally:
            try:
                if runner is not None:
                    runner.close()
                if device is not None:
                    device.close()
            except BaseException as failure:
                with self._lock:
                    self._error = f"device cleanup failed: {type(failure).__name__}: {failure}"
            finally:
                if store is not None:
                    store.close()
                with self._lock:
                    self._closed = True
                    self._profile_request = None
                    self._revision += 1
