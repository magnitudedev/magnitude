"""Single-owner request orchestration over an injected generation runtime."""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter_ns
from uuid import uuid4

from magnitude_engine.components import component
from magnitude_engine.generation.constraint_spec import ConstraintError
from magnitude_engine.generation.runtime import (
    GenerationRuntime,
    GenerationSequence,
)
from magnitude_engine.models.context import StateCheckpoint
from magnitude_engine.models.prompt import Prompt

from .delivery import Delivery, Finished, PrefillProgress
from .prefixes.contracts import PrefixIndex
from .prefixes.index import PrefixIdentity
from .requests import GenerationRequest, RequestHandle
from .scheduler.contracts import CompletedService, Runnable, Scheduler, Service


@dataclass
class ActiveRequest[S, C: StateCheckpoint]:
    handle: RequestHandle
    sequence: GenerationSequence[S, C]
    cached: int
    admitted_ns: int
    first_token_ns: int | None = None
    proposed: int = 0
    accepted: int = 0
    forced: int = 0
    prefill_ns: int = 0
    decode_ns: int = 0
    first_decode_ns: int = 0
    preparation_ns: int = 0


@dataclass(frozen=True)
class ServiceMeasurement:
    request_id: str
    phase: str
    input_tokens: int
    output_tokens: int
    elapsed_ns: int
    batch_size: int = 1


@component("ENGINE:INFERENCE:MAG:STANDARD")
class Engine[S, C: StateCheckpoint]:
    def __init__(
        self,
        generation: GenerationRuntime[S, C],
        *,
        namespace: bytes,
        scheduler: Scheduler,
        prefixes: PrefixIndex,
        clock: Callable[[], int] = perf_counter_ns,
    ):
        if not namespace:
            raise ValueError("engine needs a compatibility namespace")
        self.generation, self.namespace = generation, namespace
        self.scheduler = scheduler
        self.prefixes = prefixes
        self.clock = clock
        self.wake = Event()
        self._lock = Lock()
        self._queued: deque[RequestHandle] = deque()
        self._identities: set[str] = set()
        self._active: dict[str, ActiveRequest[S, C]] = {}
        self._closed = False
        self._failed = False
        self.last_service: CompletedService | None = None

    @component("SCHEDULING:ADMISSION:MAG:FIFO")
    def submit(
        self,
        request: GenerationRequest,
        *,
        identity: str | None = None,
        output_capacity: int = 64,
        progress: bool = False,
    ) -> RequestHandle:
        """Control threads may submit/cancel/read; they never execute model work."""
        identity = identity or uuid4().hex
        delivery = Delivery(output_capacity, self.wake.set, progress=progress)
        with self._lock:
            if self._closed or self._failed:
                raise RuntimeError("engine is unavailable")
            if identity in self._identities:
                raise ValueError("request identity is already live")
            if len(self._queued) >= self.scheduler.max_queued:
                raise OverflowError("request admission queue is full")
            handle = RequestHandle(identity, request, delivery, self.clock(), self.wake.set)
            self._queued.append(handle)
            self._identities.add(identity)
        self.wake.set()
        return handle

    def _identity(self, prompt: Prompt) -> PrefixIdentity:
        return PrefixIdentity(self.namespace, prompt.identities())

    def _terminal(
        self,
        handle: RequestHandle,
        reason: str,
        row: ActiveRequest[S, C] | None = None,
        message: str = "",
    ) -> None:
        now = self.clock()
        handle.delivery.terminate(
            Finished(
                reason,
                len(handle.request.prompt.tokens),
                0 if row is None else row.sequence.generated,
                0 if row is None else row.cached,
                0 if row is None else row.proposed,
                0 if row is None else row.accepted,
                (now if row is None else row.admitted_ns) - handle.created_ns,
                None
                if row is None or row.first_token_ns is None
                else row.first_token_ns - handle.created_ns,
                now - handle.created_ns,
                message,
                0 if row is None else row.forced,
                0 if row is None else row.prefill_ns,
                0 if row is None else row.decode_ns,
                0 if row is None else row.first_decode_ns,
                0 if row is None else row.preparation_ns,
                sum(
                    span.end - span.start
                    for span in handle.request.prompt.spans
                    if not span.language
                ),
                0
                if row is None or row.sequence.model.inputs is None
                else row.sequence.model.inputs.cache_hits,
            )
        )
        with self._lock:
            self._identities.remove(handle.identity)

    def _finish(self, row: ActiveRequest[S, C], reason: str, message: str = "") -> None:
        row.sequence.close()
        del self._active[row.handle.identity]
        self._terminal(row.handle, reason, row, message)

    def _admit(self) -> None:
        while len(self._active) < self.scheduler.max_active:
            with self._lock:
                if not self._queued:
                    return
                handle = self._queued[0]
                self._queued.popleft()
            if handle.cancelled.is_set():
                self._terminal(handle, "cancelled")
                continue
            if handle.request.max_tokens == 0:
                self._terminal(handle, "length")
                continue
            self.generation.model.owner.complete()
            prompt = handle.request.prompt
            lease = self.prefixes.match(
                self._identity(prompt), exclude_last=len(prompt.tokens) - prompt.anchor_start
            )
            sequence = None
            try:
                checkpoint = None if lease is None else lease.checkpoint
                request = handle.request
                sequence = self.generation.prepare(
                    request.prompt,
                    request.sampling,
                    request.max_tokens,
                    request.stop_tokens,
                    checkpoint,
                    constraint=request.constraint,
                    inputs=request.inputs,
                )
                sequence.reserve_prompt()
                self._active[handle.identity] = ActiveRequest(
                    handle, sequence, 0 if checkpoint is None else checkpoint.length, self.clock()
                )
                self._progress(self._active[handle.identity])
            except MemoryError as error:
                if sequence is not None:
                    sequence.close()
                if self._active:
                    # Capacity pressure delays the oldest request; later arrivals
                    # cannot pass it. A lone request that cannot fit must terminate.
                    with self._lock:
                        self._queued.appendleft(handle)
                    return
                self._terminal(handle, "error", message=str(error))
            except ConstraintError as error:
                self._terminal(handle, "error", message=str(error))
            except BaseException:
                # Keep the popped request reachable by fatal cleanup.
                with self._lock:
                    self._queued.appendleft(handle)
                if sequence is not None:
                    sequence.close()
                raise
            finally:
                if lease is not None:
                    lease.close()

    def _retain(self, sequence: GenerationSequence[S, C]) -> None:
        if not self.prefixes.enabled:
            return
        try:
            checkpoint = sequence.checkpoint()
        except MemoryError:
            # Retention is optional work; a live request does not fail for a cache miss.
            return
        if not checkpoint.length:
            checkpoint.close()
            return
        try:
            self.prefixes.retain(
                PrefixIdentity(self.namespace, checkpoint.prompt.identities()), checkpoint
            )
        except BaseException:
            checkpoint.close()
            raise

    def _progress(self, row: ActiveRequest[S, C]) -> None:
        if row.handle.delivery.progress_enabled:
            total = row.handle.request.prompt.anchor_start
            row.handle.delivery.report(
                PrefillProgress(
                    total - row.sequence.prefill_remaining,
                    total,
                    row.cached,
                    row.prefill_ns,
                )
            )

    def tick(self) -> tuple[ServiceMeasurement, ...]:
        self.generation.model.owner.check()
        if self._closed or self._failed:
            raise RuntimeError("engine is unavailable")
        self.wake.clear()
        self.last_service = None
        try:
            self.prefixes.maintain()
            with self._lock:
                terminal = tuple(
                    handle
                    for handle in self._queued
                    if handle.cancelled.is_set() or handle.request.max_tokens == 0
                )
                self._queued = deque(handle for handle in self._queued if handle not in terminal)
            for handle in terminal:
                self._terminal(handle, "cancelled" if handle.cancelled.is_set() else "length")
            for row in tuple(self._active.values()):
                if row.handle.cancelled.is_set():
                    self._finish(row, "cancelled")
            self._admit()
            prompt_groups = self.generation.prefill_groups(
                tuple(
                    row.sequence
                    for row in self._active.values()
                    if row.sequence.prefill_remaining and row.handle.delivery.credit
                )
            )
            membership = {
                sequence: index for index, group in enumerate(prompt_groups) for sequence in group
            }
            facts = tuple(
                Runnable(
                    row.handle.identity,
                    row.sequence.prefill_remaining,
                    row.handle.delivery.credit,
                    membership.get(row.sequence),
                )
                for row in self._active.values()
            )
            plan = self.scheduler.select(facts)
            if plan is None:
                return ()
            if plan.phase == "decode":
                return self._decode(plan.services, plan.budget_ns)
            return self._prefill(plan.services, plan.budget_ns)
        except BaseException as error:
            with self._lock:
                self._failed = True
            try:
                self._abort_all("error", str(error))
            except BaseException as cleanup:
                raise BaseExceptionGroup(
                    "engine failure and cleanup failure", [error, cleanup]
                ) from error
            raise

    def _prefill(
        self, services: tuple[Service, ...], budget_ns: int | None = None
    ) -> tuple[ServiceMeasurement, ...]:
        scheduled = []
        for service in services:
            row = self._active[service.identity]
            if row.handle.cancelled.is_set():
                self._finish(row, "cancelled")
            else:
                scheduled.append((row, service))
        start = self.clock()
        outcomes = self.generation.prefill_many(
            tuple(row.sequence for row, _ in scheduled),
            tuple(
                self.prefixes.prefill_allowance(
                    row.sequence.prefilled,
                    service.tokens,
                    row.sequence.prompt.retention_boundaries,
                )
                for row, service in scheduled
            ),
            clock=self.clock,
            budget_ns=budget_ns,
        )
        self._observe(
            CompletedService(
                "prefill",
                max(0, self.clock() - start) if scheduled else 0,
                sum(result.outcome for result in outcomes if isinstance(result.outcome, int)),
                max((result.preparation_ns for result in outcomes), default=0),
            )
        )
        measurements = []
        for (row, service), measured in zip(scheduled, outcomes, strict=True):
            result = measured.outcome
            row.prefill_ns += measured.elapsed_ns
            row.preparation_ns += measured.row_preparation_ns
            if isinstance(result, (MemoryError, ConstraintError)):
                self._finish(row, "error", str(result))
                continue
            measurements.append(
                ServiceMeasurement(
                    service.identity,
                    "prefill",
                    0 if result is None else result,
                    0,
                    measured.elapsed_ns,
                    measured.batch_size,
                )
            )
            self._progress(row)
            if isinstance(result, int) and (
                not row.sequence.prefill_remaining
                or row.sequence.prefilled in row.sequence.prompt.retention_boundaries
            ):
                self._retain(row.sequence)
        return tuple(measurements)

    def _decode(
        self, services: tuple[Service, ...], budget_ns: int | None
    ) -> tuple[ServiceMeasurement, ...]:
        scheduled = []
        for service in services:
            row = self._active[service.identity]
            if row.handle.cancelled.is_set():
                self._finish(row, "cancelled")
            else:
                scheduled.append((row, service))
        start = self.clock()
        outcomes = self.generation.step_many(
            tuple(row.sequence for row, _ in scheduled),
            # Publish the first token on its own. Causal execution may already
            # have submitted one successor, but it does not wait for that result.
            tuple(
                1 if row.first_token_ns is None else service.tokens for row, service in scheduled
            ),
            clock=self.clock,
            budget_ns=budget_ns,
        )
        # Per-row metrics include shared batch time for each participant. Fairness
        # accounts for the completed round once, before delivery and prefix retention.
        self._observe(CompletedService("decode", max(0, self.clock() - start) if scheduled else 0))
        measurements = []
        for (row, service), measured in zip(scheduled, outcomes, strict=True):
            result = measured.outcome
            row.decode_ns += measured.elapsed_ns
            if row.first_token_ns is None:
                row.first_decode_ns += measured.elapsed_ns
            if result is None:
                measurements.append(
                    ServiceMeasurement(
                        service.identity,
                        "decode",
                        0,
                        0,
                        measured.elapsed_ns,
                        measured.batch_size,
                    )
                )
                continue
            if isinstance(result, (MemoryError, ConstraintError)):
                self._finish(row, "error", str(result))
                continue
            now = self.clock()
            if row.first_token_ns is None:
                row.first_token_ns = now
            row.proposed += result.proposed
            row.accepted += result.accepted
            row.forced += result.forced
            row.handle.delivery.publish(result.tokens)
            measurements.append(
                ServiceMeasurement(
                    service.identity,
                    "forced" if result.forced else "decode",
                    result.evaluated_inputs,
                    len(result.tokens),
                    measured.elapsed_ns,
                    measured.batch_size,
                )
            )
            if result.finish_reason:
                self._retain(row.sequence)
                self._finish(row, result.finish_reason)
        return tuple(measurements)

    def _observe(self, service: CompletedService) -> None:
        self.scheduler.observe(service)
        self.last_service = service

    def _abort_all(self, reason: str, message: str = "") -> None:
        failures = []
        for row in tuple(self._active.values()):
            try:
                self._finish(row, reason, message)
            except BaseException as error:
                failures.append(error)
                if row.handle.delivery.finish is None:
                    self._terminal(row.handle, "error", row, f"cleanup failed: {error}")
        with self._lock:
            queued = tuple(self._queued)
            self._queued.clear()
        for handle in queued:
            self._terminal(handle, reason, message=message)
        if failures:
            raise BaseExceptionGroup("request cleanup failed; worker disposal required", failures)

    def close(self) -> None:
        self.generation.model.owner.check()
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._abort_all("shutdown")
        finally:
            self.prefixes.close()
