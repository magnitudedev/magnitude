"""One serialized request owner and one physical generation batch in flight.

Transport drives step, waits on its concrete ticket, and drains output. Pressure
uses model-owned reclamation facts; this layer knows no numerical state layout.
"""

from dataclasses import dataclass
from enum import StrEnum

from magnitude_engine.data import Record
from magnitude_engine.generation.plain import (
    FinishReason,
    Generation,
    GenerationBatch,
    Options,
    OutputToken,
    Ready,
    WaitReason,
    WorkKind,
)
from magnitude_engine.models.sequence import ModelExecutor, ModelInput
from magnitude_engine.operations.sampling import SampleSelector
from magnitude_engine.platform.execution import CapacityError, Ticket
from magnitude_engine.platform.host.measurement import clock_ns
from magnitude_engine.service.policy import (
    Candidate,
    Limits,
    Phase,
    RequestId,
    Scheduler,
    Selection,
)


class Status(StrEnum):
    QUEUED = "queued"
    RUNNABLE = "runnable"
    COMPLETION = "completion"
    OUTPUT = "output"
    PREEMPTED = "preempted"
    CAPACITY = "capacity"
    TERMINAL = "terminal"


class FailureKind(StrEnum):
    CAPACITY = "capacity"
    PREPARATION = "preparation"
    EXECUTION = "execution"


class Failure(Record):
    kind: FailureKind
    message: str
    required_bytes: int | None = None
    available_bytes: int | None = None


class CapacityWait(Record):
    required_bytes: int
    available_bytes: int
    peers: tuple[RequestId, ...]


class Snapshot(Record):
    identity: RequestId
    status: Status
    resident: bool
    waiting_ns: int
    processed: int
    reconstruction_position: int | None
    queued_output: int
    finish: FinishReason | None
    failure: Failure | None
    service_ns: int
    preemptions: int
    generated_tokens: int
    prefill_ns: int
    decode_ns: int
    replay_ns: int
    capacity_wait: CapacityWait | None


@dataclass
class Request:
    identity: RequestId
    source: ModelInput
    options: Options
    waiting_since_ns: int
    generation: Generation | None = None
    finish: FinishReason | None = None
    failure: Failure | None = None
    service_ns: int = 0
    prefill_ns: int = 0
    decode_ns: int = 0
    replay_ns: int = 0
    preemptions: int = 0
    preemption_debt: int = 0
    protected_position: int | None = None
    blocked_epoch: int | None = None
    source_closed: bool = False
    capacity_wait: CapacityWait | None = None

    def retire_source(self) -> None:
        if not self.source_closed:
            self.source.close()
            self.source_closed = True


@dataclass(frozen=True)
class Submission:
    completion: Ticket
    requests: tuple[RequestId, ...]
    phase: Phase
    tokens: int


@dataclass(frozen=True)
class Idle:
    requests: tuple[Snapshot, ...]


@dataclass
class Pending:
    batch: GenerationBatch
    selection: Selection
    started_ns: int
    submission: Submission


class _NotPrepared(Exception):
    """The selected request has a recorded terminal or changing wait condition."""


class Engine:
    def __init__(
        self, model: ModelExecutor, selector: SampleSelector, limits: Limits | None = None
    ):
        if model.context is not selector.context:
            raise ValueError("service model and selector must share an execution owner")
        self.model, self.selector, self.context = model, selector, model.context
        self.scheduler = Scheduler(Limits() if limits is None else limits)
        self.requests: dict[RequestId, Request] = {}
        self.pending: Pending | None = None
        self._next = 0
        self._epoch = 0
        self.closed = False

    def check(self) -> None:
        self.context.check_thread()
        if self.closed:
            raise RuntimeError("service is closed")

    def admit(self, source: ModelInput, options: Options) -> RequestId:
        """Transfer the source on success; state allocation waits for service."""
        self.check()
        self.context.check()
        if source.model is not self.model or not source.prompt:
            raise ValueError("request source must belong to this model and have input")
        if len(self.requests) >= self.scheduler.limits.max_requests:
            raise ValueError("service request limit reached; retire completed requests")
        identity = RequestId(self._next)
        self._next += 1
        request = Request(identity, source, options, clock_ns())
        self.requests[identity] = request
        if options.max_tokens == 0:
            request.finish = FinishReason.LENGTH
            request.retire_source()
        self._epoch += 1
        return identity

    def snapshot(self, identity: RequestId) -> Snapshot:
        self.check()
        r = self.requests[identity]
        g = r.generation
        if r.finish is not None:
            status = Status.TERMINAL
        elif g is not None and g.pending is not None:
            status = Status.COMPLETION
        elif g is not None and len(g.output) >= g.options.output_capacity:
            status = Status.OUTPUT
        elif r.blocked_epoch == self._epoch:
            status = Status.CAPACITY
        elif g is None:
            status = Status.QUEUED
        elif not g.resident:
            status = Status.PREEMPTED
        else:
            status = Status.RUNNABLE
        return Snapshot(
            identity=identity,
            status=status,
            resident=g is not None and g.resident,
            waiting_ns=max(0, clock_ns() - r.waiting_since_ns),
            processed=0 if g is None else g.accepted_position,
            reconstruction_position=g.processed if g is not None and g.rebuilding else None,
            queued_output=0 if g is None else len(g.output),
            finish=r.finish,
            failure=r.failure,
            service_ns=r.service_ns,
            preemptions=r.preemptions,
            generated_tokens=0 if g is None else len(g.sampled),
            prefill_ns=r.prefill_ns,
            decode_ns=r.decode_ns,
            replay_ns=r.replay_ns,
            capacity_wait=r.capacity_wait if status == Status.CAPACITY else None,
        )

    def take(self, identity: RequestId, count: int) -> tuple[OutputToken, ...]:
        self.check()
        if type(count) is not int or count <= 0:
            raise ValueError("output collection count must be positive")
        g = self.requests[identity].generation
        output = () if g is None else g.take(count)
        if output:
            self._epoch += 1
        return output

    def cancel(self, identity: RequestId) -> None:
        self.check()
        r = self.requests[identity]
        if r.finish is not None:
            return
        if r.generation is not None:
            r.generation.cancel()
        r.finish = FinishReason.CANCELLED
        r.retire_source()
        self._epoch += 1

    def remove(self, identity: RequestId) -> None:
        self.check()
        r = self.requests[identity]
        if r.finish is None or (r.generation is not None and r.generation.output):
            raise RuntimeError("retire only terminal requests with drained output")
        if self.pending is not None and identity in self.pending.submission.requests:
            raise RuntimeError("request still belongs to a submitted batch")
        if r.generation is not None:
            r.generation.close()
        r.retire_source()
        del self.requests[identity]
        self._epoch += 1

    def _candidate(self, r: Request) -> Candidate | None:
        if r.finish is not None or r.blocked_epoch == self._epoch:
            return None
        g = r.generation
        if g is None:
            phase = Phase.PREFILL
        else:
            ready = g.ready(self.scheduler.limits.prefill_tokens)
            if ready in (WaitReason.OUTPUT, WaitReason.COMPLETION, WaitReason.FINISHED):
                return None
            phase = (
                Phase.DECODE
                if isinstance(ready, Ready) and ready.kind == WorkKind.DECODE
                else Phase.PREFILL
            )
        return Candidate(
            r.identity,
            phase,
            g is not None and bool(g.processed or g.sampled),
            g is not None and g.resident,
            r.waiting_since_ns,
            r.service_ns,
            r.preemption_debt,
        )

    def _open(self, r: Request) -> Generation:
        if r.generation is None or not r.generation.resident:
            sequence = r.source.open()
            try:
                if r.generation is None:
                    r.generation = Generation(sequence, r.source.prompt, self.selector, r.options)
                else:
                    r.generation.restore(sequence)
            except BaseException:
                sequence.close()
                raise
        return r.generation

    def _fail(self, r: Request, kind: FailureKind, error: Exception) -> None:
        if r.finish == FinishReason.CANCELLED:
            return
        if r.generation is not None:
            r.generation.fail()
        r.finish = FinishReason.FAILED
        r.failure = Failure(
            kind=kind,
            message=str(error),
            required_bytes=error.required if isinstance(error, CapacityError) else None,
            available_bytes=error.available if isinstance(error, CapacityError) else None,
        )
        r.retire_source()
        self._epoch += 1

    def _evict(self, selected: set[RequestId], error: CapacityError) -> bool:
        candidates = []
        for r in self.requests.values():
            g = r.generation
            if (
                r.identity in selected
                or r.finish is not None
                or g is None
                or not g.resident
                or g.pending is not None
                or r.protected_position is not None
            ):
                continue
            exclusive = self.model.reclaimable((g.sequence,))
            # Actual exclusive bytes per replayed input, with output-blocked
            # requests first and preemption debt protecting repeated victims.
            candidates.append(
                (
                    len(g.output) >= g.options.output_capacity,
                    -r.preemption_debt,
                    exclusive / max(1, g.processed),
                    -r.service_ns,
                    r,
                )
            )
        candidates.sort(key=lambda item: item[:-1], reverse=True)
        victims = []
        for *_, r in candidates:
            victims.append(r)
            sequences = tuple(v.generation.sequence for v in victims if v.generation is not None)
            if self.model.reclaimable(sequences) >= max(1, error.required - error.available):
                break
        if not victims:
            return False
        sequences = tuple(v.generation.sequence for v in victims if v.generation is not None)
        if self.model.reclaimable(sequences) == 0:
            return False
        before = self.context.allocated_bytes
        for r in victims:
            assert r.generation is not None
            r.protected_position = r.generation.recovery().processed
            r.generation.evict()
            r.preemptions += 1
            r.preemption_debt += 1
        self.model.reclaim()
        # No detached full checkpoint is retained. Pins/sharing are accounted by
        # the model's ownership query and actual budget before/after release.
        return self.context.allocated_bytes < before

    def _prepare(self, selection: Selection) -> tuple[GenerationBatch, Selection, int]:
        identities = list(selection.requests)
        allowance = 1 if selection.phase == Phase.DECODE else self.scheduler.limits.prefill_tokens
        reclaimed = False
        last_shape = None
        capacity: CapacityError | None = None
        while True:
            ready = []
            remaining = allowance
            current = identities[0]
            try:
                for current in identities:
                    if selection.phase == Phase.PREFILL and remaining <= 0:
                        break
                    g = self._open(self.requests[current])
                    proposal = g.ready(max(1, remaining) if selection.phase == Phase.PREFILL else 1)
                    if not isinstance(proposal, Ready):
                        raise RuntimeError("selected request lost eligibility")
                    ready.append(proposal)
                    remaining -= len(proposal.tokens)
                shape = tuple(len(r.tokens) for r in ready)
                # A soft allowance may not split an indivisible input span. If
                # shrinking would repeat the failed physical shape, evict instead.
                if last_shape == shape:
                    assert capacity is not None
                    raise capacity
                batch = GenerationBatch.prepare(tuple(ready))
                served = tuple(identities[: len(ready)])
                return batch, Selection(selection.phase, served, selection.contended), sum(shape)
            except CapacityError as error:
                capacity = error
                if not reclaimed:
                    self.model.reclaim()
                    reclaimed = True
                    continue
                if len(identities) > 1:
                    identities.pop()
                    last_shape = None
                    continue
                if allowance > 1 and last_shape != tuple(len(r.tokens) for r in ready):
                    last_shape = tuple(len(r.tokens) for r in ready)
                    allowance = max(1, allowance // 2)
                    continue
                if self._evict(set(identities), error):
                    last_shape = None
                    continue
                r = self.requests[current]
                # A peer completion/publication/cancellation can change capacity.
                # Suppress retries until the service epoch changes.
                peers = tuple(
                    p.identity
                    for p in self.requests.values()
                    if p.identity != current
                    and p.finish is None
                    and (
                        p.blocked_epoch != self._epoch
                        or (p.generation is not None and bool(p.generation.output))
                    )
                )
                if peers:
                    r.blocked_epoch = self._epoch
                    r.capacity_wait = CapacityWait(
                        required_bytes=error.required, available_bytes=error.available, peers=peers
                    )
                else:
                    self._fail(r, FailureKind.CAPACITY, error)
                raise _NotPrepared from error
            except Exception as error:
                self._fail(self.requests[current], FailureKind.PREPARATION, error)
                raise _NotPrepared from error

    def _complete(self) -> None:
        pending = self.pending
        assert pending is not None and pending.submission.completion.done
        elapsed = max(0, clock_ns() - pending.started_ns)
        try:
            pending.batch.finish()
        except Exception as error:
            for identity in pending.submission.requests:
                r = self.requests[identity]
                if r.generation is not None and r.generation.finish_reason == FinishReason.FAILED:
                    self._fail(r, FailureKind.EXECUTION, error)
        finally:
            pending.batch.close()
            self.pending = None
        self.scheduler.completed(pending.selection, elapsed)
        share, extra = divmod(elapsed, len(pending.submission.requests))
        now = clock_ns()
        for index, identity in enumerate(pending.submission.requests):
            r = self.requests[identity]
            attributed = share + (index < extra)
            r.service_ns += attributed
            kind = pending.batch.works[index].kind
            if kind == WorkKind.PREFILL:
                r.prefill_ns += attributed
            elif kind == WorkKind.DECODE:
                r.decode_ns += attributed
            else:
                r.replay_ns += attributed
            r.waiting_since_ns = now
            g = r.generation
            if g is not None:
                if r.protected_position is not None and g.processed > r.protected_position:
                    r.protected_position = None
                    r.preemption_debt = max(0, r.preemption_debt - 1)
                if g.finish_reason is not None:
                    r.finish = g.finish_reason
                    g.retire()
                    r.retire_source()
        self._epoch += 1

    def fail(self, error: Exception) -> None:
        """Fail this execution owner's dependent continuations, retaining output."""
        self.check()
        for request in self.requests.values():
            if request.finish is None:
                self._fail(request, FailureKind.EXECUTION, error)
        if self.pending is not None:
            self.pending.batch.close()
            self.pending = None

    def step(self) -> Submission | Idle:
        """Perform bounded host work; an unfinished submission is an explicit wait."""
        self.check()
        try:
            self.context.check()
        except Exception as error:
            self.fail(error)
            return Idle(tuple(self.snapshot(identity) for identity in self.requests))
        if self.pending is not None:
            if not self.pending.submission.completion.done:
                return self.pending.submission
            self._complete()
        # Failed/deferred requests are removed from this epoch's candidates.
        for _ in range(len(self.requests) + 1):
            candidates = tuple(
                c for r in self.requests.values() if (c := self._candidate(r)) is not None
            )
            selection = self.scheduler.select(candidates, clock_ns())
            if selection is None:
                return Idle(tuple(self.snapshot(identity) for identity in self.requests))
            try:
                batch, selected, tokens = self._prepare(selection)
            except _NotPrepared:
                continue
            started = clock_ns()
            try:
                ticket = self.context.submit(batch.commands)
                batch.submitted(ticket)
            except Exception as error:
                batch.close()
                for identity in selected.requests:
                    self._fail(self.requests[identity], FailureKind.EXECUTION, error)
                try:
                    self.context.check()
                except Exception as context_error:
                    for request in self.requests.values():
                        if request.finish is None:
                            self._fail(request, FailureKind.EXECUTION, context_error)
                    return Idle(tuple(self.snapshot(identity) for identity in self.requests))
                continue
            submission = Submission(ticket, selected.requests, selected.phase, tokens)
            self.pending = Pending(batch, selected, started, submission)
            return submission
        return Idle(tuple(self.snapshot(identity) for identity in self.requests))

    def close(self) -> None:
        self.context.check_thread()
        if not self.closed:
            for r in self.requests.values():
                if r.generation is not None:
                    r.generation.close()
                r.retire_source()
            if self.pending is not None:
                self.pending.batch.close()
                self.pending = None
            self.requests.clear()
            self.closed = True
