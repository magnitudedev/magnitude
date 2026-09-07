"""Drive independent generation continuations through compatible ready operations.

Forward operations stay lazy. Only explicit observations wait for device results;
there is no all-request draft/verify stage barrier or per-forward fence.
"""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from time import perf_counter_ns
from typing import cast

import mlx.core as mx

from magnitude_engine.components import component
from magnitude_engine.models.operations import (
    Complete,
    Forward,
    Observe,
    Operation,
    ProjectVocabulary,
    Repair,
    Response,
    Submit,
    Task,
)
from magnitude_engine.models.runtime import ForwardRequest

from .constraint_spec import ConstraintError


@dataclass
class Continuation[T]:
    task: Task[T]
    ready: Operation | None = None
    result: T | MemoryError | ConstraintError | None = None
    done: bool = False
    started: bool = False
    elapsed_ns: int = 0
    batch_size: int = 1

    def resume(self, value: Response, clock: Callable[[], int]) -> None:
        start = clock()
        self.started = True
        try:
            self.ready = self.task.send(value)
        except StopIteration as stop:
            self.result, self.done, self.ready = stop.value, True, None
        except (MemoryError, ConstraintError) as error:
            self.result, self.done, self.ready = error, True, None
        finally:
            self.elapsed_ns += clock() - start

    def close(self) -> None:
        self.task.close()


def execute[T](
    tasks: tuple[Task[T], ...], *, clock: Callable[[], int] = perf_counter_ns
) -> tuple[Continuation[T], ...]:
    rows = tuple(Continuation(task) for task in tasks)
    while any(not row.done for row in rows):
        serve(rows, clock=clock)
    return rows


@component("BATCHING:ASSEMBLY:MAG:READY_COMPATIBLE")
def serve[T](
    rows: tuple[Continuation[T], ...],
    *,
    clock: Callable[[], int] = perf_counter_ns,
    budget_ns: int | None = None,
) -> None:
    """Return ready results; retain peers at a completed service boundary.

    The deadline is soft: an indivisible device operation may overrun it. No
    timing fence is inserted between dependent forwards. Bounded round widths
    cap lookahead until the next required observation or service completion.
    """
    if budget_ns is not None and budget_ns < 1:
        raise ValueError("execution service budget must be positive")
    started = clock()
    completed = sum(row.done for row in rows)
    lifetime = ExitStack()
    spans = {}
    sequences = set()
    try:
        for row in rows:
            if not row.started:
                row.resume(None, clock)
        while any(not row.done for row in rows) and sum(row.done for row in rows) == completed:
            ready = tuple(row for row in rows if not row.done)
            observations = tuple(row for row in ready if isinstance(row.ready, Observe))
            submissions = tuple(row for row in ready if isinstance(row.ready, Submit))
            completions = tuple(row for row in ready if isinstance(row.ready, Complete))
            groups: dict[tuple, list[Continuation[T]]] = {}
            repairs: dict[tuple, list[Continuation[T]]] = {}
            projections: dict[tuple, list[Continuation[T]]] = {}
            for row in ready:
                op = row.ready
                if isinstance(op, Forward):
                    sequences.add(op.sequence)
                    groups.setdefault(
                        (id(op.sequence.runtime), op.inputs.count, op.request.committed_inputs), []
                    ).append(row)
                elif isinstance(op, Repair):
                    sequences.add(op.advance.sequence)
                    repairs.setdefault(
                        (
                            id(op.advance.sequence.runtime),
                            op.inputs.count,
                            id(op.advance.sequence.runtime.repair_group(op.advance.sequence)),
                        ),
                        [],
                    ).append(row)
                elif isinstance(op, ProjectVocabulary):
                    projections.setdefault(
                        (id(op.project), op.hidden.shape[1:], op.hidden.dtype), []
                    ).append(row)
            # Construct all ready neural graphs before satisfying host observations.
            # A verifier can finish while another request is still drafting.
            for group in groups.values():
                _forward(group, clock)
            for group in projections.values():
                calls = tuple(
                    row.ready for row in group if isinstance(row.ready, ProjectVocabulary)
                )
                start = clock()
                hidden = (
                    calls[0].hidden
                    if len(calls) == 1
                    else mx.concatenate([op.hidden for op in calls])
                )
                logits = calls[0].project(hidden)
                elapsed = clock() - start
                for index, row in enumerate(group):
                    row.elapsed_ns += elapsed
                    row.resume(logits if len(calls) == 1 else logits[index : index + 1], clock)
            for group in repairs.values():
                calls = tuple(row.ready for row in group if isinstance(row.ready, Repair))
                start = clock()
                try:
                    calls[0].advance.sequence.runtime.replay(
                        tuple(op.advance for op in calls), tuple(op.inputs for op in calls)
                    )
                except MemoryError as error:
                    elapsed = clock() - start
                    for row in group:
                        row.elapsed_ns += elapsed
                        row.result, row.done, row.ready = error, True, None
                    continue
                elapsed = clock() - start
                for row in group:
                    row.elapsed_ns += elapsed
                    row.resume(None, clock)
            if submissions:
                executions = {}
                for row in submissions:
                    op = row.ready
                    assert isinstance(op, Submit)
                    executions.setdefault(op.execution, []).extend(op.consumers)
                start = clock()
                for execution, consumers in executions.items():
                    owner = execution.owner
                    if execution.done:
                        owner.backend.submit(tuple(consumers))
                        continue
                    if owner not in spans or spans[owner].closed:
                        spans[owner] = lifetime.enter_context(owner.span())
                    spans[owner].submit(execution, *consumers)
                elapsed = clock() - start
                for row in submissions:
                    row.elapsed_ns += elapsed
                    row.resume(None, clock)
            if observations:
                start = clock()
                mx.eval(
                    *(
                        a
                        for row in observations
                        if isinstance(row.ready, Observe)
                        for a in row.ready.arrays
                    )
                )
                elapsed = clock() - start
                for row in observations:
                    row.elapsed_ns += elapsed
                    row.resume(None, clock)
            # Physical completion is shared, while subsequent state publication
            # remains row-local. Do not charge unrelated execution groups to peers.
            completion_groups = {}
            for row in completions:
                assert isinstance(row.ready, Complete)
                completion_groups.setdefault(row.ready.execution, []).append(row)
            for execution, group in completion_groups.items():
                start = clock()
                execution.complete()
                elapsed = clock() - start
                for row in group:
                    row.elapsed_ns += elapsed
                    row.resume(None, clock)
            if budget_ns is not None and clock() - started >= budget_ns:
                break
    except BaseException:
        for row in rows:
            row.close()
        raise
    finally:
        try:
            lifetime.close()
            for sequence in sequences:
                if sequence.pending is not None:
                    sequence.pending.complete()
                sequence.complete_committed()
        finally:
            for row in rows:
                if row.done:
                    row.close()


def _forward[T](rows: list[Continuation[T]], clock: Callable[[], int]) -> None:
    operations = tuple(row.ready for row in rows)
    assert all(isinstance(op, Forward) for op in operations)
    calls = tuple(op for op in operations if isinstance(op, Forward))
    runtime = calls[0].sequence.runtime
    if len(rows) > 1 and not runtime.can_batch(tuple(op.sequence for op in calls)):
        for row in rows:
            _forward([row], clock)
        return
    start = clock()
    # Output demands are additive, not execution incompatibilities. In particular,
    # a grammar-forced row need not split from a peer that needs sampled logits.
    # Keep causal commitments equal: they govern the state's reconciliation work.
    request = ForwardRequest(
        any(op.request.logits for op in calls),
        frozenset(feature for op in calls for feature in op.request.features),
        calls[0].request.committed_inputs,
    )
    for attempt in range(2):
        try:
            advances = (
                (runtime.forward(calls[0].sequence, calls[0].inputs, request),)
                if len(calls) == 1
                else runtime.forward_batch(
                    tuple(op.sequence for op in calls),
                    tuple(op.inputs for op in calls),
                    request,
                )
            )
            break
        except MemoryError as error:
            recoverable = all(
                not op.sequence.failed and op.sequence.pending is None for op in calls
            )
            if recoverable and attempt == 0:
                # Preparation rolled back before neural execution. Retire earlier
                # committed work before retrying this unchanged ready operation;
                # normal forwards keep their submission overlap and never fence here.
                retired = False
                for op in calls:
                    retired |= op.sequence.complete_committed()
                if retired:
                    continue
            if recoverable and len(rows) > 1:
                elapsed = clock() - start
                for row in rows:
                    row.elapsed_ns += elapsed
                    _forward([row], clock)
                return
            elapsed = clock() - start
            for row in rows:
                row.elapsed_ns += elapsed
                row.result, row.done, row.ready = error, True, None
            return
    elapsed = clock() - start
    for row, advance in zip(rows, advances, strict=True):
        row.elapsed_ns += elapsed
        row.batch_size = max(row.batch_size, len(rows))
        row.resume(advance, clock)


def run[T](task: Task[T]) -> T:
    result = execute((task,))[0].result
    if isinstance(result, (MemoryError, ConstraintError)):
        raise result
    return cast(T, result)
