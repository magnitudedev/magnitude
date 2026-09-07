"""Device-free preparation recovery through real model transactions and spans."""

from types import SimpleNamespace

import pytest

from magnitude_engine.generation.execution import execute
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.operations import Forward, Submit
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput, ModelRuntime
from magnitude_engine.resources.budget import MemoryBudget
from tests.models.test_execution import Backend


class Transaction:
    def __init__(self, state, reservation):
        self.state, self.reservation = state, reservation

    def commit_all(self):
        self.state.position += 1

    def close(self):
        self.reservation.close()


class Store:
    def __init__(self, limit):
        self.budget = MemoryBudget(limit)
        self.attempts = []
        self.errors = []

    def create(self, checkpoint=None):
        return SimpleNamespace(position=0)

    def can_batch(self, states):
        return True

    def repair_group(self, state):
        return self

    def prepare_batch(self, states, width):
        pass

    def begin(self, state, inputs, *, committed_inputs=0):
        self.attempts.append(state.position)
        try:
            charge = self.budget.reserve("tentative destination", 1)
        except MemoryError as error:
            self.errors.append(error)
            raise
        return Transaction(state, charge)

    def arrays(self, state):
        return ()

    def release(self, state):
        pass


class Program:
    features = conditioning = frozenset()

    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def forward(self, inputs, state, request, scope):
        return self.forward_batch((inputs,), (state,), request, scope)

    def forward_batch(self, inputs, states, request, scope):
        self.calls.append(tuple(state.position for state in states))
        if self.fail:
            raise MemoryError("failure after neural execution started")
        return ModelOutput()


def run(rows, limit, *, fail_program=False, fail_drain=False):
    events = []
    owner = ExecutionOwner(Backend(events, fail_drain=fail_drain))
    store, program = Store(limit), Program(fail=fail_program)
    runtime = ModelRuntime(program, store, owner)
    sequences = tuple(runtime.create() for _ in range(rows))
    inputs = SimpleNamespace(count=1, conditioning={})
    publications = []

    def continuation(row):
        for position in range(2):
            advance = yield Forward(
                sequences[row], inputs, ForwardRequest(logits=False, committed_inputs=1)
            )
            yield Submit(advance.execution, ())
            advance.accept_all_lazily()
            publications.append((row, position))
        return sequences[row].state.position

    tasks = tuple(continuation(row) for row in range(rows))
    return owner, store, program, sequences, events, publications, tasks


@pytest.mark.parametrize("rows", [1, 2])
def test_pressure_retires_shared_committed_span_and_retries_only_preparation(rows):
    owner, store, program, sequences, events, publications, tasks = run(rows, rows * 2 - 1)
    result = execute(tasks)
    assert [row.result for row in result] == [2] * rows
    assert program.calls == [(0,) * rows, (1,) * rows]
    assert sorted(publications) == [(row, pos) for row in range(rows) for pos in range(2)]
    assert len(store.errors) == 1
    assert events.count("drain") == 2
    assert store.budget.snapshot().reserved == 0
    for sequence in sequences:
        assert not sequence.failed and sequence.pending is None
        sequence.close()
    owner.close()


def test_without_pressure_all_forwards_keep_one_shared_retirement_fence():
    owner, store, program, sequences, events, publications, tasks = run(2, 4)
    assert [row.result for row in execute(tasks)] == [2, 2]
    assert not store.errors and events.count("drain") == 1
    assert len(publications) == 4 and len(program.calls) == 2
    for sequence in sequences:
        sequence.close()
    owner.close()


def test_without_outstanding_execution_original_preparation_error_is_not_retried():
    owner, store, program, sequences, events, publications, tasks = run(1, 0)
    result = execute(tasks)
    assert result[0].result is store.errors[0]
    assert store.attempts == [0]
    assert not program.calls and not publications and "drain" not in events
    sequences[0].close()
    owner.close()


def test_program_failure_is_terminal_and_never_retried():
    owner, store, program, sequences, events, publications, tasks = run(2, 4, fail_program=True)
    result = execute(tasks)
    assert all(isinstance(row.result, MemoryError) for row in result)
    assert program.calls == [(0, 0)]
    assert not publications and all(sequence.failed for sequence in sequences)
    assert store.budget.snapshot().reserved == 0
    for sequence in sequences:
        sequence.close()
    owner.close()


def test_failed_pressure_fence_preserves_unsafe_leases_and_never_retries():
    owner, store, program, sequences, events, publications, tasks = run(2, 3, fail_drain=True)
    with pytest.raises(RuntimeError, match="execution owner is unavailable"):
        execute(tasks)
    assert owner.requires_disposal
    assert program.calls == [(0, 0)]
    assert len(publications) == 2
    assert store.budget.snapshot().reserved == 2


def test_persistent_preparation_failure_retries_once_after_retirement():
    owner, store, program, sequences, events, publications, tasks = run(1, 1)
    begin = store.begin
    error = MemoryError("permanent capacity shortfall")
    failures = []

    def persistent(state, inputs, *, committed_inputs=0):
        if state.position:
            failures.append(state.position)
            raise error
        return begin(state, inputs, committed_inputs=committed_inputs)

    store.begin = persistent
    result = execute(tasks)
    assert result[0].result is error and failures == [1, 1]
    assert program.calls == [(0,)] and publications == [(0, 0)]
    assert events.count("drain") == 1
    assert store.budget.snapshot().reserved == 0
    sequences[0].close()
    owner.close()


def test_singleton_fallback_charges_failed_group_and_retirement_time_to_each_row():
    from magnitude_engine.generation.execution import Continuation, _forward

    class Clock:
        now = 0

        def __call__(self):
            return self.now

    clock = Clock()

    class Runtime:
        retired = False

        def can_batch(self, states):
            return True

        def forward_batch(self, *args):
            clock.now += 7
            raise MemoryError("group does not fit")

        def forward(self, *args):
            clock.now += 3
            return None

    runtime = Runtime()

    class Sequence:
        failed = False
        pending = None

        def __init__(self):
            self.runtime = runtime

        def complete_committed(self):
            if runtime.retired:
                return False
            clock.now += 50
            runtime.retired = True
            return True

    def task():
        yield Forward(Sequence(), SimpleNamespace(count=1), ForwardRequest())
        return "published once"

    rows = [Continuation(task()), Continuation(task())]
    for row in rows:
        row.resume(None, clock)
    _forward(rows, clock)
    assert [row.elapsed_ns for row in rows] == [67, 67]
    assert [row.result for row in rows] == ["published once", "published once"]
