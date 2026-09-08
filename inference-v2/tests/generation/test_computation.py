from contextlib import ExitStack
from dataclasses import dataclass

import mlx.core as mx
import pytest

from magnitude_engine.generation.execution import Continuation, execute, serve
from magnitude_engine.models.computation import evaluate
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.operations import Complete, compute
from magnitude_engine.resources.budget import MemoryBudget


def test_preparation_capacity_splits_before_graph_construction_without_draining():
    budget = MemoryBudget(1)
    owner = ExecutionOwner()
    calls = []

    class BoundedWork(Work):
        def reserve(self, rows):
            return budget.reserve("work", len(rows))

    rows = execute(tuple(task(BoundedWork(owner, value, calls)) for value in (2, 3)))
    assert calls == [(2,), (3,)]
    assert [row.result for row in rows] == [4, 9]
    owner.close()
    assert budget.snapshot().reserved == 0


@dataclass
class Work:
    owner: ExecutionOwner
    value: int
    calls: list

    @property
    def batch_key(self):
        return id(self.calls)

    def run_batch(self, rows, scope):
        self.calls.append(tuple(row.value for row in rows))
        values = mx.array([row.value for row in rows]) ** 2
        return tuple((values[i : i + 1],) for i in range(len(rows)))

    def reserve(self, rows):
        return ExitStack()


def task(work):
    result = yield from compute(work)
    yield Complete(result.execution)
    return result.arrays[0].item()


def test_stateless_dependencies_group_and_continue_without_decoder_state():
    owner = ExecutionOwner()
    calls = []
    results = execute(tuple(task(Work(owner, value, calls)) for value in (2, 3)))
    assert [row.result for row in results] == [4, 9]
    assert calls == [(2, 3)]
    assert all(row.batch_size == 2 for row in results)
    owner.close()


def test_stateless_work_yields_at_service_deadline_and_retains_owned_completion():
    owner = ExecutionOwner()
    calls = []
    row = Continuation(task(Work(owner, 3, calls)))
    time = iter(range(100))
    serve((row,), clock=lambda: next(time), budget_ns=1)
    assert not row.done and isinstance(row.ready, Complete)
    execution = row.ready.execution
    assert execution.done  # Physical preparation is charged to this service quantum.
    serve((row,))
    assert row.result == 9 and execution.done
    owner.close()


def test_stateless_batch_rejects_unrelated_owners_before_execution():
    first, second = ExecutionOwner(), ExecutionOwner()
    calls = []
    with pytest.raises(ValueError, match="compatible"):
        evaluate((Work(first, 1, calls), Work(second, 2, calls)))
    assert not calls
    first.close()
    second.close()
