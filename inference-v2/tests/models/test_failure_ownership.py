"""Device failure must not turn state allocations into reusable capacity."""

from types import SimpleNamespace

import pytest

from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest, ModelOutput, ModelRuntime
from magnitude_engine.resources.budget import MemoryBudget
from tests.models.test_execution import Backend


@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("phase", ["forward", "reconcile"])
@pytest.mark.parametrize("failed_drain", [False, True])
def test_state_ownership_survives_failure_until_drain(rows, phase, failed_drain):
    events = []
    budget = MemoryBudget(1024)
    backend = Backend(events, fail_drain=failed_drain)

    class Transaction:
        def __init__(self, state):
            self.state = state
            self.reservation = budget.reserve("tentative-state", 8)
            self.closed = False
            state.active = self

        def reconcile(self, accepted):
            events.append("repair-work")
            raise ValueError("state repair failed")

        def finish(self, accepted):
            raise AssertionError("failed work cannot commit")

        def close(self):
            if not self.closed:
                events.append("release-state")
                self.reservation.close()
                self.state.active = None
                self.closed = True

    class Store:
        def create(self, checkpoint=None):
            return SimpleNamespace(active=None)

        def begin(self, state, inputs, *, committed_inputs=0):
            return Transaction(state)

        def arrays(self, state):
            return ()

        def release(self, state):
            assert state.active is None

    class Program:
        features = conditioning = frozenset()

        def forward(self, inputs, state, request, scope):
            if phase == "forward":
                raise ValueError("program failed after starting device work")
            return ModelOutput()

        def forward_batch(self, inputs, states, request, scope):
            return self.forward(inputs[0], states[0], request, scope)

    runtime = ModelRuntime(Program(), Store(), ExecutionOwner(backend))
    sequences = tuple(runtime.create() for _ in range(rows))
    inputs = ModelInputs.from_tokens((1,))
    with pytest.raises(BaseExceptionGroup if failed_drain else ValueError):
        advances = (
            (runtime.forward(sequences[0], inputs, ForwardRequest(False)),)
            if rows == 1 else runtime.forward_batch(
                sequences, (inputs,) * rows, ForwardRequest(False),
            )
        )
        advances[0].accept(1)

    assert events.count("drain") == 1
    assert runtime.owner.requires_disposal == failed_drain
    if failed_drain:
        assert budget.snapshot().reserved == 8 * rows
        assert "release-state" not in events
        for sequence in sequences:
            with pytest.raises(RuntimeError, match="unavailable"):
                sequence.close()
        # This fixture owns only a host ledger, not real device storage. Simulate
        # process disposal after proving that normal cleanup cannot reclaim it.
        for sequence in sequences:
            sequence.state.active.close()
    else:
        for sequence in sequences:
            sequence.close()
        assert events.index("drain") < events.index("release-state")
        runtime.owner.close()
    assert budget.snapshot().reserved == 0
