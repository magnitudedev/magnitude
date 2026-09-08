"""Semantic state must survive the same transactions as decoder state."""

from contextlib import ExitStack
from dataclasses import dataclass

import mlx.core as mx
import pytest

from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.runtime import ForwardRequest
from tests.models.test_model_runtime import hybrid


@dataclass(frozen=True)
class Coordinates:
    values: mx.array

    def prefix(self, count):
        return Coordinates(self.values[:, :count])


@dataclass
class Context:
    cache_hits = 0

    shift: int
    block: tuple[int, int] = (1, 4)
    closed: bool = False

    def acquire(self, position, count):
        return ExitStack()

    def boundary(self, position):
        return not self.block[0] < position < self.block[1]

    def prepare(self, position, count):
        yield from ()

    def assemble(self, inputs, position):
        return ModelInputs(
            inputs.tokens,
            inputs.conditioning,
            Coordinates(mx.arange(position, position + inputs.count)[None] + self.shift),
        )

    def batch_key(self, position, count):
        return Context

    def checkpoint(self, position):
        return Snapshot(self.shift, self.block)

    def close(self):
        self.closed = True


@dataclass
class Snapshot:
    shift: int
    block: tuple[int, int]
    closed: bool = False
    reclaimable = True

    def retained_storage(self):
        return ()

    def restore(self):
        assert not self.closed
        return Context(self.shift, self.block)

    def close(self):
        self.closed = True


@dataclass(frozen=True)
class Source:
    shift: int = 3

    def bind(self, checkpoint):
        return Context(self.shift) if checkpoint is None else checkpoint.restore()


def semantic_model():
    runtime, budget = hybrid()
    program = runtime.program
    forward = program.forward

    def interpreted(inputs, state, request, scope):
        assert isinstance(inputs.data, Coordinates)
        transformed = ModelInputs((inputs.tokens + inputs.data.values).astype(mx.int32))
        return forward(transformed, state, request, scope)

    program.forward = interpreted
    return runtime, budget


def test_input_semantics_survive_checkpoint_and_rejected_recurrent_replay():
    runtime, budget = semantic_model()
    row = runtime.create(inputs=Source())
    runtime.prefill(row, (1,))
    runtime.prefill(row, (1, 1, 1))
    checkpoint = row.checkpoint()
    context = row.inputs
    row.close()
    assert context.closed and not checkpoint.inputs.closed
    restored = runtime.create(checkpoint)
    checkpoint.close()
    assert restored.position == 4 and not restored.inputs.closed
    # The replay must use the original coordinate payload, not reconstruct
    # positions from tentative state or lose the checkpoint's continuation shift.
    advance = runtime.forward(restored, (1, 1, 1))
    advance.accept(1)
    assert restored.position == restored.state.position == 5
    assert restored.state.caches[1][0].item() == sum(range(4, 9))
    runtime.prefill(restored, (1,))
    assert restored.state.caches[1][0].item() == sum(range(4, 10))
    restored.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_independent_boundaries_guard_forward_commit_checkpoint_and_rewind():
    runtime, budget = semantic_model()
    row = runtime.create(inputs=Source())
    runtime.prefill(row, (1,))
    with pytest.raises(ValueError, match="boundaries"):
        runtime.forward(row, (1,))
    with pytest.raises(ValueError, match="boundaries"):
        runtime.forward(row, (1, 1, 1), ForwardRequest(committed_inputs=1))
    advance = runtime.forward(row, (1, 1, 1))
    with pytest.raises(ValueError, match="boundary"):
        advance.accept(1)
    assert row.pending is advance and not row.failed
    advance.accept(3)
    assert row.position == 4
    row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_model_checkpoint_rejects_other_residency_and_closed_handles():
    runtime, budget = semantic_model()
    peer, peer_budget = semantic_model()
    row = runtime.create(inputs=Source())
    runtime.prefill(row, (1,))
    checkpoint = row.checkpoint()
    with pytest.raises(ValueError, match="residency"):
        peer.create(checkpoint)
    checkpoint.close()
    with pytest.raises(ValueError, match="closed"):
        runtime.create(checkpoint)
    row.close()
    runtime.owner.close()
    peer.owner.close()
    assert budget.snapshot().reserved == peer_budget.snapshot().reserved == 0
