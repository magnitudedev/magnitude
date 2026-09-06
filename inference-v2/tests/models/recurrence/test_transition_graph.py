"""Tensor tracing cannot absorb row ownership or require single-step repair images."""

import mlx.core as mx
import pytest

from magnitude_engine.models.recurrence.gated_delta import DeltaTransition
from magnitude_engine.models.state.recurrent import RecurrentBoundaries
from tests.models.architectures.qwen35.test_hybrid_model import setup
from tests.models.recurrence.fixtures import recurrent_slots


@pytest.mark.parametrize("batch", [1, 3])
@pytest.mark.parametrize("known_prefix", [False, True])
def test_compiled_graph_preserves_each_invocations_state_boundaries(batch, known_prefix):
    _, runtime, arena, budget = setup(bits=4, dtype=mx.bfloat16)
    operation = runtime.program.blocks[0].mixer.operation
    layout = operation.layout(mx.bfloat16)
    try:
        # Repeat shapes to hit the cached graph, then change width and return to
        # decode. Every invocation must stage the newly supplied logical slots.
        for width in (1, 1, 4, 4, 1):
            hidden = mx.random.normal((batch, width, 64)).astype(mx.bfloat16)
            initial = tuple(
                tuple(mx.random.normal(t.shape).astype(t.dtype) * 0.1 for t in layout.tensors)
                for _ in range(batch)
            )
            with recurrent_slots(layout, initial, budget) as slots:
                conv, memory = (mx.concatenate([values[i] for values in initial]) for i in range(2))
                expected = operation.graph(hidden, conv, memory)
                with runtime.owner.scope() as scope:
                    actual = operation.compute_batch(
                        hidden,
                        slots,
                        scope,
                        committed_inputs=width if known_prefix else 0,
                    )
                    scope.seal(
                        actual, *(a for slot in slots for a in slot.pending.values)
                    ).complete()
                assert mx.allclose(actual, expected[0], atol=2e-5, rtol=2e-5).item()
                for row, slot in enumerate(slots):
                    transition = slot.pending
                    assert isinstance(
                        transition,
                        RecurrentBoundaries if width == 1 or known_prefix else DeltaTransition,
                    )
                    assert transition.prefix(0) is initial[row]
                    assert transition.prefix(width) is transition.values
                    assert mx.allclose(
                        transition.values[1], expected[-1][row : row + 1], atol=2e-5, rtol=2e-5
                    ).item()
                    with pytest.raises(ValueError):
                        transition.prefix(width + 1)
                    with pytest.raises(ValueError):
                        transition.prefix(-1)
    finally:
        runtime.owner.close()
        arena.close()
    assert budget.snapshot().reserved == 0
