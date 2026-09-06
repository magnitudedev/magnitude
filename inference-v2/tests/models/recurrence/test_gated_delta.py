import mlx.core as mx
import pytest
from mlx_lm.models.cache import ArraysCache

from tests.models.architectures.qwen35.test_hybrid_model import setup
from tests.models.recurrence.fixtures import recurrent_slots


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("batch,count", [(1, 1), (3, 4), (1, 32)])
def test_gated_delta_matches_library_with_low_precision_state_and_multiple_rows(
    dtype, batch, count
):
    model, runtime, arena, budget = setup(bits=4, dtype=dtype)
    operation = runtime.program.blocks[0].mixer.operation
    layout = operation.layout(dtype)
    mx.random.seed(271)
    hidden = mx.random.normal((batch, count, 64)).astype(dtype)
    initial = tuple(
        tuple(mx.random.normal(spec.shape).astype(spec.dtype) * 0.1 for spec in layout.tensors)
        for _ in range(batch)
    )
    with recurrent_slots(layout, initial, budget) as slots:
        with runtime.owner.scope() as scope:
            actual = operation.compute_batch(hidden, slots, scope)
            scope.seal(actual, *(a for slot in slots for a in slot.pending.values)).complete()
        cache = ArraysCache(2)
        cache.state = [mx.concatenate([values[i] for values in initial]) for i in range(2)]
        expected = model.layers[0].linear_attn(hidden, cache=cache)
        tolerance = 2e-5
        assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
        for row, slot in enumerate(slots):
            for actual_state, expected_state in zip(slot.pending.values, cache.state, strict=True):
                assert mx.allclose(
                    actual_state, expected_state[row : row + 1], atol=tolerance, rtol=tolerance
                ).item()
            assert slot.pending.prefix(0) is initial[row]
            assert all(v.shape[0] == 1 for v in slot.pending.values)
    runtime.owner.close()
    arena.close()
    assert budget.snapshot().reserved == 0
