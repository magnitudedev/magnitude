import mlx.core as mx
import pytest

from magnitude_engine.generation.execution import execute
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.inputs import ModelInputs
from magnitude_engine.models.operations import accept
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget
from tests.models.architectures.qwen35.test_hybrid_model import setup


@pytest.mark.parametrize('prefix_length', [0, 250])
@pytest.mark.parametrize('accepted', [(1, 3, 2), (2, 2, 2)])
def test_native_batch_matches_independent_attention_and_recurrence(accepted, prefix_length):
    model, unused, arena, unused_budget = setup()
    unused.owner.close()
    arena.close()
    assert unused_budget.snapshot().reserved == 0
    budget = MemoryBudget(64 << 20)
    states = LibraryStateStore(model.make_cache, budget, lambda n: ((n + 255) // 256) * 65536)
    physical = []
    def call(tokens, cache):
        physical.append(tokens.shape)
        return model(tokens, cache=cache)
    runtime = ModelRuntime(LibraryProgram(call), states, ExecutionOwner())
    prefix = tuple(1 + i % 16 for i in range(prefix_length))
    histories = [(*prefix, 1, 2), (*prefix, 3, 4, 5, 6), (*prefix, 7)]
    rows = tuple(runtime.create() for _ in histories)
    for row, history in zip(rows, histories, strict=True):
        runtime.prefill(row, history)
    proposals = ((8, 9, 10), (11, 12, 13), (14, 15, 16))
    advances = runtime.forward_batch(rows, tuple(ModelInputs.from_tokens(p) for p in proposals))
    for advance, history, proposal in zip(advances, histories, proposals, strict=True):
        advance.complete()
        expected = model(mx.array([[*history, *proposal]]), cache=model.make_cache())[:, -3:]
        assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
    execute(tuple(accept(a, count) for a, count in zip(advances, accepted, strict=True)))
    cohort = rows[0].state.batch
    assert all(row.state.batch is cohort for row in rows)
    if len(set(accepted)) == 1:
        assert physical[-1] == (3, 2), 'recurrent repair must physically batch'
    for i, count in enumerate(accepted):
        histories[i] = (*histories[i], *proposals[i][:count])
    # Unequal acceptance, subset execution, and checkpoint restoration must all
    # retain independent attention positions and recurrent state.
    checkpoint = rows[1].checkpoint()
    branch = runtime.create(checkpoint)
    for indices in ((0, 2), (1,), (0, 1, 2)):
        selected = tuple(rows[i] for i in indices)
        advances = runtime.forward_batch(selected, (ModelInputs.from_tokens((17,)),) * len(indices))
        for i, advance in zip(indices, advances, strict=True):
            advance.complete()
            histories[i] = (*histories[i], 17)
            expected = model(mx.array([histories[i]]), cache=model.make_cache())[:, -1:]
            assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
            advance.accept(1)
        assert all(row.state.batch is cohort for row in rows)
    advance = runtime.forward(branch, (20,))
    advance.complete()
    expected = model(mx.array([[*prefix, 3, 4, 5, 6, *proposals[1][:accepted[1]], 20]]),
                     cache=model.make_cache())[:, -1:]
    assert mx.allclose(advance.output.logits, expected, atol=1e-4, rtol=1e-4).item()
    advance.accept(1)
    branch.close()
    checkpoint.close()
    for row in rows:
        row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_repairs_respect_existing_physical_storage_groups():
    from mlx_lm.models.cache import ArraysCache, KVCache

    budget = MemoryBudget(1 << 20)
    shapes = []
    def call(tokens, caches):
        shapes.append(tokens.shape)
        keys = tokens[:, None, :, None].astype(mx.float32)
        caches[0].update_and_fetch(keys, keys)
        previous = caches[1][0]
        caches[1][0] = keys.sum(axis=2) + (0 if previous is None else previous)
        return keys[:, 0]
    runtime = ModelRuntime(LibraryProgram(call), LibraryStateStore(
        lambda: [KVCache(), ArraysCache(1)], budget, lambda n: 4096,
    ), ExecutionOwner())
    rows = tuple(runtime.create() for _ in range(4))
    for row in rows:
        runtime.prefill(row, (1,))
    pending = []
    for pair in (rows[:2], rows[2:]):
        pending.extend(runtime.forward_batch(pair, (ModelInputs.from_tokens((2, 3)),) * 2))
    before = len(shapes)
    execute(tuple(accept(advance, 1) for advance in pending))
    assert shapes[before:] == [(2, 1), (2, 1)]
    for row in rows:
        assert row.state.caches[1][0].item() == 3
        row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_departed_storage_is_reused_before_request_admission_needs_more_capacity():
    from mlx_lm.models.cache import ArraysCache, KVCache

    budget = MemoryBudget(12_288)
    def call(tokens, caches):
        keys = tokens[:, None, :, None].astype(mx.float32)
        caches[0].update_and_fetch(keys, keys)
        previous = caches[1][0]
        caches[1][0] = keys.sum(axis=2) + (0 if previous is None else previous)
        return keys[:, 0]
    runtime = ModelRuntime(LibraryProgram(call), LibraryStateStore(
        lambda: [KVCache(), ArraysCache(1)], budget, lambda n: 4096,
    ), ExecutionOwner())
    rows = tuple(runtime.create() for _ in range(2))
    advances = runtime.forward_batch(rows, (ModelInputs.from_tokens((2,)),) * 2)
    for advance in advances:
        advance.accept(1)
    cohort = rows[0].state.batch
    rows[0].close()
    before = budget.snapshot().reserved
    replacement = runtime.create()
    runtime.reserve(replacement, 4)
    assert replacement.state.batch is cohort
    assert budget.snapshot().reserved == before
    runtime.prefill(replacement, (9,))
    assert replacement.state.caches[1][0].item() == 9  # No departed recurrent state.
    assert rows[1].state.caches[1][0].item() == 2
    replacement.close()
    rows[1].close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0
    assert not runtime.states._batches


def test_regrouping_keeps_old_storage_charged_until_lazy_consumers_complete():
    from mlx_lm.models.cache import KVCache

    budget = MemoryBudget(1 << 20)
    def call(tokens, caches):
        keys = tokens[:, None, :, None].astype(mx.float32)
        caches[0].update_and_fetch(keys, keys)
        return keys[:, 0]
    runtime = ModelRuntime(LibraryProgram(call), LibraryStateStore(
        lambda: [KVCache()], budget, lambda n: 4096,
    ), ExecutionOwner())
    rows = tuple(runtime.create() for _ in range(3))
    advances = runtime.forward_batch(rows[:2], (ModelInputs.from_tokens((2,)),) * 2)
    old = rows[0].state.batch
    for advance in advances:
        advance.accept_all_lazily()
    runtime.states.prepare_batch(tuple(row.state for row in rows), 1)
    assert old.closed and old.charges  # Detached from logical rows, still borrowed by device work.
    before = budget.snapshot().reserved
    advances[0].complete()
    assert not old.charges
    assert budget.snapshot().reserved == before - 8192
    for row in rows:
        row.close()
    runtime.owner.close()
    assert budget.snapshot().reserved == 0
