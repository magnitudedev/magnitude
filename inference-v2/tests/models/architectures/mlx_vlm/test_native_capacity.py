from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.cache import KVCache, RotatingKVCache

from magnitude_engine.models.architectures.mlx_vlm.loading import native_capacity
from magnitude_engine.models.architectures.mlx_vlm.program import LibraryProgram
from magnitude_engine.models.execution import ExecutionOwner
from magnitude_engine.models.runtime import ModelRuntime
from magnitude_engine.models.state.native import LibraryStateStore
from magnitude_engine.resources.budget import MemoryBudget


def arguments():
    return SimpleNamespace(
        num_key_value_heads=1, head_dim=4, hidden_size=4, num_attention_heads=1,
    )


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
@pytest.mark.parametrize("rotating", [False, True])
def test_forecast_bounds_real_cache_across_query_widths_and_rotation(dtype, rotating):
    cache = RotatingKVCache(max_size=17, keep=2) if rotating else KVCache()
    capacity = native_capacity(arguments(), [cache])
    position = 0
    for count in (1, 1, 3, 16, 1, 1, 32, 2, 1, 257, 1, 512, 3, 1):
        position += count
        forecast = capacity(position, count)
        keys = mx.full((1, 1, count, 4), position, dtype)
        cache.update_and_fetch(keys, keys + 1)
        mx.eval(cache.keys, cache.values)
        assert cache.nbytes <= forecast
    if rotating:
        # Retained admission capacity plateaus; a wide forward is still charged
        # for its complete causal window, even after a long prior history.
        assert capacity(1_000_000, 0) == 17 * 2 * 4 * 4
        assert capacity(1_000_000, 512) >= (17 + 511) * 2 * 4 * 4
    else:
        assert capacity(1_000_000, 0) >= 1_000_000 * 2 * 4 * 4


def test_rotation_reserves_extensions_and_same_size_replacement_before_forward():
    budget = MemoryBudget(1 << 20)
    capacity = native_capacity(arguments(), [RotatingKVCache(max_size=4)])
    store = LibraryStateStore(lambda: [RotatingKVCache(max_size=4)], budget, capacity)
    calls = []

    def call(tokens, caches):
        calls.append(budget.snapshot())
        values = mx.broadcast_to(tokens.astype(mx.float32).reshape(1, 1, -1, 1),
                                 (1, 1, tokens.shape[1], 4))
        keys, _ = caches[0].update_and_fetch(values, values + 1)
        return keys[:, :, -1]

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    try:
        runtime.reserve(row, 100_000)
        assert budget.snapshot().reserved == 128
        runtime.prefill(row, (1, 2, 3, 4))
        budget.limit = 287  # 128 old + 160 new, including the multi-query extension.
        with pytest.raises(MemoryError):
            runtime.prefill(row, (5, 6))
        assert len(calls) == 1 and row.state.position == 4 and not row.failed
        assert budget.snapshot().reserved == 128
        budget.limit = 288
        runtime.prefill(row, (5, 6))
        assert calls[-1].reserved == 288
        assert row.state.caches[0].keys.shape[2] == 5

        # Another same-width update replaces storage even though capacity is flat.
        budget.limit = 319
        with pytest.raises(MemoryError, match="library-cache-growth"):
            runtime.prefill(row, (7, 8))
        assert len(calls) == 2 and row.state.position == 6 and not row.failed
        assert budget.snapshot().reserved == 160
        budget.limit = 320
        runtime.prefill(row, (7, 8))
        assert calls[-1].owners["library-cache-growth"] == 160
        assert mx.array_equal(row.state.caches[0].keys[0, 0, :, 0], mx.arange(4, 9)).item()

        # Single-token decode must also cover the trim of an oversized window.
        runtime.prefill(row, (9,))
        assert calls[-1].owners["library-cache-growth"] == 160
        assert row.state.caches[0].keys.shape[2] == 4
        assert budget.snapshot().reserved == 160
    finally:
        row.close()
        runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_prepaid_high_water_mark_does_not_hide_physical_append_replacement():
    budget = MemoryBudget(4096)
    store = LibraryStateStore(lambda: [KVCache()], budget, lambda end, query: 4096)
    calls = []

    def call(tokens, caches):
        calls.append(budget.snapshot().reserved)
        values = tokens.astype(mx.float32).reshape(1, 1, -1, 1)
        keys, _ = caches[0].update_and_fetch(values, values + 1)
        return keys[:, :, -1]

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    try:
        runtime.prefill(row, tuple(range(256)))
        assert row.state.caches[0].nbytes == 2048
        # The forecast is flat, but the next token replaces 256-slot buffers.
        budget.limit = 6143
        with pytest.raises(MemoryError, match="library-cache-growth"):
            runtime.prefill(row, (256,))
        assert len(calls) == 1 and row.state.position == 256
        budget.limit = 6144
        runtime.prefill(row, (256,))
        assert calls[-1] == 6144
        assert row.state.caches[0].nbytes == 4096
    finally:
        row.close()
        runtime.owner.close()
    assert budget.snapshot().reserved == 0


def test_checkpoint_retains_oversized_window_and_rollback_restores_it():
    budget = MemoryBudget(1 << 20)
    capacity = native_capacity(arguments(), [RotatingKVCache(max_size=4)])
    store = LibraryStateStore(lambda: [RotatingKVCache(max_size=4)], budget, capacity)

    def call(tokens, caches):
        values = mx.broadcast_to(tokens.astype(mx.float32).reshape(1, 1, -1, 1),
                                 (1, 1, tokens.shape[1], 4))
        keys, _ = caches[0].update_and_fetch(values, values + 1)
        return keys[:, :, -1]

    runtime = ModelRuntime(LibraryProgram(call), store, ExecutionOwner())
    row = runtime.create()
    runtime.prefill(row, tuple(range(32)))
    checkpoint = row.checkpoint()
    branch = runtime.create(checkpoint)
    try:
        assert capacity(32, 0) == 128
        assert branch.state.capacity_bytes == 32 * 32
        original = mx.array(branch.state.caches[0].keys)
        advance = runtime.forward(branch, (40, 41))
        advance.accept(0)
        assert branch.state.position == 32
        assert mx.array_equal(branch.state.caches[0].keys, original).item()
        runtime.prefill(row, (33,))
        assert mx.array_equal(branch.state.caches[0].keys, original).item()
    finally:
        branch.close()
        checkpoint.close()
        row.close()
        runtime.owner.close()
    assert budget.snapshot().reserved == 0
