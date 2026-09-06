from itertools import pairwise

import mlx.core as mx
import pytest

from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import PageStore, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.resources.budget import MemoryBudget


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("head_sharing", [1, 2, 4])
@pytest.mark.parametrize(
    "prefix,count,dk,dv,fragmented",
    [
        (0, 1, 32, 64, False),
        (127, 3, 64, 32, True),
        (130, 8, 128, 128, False),
        (1023, 1, 256, 256, True),
        (4093, 3, 256, 256, True),
        (257, 3, 512, 512, True),
    ],
)
def test_paged_attention_matches_causal_sdpa_with_poisoned_unused_storage(
    dtype,
    head_sharing,
    prefix,
    count,
    dk,
    dv,
    fragmented,
):
    mx.random.seed(131)
    total = prefix + count
    pages = (total + 15) // 16
    budget = MemoryBudget(256 << 20)
    arena = KVArena(
        (LayerGeometry(2, dk, dv),),
        page_size=16,
        slab_pages=4,
        max_pages=max(4, 2 * pages + 4),
        budget=budget,
        dtype=dtype,
    )
    guards = ()
    if fragmented:
        allocated = arena.allocate(2 * pages)
        guards = allocated[::2]
        arena.release(allocated[1::2])
    state = PageStore(arena).create()
    state.reserve(total)
    if fragmented:
        assert any(b != a + 1 for a, b in pairwise(state.addresses))
    arena.keys = tuple(mx.full(a.shape, float("nan"), dtype) for a in arena.keys)
    arena.values = tuple(mx.full(a.shape, float("nan"), dtype) for a in arena.values)
    keys = mx.random.normal((1, 2, total, dk)).astype(dtype)
    values = mx.random.normal((1, 2, total, dv)).astype(dtype)
    queries = mx.random.normal((1, 8, count, dk)).astype(dtype)
    # Include sharp distributions to exercise stable online-softmax merging.
    if prefix == 130:
        queries = queries * 25
    if prefix:
        state.write(0, 0, keys[0, :, :prefix], values[0, :, :prefix])
        state.commit(prefix)
    append_layer((state,), 0, keys[:, :, prefix:], values[:, :, prefix:])
    actual = MetalPagedAttention(heads_per_group=head_sharing).compute(
        queries, read_layer((state,), 0, pending_tokens=count), dk**-0.5
    )
    mask = mx.arange(total)[None, :] <= (prefix + mx.arange(count))[:, None]
    # Unequal key/value widths can route the library to BF16 intermediate
    # matmuls. Qualify the FP32 accumulation contract against FP32 attention,
    # rounded once at output, rather than inheriting that fallback's rounding.
    expected = mx.fast.scaled_dot_product_attention(
        queries.astype(mx.float32),
        keys.astype(mx.float32),
        values.astype(mx.float32),
        scale=dk**-0.5,
        mask=mask,
    ).astype(dtype)
    mx.eval(actual, expected)
    tolerance = 1e-5 if dtype == mx.float32 else 2e-3
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item(), mx.max(
        mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
    ).item()
    state.commit(total)
    state.close()
    arena.release(guards)
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("operator", ["native", "gathered"])
@pytest.mark.parametrize("prefixes,count", [((0, 129), 1), ((3, 130, 1, 1023), 3)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_attention_batches_distinct_positions_and_shared_checkpoint(
    operator, prefixes, count, dtype
):
    from magnitude_engine.models.attention.gathered import GatheredAttention

    mx.random.seed(63)
    budget = MemoryBudget(64 << 20)
    arena = KVArena(
        (LayerGeometry(2, 32, 64),),
        page_size=16,
        slab_pages=8,
        max_pages=160,
        budget=budget,
        dtype=dtype,
    )
    store = PageStore(arena)
    states = tuple(store.create() for _ in prefixes)
    histories = []
    for state, prefix in zip(states, prefixes, strict=True):
        state.reserve(prefix + count)
        k = mx.random.normal((2, prefix + count, 32)).astype(dtype)
        v = mx.random.normal((2, prefix + count, 64)).astype(dtype)
        if prefix:
            state.write(0, 0, k[:, :prefix], v[:, :prefix])
            state.commit(prefix)
        histories.append((k, v))
    checkpoint = states[-1].checkpoint()
    branch = store.create(checkpoint)
    branch.reserve(prefixes[-1] + count)
    # A branch shares full pages and owns its append boundary independently.
    states = (*states, branch)
    prefixes = (*prefixes, prefixes[-1])
    histories.append(histories[-1])
    queries = mx.random.normal((len(states), 8, count, 32)).astype(dtype)
    keys = mx.stack([k[:, -count:] for k, _ in histories])
    values = mx.stack([v[:, -count:] for _, v in histories])
    op = MetalPagedAttention(heads_per_group=2) if operator == "native" else GatheredAttention()
    append_layer(states, 0, keys, values)
    actual = op.compute(queries, read_layer(states, 0, pending_tokens=count), 32**-0.5)
    expected = []
    oracle_dtype = mx.float32 if operator == "native" else dtype
    for row, ((k, v), prefix) in enumerate(zip(histories, prefixes, strict=True)):
        mask = mx.arange(prefix + count)[None, :] <= (prefix + mx.arange(count))[:, None]
        expected.append(
            mx.fast.scaled_dot_product_attention(
                queries[row : row + 1].astype(oracle_dtype),
                k[None].astype(oracle_dtype),
                v[None].astype(oracle_dtype),
                scale=32**-0.5,
                mask=mask,
            ).astype(dtype)
        )
    expected = mx.concatenate(expected)
    mx.eval(actual, expected)
    tolerance = 1e-5 if dtype == mx.float32 else 2e-3
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
    for state, prefix in zip(states, prefixes, strict=True):
        state.commit(prefix + count)
        state.close()
    checkpoint.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("operator", ["native", "gathered"])
@pytest.mark.parametrize("count,window", [(1, 1), (3, 7), (8, 128), (33, 17)])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_read_only_window_attention_excludes_poisoned_past_and_can_share_kv(
    operator, count, window, dtype
):
    from magnitude_engine.models.attention.gathered import GatheredAttention

    mx.random.seed(562)
    budget = MemoryBudget(32 << 20)
    arena = KVArena(
        (LayerGeometry(2, 32, 32),),
        page_size=16,
        slab_pages=8,
        max_pages=160,
        budget=budget,
        dtype=dtype,
    )
    store = PageStore(arena)
    prefixes = (0, 69, 1029)
    states = tuple(store.create() for _ in prefixes)
    histories = []
    for state, prefix in zip(states, prefixes, strict=True):
        length = prefix + count
        state.reserve(length)
        k = mx.random.normal((2, length, 32)).astype(dtype)
        v = mx.random.normal(k.shape).astype(dtype)
        start = max(0, prefix + 1 - window)
        k[:, :start] = float("nan")
        v[:, :start] = float("nan")
        if prefix:
            state.write(0, 0, k[:, :prefix], v[:, :prefix])
            state.commit(prefix)
        histories.append((k, v))
    with pytest.raises(ValueError, match="not been staged"):
        read_layer(states, 0, pending_tokens=count)
    append_layer(
        states,
        0,
        mx.stack([k[:, -count:] for k, _ in histories]),
        mx.stack([v[:, -count:] for _, v in histories]),
    )
    view = read_layer(states, 0, pending_tokens=count)
    writes = arena.counters.copy()
    for heads in (4, 8):  # Different consuming layers reuse the same producer without appending.
        q = mx.random.normal((len(states), heads, count, 32)).astype(dtype)
        op = MetalPagedAttention() if operator == "native" else GatheredAttention()
        actual = op.compute(q, view, 32**-0.5, window=window)
        expected = []
        oracle_dtype = mx.float32 if operator == "native" and count <= 8 else dtype
        for row, ((k, v), prefix) in enumerate(zip(histories, prefixes, strict=True)):
            start = max(0, prefix + 1 - window)
            key_at = mx.arange(start, prefix + count)[None]
            query_at = (prefix + mx.arange(count))[:, None]
            mask = (key_at <= query_at) & (key_at > query_at - window)
            expected.append(
                mx.fast.scaled_dot_product_attention(
                    q[row : row + 1].astype(oracle_dtype),
                    k[None, :, start:].astype(oracle_dtype),
                    v[None, :, start:].astype(oracle_dtype),
                    scale=32**-0.5,
                    mask=mask,
                )
            )
        expected = mx.concatenate(expected)
        if operator == "native" and count <= 8 and dtype == mx.bfloat16:
            from dataclasses import replace

            fp32 = op.compute(
                q.astype(mx.float32),
                replace(
                    view, keys=view.keys.astype(mx.float32), values=view.values.astype(mx.float32)
                ),
                32**-0.5,
                window=window,
            )
            assert mx.allclose(fp32, expected, atol=1e-5, rtol=1e-5).item()
            # FP32 reductions can straddle an exact BF16 rounding midpoint. Check
            # the actual quantization bound against unrounded FP32, not equality
            # to one reduction order's rounded result.
            spacing = mx.power(2.0, mx.floor(mx.log2(mx.maximum(mx.abs(expected), 1e-30))) - 7)
            assert mx.all(mx.abs(actual.astype(mx.float32) - expected) <= spacing / 2 + 1e-5).item()
        else:
            tolerance = 1e-5 if dtype == mx.float32 else 2e-3
            assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()
        assert arena.counters == writes
        assert tuple(state.length for state in states) == prefixes
    for state, prefix in zip(states, prefixes, strict=True):
        state.commit(prefix + count)
        state.close()
    arena.close()
    assert budget.snapshot().reserved == 0
