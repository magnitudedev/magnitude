from itertools import pairwise

import mlx.core as mx
import pytest

from magnitude_engine.models.attention.metal import MetalPagedAttention
from magnitude_engine.models.state.arena import KVArena, LayerGeometry
from magnitude_engine.models.state.pages import PageStore, append_layer
from magnitude_engine.models.state.views import read_layer
from magnitude_engine.resources.budget import MemoryBudget


@pytest.mark.parametrize("window", [3, 32])
def test_image_block_local_visibility_matches_independent_rectangular_mask(window):
    mx.random.seed(72)
    budget = MemoryBudget(1 << 20)
    arena = KVArena(
        (LayerGeometry(1, 32, 32),),
        page_size=4,
        slab_pages=4,
        max_pages=16,
        budget=budget,
        dtype=mx.float32,
    )
    store = PageStore(arena)
    states = (store.create(), store.create())
    count, prefixes = 6, (3, 7)
    ends, histories = [], []
    for state, prefix in zip(states, prefixes, strict=True):
        total = prefix + count
        state.reserve(total)
        keys, values = mx.random.normal((1, total, 32)), mx.random.normal((1, total, 32))
        state.write(0, 0, keys[:, :prefix], values[:, :prefix])
        state.commit(prefix)
        append_layer((state,), 0, keys[None, :, prefix:], values[None, :, prefix:])
        histories.append((keys, values))
        # Four soft image tokens surrounded by ordinary text.
        ends.append([prefix + 1, prefix + 5, prefix + 5, prefix + 5, prefix + 5, total])
    queries = mx.random.normal((2, 2, count, 32))
    actual = MetalPagedAttention().compute(
        queries,
        read_layer(states, 0, pending_tokens=count),
        32**-0.5,
        window=window,
        key_ends=mx.array(ends, mx.int32),
    )
    expected = []
    for index, ((keys, values), prefix) in enumerate(zip(histories, prefixes, strict=True)):
        mask = mx.array(
            [
                [
                    ((k <= q) or (prefix + 1 <= q < prefix + 5 and prefix + 1 <= k < prefix + 5))
                    and k > q - window
                    for k in range(prefix + count)
                ]
                for q in range(prefix, prefix + count)
            ]
        )
        expected.append(
            mx.fast.scaled_dot_product_attention(
                queries[index : index + 1], keys[None], values[None], scale=32**-0.5, mask=mask
            )
        )
    assert mx.allclose(actual, mx.concatenate(expected), atol=2e-6).item()
    for state in states:
        state.close()
    arena.close()
    assert budget.snapshot().reserved == 0


@pytest.mark.parametrize("batch", [2, 9])
def test_short_attention_reduction_does_not_depend_on_peer_count(batch):
    mx.random.seed(19)
    queries = mx.random.normal((batch, 32, 1, 256)).astype(mx.bfloat16)
    keys = mx.random.normal((2, 512, 256)).astype(mx.bfloat16)
    values = mx.random.normal((2, 512, 256)).astype(mx.bfloat16)
    mx.eval(queries, keys, values)
    operation = MetalPagedAttention()

    def apply(q):
        rows = q.shape[0]
        return operation.apply(
            q,
            keys,
            values,
            mx.broadcast_to(mx.array([[0, 1]], mx.int32), (rows, 2)),
            mx.full((rows,), 382, mx.int32),
            page_size=256,
            table_width=2,
            covered=512,
            scale=256**-0.5,
            window=None,
        )

    expected = mx.concatenate([apply(row[None]) for row in queries])
    assert mx.array_equal(apply(queries), expected).item()


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
        (8193, 8, 128, 256, True),
        (8193, 5, 512, 512, False),
        (257, 3, 512, 512, True),
        (129, 1, 64, 64, False),
        (1023, 1, 128, 128, False),
        (4093, 1, 256, 256, False),
        (257, 1, 512, 512, False),
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
    operation = MetalPagedAttention(heads_per_group=head_sharing)
    view = read_layer((state,), 0, pending_tokens=count)
    actual = operation.compute(queries, view, dk**-0.5)
    if prefix > 8192:
        padded = operation.apply(
            queries,
            view.keys,
            view.values,
            view.table.device,
            mx.array([prefix], mx.int32),
            page_size=view.page_size,
            table_width=view.table.width,
            covered=((total + 511) // 512) * 512,
            scale=dk**-0.5,
        )
        # Compiled capacity buckets must not alter reductions over identical logical KV.
        assert mx.array_equal(actual, padded).item()
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
@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("window", [1, 17, 129, None])
def test_single_query_views_exclude_poisoned_history_and_unused_physical_pages(
    operator, dtype, window
):
    from magnitude_engine.models.attention.gathered import GatheredAttention
    from magnitude_engine.models.state.table import PageMap, PageTable
    from magnitude_engine.models.state.views import PagedKV

    mx.random.seed(892)
    length, width, page_size = 133, 128, 16
    start = 0 if window is None else max(0, length - window)
    # The requested window is adjacent even though the earlier history is not.
    pages = (1, 4, 5, 6, 7, 8, 9, 10, 11)
    keys = mx.full((2, 14 * page_size, width), float("nan"), dtype)
    values = mx.full(keys.shape, float("nan"), dtype)
    logical_k = mx.random.normal((2, length - start, width)).astype(dtype)
    logical_v = mx.random.normal(logical_k.shape).astype(dtype)
    for index in range(start, length):
        physical = pages[index // page_size] * page_size + index % page_size
        keys[:, physical] = logical_k[:, index - start]
        values[:, physical] = logical_v[:, index - start]
    kv = PagedKV(keys, values, page_size, PageTable((PageMap(pages, 14),)), (length,))
    q = mx.random.normal((1, 8, 1, width)).astype(dtype)
    op = MetalPagedAttention() if operator == "native" else GatheredAttention()
    actual = op.compute(q, kv, width**-0.5, window=window)
    expected = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32),
        logical_k[None].astype(mx.float32),
        logical_v[None].astype(mx.float32),
        scale=width**-0.5,
    ).astype(dtype)
    tolerance = 1e-5 if dtype == mx.float32 else 2e-3
    assert mx.allclose(actual, expected, atol=tolerance, rtol=tolerance).item()


@pytest.mark.parametrize("operator", ["native", "gathered"])
@pytest.mark.parametrize(
    "prefixes,count,tail",
    [
        ((0, 129), 1, False),
        ((3, 130, 1, 1023), 3, False),
        ((1, 8193), 5, False),
        ((17, 65533), 5, True),
    ],
)
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
@pytest.mark.parametrize("query_heads", [2, 6, 16])
def test_attention_batches_distinct_positions_and_shared_checkpoint(
    operator, prefixes, count, tail, dtype, query_heads
):
    from magnitude_engine.models.attention.gathered import GatheredAttention

    mx.random.seed(63)
    budget = MemoryBudget(64 << 20)
    arena = KVArena(
        (LayerGeometry(2, 32, 64),),
        page_size=16,
        slab_pages=8,
        max_pages=max(160, sum((p + count + 15) // 16 for p in prefixes) + 8),
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
    queries = mx.random.normal((len(states), query_heads, count, 32)).astype(dtype)
    keys = mx.stack([k[:, -count:] for k, _ in histories])
    values = mx.stack([v[:, -count:] for _, v in histories])
    op = MetalPagedAttention(heads_per_group=2) if operator == "native" else GatheredAttention()
    append_layer(states, 0, keys, values)
    view = read_layer(states, 0, pending_tokens=count)
    if tail:
        from dataclasses import replace

        from magnitude_engine.models.state.views import AppendView

        physical_k, physical_v = mx.array(view.keys), mx.array(view.values)
        # A page-only peer shares the batch with bounded append buffers.
        tails: list[AppendView | None] = [None]
        for row in range(1, len(states)):
            start = prefixes[row] - 3
            k, v = histories[row]
            tk = mx.full((2, 512, 32), float("nan"), dtype)
            tv = mx.full((2, 512, 64), float("nan"), dtype)
            tk[:, : count + 3], tv[:, : count + 3] = k[:, start:], v[:, start:]
            tails.append(AppendView(tk, tv, start))
            for pos in range(start, prefixes[row] + count):
                address = view.pages[row][pos // 16] * 16 + pos % 16
                physical_k[:, address] = float("nan")
                physical_v[:, address] = float("nan")
        view = replace(view, keys=physical_k, values=physical_v, tails=tuple(tails))
    actual = op.compute(queries, view, 32**-0.5)
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
