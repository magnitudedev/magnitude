"""Physical read coalescing does not change logical attention or extent lifetimes."""

import os
from contextlib import ExitStack

import numpy as np
import pytest

from magnitude_engine.kernels.precision import REFERENCE_F32
from magnitude_engine.models.qwen35.arena import Arena
from magnitude_engine.operations.attention import CausalAttention, KVAppend
from magnitude_engine.operations.kv_binding import KVBinding
from magnitude_engine.operations.preparation import Preparation
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, TensorSpec
from magnitude_engine.platform.host.machine import open_context
from magnitude_engine.state.kv import KVLayout, KVPool, KVSpan, KVWrite, physical_runs
from performance.attention import attention_error_bound
from performance.precision import decode, encode, rounded


@pytest.mark.device
@pytest.mark.parametrize("first_length", [100, 128])
def test_allocations_coalesce_only_without_a_physical_visibility_gap(first_length):
    rng = np.random.default_rng(184)
    length, heads, kv_heads, width = 211, 4, 2, 16
    keys = rng.normal(size=(length, kv_heads, width)).astype(np.float32)
    values = rng.normal(size=keys.shape).astype(np.float32)
    queries = rng.normal(size=(2, heads, width)).astype(np.float32)
    positions = (200, 210)
    scores = np.einsum(
        "rhd,thd->rht", queries.astype(np.float64), keys[:, [0, 0, 1, 1]].astype(np.float64)
    ) / np.sqrt(width)
    scores = np.where(
        np.arange(length)[None, None, :] <= np.array(positions)[:, None, None], scores, -np.inf
    )
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("rht,thd->rhd", probabilities, values[:, [0, 0, 1, 1]])
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")),
            16 * 1024**2,
            0,
        )
        cleanup.callback(context.close)
        pool = KVPool(context, KVLayout(1, kv_heads, width), slab_pages=4)
        cleanup.callback(pool.close)
        (first,) = pool.reserve(128)
        cleanup.callback(first.close)
        (second,) = pool.reserve(128, after=first)
        cleanup.callback(second.close)
        history = physical_runs(
            (
                KVSpan(first, 0, first_length),
                KVSpan(second, first_length, length - first_length),
            )
        )
        assert len(history) == (1 if first_length == 128 else 2)
        writes = (
            KVWrite(first, 0, 0, first_length),
            KVWrite(second, 0, first_length, length - first_length),
        )
        append = KVAppend(context, kv_heads, width)
        arena = Arena(context, REFERENCE_F32)
        cleanup.callback(arena.close)
        attention = CausalAttention(context, heads, kv_heads, width, REFERENCE_F32, arena)
        cleanup.callback(append.close)
        cleanup.callback(attention.close)
        output = context.allocate(TensorSpec(queries.shape, DType.F32))
        cleanup.callback(output.close)
        with Preparation(context) as p:
            k = p.upload(TensorSpec(keys.shape, DType.F32), keys.tobytes())
            v = p.upload(TensorSpec(values.shape, DType.F32), values.tobytes())
            q = p.upload(TensorSpec(queries.shape, DType.F32), queries.tobytes())
            position = p.indices(positions)
            binding = KVBinding(p, history, writes)
            p.add(*append.prepare(k, v, binding.appends(0)))
            p.add(*attention.prepare(q, position, binding.reads(0), output))
            commands = p.finish()
            assert len(binding.groups[0].runs) == (1 if first_length == 128 else 2)
        first.close()
        second.close()
        ticket = context.submit(commands)
        actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(
            queries.shape
        )
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=5e-6)
        assert pool.slab_count == 0
        output.close()
        attention.close()
        arena.close()
        assert context.allocated_bytes == 0


@pytest.mark.device
@pytest.mark.parametrize("rows", [3, 17])
@pytest.mark.parametrize("dtype", [DType.F32, DType.BF16])
def test_grouped_reads_mask_nan_gaps_and_nonmonotonic_physical_placement(rows, dtype):
    rng = np.random.default_rng(428)
    heads, kh, width = 4, 2, 40
    q = rng.normal(size=(rows, heads, width)).astype(np.float32)
    positions = np.linspace(-1, 180, rows, dtype=np.int32)
    # Logical order reverses physical placement. Both allocation tails and the
    # intervening peer contain NaNs, so reading any gap corrupts the answer.
    keys = [rng.normal(size=(n, kh, width)).astype(np.float32) for n in (65, 89)]
    values = [rng.normal(size=k.shape).astype(np.float32) for k in keys]
    q = rounded(q, dtype)
    keys, values = ([rounded(a, dtype) for a in arrays] for arrays in (keys, values))
    logical = np.concatenate((np.arange(65), np.arange(80, 169)))
    k, v = np.concatenate(keys), np.concatenate(values)
    expected = np.zeros_like(q, dtype=np.float64)
    magnitude = np.zeros_like(expected)
    for row, position in enumerate(positions):
        visible = logical <= position
        if not visible.any():
            continue
        for head in range(heads):
            kv = head // (heads // kh)
            score = k[visible, kv].astype(np.float64) @ q[row, head] / np.sqrt(width)
            probability = np.exp(score - score.max())
            expected[row, head] = probability @ v[visible, kv] / probability.sum()
            magnitude[row, head] = probability @ np.abs(v[visible, kv]) / probability.sum()
    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")), 16 * 1024**2, 0
        )
        cleanup.callback(context.close)
        pool = KVPool(context, KVLayout(1, kh, width, dtype), slab_pages=3)
        cleanup.callback(pool.close)
        first, peer, last = (pool.reserve(128)[0] for _ in range(3))
        for run in (first, peer, last):
            cleanup.callback(run.close)
        history = physical_runs((KVSpan(last, 0, 65), KVSpan(first, 80, 89)))
        arena = Arena(context, REFERENCE_F32)
        cleanup.callback(arena.close)
        append = KVAppend(context, kh, width)
        attention = CausalAttention(context, heads, kh, width, REFERENCE_F32, arena)
        cleanup.callback(append.close)
        cleanup.callback(attention.close)
        output = context.allocate(TensorSpec(q.shape, dtype))
        cleanup.callback(output.close)
        with Preparation(context) as p:
            for index, run in enumerate((last, first, peer)):
                padded = [np.full((128, kh, width), np.nan, np.float32) for _ in range(2)]
                if index < 2:
                    padded[0][: len(keys[index])] = keys[index]
                    padded[1][: len(values[index])] = values[index]
                operands = [p.upload(TensorSpec(a.shape, dtype), encode(a, dtype)) for a in padded]
                binding = KVBinding(p, (), (KVWrite(run, 0, 0, 128),))
                p.add(*append.prepare(*operands, binding.appends(0)))
            query = p.upload(TensorSpec(q.shape, dtype), encode(q, dtype))
            position = p.upload(TensorSpec(positions.shape, DType.I32), positions.tobytes())
            reads = attention.prepare(query, position, KVBinding(p, history, ()).reads(0), output)
            p.add(*reads)
            commands = p.finish()
        for run in (first, peer, last):
            run.close()
        ticket = context.submit(commands)
        actual = decode(context.read(output, after=ticket), dtype).reshape(q.shape)
        error = np.abs(actual - expected)
        assert np.isfinite(actual).all()
        assert np.max(error / attention_error_bound(expected, magnitude, dtype, rows)) <= 1
        np.testing.assert_array_equal(actual[0], 0)
        assert pool.slab_count == 0
        output.close()
        # Reusable numerical scratch belongs to the arena, not to the KV runs.
        attention.close()
        arena.close()
        assert context.allocated_bytes == 0


@pytest.mark.device
def test_attention_scratch_reuses_backing_and_pins_growth_until_completion():
    from performance.attention import AttentionCase, AttentionWorkload

    with ExitStack() as cleanup:
        context = open_context(
            Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal")), 16 * 1024**2, 0
        )
        cleanup.callback(context.close)
        arena = Arena(context, REFERENCE_F32)
        cleanup.callback(arena.close)
        attention = CausalAttention(context, 4, 2, 40, REFERENCE_F32, arena)
        cleanup.callback(attention.close)
        cases = tuple(
            AttentionCase(
                attention,
                AttentionWorkload(rows=3, history_tokens=length, seed=seed, physical_groups=groups),
            )
            for length, seed, groups in ((256, 8, 2), (256, 19, 2), (1024, 42, 4))
        )
        for case in cases:
            cleanup.callback(case.close)
        commands = []
        for index, case in enumerate(cases):
            before = context.allocated_bytes
            commands.extend(
                attention.prepare(case.queries, case.positions, case.history, case.output)
            )
            if index == 1:
                # A second complete invocation needs no second scratch allocation.
                assert context.allocated_bytes == before
        # Growing scratch must preserve operands in the earlier prepared work;
        # releasing the arena must preserve every submitted consumer's backing.
        attention.close()
        arena.close()
        ticket = context.submit(tuple(commands))
        ticket.wait()
        assert all(case.validate(ticket).passed for case in cases)
        for case in cases:
            case.close()
        assert context.allocated_bytes == 0
