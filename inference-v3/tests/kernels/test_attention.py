import os

import numpy as np
import pytest

from magnitude_engine.kernels.attention.merge import merge_runs
from magnitude_engine.kernels.attention.portable import run_attention
from magnitude_engine.kernels.attention.serial import run_attention as run_attention_cpu
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.host.machine import open_context


@pytest.mark.device
@pytest.mark.parametrize("width,visible", [(40, 63), (256, 73), (256, 0)])
@pytest.mark.parametrize("head_tile", [1, 2])
def test_causal_run_attention_masked_rows_partial_tiles_and_online_statistics(
    width, visible, head_tile
):
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    if backend == Backend.LLVM and head_tile != 1:
        pytest.skip("CPU uses its scalar head schedule")
    rows, heads, kvheads, capacity = 19, 4, 2, 79
    rng = np.random.default_rng(384)
    q = rng.normal(size=(rows, heads, width)).astype(np.float32) * 3
    k = rng.normal(size=(capacity, kvheads, width)).astype(np.float32) * 3
    v = rng.normal(size=k.shape).astype(np.float32)
    start = 11
    positions = np.linspace(0, start + capacity + 20, rows, dtype=np.int32)
    span = np.array([[0, start, visible]], np.int32)
    expected = np.zeros_like(q, dtype=np.float64)
    expected_stats = np.zeros((rows, heads, 2), dtype=np.float64)
    expected_stats[:, :, 0] = np.float32(-3.402823466e38)
    for row in range(rows):
        count = min(visible, max(0, int(positions[row]) - start + 1))
        if count == 0:
            continue
        for head in range(heads):
            kv = head // (heads // kvheads)
            scores = (
                k[:count, kv].astype(np.float64) @ q[row, head].astype(np.float64) / np.sqrt(width)
            )
            weights = np.exp(scores - scores.max())
            expected[row, head] = weights @ v[:count, kv].astype(np.float64) / weights.sum()
            expected_stats[row, head] = (scores.max(), weights.sum())
    context = open_context(backend, 16 * 1024**2, 0)
    inputs = [
        context.upload(TensorSpec(a.shape, dtype), a.tobytes())
        for a, dtype in (
            (q, DType.F32),
            (k, DType.F32),
            (v, DType.F32),
            (positions, DType.I32),
            (span, DType.I32),
        )
    ]
    output = context.allocate(TensorSpec((1, *q.shape), DType.F32))
    stats = context.allocate(TensorSpec((1, *expected_stats.shape), DType.F32))
    factory = run_attention_cpu if backend == Backend.LLVM else run_attention
    program = (
        factory(rows, heads, kvheads, width, capacity)
        if backend == Backend.LLVM
        else factory(rows, heads, kvheads, width, capacity, head_tile=head_tile)
    )
    kernel = context.compile(program)
    ticket = context.submit([Prepared(context, kernel, [*inputs, output, stats])])
    actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(q.shape)
    actual_stats = np.frombuffer(context.read(stats, after=ticket), np.float32).reshape(
        expected_stats.shape
    )
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=8e-6)
    np.testing.assert_allclose(actual_stats, expected_stats, rtol=1e-5, atol=1e-5)
    for tensor in [*inputs, output, stats]:
        tensor.close()
    assert context.allocated_bytes == 0
    context.close()


@pytest.mark.device
@pytest.mark.parametrize("capacity", [97, 65536])
@pytest.mark.parametrize("partitions", [1, 4])
@pytest.mark.parametrize("head_tile", [1, 2])
def test_independent_physical_runs_merge_to_contiguous_causal_attention(
    capacity, partitions, head_tile
):
    backend = Backend(os.environ.get("MAGNITUDE_TEST_BACKEND", "metal"))
    if backend == Backend.LLVM and (partitions != 1 or head_tile != 1):
        pytest.skip("CPU schedule uses one partial per physical run")
    rows, heads, kh = 4, 4, 2
    width = 256 if capacity == 65536 else 64
    rng = np.random.default_rng(973)
    q = rng.normal(size=(rows, heads, width)).astype(np.float32)
    k = rng.normal(size=(capacity, kh, width)).astype(np.float32)
    v = rng.normal(size=k.shape).astype(np.float32)
    positions = np.array([-1, 7, capacity // 2 - 1, capacity - 1], np.int32)
    expected = np.zeros_like(q, dtype=np.float64)
    for row, position in enumerate(positions):
        if position < 0:
            continue
        for head in range(heads):
            kv = head // (heads // kh)
            score = (
                k[: position + 1, kv].astype(np.float64)
                @ q[row, head].astype(np.float64)
                / np.sqrt(width)
            )
            probability = np.exp(score - score.max())
            expected[row, head] = probability @ v[: position + 1, kv] / probability.sum()
    context = open_context(backend, k.nbytes + v.nbytes + 16 * 1024**2, 0)
    resources = []

    def upload(array, dtype=DType.F32):
        tensor = context.upload(TensorSpec(array.shape, dtype), array.tobytes())
        resources.append(tensor)
        return tensor

    query, coordinates = upload(q), upload(positions, DType.I32)
    values = context.allocate(TensorSpec((3 * partitions, rows, heads, width), DType.F32))
    statistics = context.allocate(TensorSpec((3 * partitions, rows, heads, 2), DType.F32))
    output = context.allocate(TensorSpec(q.shape, DType.F32))
    resources.extend((values, statistics))
    commands = []
    for index, (start, end) in enumerate(((0, 31), (31, 67), (67, capacity))):
        keys, vals = upload(k[start:end]), upload(v[start:end])
        span = upload(np.array([[0, start, end - start]], np.int32), DType.I32)
        result_spec = TensorSpec((partitions, *q.shape), DType.F32)
        stats_spec = TensorSpec((partitions, rows, heads, 2), DType.F32)
        result_view = values.view(result_spec, index * result_spec.nbytes)
        stats_view = statistics.view(stats_spec, index * stats_spec.nbytes)
        args = (rows, heads, kh, width, end - start)
        program = (
            run_attention_cpu(*args)
            if backend == Backend.LLVM
            else run_attention(*args, partitions=partitions, head_tile=head_tile)
        )
        kernel = context.compile(program)
        commands.append(
            Prepared(
                context, kernel, [query, keys, vals, coordinates, span, result_view, stats_view]
            )
        )
        result_view.close()
        stats_view.close()
    merger = context.compile(
        merge_runs(rows, heads, width, 3 * partitions, capability=context.capability)
    )
    commands.append(Prepared(context, merger, [values, statistics, output]))
    for resource in resources:
        resource.close()
    ticket = context.submit(commands)
    actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(q.shape)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    output.close()
    assert context.allocated_bytes == 0
    context.close()
