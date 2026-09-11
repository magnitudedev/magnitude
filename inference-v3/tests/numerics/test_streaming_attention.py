"""Logical fragments preserve streaming attention's physical-run contract."""

import os
from contextlib import ExitStack

import numpy as np
import pytest

from magnitude_engine.numerics.metal_attention import run_attention
from magnitude_engine.platform.backend import Backend
from magnitude_engine.platform.execution import DType, Prepared, TensorSpec
from magnitude_engine.platform.machine import open_context
from performance.precision import encode, rounded


@pytest.mark.device
@pytest.mark.parametrize("rows", [19, 37])
@pytest.mark.parametrize("dtype", [DType.F32, DType.BF16])
@pytest.mark.parametrize("width", [40, 256])
@pytest.mark.parametrize("partitions", [1, 4])
def test_streaming_fragments_mask_tails_empty_runs_and_partition_statistics(
    rows, dtype, width, partitions
):
    if os.environ.get("MAGNITUDE_TEST_BACKEND", "metal") != "metal":
        pytest.skip("Metal streaming schedule")
    heads, kv_heads, capacity, span = 4, 2, 384, 128
    rng = np.random.default_rng(384)
    q = rounded(rng.normal(size=(rows, heads, width)).astype(np.float32) * 3, dtype)
    k = np.full((capacity, kv_heads, width), np.nan, np.float32)
    v = np.full_like(k, np.nan)
    runs = np.array([[128, 0, 65], [0, 80, 39], [256, 200, 0]], np.int32)
    for physical, _, length in runs:
        k[physical : physical + length] = rounded(
            rng.normal(size=(length, kv_heads, width)).astype(np.float32) * 3, dtype
        )
        v[physical : physical + length] = rounded(
            rng.normal(size=(length, kv_heads, width)).astype(np.float32), dtype
        )
    positions = np.linspace(-1, 150, rows, dtype=np.int32)
    expected = np.zeros((len(runs) * partitions, rows, heads, width), np.float64)
    statistics = np.zeros((*expected.shape[:-1], 2), np.float64)
    statistics[..., 0] = np.float32(-3.402823466e38)
    chunk_span = ((span + partitions * 32 - 1) // (partitions * 32)) * 32
    for segment, (physical, logical, length) in enumerate(runs):
        for partition in range(partitions):
            partial = segment * partitions + partition
            first = partition * chunk_span
            for row, position in enumerate(positions):
                end = min(length, first + chunk_span, max(0, position - logical + 1))
                if end <= first:
                    continue
                for head in range(heads):
                    kh = head // (heads // kv_heads)
                    scores = (
                        k[physical + first : physical + end, kh].astype(np.float64)
                        @ q[row, head].astype(np.float64)
                    ) / np.sqrt(width)
                    weights = np.exp(scores - scores.max())
                    expected[partial, row, head] = (
                        weights @ v[physical + first : physical + end, kh] / weights.sum()
                    )
                    statistics[partial, row, head] = scores.max(), weights.sum()
    with ExitStack() as cleanup:
        context = open_context(Backend.METAL, 16 * 1024**2, 0)
        cleanup.callback(context.close)
        inputs = []
        for array, storage in (
            (q, dtype),
            (k, dtype),
            (v, dtype),
            (positions, DType.I32),
            (runs, DType.I32),
        ):
            content = array.tobytes() if storage == DType.I32 else encode(array, storage)
            tensor = context.upload(TensorSpec(array.shape, storage), content)
            cleanup.callback(tensor.close)
            inputs.append(tensor)
        output = context.allocate(TensorSpec(expected.shape, DType.F32))
        stats = context.allocate(TensorSpec(statistics.shape, DType.F32))
        cleanup.callback(output.close)
        cleanup.callback(stats.close)
        kernel = context.compile(
            run_attention(
                rows,
                heads,
                kv_heads,
                width,
                capacity,
                segments=len(runs),
                segment_capacity=span,
                partitions=partitions,
                dtype=dtype,
            )
        )
        ticket = context.submit([Prepared(context, kernel, [*inputs, output, stats])])
        actual = np.frombuffer(context.read(output, after=ticket), np.float32).reshape(
            expected.shape
        )
        actual_stats = np.frombuffer(context.read(stats, after=ticket), np.float32).reshape(
            statistics.shape
        )
        np.testing.assert_allclose(actual_stats, statistics, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(
            actual, expected, rtol=5e-5, atol=6e-3 if dtype == DType.BF16 else 2e-5
        )
        np.testing.assert_array_equal(actual[statistics[..., 1] == 0], 0)
