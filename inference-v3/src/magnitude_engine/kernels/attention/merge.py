"""Combine disjoint logical runs from their softmax sufficient statistics."""

import tilelang.language as T

from magnitude_engine.kernels.capabilities import Capability
from magnitude_engine.kernels.launch import serial
from magnitude_engine.platform.execution import DType


def merge_runs(
    rows: int,
    heads: int,
    width: int,
    runs: int,
    *,
    capability: Capability,
    threads: int = 128,
    output_dtype: DType = DType.F32,
):
    """Combine disjoint logical runs using their softmax sufficient statistics.

    This is not a candidate: it is the fixed second stage that any realization
    producing more than one partial uses.
    """
    if min(rows, heads, width, runs, threads) <= 0:
        raise ValueError("invalid attention merge geometry")
    cpu = serial(capability)

    @T.macro
    def merge(Values, Statistics, Output, row, h, d):
        maximum = T.alloc_local((1,), "float32")
        denominator = T.alloc_local((1,), "float32")
        answer = T.alloc_local((1,), "float32")
        maximum[0] = -3.402823466e38
        denominator[0] = 0
        answer[0] = 0
        for run in T.serial(runs):
            if Statistics[run, row, h, 1] > 0:
                maximum[0] = T.max(maximum[0], Statistics[run, row, h, 0])
        for run in T.serial(runs):
            if Statistics[run, row, h, 1] > 0:
                weight = T.exp(Statistics[run, row, h, 0] - maximum[0]) * Statistics[run, row, h, 1]
                denominator[0] += weight
                answer[0] += weight * Values[run, row, h, d]
        Output[row, h, d] = T.if_then_else(denominator[0] > 0, answer[0] / denominator[0], 0)

    @T.prim_func
    def main(
        Values: T.Tensor((runs, rows, heads, width), "float32"),
        Statistics: T.Tensor((runs, rows, heads, 2), "float32"),
        Output: T.Tensor((rows, heads, width), output_dtype.value),
    ):
        if cpu:
            for row, h, d in T.Parallel(rows, heads, width):
                merge(Values, Statistics, Output, row, h, d)
        else:
            with T.Kernel(T.ceildiv(width, threads), heads, rows, threads=threads) as (
                block,
                h,
                row,
            ):
                d = block * threads + T.get_thread_binding(0)
                if d < width:
                    merge(Values, Statistics, Output, row, h, d)

    return main
